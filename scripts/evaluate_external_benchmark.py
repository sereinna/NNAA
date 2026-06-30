#!/usr/bin/env python3
"""中文说明：外部文献验证集评估脚本，检测训练集重叠、模型打分并计算 top-k recovery 指标。"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cedg.data import CEDGCollatorWithESM, CEDGDataset, build_vocabs, collate_cedg, load_jsonl  # noqa: E402
from cedg.metrics import group_ndcg_at_k, group_pairwise_accuracy  # noqa: E402
from cedg.model import CEDGScoreModel  # noqa: E402
from cedg.plm import ESMBatchConverter, esm_model_spec, load_esm_model  # noqa: E402
from cedg.utils import move_to_device  # noqa: E402


TRAIN_DATASET = ROOT / "data" / "final" / "model_ready" / "peptide_component" / "cedg_score_dataset.jsonl"


def safe_spearman(values: list[float], scores: list[float]) -> float:
    if len(values) < 2 or len(set(values)) < 2 or len(set(scores)) < 2:
        return float("nan")
    value = spearmanr(values, scores).statistic
    return float(value) if not math.isnan(value) else float("nan")


def edit_summary(edit_set: list[dict[str, object]]) -> str:
    return " ; ".join(
        (
            f"site{event.get('site_index')}:"
            f"{event.get('original_monomer')}->{event.get('modified_monomer')}"
            f"|{event.get('final_edit_scope')}"
            f"|{event.get('attachment_type')}"
            f"|{event.get('final_chemical_payload')}"
        )
        for event in edit_set
    )


def record_key(record: dict[str, object], fields: tuple[str, ...]) -> str:
    return "|".join(str(record.get(field, "") or "") for field in fields)


def load_external_records(path: Path) -> list[dict[str, object]]:
    if path.suffix.lower() == ".jsonl":
        return load_jsonl(path)
    if path.suffix.lower() != ".csv":
        raise ValueError("External benchmark must be .jsonl or .csv.")
    table = pd.read_csv(path)
    records: list[dict[str, object]] = []
    json_columns = {"edit_set", "parent_monomer_list", "modified_monomer_list"}
    for row in table.to_dict(orient="records"):
        record = {}
        for key, value in row.items():
            if key in json_columns and isinstance(value, str) and value.strip():
                record[key] = json.loads(value)
            elif pd.isna(value):
                record[key] = ""
            else:
                record[key] = value
        records.append(record)
    return records


def add_candidate_group_id(record: dict[str, object]) -> None:
    if record.get("candidate_group_id"):
        return
    record["candidate_group_id"] = (
        f"{record.get('source_slug', 'external')}:"
        f"{record.get('assay_type', 'PAMPA')}:"
        f"parent_{record.get('parent_peptide_id', record.get('parent_name', 'unknown'))}:"
        f"shadow_{str(record.get('parent_shadow_sequence', '')).replace('.', '_').lower()}"
    )


def normalize_external_records(records: list[dict[str, object]]) -> list[dict[str, object]]:
    normalized: list[dict[str, object]] = []
    for index, record in enumerate(records, start=1):
        item = dict(record)
        item["split"] = "score"
        item.setdefault("sample_id", f"external_sample_{index:06d}")
        item.setdefault("source_id", item.get("source_slug", "external"))
        item.setdefault("source_slug", str(item.get("source_id", "external")).lower())
        item.setdefault("table_type", "external")
        item.setdefault("assay_type", "PAMPA")
        item.setdefault("sample_model_use_tier", "external_benchmark")
        item.setdefault("candidate_generation_strategy", "external_literature")
        item.setdefault("sample_weight", 1.0)
        item.setdefault("confidence_weight", 1.0)
        item.setdefault("censored_property_flag", False)
        item.setdefault("contains_manual_curation_monomer_final", False)
        item.setdefault("contains_residue_graph_replacement", False)
        item.setdefault("contains_unknown_replacement", False)
        item.setdefault("parent_smiles", "")
        item.setdefault("modified_smiles", "")
        item.setdefault("unordered_pair_id", "")
        item.setdefault("parent_peptide_id", item.get("parent_name", -1))
        item.setdefault("modified_peptide_id", item.get("modified_name", -1))
        item.setdefault("edit_count", len(item.get("edit_set", []) or []))
        item.setdefault("property_before", 0.0)
        item.setdefault("property_after", 0.0)
        item.setdefault("delta_property", float(item.get("property_after", 0.0)) - float(item.get("property_before", 0.0)))
        item.setdefault("direction_label", int(float(item.get("delta_property", 0.0)) > 0.0))
        item.setdefault("canonical_shadow_sequence_final", item.get("parent_shadow_sequence", ""))
        item.setdefault("parent_monomer_list", str(item.get("parent_shadow_sequence", "")).split("."))
        item.setdefault("modified_monomer_list", item.get("parent_monomer_list", []))
        add_candidate_group_id(item)
        normalized.append(item)
    return normalized


def overlap_report(train_records: list[dict[str, object]], external_records: list[dict[str, object]]) -> dict[str, object]:
    checks = {
        "sample_id": ("sample_id",),
        "unordered_pair_id": ("unordered_pair_id",),
        "modified_smiles": ("modified_smiles",),
        "parent_modified_smiles": ("parent_smiles", "modified_smiles"),
        "shadow_modified_name": ("parent_shadow_sequence", "modified_name"),
    }
    report: dict[str, object] = {}
    for name, fields in checks.items():
        train_keys = {
            record_key(record, fields)
            for record in train_records
            if all(str(record.get(field, "") or "") for field in fields)
        }
        external_keys = [
            record_key(record, fields)
            for record in external_records
            if all(str(record.get(field, "") or "") for field in fields)
        ]
        hits = sorted({key for key in external_keys if key in train_keys})
        report[f"{name}_overlap_count"] = len(hits)
        report[f"{name}_overlap_examples"] = hits[:10]
    return report


def load_model(args: argparse.Namespace, vocabs: object) -> tuple[CEDGScoreModel, object]:
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model_args = checkpoint.get("args", {})
    esm_model = None
    esm_repr_layer = None
    collate_fn = collate_cedg
    if args.use_live_esm:
        esm_model, alphabet, esm_repr_layer, esm_dim = load_esm_model(args.esm_model)
        collate_fn = CEDGCollatorWithESM(ESMBatchConverter(alphabet, max_length=args.max_plm_length))
    else:
        _, esm_dim = esm_model_spec(args.esm_model)
    model = CEDGScoreModel(
        vocabs=vocabs,
        emb_dim=int(model_args.get("emb_dim", 64)),
        hidden_dim=int(model_args.get("hidden_dim", 128)),
        dropout=float(model_args.get("dropout", 0.1)),
        peptide_layers=int(model_args.get("peptide_layers", 2)),
        edit_layers=int(model_args.get("edit_layers", 2)),
        num_heads=int(model_args.get("num_heads", 4)),
        max_len=int(model_args.get("max_len", 128)),
        esm_model=esm_model,
        esm_repr_layer=esm_repr_layer,
        esm_dim=esm_dim,
        freeze_esm=True,
    ).to(args.device)
    model.load_state_dict(checkpoint["model_state"], strict=False)
    model.eval()
    return model, collate_fn


def score_records(
    records: list[dict[str, object]],
    train_records: list[dict[str, object]],
    args: argparse.Namespace,
) -> pd.DataFrame:
    vocabs = build_vocabs(train_records)
    model, collate_fn = load_model(args, vocabs)
    graph_cache_dir = args.graph_cache_dir or ROOT / "data" / "final" / "cache" / "edit_graphs" / "external_benchmark"
    plm_cache_dir = args.plm_cache_dir or ROOT / "data" / "final" / "cache" / "plm_embeddings" / "external_benchmark" / args.esm_model
    dataset = CEDGDataset(
        records,
        vocabs,
        "score",
        graph_cache_dir=graph_cache_dir,
        use_graph_cache=True,
        plm_cache_dir=plm_cache_dir,
        esm_model_name=args.esm_model,
        use_plm_cache=not args.use_live_esm,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    rows: list[dict[str, object]] = []
    offset = 0
    with torch.no_grad():
        for batch in loader:
            batch = move_to_device(batch, torch.device(args.device))
            outputs = model(batch)
            for local_idx in range(len(batch["sample_id"])):
                record = records[offset + local_idx]
                rows.append(
                    {
                        "sample_id": record["sample_id"],
                        "source_slug": record.get("source_slug", ""),
                        "candidate_group_id": record.get("candidate_group_id", ""),
                        "parent_name": record.get("parent_name", ""),
                        "modified_name": record.get("modified_name", ""),
                        "edit_count": record.get("edit_count", 0),
                        "edit_summary": edit_summary(record.get("edit_set", []) or []),
                        "observed_delta": float(record.get("delta_property", 0.0)),
                        "observed_property_after": float(record.get("property_after", 0.0)),
                        "pred_delta": float(outputs["delta"][local_idx].detach().cpu()),
                        "pred_property_after": float(outputs["property_after"][local_idx].detach().cpu()),
                        "ranking_score": float(outputs["ranking_score"][local_idx].detach().cpu()),
                        "uncertainty": float(outputs["uncertainty"][local_idx].detach().cpu()),
                    }
                )
            offset += len(batch["sample_id"])
    scored = pd.DataFrame(rows)
    scored["recommendation_score"] = scored["ranking_score"] - args.uncertainty_penalty * scored["uncertainty"]
    return scored


def benchmark_group(group: pd.DataFrame, top_ks: list[int], positive_threshold: float) -> dict[str, object]:
    ordered = group.sort_values("recommendation_score", ascending=False).reset_index(drop=True)
    best_idx = int(group["observed_delta"].astype(float).idxmax())
    best_sample = str(group.loc[best_idx, "sample_id"])
    best_rank = int(ordered.index[ordered["sample_id"] == best_sample][0]) + 1
    row: dict[str, object] = {
        "candidate_group_id": group["candidate_group_id"].iloc[0],
        "source_slug": group["source_slug"].iloc[0],
        "n_candidates": len(group),
        "best_observed_delta": float(group["observed_delta"].max()),
        "best_observed_sample_id": best_sample,
        "best_observed_rank": best_rank,
        "best_observed_reciprocal_rank": 1.0 / best_rank,
        "group_spearman": safe_spearman(group["observed_delta"].tolist(), group["recommendation_score"].tolist()),
        "top1_sample_id": ordered["sample_id"].iloc[0],
        "top1_observed_delta": float(ordered["observed_delta"].iloc[0]),
        "top1_recommendation_score": float(ordered["recommendation_score"].iloc[0]),
        "mean_observed_delta": float(group["observed_delta"].mean()),
    }
    positives = set(group.loc[group["observed_delta"] > positive_threshold, "sample_id"].astype(str))
    row["has_positive"] = bool(positives)
    for k in top_ks:
        top_ids = set(ordered.head(k)["sample_id"].astype(str))
        row[f"best_hit_at_{k}"] = best_rank <= k
        row[f"positive_hit_at_{k}"] = bool(positives & top_ids) if positives else None
        row[f"top{k}_mean_observed_delta"] = float(ordered.head(k)["observed_delta"].mean())
        row[f"top{k}_max_observed_delta"] = float(ordered.head(k)["observed_delta"].max())
    return row


def summarize_benchmark(scored: pd.DataFrame, top_ks: list[int], positive_threshold: float) -> tuple[pd.DataFrame, dict[str, object]]:
    groups = [group for _, group in scored.groupby("candidate_group_id") if len(group) >= 2]
    group_rows = [benchmark_group(group, top_ks, positive_threshold) for group in groups]
    group_summary = pd.DataFrame(group_rows)
    y = scored["observed_delta"].to_numpy(dtype=float)
    score = scored["recommendation_score"].to_numpy(dtype=float)
    group_names = scored["candidate_group_id"].astype(str).tolist()
    metrics: dict[str, object] = {
        "n_rows": int(len(scored)),
        "n_groups": int(len(group_summary)),
        "positive_threshold": positive_threshold,
        "overall_spearman": safe_spearman(y.tolist(), score.tolist()),
        "group_pairwise_accuracy": group_pairwise_accuracy(group_names, y, score),
        "mean_group_spearman": float(group_summary["group_spearman"].dropna().mean()) if len(group_summary) else float("nan"),
        "mean_best_observed_rank": float(group_summary["best_observed_rank"].mean()) if len(group_summary) else float("nan"),
        "mean_reciprocal_rank": float(group_summary["best_observed_reciprocal_rank"].mean()) if len(group_summary) else float("nan"),
    }
    for k in top_ks:
        metrics[f"best_hit_at_{k}"] = float(group_summary[f"best_hit_at_{k}"].mean()) if len(group_summary) else float("nan")
        positive_groups = group_summary[group_summary["has_positive"]]
        metrics[f"positive_hit_at_{k}"] = (
            float(positive_groups[f"positive_hit_at_{k}"].mean()) if len(positive_groups) else float("nan")
        )
        metrics[f"ndcg_at_{k}"] = group_ndcg_at_k(group_names, y, score, k=k)
        metrics[f"top{k}_mean_delta_minus_group_mean"] = (
            float((group_summary[f"top{k}_mean_observed_delta"] - group_summary["mean_observed_delta"]).mean())
            if len(group_summary)
            else float("nan")
        )
    return group_summary, metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--external-dataset", type=Path, required=True)
    parser.add_argument("--train-dataset", type=Path, default=TRAIN_DATASET)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--top-ks", default="1,3,5,10")
    parser.add_argument("--positive-threshold", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--esm-model", default="esm2_t6_8M_UR50D")
    parser.add_argument("--use-live-esm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-plm-length", type=int, default=256)
    parser.add_argument("--graph-cache-dir", type=Path, default=None)
    parser.add_argument("--plm-cache-dir", type=Path, default=None)
    parser.add_argument("--uncertainty-penalty", type=float, default=0.1)
    args = parser.parse_args()

    top_ks = [int(value) for value in args.top_ks.split(",") if value.strip()]
    train_records = load_jsonl(args.train_dataset)
    external_records = normalize_external_records(load_external_records(args.external_dataset))
    report = overlap_report(train_records, external_records)
    scored = score_records(external_records, train_records, args)
    group_summary, metrics = summarize_benchmark(scored, top_ks, args.positive_threshold)
    metrics["overlap_report"] = report
    metrics["external_dataset"] = str(args.external_dataset)
    metrics["checkpoint"] = str(args.checkpoint)
    metrics["uncertainty_penalty"] = args.uncertainty_penalty

    args.out_dir.mkdir(parents=True, exist_ok=True)
    scored.sort_values(["candidate_group_id", "recommendation_score"], ascending=[True, False]).to_csv(
        args.out_dir / "scored_candidates.csv",
        index=False,
    )
    group_summary.to_csv(args.out_dir / "group_topk_recovery.csv", index=False)
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
