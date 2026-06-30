#!/usr/bin/env python3
"""中文说明：给定新肽序列，按 edit library 自适应枚举 site/edit 候选并输出 top-k 优化建议。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cedg.candidates import combine_edit_records, enumerate_single_edit_records  # noqa: E402
from cedg.data import CEDGCollatorWithESM, CEDGDataset, build_vocabs, collate_cedg, load_jsonl  # noqa: E402
from cedg.model import CEDGScoreModel  # noqa: E402
from cedg.plm import ESMBatchConverter, esm_model_spec, load_esm_model  # noqa: E402
from cedg.utils import move_to_device  # noqa: E402


DATASET = ROOT / "data" / "final" / "model_ready" / "peptide_component" / "cedg_score_dataset.jsonl"
LIBRARY = ROOT / "data" / "final" / "edit_libraries" / "local_edit_library.csv"


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


def edit_set_columns(edit_set: list[dict[str, object]]) -> dict[str, str]:
    return {
        "sites": ";".join(str(event.get("site_index", "")) for event in edit_set),
        "edit_ids": ";".join(str(event.get("candidate_edit_id", "")) for event in edit_set),
        "operations": ";".join(str(event.get("edit_event_subclass", "")) for event in edit_set),
        "payloads": ";".join(str(event.get("final_chemical_payload", "")) for event in edit_set),
        "attachment_types": ";".join(str(event.get("attachment_type", "")) for event in edit_set),
        "synthetic_risks": ";".join(str(event.get("synthetic_risk", "")) for event in edit_set),
    }


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
    model: CEDGScoreModel,
    vocabs: object,
    args: argparse.Namespace,
    collate_fn: object,
) -> pd.DataFrame:
    graph_cache_dir = args.graph_cache_dir or ROOT / "data" / "final" / "cache" / "edit_graphs" / "recommendations"
    plm_cache_dir = args.plm_cache_dir or ROOT / "data" / "final" / "cache" / "plm_embeddings" / args.dataset.parent.name / args.esm_model
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
                edit_columns = edit_set_columns(record["edit_set"])
                rows.append(
                    {
                        "sample_id": record["sample_id"],
                        "candidate_group_id": record.get("candidate_group_id", ""),
                        "parent_name": record["parent_name"],
                        "modified_name": record["modified_name"],
                        "edit_count": record["edit_count"],
                        "modified_shadow_sequence": ".".join(str(token) for token in record.get("modified_monomer_list", [])),
                        "edit_summary": edit_summary(record["edit_set"]),
                        **edit_columns,
                        "pred_delta": float(outputs["delta"][local_idx].detach().cpu()),
                        "pred_property_after": float(outputs["property_after"][local_idx].detach().cpu()),
                        "ranking_score": float(outputs["ranking_score"][local_idx].detach().cpu()),
                        "uncertainty": float(outputs["uncertainty"][local_idx].detach().cpu()),
                        "edit_set_json": json.dumps(record["edit_set"], ensure_ascii=False, sort_keys=True),
                        "record_json": json.dumps(record, ensure_ascii=False, sort_keys=True),
                    }
                )
            offset += len(batch["sample_id"])
    scored = pd.DataFrame(rows)
    scored["recommendation_score"] = scored["ranking_score"] - args.uncertainty_penalty * scored["uncertainty"]
    return scored.sort_values("recommendation_score", ascending=False).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--parent-shadow-sequence", required=True)
    parser.add_argument("--parent-name", default="new_peptide")
    parser.add_argument("--assay-type", default="PAMPA")
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--edit-library", type=Path, default=LIBRARY)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-jsonl", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--beam-size", type=int, default=30)
    parser.add_argument("--max-edit-count", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--esm-model", default="esm2_t6_8M_UR50D")
    parser.add_argument("--use-live-esm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-plm-length", type=int, default=256)
    parser.add_argument("--plm-cache-dir", type=Path, default=None)
    parser.add_argument("--graph-cache-dir", type=Path, default=None)
    parser.add_argument("--uncertainty-penalty", type=float, default=0.1)
    parser.add_argument("--out-all-csv", type=Path, default=None)
    parser.add_argument("--out-candidates-jsonl", type=Path, default=None)
    args = parser.parse_args()

    train_records = load_jsonl(args.dataset)
    vocabs = build_vocabs(train_records)
    model, collate_fn = load_model(args, vocabs)

    single_records = enumerate_single_edit_records(
        args.parent_shadow_sequence,
        parent_name=args.parent_name,
        assay_type=args.assay_type,
        library_path=args.edit_library,
    )
    if not single_records:
        raise ValueError("No compatible single-edit candidates were generated.")
    for record in single_records:
        record["split"] = "score"

    single_scored = score_records(single_records, model, vocabs, args, collate_fn)
    beam_records = [
        json.loads(row["record_json"])
        for _, row in single_scored.head(args.beam_size).iterrows()
    ]
    combo_records = combine_edit_records(
        beam_records,
        max_edit_count=args.max_edit_count,
        beam_size=args.beam_size,
        parent_shadow_sequence=args.parent_shadow_sequence,
        parent_name=args.parent_name,
        assay_type=args.assay_type,
    )
    for record in combo_records:
        record["split"] = "score"
    all_records = single_records + combo_records
    scored = score_records(all_records, model, vocabs, args, collate_fn)
    top = scored.head(args.top_k)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    top.drop(columns=["record_json"]).to_csv(args.out_csv, index=False)
    if args.out_all_csv is not None:
        args.out_all_csv.parent.mkdir(parents=True, exist_ok=True)
        scored.drop(columns=["record_json"]).to_csv(args.out_all_csv, index=False)
    if args.out_jsonl is not None:
        with args.out_jsonl.open("w", encoding="utf-8") as handle:
            for row in top.drop(columns=["record_json"]).to_dict(orient="records"):
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    if args.out_candidates_jsonl is not None:
        args.out_candidates_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.out_candidates_jsonl.open("w", encoding="utf-8") as handle:
            for record in all_records:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"single_candidates={len(single_records)} combo_candidates={len(combo_records)} wrote={len(top)}")
    print(f"out_csv={args.out_csv}")


if __name__ == "__main__":
    main()
