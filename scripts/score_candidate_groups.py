#!/usr/bin/env python3
"""中文说明：对 CEDG candidate group 批量打分并导出 parent-level top-k edit set 优化建议。"""

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

from cedg.data import CEDGDataset, build_vocabs, collate_cedg, load_jsonl  # noqa: E402
from cedg.model import CEDGScoreModel  # noqa: E402
from cedg.plm import esm_model_spec  # noqa: E402
from cedg.utils import move_to_device  # noqa: E402


DATASET = ROOT / "data" / "final" / "model_ready" / "peptide_component" / "cedg_score_dataset.jsonl"


def edit_summary(edit_set: list[dict[str, object]]) -> str:
    parts = []
    for event in edit_set:
        parts.append(
            (
                f"site{event.get('site_index')}:"
                f"{event.get('original_monomer')}->{event.get('modified_monomer')}"
                f"|{event.get('final_edit_scope')}"
                f"|{event.get('attachment_type')}"
                f"|{event.get('final_chemical_payload')}"
            )
        )
    return " ; ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-jsonl", type=Path, default=None)
    parser.add_argument("--parent-peptide-id", type=int, default=None)
    parser.add_argument("--candidate-group-id", default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--graph-cache-dir", type=Path, default=None)
    parser.add_argument("--plm-cache-dir", type=Path, default=None)
    parser.add_argument("--esm-model", default="esm2_t6_8M_UR50D")
    args = parser.parse_args()

    records = load_jsonl(args.dataset)
    if args.parent_peptide_id is not None:
        records = [record for record in records if int(record["parent_peptide_id"]) == args.parent_peptide_id]
    if args.candidate_group_id is not None:
        records = [record for record in records if str(record.get("candidate_group_id")) == args.candidate_group_id]
    if not records:
        raise ValueError("No candidate records matched the requested filter.")

    all_records_for_vocab = load_jsonl(args.dataset)
    vocabs = build_vocabs(all_records_for_vocab)
    cache_dir = args.graph_cache_dir or ROOT / "data" / "final" / "cache" / "edit_graphs" / args.dataset.parent.name
    plm_cache_dir = args.plm_cache_dir or ROOT / "data" / "final" / "cache" / "plm_embeddings" / args.dataset.parent.name / args.esm_model

    for record in records:
        record["split"] = "score"
    dataset = CEDGDataset(
        records,
        vocabs,
        "score",
        graph_cache_dir=cache_dir,
        use_graph_cache=True,
        plm_cache_dir=plm_cache_dir,
        esm_model_name=args.esm_model,
        use_plm_cache=True,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_cedg)

    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model_args = checkpoint.get("args", {})
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
        esm_model=None,
        esm_repr_layer=None,
        esm_dim=esm_dim,
        freeze_esm=True,
    ).to(args.device)
    model.load_state_dict(checkpoint["model_state"], strict=False)
    model.eval()

    rows: list[dict[str, object]] = []
    offset = 0
    with torch.no_grad():
        for batch in loader:
            batch = move_to_device(batch, torch.device(args.device))
            outputs = model(batch)
            batch_size = len(batch["sample_id"])
            for local_idx in range(batch_size):
                record = records[offset + local_idx]
                rows.append(
                    {
                        "sample_id": record["sample_id"],
                        "candidate_group_id": record.get("candidate_group_id"),
                        "parent_peptide_id": record.get("parent_peptide_id"),
                        "parent_name": record.get("parent_name"),
                        "modified_peptide_id": record.get("modified_peptide_id"),
                        "modified_name": record.get("modified_name"),
                        "edit_count": record.get("edit_count"),
                        "edit_summary": edit_summary(record.get("edit_set", [])),
                        "pred_delta": float(outputs["delta"][local_idx].detach().cpu()),
                        "pred_property_after": float(outputs["property_after"][local_idx].detach().cpu()),
                        "ranking_score": float(outputs["ranking_score"][local_idx].detach().cpu()),
                        "uncertainty": float(outputs["uncertainty"][local_idx].detach().cpu()),
                        "observed_delta": record.get("delta_property"),
                        "observed_property_after": record.get("property_after"),
                        "edit_set_json": json.dumps(record.get("edit_set", []), ensure_ascii=False, sort_keys=True),
                    }
                )
            offset += batch_size

    scored = pd.DataFrame(rows)
    scored["recommendation_score"] = scored["ranking_score"] - 0.1 * scored["uncertainty"]
    top = (
        scored.sort_values(["candidate_group_id", "recommendation_score"], ascending=[True, False])
        .groupby("candidate_group_id", as_index=False)
        .head(args.top_k)
        .reset_index(drop=True)
    )
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    top.to_csv(args.out_csv, index=False)
    if args.out_jsonl is not None:
        with args.out_jsonl.open("w", encoding="utf-8") as handle:
            for row in top.to_dict(orient="records"):
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"wrote {len(top)} rows to {args.out_csv}")


if __name__ == "__main__":
    main()
