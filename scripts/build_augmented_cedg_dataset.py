#!/usr/bin/env python3
"""中文说明：合并原始 CEDG model-ready 数据和外部局部 SAR 数据，生成可训练的增强版 JSONL。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = ROOT / "data" / "final" / "model_ready" / "peptide_component" / "cedg_score_dataset.jsonl"
DEFAULT_OUT = ROOT / "data" / "final" / "model_ready" / "peptide_component_plus_faris" / "cedg_score_dataset.jsonl"
MODEL_READY_FIELDS = [
    "assay_type",
    "candidate_generation_strategy",
    "candidate_group_id",
    "canonical_shadow_sequence_final",
    "censored_property_flag",
    "confidence_weight",
    "contains_manual_curation_monomer_final",
    "contains_residue_graph_replacement",
    "contains_unknown_replacement",
    "delta_property",
    "direction_label",
    "edit_count",
    "edit_set",
    "modified_monomer_list",
    "modified_name",
    "modified_peptide_id",
    "modified_smiles",
    "parent_monomer_list",
    "parent_name",
    "parent_peptide_id",
    "parent_shadow_sequence",
    "parent_smiles",
    "property_after",
    "property_before",
    "sample_id",
    "sample_model_use_tier",
    "sample_weight",
    "source_id",
    "source_slug",
    "split",
    "table_type",
    "unordered_pair_id",
]


def load_jsonl(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def group_split(group_id: str, seed: int, train_frac: float, val_frac: float) -> str:
    digest = hashlib.sha1(f"{seed}:{group_id}".encode("utf-8")).hexdigest()
    value = int(digest[:12], 16) / float(16**12)
    if value < train_frac:
        return "train"
    if value < train_frac + val_frac:
        return "val"
    return "test"


def normalize_external_record(record: dict[str, object], split: str, weight: float, source_tag: str) -> dict[str, object]:
    item = dict(record)
    item["split"] = split
    item["source_slug"] = str(item.get("source_slug") or source_tag)
    item["source_id"] = str(item.get("source_id") or source_tag)
    item["sample_id"] = f"{source_tag}_{item.get('sample_id')}"
    item["sample_model_use_tier"] = "external_training_augmented"
    item["sample_weight"] = float(weight)
    item["confidence_weight"] = float(item.get("confidence_weight") or 1.0) * float(weight)
    item["candidate_generation_strategy"] = "external_local_neighbor_augmented_training"
    item["edit_count"] = len(item.get("edit_set", []) or [])
    item.setdefault("direction_label", int(float(item.get("delta_property", 0.0)) > 0.0))
    item.setdefault("censored_property_flag", False)
    item.setdefault("contains_manual_curation_monomer_final", False)
    item.setdefault("contains_unknown_replacement", False)
    item.setdefault("contains_residue_graph_replacement", True)
    return {field: item.get(field) for field in MODEL_READY_FIELDS}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dataset", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--external-dataset", type=Path, required=True)
    parser.add_argument("--out-jsonl", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--out-summary", type=Path, default=None)
    parser.add_argument("--source-tag", default="faris2024")
    parser.add_argument("--external-weight", type=float, default=0.7)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    base_records = load_jsonl(args.base_dataset)
    external_records_raw = load_jsonl(args.external_dataset)
    group_to_split = {
        str(record.get("candidate_group_id")): group_split(
            str(record.get("candidate_group_id")),
            args.seed,
            args.train_frac,
            args.val_frac,
        )
        for record in external_records_raw
    }
    external_records = [
        normalize_external_record(
            record,
            group_to_split[str(record.get("candidate_group_id"))],
            args.external_weight,
            args.source_tag,
        )
        for record in external_records_raw
    ]
    merged = base_records + external_records
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_jsonl.open("w", encoding="utf-8") as handle:
        for record in merged:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    summary: dict[str, object] = {
        "base_dataset": str(args.base_dataset),
        "external_dataset": str(args.external_dataset),
        "out_jsonl": str(args.out_jsonl),
        "base_rows": len(base_records),
        "external_rows": len(external_records),
        "total_rows": len(merged),
        "external_weight": args.external_weight,
        "external_group_count": len(group_to_split),
        "split_counts": {},
        "external_split_counts": {},
    }
    for record in merged:
        split = str(record.get("split"))
        summary["split_counts"][split] = int(summary["split_counts"].get(split, 0)) + 1
    for record in external_records:
        split = str(record.get("split"))
        summary["external_split_counts"][split] = int(summary["external_split_counts"].get(split, 0)) + 1
    out_summary = args.out_summary or (args.out_jsonl.parent / "cedg_score_dataset_summary.json")
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
