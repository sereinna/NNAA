#!/usr/bin/env python3
"""中文说明：把 split 后的 CEDG 训练 CSV 导出为模型可读 JSONL，保留 ranking 候选组和 attachment。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPLIT_ROOT = ROOT / "data" / "final" / "splits"
OUT_ROOT = ROOT / "data" / "final" / "model_ready"

EDIT_FIELDS = [
    "site_index",
    "anchor_for_alignment",
    "original_monomer",
    "modified_monomer",
    "original_model_use_tier",
    "modified_model_use_tier",
    "edit_model_use_tier",
    "edit_event_class",
    "edit_event_subclass",
    "final_edit_scope",
    "attachment_type",
    "final_chemical_payload_type",
    "final_chemical_payload",
    "local_model_eligibility",
    "atom_model_eligibility",
    "use_for_local_edit_model",
    "use_for_atom_edit_model",
    "requires_manual_curation_final",
    "quality_flag_final",
]


def parse_monomer_list(value: object) -> list[str]:
    text = "" if pd.isna(value) else str(value)
    return [part for part in text.split(";") if part]


def clean_value(value: object) -> object:
    if pd.isna(value):
        return None
    if isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def single_edit_set(row: pd.Series) -> list[dict[str, object]]:
    return [{field: clean_value(row[field]) if field in row else None for field in EDIT_FIELDS}]


def multi_edit_set(row: pd.Series) -> list[dict[str, object]]:
    edits = json.loads(str(row["edit_set_json"]))
    return [{key: clean_value(value) for key, value in edit.items()} for edit in edits]


def sample_weight(row: pd.Series) -> float:
    weight = float(row["confidence_weight"])
    if bool(row["censored_property_flag"]):
        weight *= 0.7
    if str(row["sample_model_use_tier"]) == "atom_replacement_auxiliary":
        weight *= 0.7
    return round(weight, 4)


def row_to_record(row: pd.Series, table_type: str) -> dict[str, object]:
    edit_set = single_edit_set(row) if table_type == "single" else multi_edit_set(row)
    return {
        "sample_id": row["sample_id"],
        "split": row["split"],
        "source_id": row["source_id"],
        "source_slug": row["source_slug"],
        "table_type": table_type,
        "assay_type": row["assay_type"],
        "parent_peptide_id": int(row["parent_peptide_id"]),
        "modified_peptide_id": int(row["modified_peptide_id"]),
        "unordered_pair_id": row["unordered_pair_id"],
        "parent_name": clean_value(row["parent_name"]),
        "modified_name": clean_value(row["modified_name"]),
        "parent_shadow_sequence": row["canonical_shadow_sequence_final"],
        "canonical_shadow_sequence_final": row["canonical_shadow_sequence_final"],
        "parent_monomer_list": parse_monomer_list(row["parent_monomer_list"]),
        "modified_monomer_list": parse_monomer_list(row["modified_monomer_list"]),
        "parent_smiles": clean_value(row["parent_smiles"]),
        "modified_smiles": clean_value(row["modified_smiles"]),
        "edit_count": int(row["edit_count"]),
        "edit_set": edit_set,
        "property_before": float(row["property_before"]),
        "property_after": float(row["property_after"]),
        "delta_property": float(row["delta_property"]),
        "direction_label": int(float(row["delta_property"]) > 0),
        "censored_property_flag": bool(row["censored_property_flag"]),
        "confidence_weight": float(row["confidence_weight"]),
        "sample_weight": sample_weight(row),
        "sample_model_use_tier": row["sample_model_use_tier"],
        "contains_manual_curation_monomer_final": bool(
            row["contains_manual_curation_monomer_final"]
        ),
        "contains_unknown_replacement": bool(row["contains_unknown_replacement"]),
        "contains_residue_graph_replacement": bool(
            row["contains_residue_graph_replacement"]
        ),
        "candidate_generation_strategy": row["candidate_generation_strategy"],
        "candidate_group_id": clean_value(
            row["candidate_group_id"]
            if "candidate_group_id" in row
            else f"{row['source_id']}:{row['assay_type']}:parent_{row['parent_peptide_id']}"
        ),
    }


def export_jsonl(split_scheme: str) -> None:
    split_dir = SPLIT_ROOT / split_scheme
    single = pd.read_csv(split_dir / "training_single_site.csv")
    multi = pd.read_csv(split_dir / "training_multi_site.csv")
    out_dir = OUT_ROOT / split_scheme
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "cedg_score_dataset.jsonl"

    count = 0
    split_counts: dict[str, int] = {}
    with out_path.open("w", encoding="utf-8") as handle:
        for _, row in single.iterrows():
            record = row_to_record(row, "single")
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
            split_counts[record["split"]] = split_counts.get(record["split"], 0) + 1
        for _, row in multi.iterrows():
            record = row_to_record(row, "multi")
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
            split_counts[record["split"]] = split_counts.get(record["split"], 0) + 1

    summary = pd.DataFrame(
        [
            {"split": split_name, "rows": rows}
            for split_name, rows in sorted(split_counts.items())
        ]
    )
    summary.to_csv(out_dir / "cedg_score_dataset_summary.csv", index=False)
    print(f"{split_scheme}: wrote {out_path.relative_to(ROOT)} rows={count}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-scheme", default="peptide_component")
    args = parser.parse_args()
    export_jsonl(args.split_scheme)


if __name__ == "__main__":
    main()
