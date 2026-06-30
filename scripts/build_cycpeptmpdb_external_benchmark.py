#!/usr/bin/env python3
"""中文说明：把新版 CycPeptMPDB source CSV 转成外部验证用 CEDG model-ready JSONL。"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cedg.edit_rules import anchor_for_monomer_with_rules, infer_pair_edit_annotation  # noqa: E402

MONOMER_TABLE = ROOT / "data" / "final" / "annotations" / "monomer_anchor_table.csv"


def load_monomer_rules(path: Path = MONOMER_TABLE) -> dict[str, dict[str, object]]:
    if not path.exists():
        return {}
    table = pd.read_csv(path)
    rules: dict[str, dict[str, object]] = {}
    for row in table.to_dict(orient="records"):
        for key in {row.get("source_monomer_name"), row.get("standardized_name"), row.get("base_monomer_name")}:
            if isinstance(key, str) and key:
                rules[key] = row
    return rules


def parse_sequence(value: object) -> list[str]:
    if pd.isna(value):
        return []
    parsed = ast.literal_eval(str(value))
    return [str(item) for item in parsed]


def anchor_for(token: str, rules: dict[str, dict[str, object]]) -> str:
    return anchor_for_monomer_with_rules(token, rules)


def shadow_sequence(tokens: list[str], rules: dict[str, dict[str, object]]) -> str:
    return ".".join(anchor_for(token, rules) for token in tokens)


def monomer_rule(token: str, rules: dict[str, dict[str, object]]) -> dict[str, object]:
    return rules.get(token, {})


def edit_event(site: int, original: str, modified: str, rules: dict[str, dict[str, object]]) -> dict[str, object]:
    original_rule = monomer_rule(original, rules)
    modified_rule = monomer_rule(modified, rules)
    pair_annotation = infer_pair_edit_annotation(original, modified, lambda token: anchor_for(token, rules))
    anchor = anchor_for(original, rules)
    payloads: list[str] = []
    scopes: list[str] = []
    operations: list[str] = []
    attachments: list[str] = []
    payload_type = str(modified_rule.get("final_chemical_payload_type") or "external_delta")
    if original != modified:
        if pair_annotation is not None:
            anchor = pair_annotation.anchor_for_alignment
            scopes.append(pair_annotation.final_edit_scope)
            operations.append(pair_annotation.edit_event_subclass)
            attachments.append(pair_annotation.attachment_type)
            payloads.append(pair_annotation.final_chemical_payload)
            payload_type = pair_annotation.final_chemical_payload_type
        else:
            fallback_payload = modified_rule.get("final_chemical_payload") or modified_rule.get("chemical_payload_template")
            if fallback_payload is None or pd.isna(fallback_payload) or str(fallback_payload) == "nan":
                fallback_payload = f"{original}_to_{modified}"
            payloads.append(str(fallback_payload))
            fallback_scope = modified_rule.get("final_edit_scope") or modified_rule.get("edit_scope_default")
            if fallback_scope is None or pd.isna(fallback_scope) or str(fallback_scope) == "nan" or str(fallback_scope) == "none":
                fallback_scope = "R_group_edit"
            scopes.append(str(fallback_scope))
            fallback_operation = modified_rule.get("edit_event_subclass_default") or modified_rule.get("operation_default")
            if (
                fallback_operation is None
                or pd.isna(fallback_operation)
                or str(fallback_operation) == "nan"
                or str(fallback_operation) == "natural_reference_residue"
            ):
                fallback_operation = "monomer_replacement"
            operations.append(
                str(fallback_operation)
            )
            fallback_attachment = modified_rule.get("attachment_default")
            if fallback_attachment is None or pd.isna(fallback_attachment) or str(fallback_attachment) == "nan":
                fallback_attachment = "side_chain_or_backbone"
            attachments.append(str(fallback_attachment))
    subclass = "+".join(dict.fromkeys(part for op in operations for part in str(op).split("+"))) or "none"
    scope_text = "+".join(dict.fromkeys(part for scope in scopes for part in str(scope).split("+"))) or "none"
    attachment_text = ";".join(dict.fromkeys(part for att in attachments for part in str(att).split(";") if part))
    payload_text = ";".join(dict.fromkeys(part for payload in payloads for part in str(payload).split(";") if part))
    return {
        "site_index": site,
        "anchor_for_alignment": anchor,
        "original_monomer": original,
        "modified_monomer": modified,
        "original_model_use_tier": "external_reference",
        "modified_model_use_tier": "external_candidate",
        "edit_model_use_tier": "external_benchmark",
        "edit_event_class": "external_observed_edit",
        "edit_event_subclass": subclass,
        "final_edit_scope": scope_text,
        "attachment_type": attachment_text,
        "final_chemical_payload_type": payload_type,
        "final_chemical_payload": payload_text,
        "local_model_eligibility": "external_benchmark",
        "atom_model_eligibility": "external_benchmark",
        "use_for_local_edit_model": True,
        "use_for_atom_edit_model": True,
        "requires_manual_curation_final": False,
        "quality_flag_final": "external_auto_pair",
    }


def make_record(
    parent: pd.Series,
    child: pd.Series,
    parent_tokens: list[str],
    child_tokens: list[str],
    source_slug: str,
    rules: dict[str, dict[str, object]],
    group_id: str | None = None,
) -> dict[str, object]:
    edits = [
        edit_event(site, old, new, rules)
        for site, (old, new) in enumerate(zip(parent_tokens, child_tokens), start=1)
        if old != new
    ]
    parent_value = float(parent["PAMPA"])
    child_value = float(child["PAMPA"])
    if group_id is None:
        group_id = f"{source_slug}:PAMPA:parent_{int(parent['ID'])}:local_neighbors"
    return {
        "sample_id": f"{source_slug}_{int(parent['ID'])}_{int(child['ID'])}",
        "split": "score",
        "source_id": str(child["Source"]),
        "source_slug": source_slug,
        "table_type": "external_cycpeptmpdb_source",
        "assay_type": "PAMPA",
        "parent_peptide_id": int(parent["ID"]),
        "modified_peptide_id": int(child["ID"]),
        "unordered_pair_id": f"{source_slug}:{int(parent['ID'])}_{int(child['ID'])}",
        "parent_name": str(parent["Original_Name_in_Source_Literature"]),
        "modified_name": str(child["Original_Name_in_Source_Literature"]),
        "parent_shadow_sequence": shadow_sequence(parent_tokens, rules),
        "canonical_shadow_sequence_final": shadow_sequence(parent_tokens, rules),
        "parent_monomer_list": parent_tokens,
        "modified_monomer_list": child_tokens,
        "parent_smiles": str(parent["SMILES"]),
        "modified_smiles": str(child["SMILES"]),
        "edit_count": len(edits),
        "edit_set": edits,
        "property_before": parent_value,
        "property_after": child_value,
        "delta_property": child_value - parent_value,
        "direction_label": int(child_value > parent_value),
        "censored_property_flag": False,
        "confidence_weight": 1.0,
        "sample_weight": 1.0,
        "sample_model_use_tier": "external_benchmark",
        "contains_manual_curation_monomer_final": False,
        "contains_unknown_replacement": False,
        "contains_residue_graph_replacement": True,
        "candidate_generation_strategy": "external_source_vs_reference_parent",
        "candidate_group_id": group_id,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-csv", type=Path, required=True)
    parser.add_argument("--out-jsonl", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, required=True)
    parser.add_argument("--source-slug", required=True)
    parser.add_argument("--parent-strategy", choices=["best", "worst", "median", "first"], default="median")
    parser.add_argument("--pair-mode", choices=["reference-parent", "local-neighbor"], default="reference-parent")
    parser.add_argument("--max-hamming-distance", type=int, default=1)
    parser.add_argument("--min-abs-delta", type=float, default=0.0)
    args = parser.parse_args()

    table = pd.read_csv(args.source_csv, low_memory=False)
    table = table[pd.to_numeric(table["PAMPA"], errors="coerce").notna()].copy()
    table["PAMPA"] = pd.to_numeric(table["PAMPA"], errors="coerce")
    table["tokens"] = table["Sequence"].map(parse_sequence)
    rules = load_monomer_rules()
    records = []
    parent = None
    group_id = ""
    if args.pair_mode == "local-neighbor":
        for left_idx, left in table.iterrows():
            left_tokens = left["tokens"]
            for right_idx, right in table.iterrows():
                if right_idx == left_idx:
                    continue
                right_tokens = right["tokens"]
                if len(left_tokens) != len(right_tokens):
                    continue
                distance = sum(old != new for old, new in zip(left_tokens, right_tokens))
                if distance < 1 or distance > args.max_hamming_distance:
                    continue
                delta = float(right["PAMPA"]) - float(left["PAMPA"])
                if abs(delta) < args.min_abs_delta:
                    continue
                records.append(make_record(left, right, left_tokens, right_tokens, args.source_slug, rules))
        summary = {
            "source_csv": str(args.source_csv),
            "out_jsonl": str(args.out_jsonl),
            "source_slug": args.source_slug,
            "pair_mode": args.pair_mode,
            "max_hamming_distance": args.max_hamming_distance,
            "min_abs_delta": args.min_abs_delta,
            "rows": len(records),
            "source_rows": int(len(table)),
            "pampa_min": float(table["PAMPA"].min()),
            "pampa_median": float(table["PAMPA"].median()),
            "pampa_max": float(table["PAMPA"].max()),
        }
        args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.out_jsonl.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        args.out_summary.parent.mkdir(parents=True, exist_ok=True)
        args.out_summary.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(summary, sort_keys=True))
        return

    if args.parent_strategy == "best":
        parent = table.loc[table["PAMPA"].idxmax()]
    elif args.parent_strategy == "worst":
        parent = table.loc[table["PAMPA"].idxmin()]
    elif args.parent_strategy == "first":
        parent = table.iloc[0]
    else:
        median = table["PAMPA"].median()
        parent = table.iloc[(table["PAMPA"] - median).abs().argsort().iloc[0]]
    parent_tokens = parse_sequence(parent["Sequence"])
    group_id = f"{args.source_slug}:PAMPA:reference_{int(parent['ID'])}:strategy_{args.parent_strategy}"
    for _, child in table.iterrows():
        if int(child["ID"]) == int(parent["ID"]):
            continue
        child_tokens = child["tokens"]
        if len(child_tokens) != len(parent_tokens):
            continue
        records.append(make_record(parent, child, parent_tokens, child_tokens, args.source_slug, rules, group_id))

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_jsonl.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {
        "source_csv": str(args.source_csv),
        "out_jsonl": str(args.out_jsonl),
        "source_slug": args.source_slug,
        "parent_strategy": args.parent_strategy,
        "parent_id": int(parent["ID"]),
        "parent_name": str(parent["Original_Name_in_Source_Literature"]),
        "parent_pampa": float(parent["PAMPA"]),
        "rows": len(records),
        "pampa_min": float(table["PAMPA"].min()),
        "pampa_median": float(table["PAMPA"].median()),
        "pampa_max": float(table["PAMPA"].max()),
    }
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
