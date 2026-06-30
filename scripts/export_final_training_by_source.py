#!/usr/bin/env python3
"""中文说明：用最终单体注释导出 CEDG 第一阶段训练表，保留 edit set、attachment 和候选组信息。"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FINAL_DIR = ROOT / "data" / "final"
PEPTIDE_TABLE = FINAL_DIR / "annotations" / "peptide_sample_table.csv"
MONOMER_TABLE = FINAL_DIR / "annotations" / "monomer_anchor_table.csv"

INCLUDE_TIERS_DEFAULT = {
    "core_local_edit",
    "atom_replacement_auxiliary",
    "atom_unknown_auxiliary",
    "local_candidate_needs_review",
}
EXCLUDE_TIERS_DEFAULT = {
    "reference_or_no_edit",
    "exclude_until_curated",
    "terminal_context_only",
}

BASE_COLUMNS = [
    "sample_id",
    "source_id",
    "parent_peptide_id",
    "modified_peptide_id",
    "unordered_pair_id",
    "parent_name",
    "modified_name",
    "canonical_shadow_sequence_final",
    "parent_monomer_list",
    "modified_monomer_list",
    "parent_smiles",
    "modified_smiles",
    "assay_type",
    "property_before",
    "property_after",
    "delta_property",
    "max_edits_allowed",
    "censored_property_flag",
    "confidence_weight",
    "directional_pair",
    "reverse_pair_removed_flag",
    "pair_direction_rule",
    "candidate_generation_strategy",
    "candidate_group_id",
    "sample_model_use_tier",
    "contains_manual_curation_monomer_final",
    "contains_unknown_replacement",
    "contains_residue_graph_replacement",
    "notes",
]

EDIT_COLUMNS = [
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

SINGLE_COLUMNS = BASE_COLUMNS + EDIT_COLUMNS + ["edit_count"]
MULTI_COLUMNS = BASE_COLUMNS + ["edit_count", "edit_set_json"]


def normalize_source_id(source_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", source_id).strip("_").lower()


def parse_monomers(value: object) -> list[str]:
    text = str(value)
    if ";" in text and not text.startswith("["):
        return [part for part in text.split(";") if part]
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return []
    return [str(x).strip() for x in parsed] if isinstance(parsed, list) else []


def bool_value(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def monomer_base_name(monomer: str) -> str:
    token = str(monomer).strip().strip("[]")
    token = token.replace("N-Me-", "me")
    token = token.replace("NMe", "me")
    changed = True
    while changed:
        before = token
        token = re.sub(r"^(me|Me)_?", "", token)
        token = re.sub(r"^D-", "", token)
        if len(token) > 1 and token.startswith("d"):
            token = token[1:]
        if len(token) == 2 and token[0] == "m" and token[1].isupper():
            token = token[1]
        changed = token != before
    return token


def is_d_monomer(monomer: str) -> bool:
    token = str(monomer)
    return token.startswith("d") or token.startswith("D-") or token.startswith("Me_d")


def is_n_methyl_monomer(monomer: str) -> bool:
    token = str(monomer)
    return token.startswith("me") or token.startswith("Me_") or token.startswith("N-Me")


def censored_value(value: object) -> bool:
    return pd.notna(value) and float(value) <= -9.99


def confidence_weight(tiers: list[str], censored: bool, manual: bool) -> float:
    if any(t == "atom_unknown_auxiliary" for t in tiers):
        base = 0.35
    elif any(t == "local_candidate_needs_review" for t in tiers):
        base = 0.45
    elif any(t == "atom_replacement_auxiliary" for t in tiers):
        base = 0.6
    else:
        base = 1.0
    if censored:
        base *= 0.7
    if manual:
        base *= 0.8
    return round(base, 3)


def sample_tier(tiers: list[str]) -> str:
    if any(t == "atom_unknown_auxiliary" for t in tiers):
        return "atom_unknown_auxiliary"
    if any(t == "local_candidate_needs_review" for t in tiers):
        return "local_candidate_needs_review"
    if any(t == "atom_replacement_auxiliary" for t in tiers):
        return "atom_replacement_auxiliary"
    return "core_local_edit"


def monomer_lookup(table: pd.DataFrame) -> dict[str, pd.Series]:
    return {str(row["source_monomer_name"]): row for _, row in table.iterrows()}


STATE_PAYLOADS = {
    "N-H_to_N-CH3",
    "N-CH3_to_N-H",
    "L_to_D",
    "D_to_L",
}


def split_payload_parts(value: object) -> list[str]:
    return [part for part in str(value).split(";") if part and part != "nan"]


def pair_specific_edit_annotation(
    original: str,
    modified: str,
    modified_ann: pd.Series,
) -> dict[str, object]:
    original_base = monomer_base_name(original)
    modified_base = monomer_base_name(modified)
    original_is_d = is_d_monomer(original)
    modified_is_d = is_d_monomer(modified)
    original_is_n_methyl = is_n_methyl_monomer(original)
    modified_is_n_methyl = is_n_methyl_monomer(modified)

    scope_parts: list[str] = []
    state_payload_parts: list[str] = []

    if original_is_n_methyl != modified_is_n_methyl:
        scope_parts.append("backbone")
        state_payload_parts.append("N-H_to_N-CH3" if modified_is_n_methyl else "N-CH3_to_N-H")

    if original_is_d != modified_is_d:
        scope_parts.append("stereochemistry")
        state_payload_parts.append("D_to_L" if original_is_d and not modified_is_d else "L_to_D")

    # For same-anchor changes, derive the local event from state deltas rather than only
    # the modified monomer default. This avoids missing dL -> meL as N-methylation + D_to_L.
    if state_payload_parts and original_base == modified_base:
        if scope_parts == ["backbone"]:
            scope = "backbone_edit"
            subclass = "backbone_N_substitution_or_N_methylation"
            payload_type = "delta_graph"
            attachment = "backbone_N"
        elif scope_parts == ["stereochemistry"]:
            scope = "stereochemistry_edit"
            subclass = "stereochemistry_edit"
            payload_type = "chirality_change_flag"
            attachment = "alpha_C"
        elif set(scope_parts) == {"backbone", "stereochemistry"}:
            scope = "backbone_and_stereochemistry_edit"
            subclass = "backbone_N_substitution_plus_stereochemistry"
            payload_type = "combined_delta"
            attachment = "backbone_N;alpha_C"
        else:
            scope = str(modified_ann["final_edit_scope"])
            subclass = str(modified_ann["edit_event_subclass_default"])
            payload_type = str(modified_ann["final_chemical_payload_type"])
            attachment = str(modified_ann["attachment_default"])
        return {
            "edit_event_subclass": subclass,
            "final_edit_scope": scope,
            "attachment_type": attachment,
            "final_chemical_payload_type": payload_type,
            "final_chemical_payload": ";".join(state_payload_parts),
        }

    base_payload_parts = [
        part for part in split_payload_parts(modified_ann["final_chemical_payload"])
        if part not in STATE_PAYLOADS
    ]
    annotation = {
        "edit_event_subclass": modified_ann["edit_event_subclass_default"],
        "final_edit_scope": modified_ann["final_edit_scope"],
        "attachment_type": modified_ann["attachment_default"],
        "final_chemical_payload_type": modified_ann["final_chemical_payload_type"],
        "final_chemical_payload": ";".join(base_payload_parts)
        if base_payload_parts
        else modified_ann["final_chemical_payload"],
    }
    if state_payload_parts and str(modified_ann["final_edit_scope"]) not in {"unknown_replacement"}:
        combined_payload_parts = base_payload_parts + state_payload_parts
        annotation["final_edit_scope"] = (
            "stereochemistry_and_R_group_edit"
            if scope_parts == ["stereochemistry"]
            else "backbone_and_R_group_edit"
            if scope_parts == ["backbone"]
            else "backbone_stereochemistry_and_R_group_edit"
        )
        base_attachment = str(modified_ann["attachment_default"])
        state_attachment = (
            "alpha_C"
            if scope_parts == ["stereochemistry"]
            else "backbone_N"
            if scope_parts == ["backbone"]
            else "backbone_N;alpha_C"
        )
        annotation["attachment_type"] = ";".join(
            part for part in [base_attachment, state_attachment] if part and part != "nan"
        )
        annotation["final_chemical_payload_type"] = "combined_delta"
        annotation["final_chemical_payload"] = ";".join(combined_payload_parts)
    return annotation


def build_edit(
    site_index: int,
    original: str,
    modified: str,
    annotations: dict[str, pd.Series],
) -> dict[str, object] | None:
    original_ann = annotations.get(original)
    modified_ann = annotations.get(modified)
    if original_ann is None or modified_ann is None:
        return None

    original_tier = str(original_ann["model_use_tier_default"])
    modified_tier = str(modified_ann["model_use_tier_default"])
    if modified_tier in EXCLUDE_TIERS_DEFAULT:
        return None
    if modified_tier not in INCLUDE_TIERS_DEFAULT:
        return None

    pair_annotation = pair_specific_edit_annotation(original, modified, modified_ann)

    return {
        "site_index": site_index,
        "anchor_for_alignment": modified_ann["anchor_for_alignment"],
        "original_monomer": original,
        "modified_monomer": modified,
        "original_model_use_tier": original_tier,
        "modified_model_use_tier": modified_tier,
        "edit_model_use_tier": modified_tier,
        "edit_event_class": modified_ann["edit_event_class_default"],
        "edit_event_subclass": pair_annotation["edit_event_subclass"],
        "final_edit_scope": pair_annotation["final_edit_scope"],
        "attachment_type": pair_annotation["attachment_type"],
        "final_chemical_payload_type": pair_annotation["final_chemical_payload_type"],
        "final_chemical_payload": pair_annotation["final_chemical_payload"],
        "local_model_eligibility": modified_ann["local_model_eligibility"],
        "atom_model_eligibility": modified_ann["atom_model_eligibility"],
        "use_for_local_edit_model": bool_value(modified_ann["use_for_local_edit_model"]),
        "use_for_atom_edit_model": bool_value(modified_ann["use_for_atom_edit_model"]),
        "requires_manual_curation_final": bool_value(modified_ann["requires_manual_curation_final"]),
        "quality_flag_final": modified_ann["quality_flag_final"],
    }


def build_tables(
    peptides: pd.DataFrame,
    monomers: pd.DataFrame,
    source_id: str,
    assay: str,
    max_edits: int,
    max_edit_fraction: float,
    max_group_size: int,
    large_group_strategy: str,
    max_pairs_per_large_group: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    annotations = monomer_lookup(monomers)
    source = peptides[peptides["source_id"].eq(source_id)].copy()
    if source.empty:
        raise ValueError(f"No rows found for source_id={source_id}")
    if assay not in source.columns:
        raise ValueError(f"Assay column not found: {assay}")
    source = source[source[assay].notna()].copy()
    source["monomers"] = source["standardized_monomer_list"].map(parse_monomers)

    single_rows: list[dict[str, object]] = []
    multi_rows: list[dict[str, object]] = []
    skipped_rows: list[dict[str, object]] = []
    reverse_duplicate_rows_removed = 0
    sample_index = 1

    def max_edits_allowed(length: int) -> int:
        if length <= 0:
            return 0
        return min(max_edits, max(2, math.ceil(length * max_edit_fraction)))

    def pair_key(candidate: dict[str, object]) -> tuple[int, int]:
        parent_id = int(candidate["parent"]["peptide_id"])
        modified_id = int(candidate["modified"]["peptide_id"])
        return tuple(sorted((parent_id, modified_id)))

    def choose_pair_direction(candidates: list[dict[str, object]]) -> tuple[list[dict[str, object]], int]:
        best_by_pair: dict[tuple[int, int], dict[str, object]] = {}
        counts_by_pair: dict[tuple[int, int], int] = {}

        def rank(candidate: dict[str, object]) -> tuple[bool, float, bool, int]:
            parent_id = int(candidate["parent"]["peptide_id"])
            modified_id = int(candidate["modified"]["peptide_id"])
            delta = float(candidate["delta_property"])
            return (
                delta > 0,
                delta,
                parent_id < modified_id,
                -parent_id,
            )

        for candidate in candidates:
            key = pair_key(candidate)
            counts_by_pair[key] = counts_by_pair.get(key, 0) + 1
            current = best_by_pair.get(key)
            if current is None or rank(candidate) > rank(current):
                best_by_pair[key] = candidate

        kept: list[dict[str, object]] = []
        removed = 0
        for key, candidate in best_by_pair.items():
            candidate["reverse_pair_removed_flag"] = counts_by_pair[key] > 1
            candidate["pair_direction_rule"] = (
                "prefer_positive_delta_when_reverse_exists"
                if counts_by_pair[key] > 1
                else "only_generated_direction"
            )
            removed += counts_by_pair[key] - 1
            kept.append(candidate)
        return kept, removed

    def candidate_rows_for_group(shadow: str, group: pd.DataFrame) -> list[dict[str, object]]:
        candidates: list[dict[str, object]] = []
        group = group.sort_values("peptide_id")
        for _, parent in group.iterrows():
            for _, modified in group.iterrows():
                if int(parent["peptide_id"]) == int(modified["peptide_id"]):
                    continue
                parent_monomers = parent["monomers"]
                modified_monomers = modified["monomers"]
                if len(parent_monomers) != len(modified_monomers) or not parent_monomers:
                    continue
                allowed_edits = max_edits_allowed(len(parent_monomers))

                edits: list[dict[str, object]] = []
                unsupported = False
                raw_diff_count = 0
                for idx, (orig, mod) in enumerate(zip(parent_monomers, modified_monomers), start=1):
                    if orig == mod:
                        continue
                    raw_diff_count += 1
                    if raw_diff_count > allowed_edits:
                        unsupported = True
                        break
                    edit = build_edit(idx, orig, mod, annotations)
                    if edit is None:
                        unsupported = True
                        break
                    edits.append(edit)

                edit_count = len(edits)
                if unsupported or edit_count == 0 or edit_count > allowed_edits:
                    continue

                before = float(parent[assay])
                after = float(modified[assay])
                candidates.append(
                    {
                        "shadow": shadow,
                        "parent": parent,
                        "modified": modified,
                        "edits": edits,
                        "edit_count": edit_count,
                        "property_before": before,
                        "property_after": after,
                        "delta_property": after - before,
                        "abs_delta_property": abs(after - before),
                        "max_edits_allowed": allowed_edits,
                    }
                )
        return candidates

    def add_candidate(candidate: dict[str, object], generation_strategy: str) -> None:
        nonlocal sample_index
        parent = candidate["parent"]
        modified = candidate["modified"]
        edits = candidate["edits"]
        edit_count = int(candidate["edit_count"])
        before = float(candidate["property_before"])
        after = float(candidate["property_after"])
        censored = censored_value(before) or censored_value(after)
        tiers = [str(edit["edit_model_use_tier"]) for edit in edits]
        manual = any(bool(edit["requires_manual_curation_final"]) for edit in edits)
        weight = confidence_weight(tiers, censored, manual)
        tier = sample_tier(tiers)

        base = {
            "sample_id": f"{normalize_source_id(source_id)}_final_sample_{sample_index:06d}",
            "source_id": source_id,
            "parent_peptide_id": int(parent["peptide_id"]),
            "modified_peptide_id": int(modified["peptide_id"]),
            "unordered_pair_id": (
                f"{normalize_source_id(source_id)}:"
                f"{min(int(parent['peptide_id']), int(modified['peptide_id']))}_"
                f"{max(int(parent['peptide_id']), int(modified['peptide_id']))}"
            ),
            "parent_name": parent["peptide_name"],
            "modified_name": modified["peptide_name"],
            "canonical_shadow_sequence_final": candidate["shadow"],
            "parent_monomer_list": parent["standardized_monomer_list"],
            "modified_monomer_list": modified["standardized_monomer_list"],
            "parent_smiles": parent["smiles_raw"],
            "modified_smiles": modified["smiles_raw"],
            "assay_type": assay,
            "property_before": before,
            "property_after": after,
            "delta_property": after - before,
            "max_edits_allowed": int(candidate["max_edits_allowed"]),
            "censored_property_flag": censored,
            "confidence_weight": weight,
            "directional_pair": True,
            "reverse_pair_removed_flag": bool(candidate["reverse_pair_removed_flag"]),
            "pair_direction_rule": candidate["pair_direction_rule"],
            "candidate_generation_strategy": generation_strategy,
            "candidate_group_id": (
                f"{normalize_source_id(source_id)}:{assay}:"
                f"parent_{int(parent['peptide_id'])}:shadow_{normalize_source_id(candidate['shadow'])}"
            ),
            "sample_model_use_tier": tier,
            "contains_manual_curation_monomer_final": manual,
            "contains_unknown_replacement": tier == "atom_unknown_auxiliary",
            "contains_residue_graph_replacement": any(
                str(edit["quality_flag_final"]) == "residue_graph_replacement"
                for edit in edits
            ),
            "notes": (
                f"{assay} -10 retained with censored_property_flag=True."
                if censored
                else ""
            ),
        }
        sample_index += 1

        if edit_count == 1:
            single_rows.append({**base, **edits[0], "edit_count": 1})
        else:
            multi_rows.append(
                {
                    **base,
                    "edit_count": edit_count,
                    "edit_set_json": json.dumps(edits, ensure_ascii=False, sort_keys=True),
                }
            )

    for shadow, group in source.groupby("canonical_shadow_sequence_final"):
        if len(group) < 2:
            continue
        if len(group) > max_group_size:
            if large_group_strategy == "skip":
                skipped_rows.append(
                    {
                        "source_id": source_id,
                        "canonical_shadow_sequence_final": shadow,
                        "group_size": len(group),
                        "skip_reason": "group_size_exceeds_limit",
                    }
                )
                continue
            candidates = candidate_rows_for_group(shadow, group)
            candidates, removed = choose_pair_direction(candidates)
            reverse_duplicate_rows_removed += removed
            if large_group_strategy == "top-delta":
                candidates = sorted(
                    candidates,
                    key=lambda x: (
                        x["abs_delta_property"],
                        -int(x["edit_count"]),
                    ),
                    reverse=True,
                )[:max_pairs_per_large_group]
                for candidate in candidates:
                    add_candidate(candidate, "large_group_top_abs_delta_filtered")
                skipped_rows.append(
                    {
                        "source_id": source_id,
                        "canonical_shadow_sequence_final": shadow,
                        "group_size": len(group),
                        "skip_reason": f"large_group_top_delta_kept_{len(candidates)}",
                    }
                )
                continue
            raise ValueError(f"Unsupported large_group_strategy={large_group_strategy}")

        candidates = candidate_rows_for_group(shadow, group)
        candidates, removed = choose_pair_direction(candidates)
        reverse_duplicate_rows_removed += removed
        for candidate in candidates:
            add_candidate(candidate, "same_source_same_assay_same_final_shadow_all_pairs_filtered")

    single = pd.DataFrame(single_rows, columns=SINGLE_COLUMNS)
    multi = pd.DataFrame(multi_rows, columns=MULTI_COLUMNS)
    summary = pd.DataFrame(
        [
            {
                "source_id": source_id,
                "assay_type": assay,
                "table": "training_single_site",
                "rows": len(single),
                "censored_rows": int(single["censored_property_flag"].sum()) if not single.empty else 0,
            },
            {
                "source_id": source_id,
                "assay_type": assay,
                "table": "training_multi_site",
                "rows": len(multi),
                "censored_rows": int(multi["censored_property_flag"].sum()) if not multi.empty else 0,
            },
            {
                "source_id": source_id,
                "assay_type": assay,
                "table": "skipped_shadow_groups",
                "rows": len(skipped_rows),
                "censored_rows": 0,
            },
            {
                "source_id": source_id,
                "assay_type": assay,
                "table": "reverse_direction_duplicates_removed",
                "rows": reverse_duplicate_rows_removed,
                "censored_rows": 0,
            },
        ]
    )
    skipped = pd.DataFrame(skipped_rows)
    return single, multi, pd.concat([summary, skipped], axis=0, ignore_index=True)


def export_source(
    source_id: str,
    assay: str,
    max_edits: int,
    max_edit_fraction: float,
    max_group_size: int,
    large_group_strategy: str,
    max_pairs_per_large_group: int,
) -> None:
    peptides = pd.read_csv(PEPTIDE_TABLE)
    monomers = pd.read_csv(MONOMER_TABLE)
    slug = normalize_source_id(source_id)
    out_dir = FINAL_DIR / "training" / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    single, multi, summary = build_tables(
        peptides,
        monomers,
        source_id,
        assay,
        max_edits,
        max_edit_fraction,
        max_group_size,
        large_group_strategy,
        max_pairs_per_large_group,
    )
    single.to_csv(out_dir / "training_single_site.csv", index=False)
    multi.to_csv(out_dir / "training_multi_site.csv", index=False)
    summary.to_csv(out_dir / "training_summary.csv", index=False)
    print(f"{source_id}: single={len(single)} multi={len(multi)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-id", action="append", required=True)
    parser.add_argument("--assay", default="PAMPA")
    parser.add_argument(
        "--max-edits",
        type=int,
        default=4,
        help="Hard cap for adaptive edit count.",
    )
    parser.add_argument(
        "--max-edit-fraction",
        type=float,
        default=0.25,
        help="Adaptive edit limit per peptide length; ceil(length * fraction), capped by --max-edits.",
    )
    parser.add_argument("--max-group-size", type=int, default=50)
    parser.add_argument(
        "--large-group-strategy",
        choices=["skip", "top-delta"],
        default="skip",
    )
    parser.add_argument("--max-pairs-per-large-group", type=int, default=200)
    args = parser.parse_args()
    for source_id in args.source_id:
        export_source(
            source_id,
            args.assay,
            args.max_edits,
            args.max_edit_fraction,
            args.max_group_size,
            args.large_group_strategy,
            args.max_pairs_per_large_group,
        )


if __name__ == "__main__":
    main()
