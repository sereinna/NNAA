"""中文说明：CEDG 新肽候选生成模块，基于 anchor-compatible edit library 枚举并组合局部编辑建议。"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import pandas as pd

from .data import normalize_token, shadow_tokens


DEFAULT_LIBRARY = Path(__file__).resolve().parents[2] / "data" / "final" / "edit_libraries" / "local_edit_library.csv"
NO_BACKBONE_NH = {"P", "Pro", "me", "N-Me"}
AROMATIC_ANCHORS = {"F", "W", "Y", "Phe", "Trp", "Tyr", "Phe-like", "Trp-like"}
SIDECHAIN_NUCLEOPHILES = {"S", "T", "C", "K", "R", "D", "E", "Ser", "Thr", "Cys", "Lys", "Arg", "Asp", "Glu"}


def load_edit_library(path: Path = DEFAULT_LIBRARY) -> list[dict[str, object]]:
    table = pd.read_csv(path)
    return [row.dropna().to_dict() for _, row in table.iterrows()]


def split_allowed(value: object) -> set[str]:
    return {part.strip() for part in str(value).split(";") if part.strip()}


def compatible(anchor: str, edit: dict[str, object]) -> bool:
    allowed = split_allowed(edit.get("allowed_anchors", ""))
    if anchor not in allowed:
        return False
    if str(edit.get("requires_backbone_nh", "false")).lower() == "true" and anchor in NO_BACKBONE_NH:
        return False
    if str(edit.get("requires_aromatic", "false")).lower() == "true" and anchor not in AROMATIC_ANCHORS:
        return False
    if str(edit.get("requires_sidechain_nucleophile", "false")).lower() == "true" and anchor not in SIDECHAIN_NUCLEOPHILES:
        return False
    return True


def modified_monomer(anchor: str, edit: dict[str, object]) -> str:
    template = str(edit.get("modified_monomer_template", "{anchor}"))
    return template.replace("{anchor}", anchor)


def make_edit_event(site_index: int, anchor: str, edit: dict[str, object]) -> dict[str, object]:
    return {
        "site_index": int(site_index),
        "anchor_for_alignment": anchor,
        "original_monomer": anchor,
        "modified_monomer": modified_monomer(anchor, edit),
        "edit_event_class": "local_interpretable_edit",
        "edit_event_subclass": edit.get("operation"),
        "final_edit_scope": edit.get("edit_scope"),
        "attachment_type": edit.get("attachment_type"),
        "final_chemical_payload_type": edit.get("chemical_payload_type"),
        "final_chemical_payload": edit.get("chemical_payload"),
        "edit_model_use_tier": "library_candidate",
        "local_model_eligibility": "candidate",
        "atom_model_eligibility": "candidate",
        "quality_flag_final": "library_candidate",
        "requires_manual_curation_final": False,
        "use_for_local_edit_model": True,
        "use_for_atom_edit_model": True,
        "candidate_edit_id": edit.get("edit_id"),
        "synthetic_risk": edit.get("synthetic_risk", "unknown"),
    }


def base_candidate_record(
    parent_shadow_sequence: str,
    parent_name: str,
    assay_type: str,
    candidate_group_id: str,
) -> dict[str, object]:
    tokens = shadow_tokens({"parent_shadow_sequence": parent_shadow_sequence})
    return {
        "sample_id": "",
        "split": "score",
        "source_id": "user_candidate",
        "source_slug": "user_candidate",
        "table_type": "candidate",
        "assay_type": assay_type,
        "parent_peptide_id": -1,
        "modified_peptide_id": -1,
        "unordered_pair_id": "",
        "parent_name": parent_name,
        "modified_name": "",
        "parent_shadow_sequence": parent_shadow_sequence,
        "canonical_shadow_sequence_final": parent_shadow_sequence,
        "parent_monomer_list": tokens,
        "modified_monomer_list": tokens,
        "parent_smiles": "",
        "modified_smiles": "",
        "edit_count": 0,
        "edit_set": [],
        "property_before": 0.0,
        "property_after": 0.0,
        "delta_property": 0.0,
        "direction_label": 0,
        "censored_property_flag": False,
        "confidence_weight": 1.0,
        "sample_weight": 1.0,
        "sample_model_use_tier": "candidate",
        "contains_manual_curation_monomer_final": False,
        "contains_unknown_replacement": False,
        "contains_residue_graph_replacement": False,
        "candidate_generation_strategy": "library_enumeration",
        "candidate_group_id": candidate_group_id,
    }


def enumerate_single_edit_records(
    parent_shadow_sequence: str,
    parent_name: str = "new_parent",
    assay_type: str = "PAMPA",
    library_path: Path = DEFAULT_LIBRARY,
) -> list[dict[str, object]]:
    tokens = shadow_tokens({"parent_shadow_sequence": parent_shadow_sequence})
    library = load_edit_library(library_path)
    group_id = f"user:{parent_name}:library_candidates"
    records: list[dict[str, object]] = []
    for site_index, anchor in enumerate(tokens, start=1):
        for edit in library:
            if not compatible(anchor, edit):
                continue
            event = make_edit_event(site_index, anchor, edit)
            record = base_candidate_record(parent_shadow_sequence, parent_name, assay_type, group_id)
            record["sample_id"] = f"{parent_name}_single_site{site_index}_{edit['edit_id']}"
            record["modified_name"] = f"{parent_name}|{event['modified_monomer']}@{site_index}"
            record["modified_monomer_list"] = [
                event["modified_monomer"] if idx == site_index else token
                for idx, token in enumerate(tokens, start=1)
            ]
            record["edit_count"] = 1
            record["edit_set"] = [event]
            records.append(record)
    return records


def combine_edit_records(
    single_records: list[dict[str, object]],
    max_edit_count: int,
    beam_size: int,
    parent_shadow_sequence: str,
    parent_name: str,
    assay_type: str,
) -> list[dict[str, object]]:
    if max_edit_count <= 1:
        return []
    pool = single_records[:beam_size]
    tokens = shadow_tokens({"parent_shadow_sequence": parent_shadow_sequence})
    group_id = f"user:{parent_name}:library_candidates"
    combined: list[dict[str, object]] = []
    for edit_count in range(2, max_edit_count + 1):
        for combo in itertools.combinations(pool, edit_count):
            events = [record["edit_set"][0] for record in combo]
            sites = [int(event["site_index"]) for event in events]
            if len(set(sites)) != len(sites):
                continue
            record = base_candidate_record(parent_shadow_sequence, parent_name, assay_type, group_id)
            record["sample_id"] = f"{parent_name}_combo_" + "_".join(
                f"s{event['site_index']}_{event['candidate_edit_id']}" for event in events
            )
            modified = list(tokens)
            for event in events:
                modified[int(event["site_index"]) - 1] = event["modified_monomer"]
            record["modified_name"] = f"{parent_name}|combo_{edit_count}"
            record["modified_monomer_list"] = modified
            record["edit_count"] = edit_count
            record["edit_set"] = events
            combined.append(record)
    return combined


def records_to_jsonl(records: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
