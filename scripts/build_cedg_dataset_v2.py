#!/usr/bin/env python3
"""中文说明：构建 CEDG v2.0 数据集，控制 Townsend 占比、扩展少数来源并保留正负方向。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import export_final_training_by_source as export_v1  # noqa: E402
from export_model_ready_cedg import row_to_record  # noqa: E402


DEFAULT_SOURCES = [
    "2013_CHUGAI",
    "2015_Wang",
    "2020_Le Roux",
    "2022_Taechalertpaisarn",
    "2021_Golosov",
    "2015_Bockus_2",
    "2018_Naylor",
    "2016_Furukawa",
    "2020_Townsend",
    "2021_Kelly",
]
SOURCE_SLUGS = {source: export_v1.normalize_source_id(source) for source in DEFAULT_SOURCES}
DEFAULT_FARIS_TRAINING = (
    ROOT / "data" / "final" / "model_ready" / "peptide_component_plus_faris" / "cedg_score_dataset.jsonl"
)
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


def stable_float(key: str, seed: int) -> float:
    digest = hashlib.sha1(f"{seed}:{key}".encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12)


def stable_split(key: str, seed: int, train_frac: float, val_frac: float) -> str:
    value = stable_float(key, seed)
    if value < train_frac:
        return "train"
    if value < train_frac + val_frac:
        return "val"
    return "test"


def max_edits_allowed(length: int, max_edits: int, max_edit_fraction: float) -> int:
    if length <= 0:
        return 0
    return min(max_edits, max(2, math.ceil(length * max_edit_fraction)))


def candidate_rows_for_group(
    source_id: str,
    assay: str,
    shadow: str,
    group: pd.DataFrame,
    annotations: dict[str, pd.Series],
    max_edits: int,
    max_edit_fraction: float,
) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    group = group.sort_values("peptide_id")
    for _, parent in group.iterrows():
        for _, modified in group.iterrows():
            parent_id = int(parent["peptide_id"])
            modified_id = int(modified["peptide_id"])
            if parent_id == modified_id:
                continue
            parent_monomers = parent["monomers"]
            modified_monomers = modified["monomers"]
            if len(parent_monomers) != len(modified_monomers) or not parent_monomers:
                continue
            allowed_edits = max_edits_allowed(len(parent_monomers), max_edits, max_edit_fraction)

            edits: list[dict[str, object]] = []
            unsupported = False
            raw_diff_count = 0
            for idx, (original, modified_monomer) in enumerate(
                zip(parent_monomers, modified_monomers),
                start=1,
            ):
                if original == modified_monomer:
                    continue
                raw_diff_count += 1
                if raw_diff_count > allowed_edits:
                    unsupported = True
                    break
                edit = export_v1.build_edit(idx, original, modified_monomer, annotations)
                if edit is None:
                    unsupported = True
                    break
                edits.append(edit)

            edit_count = len(edits)
            if unsupported or edit_count == 0 or edit_count > allowed_edits:
                continue

            before = float(parent[assay])
            after = float(modified[assay])
            pair_key = (
                f"{source_id}:{shadow}:"
                f"{min(parent_id, modified_id)}_{max(parent_id, modified_id)}"
            )
            candidates.append(
                {
                    "source_id": source_id,
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
                    "unordered_pair_key": pair_key,
                    "direction_key": f"{pair_key}:{parent_id}->{modified_id}",
                }
            )
    return candidates


def confidence_weight(tiers: list[str], censored: bool, manual: bool) -> float:
    return export_v1.confidence_weight(tiers, censored, manual)


def sample_tier(tiers: list[str]) -> str:
    return export_v1.sample_tier(tiers)


def candidate_to_flat_row(
    candidate: dict[str, object],
    sample_id: str,
    generation_strategy: str,
    split: str,
) -> dict[str, object]:
    source_id = str(candidate["source_id"])
    slug = export_v1.normalize_source_id(source_id)
    parent = candidate["parent"]
    modified = candidate["modified"]
    edits = candidate["edits"]
    edit_count = int(candidate["edit_count"])
    before = float(candidate["property_before"])
    after = float(candidate["property_after"])
    censored = export_v1.censored_value(before) or export_v1.censored_value(after)
    tiers = [str(edit["edit_model_use_tier"]) for edit in edits]
    manual = any(bool(edit["requires_manual_curation_final"]) for edit in edits)
    tier = sample_tier(tiers)
    parent_id = int(parent["peptide_id"])
    modified_id = int(modified["peptide_id"])
    base = {
        "split": split,
        "sample_id": sample_id,
        "source_id": source_id,
        "source_slug": slug,
        "parent_peptide_id": parent_id,
        "modified_peptide_id": modified_id,
        "unordered_pair_id": (
            f"{slug}:{min(parent_id, modified_id)}_{max(parent_id, modified_id)}:"
            f"{parent_id}_to_{modified_id}"
        ),
        "parent_name": parent["peptide_name"],
        "modified_name": modified["peptide_name"],
        "canonical_shadow_sequence_final": candidate["shadow"],
        "parent_monomer_list": parent["standardized_monomer_list"],
        "modified_monomer_list": modified["standardized_monomer_list"],
        "parent_smiles": parent["smiles_raw"],
        "modified_smiles": modified["smiles_raw"],
        "assay_type": "PAMPA",
        "property_before": before,
        "property_after": after,
        "delta_property": after - before,
        "max_edits_allowed": int(candidate["max_edits_allowed"]),
        "censored_property_flag": censored,
        "confidence_weight": confidence_weight(tiers, censored, manual),
        "directional_pair": True,
        "reverse_pair_removed_flag": False,
        "pair_direction_rule": "v2_keep_both_directions",
        "candidate_generation_strategy": generation_strategy,
        "candidate_group_id": (
            f"{slug}:PAMPA:parent_{parent_id}:"
            f"shadow_{export_v1.normalize_source_id(str(candidate['shadow']))}"
        ),
        "sample_model_use_tier": tier,
        "contains_manual_curation_monomer_final": manual,
        "contains_unknown_replacement": tier == "atom_unknown_auxiliary",
        "contains_residue_graph_replacement": any(
            str(edit["quality_flag_final"]) == "residue_graph_replacement" for edit in edits
        ),
        "notes": "v2.0 keeps positive and negative pair directions.",
        "edit_count": edit_count,
    }
    if edit_count == 1:
        return {**base, **edits[0], "table_type": "single"}
    return {
        **base,
        "edit_set_json": json.dumps(edits, ensure_ascii=False, sort_keys=True),
        "table_type": "multi",
    }


def selected_candidates_for_source(
    source_id: str,
    peptides: pd.DataFrame,
    annotations: dict[str, pd.Series],
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    source = peptides[peptides["source_id"].eq(source_id)].copy()
    source = source[source[args.assay].notna()].copy()
    source["monomers"] = source["standardized_monomer_list"].map(export_v1.parse_monomers)
    selected: list[dict[str, object]] = []
    raw_directional = 0
    source_cap_pool: list[dict[str, object]] = []
    large_group_count = 0

    for shadow, group in source.groupby("canonical_shadow_sequence_final"):
        if len(group) < 2:
            continue
        candidates = candidate_rows_for_group(
            source_id,
            args.assay,
            shadow,
            group,
            annotations,
            args.max_edits,
            args.max_edit_fraction,
        )
        raw_directional += len(candidates)
        if len(group) > args.max_group_size:
            large_group_count += 1
            candidates = sorted(
                candidates,
                key=lambda item: (
                    float(item["abs_delta_property"]),
                    -int(item["edit_count"]),
                    str(item["direction_key"]),
                ),
                reverse=True,
            )[: args.max_pairs_per_large_group]
            for candidate in candidates:
                candidate["generation_strategy"] = "v2_large_group_top_abs_delta_directional_cap500"
        else:
            for candidate in candidates:
                candidate["generation_strategy"] = "v2_same_shadow_all_directional_pairs_filtered"
        if source_id == args.townsend_source_id:
            source_cap_pool.extend(candidates)
        else:
            selected.extend(candidates)

    if source_id == args.townsend_source_id:
        source_cap_pool = sorted(
            source_cap_pool,
            key=lambda item: (
                float(item["abs_delta_property"]),
                -int(item["edit_count"]),
                str(item["direction_key"]),
            ),
            reverse=True,
        )
        for candidate in source_cap_pool[: args.townsend_max_rows]:
            candidate["generation_strategy"] = (
                f"{candidate['generation_strategy']}_source_cap{args.townsend_max_rows}"
            )
            selected.append(candidate)

    summary = {
        "source_id": source_id,
        "raw_directional_candidates": raw_directional,
        "selected_rows": len(selected),
        "large_groups": large_group_count,
        "source_cap": args.townsend_max_rows if source_id == args.townsend_source_id else None,
    }
    return selected, summary


def assign_pair_component_splits(
    records: list[dict[str, object]],
    seed: int,
    train_frac: float,
    val_frac: float,
) -> dict[str, str]:
    component_samples: dict[str, set[str]] = {}
    component_sources: dict[str, str] = {}
    peptide_to_component: dict[str, str] = {}

    def find(item: str) -> str:
        if item not in peptide_to_component:
            peptide_to_component[item] = item
        if peptide_to_component[item] != item:
            peptide_to_component[item] = find(peptide_to_component[item])
        return peptide_to_component[item]

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        if left_root < right_root:
            peptide_to_component[right_root] = left_root
        else:
            peptide_to_component[left_root] = right_root

    for record in records:
        left = f"{record['source_id']}:{record['parent_peptide_id']}"
        right = f"{record['source_id']}:{record['modified_peptide_id']}"
        union(left, right)

    for record in records:
        root = find(f"{record['source_id']}:{record['parent_peptide_id']}")
        component_samples.setdefault(root, set()).add(str(record["sample_id"]))
        component_sources[root] = str(record["source_id"])

    sample_to_split: dict[str, str] = {}
    test_frac = 1.0 - train_frac - val_frac
    for source_id in sorted(set(component_sources.values())):
        source_components = [
            {
                "root": root,
                "rows": len(component_samples[root]),
                "jitter": stable_float(root, seed),
            }
            for root, active_source in component_sources.items()
            if active_source == source_id
        ]
        source_components.sort(key=lambda item: (-int(item["rows"]), float(item["jitter"])))
        if len(source_components) >= 3:
            forced = ["train", "val", "test"]
        elif len(source_components) == 2:
            forced = ["train", "test"]
        else:
            forced = ["train"]

        split_rows = {"train": 0, "val": 0, "test": 0}
        total_rows = sum(int(item["rows"]) for item in source_components)
        targets = {
            "train": total_rows * train_frac,
            "val": total_rows * val_frac,
            "test": total_rows * test_frac,
        }

        for item, split in zip(source_components, forced):
            item["split"] = split
            split_rows[split] += int(item["rows"])

        for item in source_components[len(forced) :]:
            best_split = None
            best_score = None
            for split in ("train", "val", "test"):
                candidate_rows = dict(split_rows)
                candidate_rows[split] += int(item["rows"])
                score = (
                    sum(abs(candidate_rows[name] - targets[name]) for name in ("train", "val", "test")),
                    -targets[split],
                    split,
                )
                if best_score is None or score < best_score:
                    best_score = score
                    best_split = split
            assert best_split is not None
            item["split"] = best_split
            split_rows[best_split] += int(item["rows"])

        for item in source_components:
            for sample_id in component_samples[str(item["root"])]:
                sample_to_split[sample_id] = str(item["split"])
    return sample_to_split


def write_model_ready(records: list[dict[str, object]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "cedg_score_dataset.jsonl"
    csv_path = out_dir / "cedg_score_dataset.csv"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    pd.DataFrame(records).to_csv(csv_path, index=False)

    rows: list[dict[str, object]] = []
    for (source_id, split), group in pd.DataFrame(records).groupby(["source_id", "split"]):
        rows.append({"source_id": source_id, "split": split, "rows": len(group)})
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "cedg_score_dataset_summary.csv", index=False)


def load_faris_training_records(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("source_id") == "2024_Faris":
                records.append({field: record.get(field) for field in MODEL_READY_FIELDS})
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assay", default="PAMPA")
    parser.add_argument("--max-edits", type=int, default=4)
    parser.add_argument("--max-edit-fraction", type=float, default=0.25)
    parser.add_argument("--max-group-size", type=int, default=50)
    parser.add_argument("--max-pairs-per-large-group", type=int, default=500)
    parser.add_argument("--townsend-source-id", default="2020_Townsend")
    parser.add_argument("--townsend-max-rows", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--out-root", type=Path, default=ROOT / "data" / "final" / "model_ready")
    parser.add_argument("--ori-name", default="ori_v2_0")
    parser.add_argument("--plus-name", default="ori_v2_0_plus_faris2024")
    parser.add_argument("--faris-training-jsonl", type=Path, default=DEFAULT_FARIS_TRAINING)
    args = parser.parse_args()

    peptides = pd.read_csv(export_v1.PEPTIDE_TABLE)
    monomers = pd.read_csv(export_v1.MONOMER_TABLE)
    annotations = export_v1.monomer_lookup(monomers)

    flat_rows: list[dict[str, object]] = []
    source_summaries: list[dict[str, object]] = []
    sample_index = 1
    for source_id in DEFAULT_SOURCES:
        candidates, summary = selected_candidates_for_source(source_id, peptides, annotations, args)
        source_summaries.append(summary)
        for candidate in candidates:
            flat_rows.append(
                candidate_to_flat_row(
                    candidate,
                    sample_id=f"{SOURCE_SLUGS[source_id]}_v2_sample_{sample_index:06d}",
                    generation_strategy=str(candidate["generation_strategy"]),
                    split="",
                )
            )
            sample_index += 1

    sample_to_split = assign_pair_component_splits(flat_rows, args.seed, args.train_frac, args.val_frac)
    records: list[dict[str, object]] = []
    for flat in flat_rows:
        flat["split"] = sample_to_split[str(flat["sample_id"])]
        table_type = str(flat.pop("table_type"))
        records.append(row_to_record(pd.Series(flat), table_type))

    ori_dir = args.out_root / args.ori_name
    write_model_ready(records, ori_dir)

    faris_records = load_faris_training_records(args.faris_training_jsonl)
    plus_dir = args.out_root / args.plus_name
    write_model_ready(records + faris_records, plus_dir)

    summary = {
        "version": "v2.0",
        "rules": {
            "keep_reverse_directions": True,
            "max_pairs_per_large_group": args.max_pairs_per_large_group,
            "townsend_source_cap": args.townsend_max_rows,
            "split_scheme": "source-stratified peptide component balanced split",
        },
        "ori_rows": len(records),
        "faris2024_rows": len(faris_records),
        "ori_plus_faris2024_rows": len(records) + len(faris_records),
        "source_summaries": source_summaries,
    }
    for out_dir in (ori_dir, plus_dir):
        (out_dir / "cedg_v2_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
