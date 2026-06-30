#!/usr/bin/env python3
"""中文说明：为最终 CEDG 训练表构建避免 peptide/scaffold 泄漏的训练、验证、测试划分。"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TRAINING_DIR = ROOT / "data" / "final" / "training"
SPLIT_DIR = ROOT / "data" / "final" / "splits"

SPLITS = ("train", "val", "test")
DEFAULT_FRACTIONS = {"train": 0.8, "val": 0.1, "test": 0.1}


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, item: str) -> str:
        if item not in self.parent:
            self.parent[item] = item
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if left_root < right_root:
            self.parent[right_root] = left_root
        else:
            self.parent[left_root] = right_root


def stable_slug(value: object, prefix: str) -> str:
    digest = hashlib.sha1(str(value).encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    singles: list[pd.DataFrame] = []
    multis: list[pd.DataFrame] = []
    for source_dir in sorted(TRAINING_DIR.iterdir()):
        if not source_dir.is_dir():
            continue
        single_path = source_dir / "training_single_site.csv"
        multi_path = source_dir / "training_multi_site.csv"
        if single_path.exists():
            single = pd.read_csv(single_path)
            single["source_slug"] = source_dir.name
            single["table_type"] = "single"
            singles.append(single)
        if multi_path.exists():
            multi = pd.read_csv(multi_path)
            multi["source_slug"] = source_dir.name
            multi["table_type"] = "multi"
            multis.append(multi)

    single_all = pd.concat(singles, ignore_index=True) if singles else pd.DataFrame()
    multi_all = pd.concat(multis, ignore_index=True) if multis else pd.DataFrame()
    all_rows = pd.concat([single_all, multi_all], ignore_index=True)
    return single_all, multi_all, all_rows


def add_peptide_component_ids(all_rows: pd.DataFrame) -> pd.Series:
    uf = UnionFind()
    for row in all_rows.itertuples(index=False):
        left = f"{row.source_id}:{int(row.parent_peptide_id)}"
        right = f"{row.source_id}:{int(row.modified_peptide_id)}"
        uf.union(left, right)

    root_to_component: dict[str, str] = {}
    component_ids: list[str] = []
    for row in all_rows.itertuples(index=False):
        root = uf.find(f"{row.source_id}:{int(row.parent_peptide_id)}")
        if root not in root_to_component:
            root_to_component[root] = stable_slug(root, "pc")
        component_ids.append(root_to_component[root])
    return pd.Series(component_ids, index=all_rows.index)


def add_source_shadow_ids(all_rows: pd.DataFrame) -> pd.Series:
    values = (
        all_rows["source_id"].astype(str)
        + "::"
        + all_rows["canonical_shadow_sequence_final"].astype(str)
    )
    return values.map(lambda value: stable_slug(value, "ss"))


def assign_groups(
    all_rows: pd.DataFrame,
    group_col: str,
    fractions: dict[str, float],
    stratify_by_source: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_stats = (
        all_rows.groupby(group_col)
        .agg(
            rows=("sample_id", "count"),
            censored_rows=("censored_property_flag", "sum"),
            source_count=("source_id", "nunique"),
            source_ids=("source_id", lambda x: "|".join(sorted(set(map(str, x))))),
            shadow_count=("canonical_shadow_sequence_final", "nunique"),
            peptide_endpoint_count=(
                "parent_peptide_id",
                lambda x: 0,
            ),
        )
        .reset_index()
    )

    endpoint_counts = []
    for group_id, group in all_rows.groupby(group_col):
        endpoints = set(group["parent_peptide_id"].astype(int)) | set(
            group["modified_peptide_id"].astype(int)
        )
        endpoint_counts.append((group_id, len(endpoints)))
    endpoint_df = pd.DataFrame(endpoint_counts, columns=[group_col, "peptide_endpoint_count"])
    group_stats = group_stats.drop(columns=["peptide_endpoint_count"]).merge(
        endpoint_df, on=group_col, how="left"
    )

    assignments: dict[str, str] = {}

    def total_deviation(candidate_rows: dict[str, int]) -> float:
        return sum(abs(candidate_rows[split] - active_targets[split]) for split in SPLITS)

    strata = (
        sorted(group_stats["source_ids"].unique())
        if stratify_by_source
        else ["__all_sources__"]
    )
    for stratum in strata:
        if stratify_by_source:
            stratum_groups = group_stats[group_stats["source_ids"].eq(stratum)].copy()
        else:
            stratum_groups = group_stats.copy()
        stratum_total = int(stratum_groups["rows"].sum())
        active_targets = {split: stratum_total * fractions[split] for split in SPLITS}
        split_rows = {split: 0 for split in SPLITS}
        ordered = stratum_groups.sort_values(["rows", group_col], ascending=[False, True])

        for row in ordered.itertuples(index=False):
            group_id = getattr(row, group_col)
            rows = int(row.rows)
            best_split = None
            best_score = None
            for split in SPLITS:
                candidate = dict(split_rows)
                candidate[split] += rows
                score = (
                    total_deviation(candidate),
                    -active_targets[split],
                    split,
                )
                if best_score is None or score < best_score:
                    best_score = score
                    best_split = split
            assert best_split is not None
            assignments[group_id] = best_split
            split_rows[best_split] += rows

    assignment = group_stats.copy()
    assignment["split"] = assignment[group_col].map(assignments)
    summary = (
        assignment.groupby("split")
        .agg(
            groups=(group_col, "count"),
            rows=("rows", "sum"),
            censored_rows=("censored_rows", "sum"),
            source_count=("source_ids", lambda x: len(set("|".join(x).split("|")))),
            peptide_endpoint_count=("peptide_endpoint_count", "sum"),
        )
        .reset_index()
    )
    summary["target_fraction"] = summary["split"].map(fractions)
    summary["actual_fraction"] = summary["rows"] / len(all_rows)
    return assignment, summary


def apply_assignment(
    table: pd.DataFrame,
    all_rows: pd.DataFrame,
    group_col: str,
    assignment: pd.DataFrame,
) -> pd.DataFrame:
    cols = ["sample_id", group_col]
    sample_groups = all_rows[cols].drop_duplicates()
    split_map = assignment[[group_col, "split"]]
    out = table.merge(sample_groups, on="sample_id", how="left").merge(
        split_map, on=group_col, how="left"
    )
    first_cols = ["split", group_col]
    other_cols = [col for col in out.columns if col not in first_cols]
    return out[first_cols + other_cols]


def validate_no_peptide_leakage(single: pd.DataFrame, multi: pd.DataFrame) -> pd.DataFrame:
    all_rows = pd.concat([single, multi], ignore_index=True)
    records: list[dict[str, object]] = []
    peptide_to_splits: dict[str, set[str]] = {}
    for row in all_rows.itertuples(index=False):
        for peptide_id in (int(row.parent_peptide_id), int(row.modified_peptide_id)):
            key = f"{row.source_id}:{peptide_id}"
            peptide_to_splits.setdefault(key, set()).add(str(row.split))
    leaked = {key: splits for key, splits in peptide_to_splits.items() if len(splits) > 1}
    records.append(
        {
            "check": "peptide_node_split_leakage",
            "passed": len(leaked) == 0,
            "violations": len(leaked),
        }
    )

    pair_dups = int(all_rows["unordered_pair_id"].duplicated().sum())
    records.append(
        {
            "check": "unordered_pair_duplicate_rows",
            "passed": pair_dups == 0,
            "violations": pair_dups,
        }
    )
    return pd.DataFrame(records)


def write_scheme(
    scheme: str,
    group_col: str,
    single: pd.DataFrame,
    multi: pd.DataFrame,
    all_rows: pd.DataFrame,
    fractions: dict[str, float],
    stratify_by_source: bool,
) -> None:
    out_dir = SPLIT_DIR / scheme
    out_dir.mkdir(parents=True, exist_ok=True)
    assignment, summary = assign_groups(all_rows, group_col, fractions, stratify_by_source)
    single_out = apply_assignment(single, all_rows, group_col, assignment)
    multi_out = apply_assignment(multi, all_rows, group_col, assignment)
    validation = validate_no_peptide_leakage(single_out, multi_out)

    assignment.to_csv(out_dir / "split_assignment.csv", index=False)
    summary.to_csv(out_dir / "split_summary.csv", index=False)
    validation.to_csv(out_dir / "split_validation.csv", index=False)
    single_out.to_csv(out_dir / "training_single_site.csv", index=False)
    multi_out.to_csv(out_dir / "training_multi_site.csv", index=False)

    print(
        f"{scheme}: single={len(single_out)} multi={len(multi_out)} "
        f"validation_passed={bool(validation['passed'].all())}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument(
        "--no-source-stratify",
        action="store_true",
        help="Assign groups globally instead of independently within each source.",
    )
    args = parser.parse_args()

    fractions = {
        "train": args.train_frac,
        "val": args.val_frac,
        "test": args.test_frac,
    }
    total_fraction = sum(fractions.values())
    if abs(total_fraction - 1.0) > 1e-9:
        raise ValueError(f"Split fractions must sum to 1.0, got {total_fraction}")

    single, multi, all_rows = load_tables()
    if all_rows.empty:
        raise ValueError(f"No final training rows found under {TRAINING_DIR}")

    all_rows = all_rows.copy()
    all_rows["peptide_component_id"] = add_peptide_component_ids(all_rows)
    all_rows["source_shadow_id"] = add_source_shadow_ids(all_rows)

    stratify_by_source = not args.no_source_stratify
    write_scheme(
        "peptide_component",
        "peptide_component_id",
        single,
        multi,
        all_rows,
        fractions,
        stratify_by_source,
    )
    write_scheme(
        "source_shadow",
        "source_shadow_id",
        single,
        multi,
        all_rows,
        fractions,
        stratify_by_source,
    )


if __name__ == "__main__":
    main()
