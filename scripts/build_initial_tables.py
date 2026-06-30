#!/usr/bin/env python3
"""中文说明：从 CycPeptMPDB 导出文件构建第一版 CEDG 原始样本表。"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw_sources" / "cycpeptmpdb"
NOTE_DIR = ROOT / "data" / "notes"
OUT_DIR = NOTE_DIR / "initial_build_outputs"

PEPTIDE_RAW = RAW_DIR / "CycPeptMPDB_Peptide_All.csv"
MONOMER_RAW = RAW_DIR / "monomer_table.csv"
UNIQUE_MONOMER_RAW = RAW_DIR / "unique_monomer.csv"

PEPTIDE_OUT = OUT_DIR / "peptide_sample_table.csv"
SOURCE_QUEUE_OUT = OUT_DIR / "source_paper_queue.csv"
SOURCE_SUMMARY_OUT = OUT_DIR / "source_summary.csv"
MONOMER_COUNTS_OUT = OUT_DIR / "monomer_frequency.csv"
REPORT_OUT = NOTE_DIR / "initial_table_build_report.md"


CORE_COLUMNS = [
    "CycPeptMPDB_ID",
    "Source",
    "Year",
    "Original_Name_in_Source_Literature",
    "Structurally_Unique_ID",
    "Same_Peptides_ID",
    "SMILES",
    "HELM",
    "Sequence",
    "Monomer_Length",
    "Monomer_Length_in_Main_Chain",
    "Molecule_Shape",
    "Permeability",
    "PAMPA",
    "Caco2",
    "MDCK",
    "RRCK",
]

ASSAY_COLUMNS = ["PAMPA", "Caco2", "MDCK", "RRCK"]


def parse_sequence(value: object) -> list[str]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item).strip() for item in parsed]


def normalize_anchor(monomer: str) -> str:
    """Coarse shadow mapping for triage, not final anchor annotation."""
    token = monomer.strip().strip("[]")
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


def shadow_sequence(monomers: list[str]) -> str:
    return ".".join(normalize_anchor(m) for m in monomers)


def count_changed_sites(a: str, b: str) -> int | None:
    left = a.split(".") if isinstance(a, str) and a else []
    right = b.split(".") if isinstance(b, str) and b else []
    if len(left) != len(right) or not left:
        return None
    return sum(x != y for x, y in zip(left, right))


def priority_for_source(row: pd.Series) -> str:
    source = row["source_id"]
    if source in {"2020_Townsend", "2021_Kelly", "2015_Wang"}:
        return "A"
    if source in {
        "2016_Furukawa",
        "2020_Le Roux",
        "2022_Taechalertpaisarn",
        "2021_Golosov",
        "2015_Bockus_2",
        "2015_Hewitt",
        "2018_Naylor",
    }:
        return "B"
    if row["samples_in_repeated_shadow_groups"] >= 20 or row["n_pampa"] >= 100:
        return "C"
    return "D"


def build_peptide_sample_table(raw: pd.DataFrame) -> pd.DataFrame:
    missing = [col for col in CORE_COLUMNS if col not in raw.columns]
    if missing:
        raise ValueError(f"Missing required raw columns: {missing}")

    df = raw[CORE_COLUMNS].copy()
    df = df.rename(
        columns={
            "CycPeptMPDB_ID": "peptide_id",
            "Source": "source_id",
            "Year": "source_year",
            "Original_Name_in_Source_Literature": "peptide_name",
            "Structurally_Unique_ID": "structurally_unique_id",
            "Same_Peptides_ID": "same_peptides_id",
            "SMILES": "smiles_raw",
            "HELM": "helm_raw",
            "Sequence": "monomer_list_raw",
            "Monomer_Length": "length",
            "Monomer_Length_in_Main_Chain": "main_chain_length",
            "Molecule_Shape": "molecule_shape",
            "Permeability": "permeability",
        }
    )

    monomer_lists = df["monomer_list_raw"].map(parse_sequence)
    df["standardized_monomer_list"] = monomer_lists.map(lambda xs: ";".join(xs))
    df["canonical_shadow_sequence_v0"] = monomer_lists.map(shadow_sequence)
    df["monomer_count_from_sequence"] = monomer_lists.map(len)
    df["is_cyclic"] = df["molecule_shape"].astype(str).str.contains(
        "Circle|Lariat|Cyclic", case=False, na=False
    )
    df["has_smiles"] = df["smiles_raw"].notna() & (df["smiles_raw"].astype(str).str.len() > 0)
    df["has_helm"] = df["helm_raw"].notna() & (df["helm_raw"].astype(str).str.len() > 0)
    df["has_monomer_list"] = df["monomer_count_from_sequence"] > 0
    df["primary_assay"] = df[ASSAY_COLUMNS].notna().idxmax(axis=1)
    df.loc[~df[ASSAY_COLUMNS].notna().any(axis=1), "primary_assay"] = ""
    df["quality_flag"] = "raw_import"
    df["manual_check"] = False
    df["notes"] = ""
    return df


def build_source_summary(peptide_table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for source, group in peptide_table.groupby("source_id", dropna=False):
        shadow_sizes = group.groupby("canonical_shadow_sequence_v0").size().sort_values(
            ascending=False
        )
        repeated = shadow_sizes[shadow_sizes >= 2]
        rows.append(
            {
                "source_id": source,
                "source_year": group["source_year"].dropna().min(),
                "n_peptides": len(group),
                "n_unique_structures": group["structurally_unique_id"].nunique(dropna=True),
                "n_pampa": group["PAMPA"].notna().sum(),
                "n_caco2": group["Caco2"].notna().sum(),
                "n_mdck": group["MDCK"].notna().sum(),
                "n_rrck": group["RRCK"].notna().sum(),
                "n_with_smiles": group["has_smiles"].sum(),
                "n_with_helm": group["has_helm"].sum(),
                "n_with_monomer_list": group["has_monomer_list"].sum(),
                "n_shadow_groups": shadow_sizes.shape[0],
                "shadow_groups_ge2": repeated.shape[0],
                "samples_in_repeated_shadow_groups": int(repeated.sum()),
                "max_shadow_group_size": int(shadow_sizes.iloc[0]) if len(shadow_sizes) else 0,
                "top_shadow_sequence_v0": shadow_sizes.index[0] if len(shadow_sizes) else "",
            }
        )
    summary = pd.DataFrame(rows).sort_values(
        ["samples_in_repeated_shadow_groups", "n_peptides"], ascending=False
    )
    summary["priority"] = summary.apply(priority_for_source, axis=1)
    summary["queue_reason"] = summary.apply(
        lambda r: (
            "manual_pair_mining"
            if r["priority"] in {"A", "B"}
            else "hold_for_later"
        ),
        axis=1,
    )
    return summary


def build_source_queue(summary: pd.DataFrame) -> pd.DataFrame:
    queue = summary[summary["priority"].isin(["A", "B", "C"])].copy()
    queue = queue[
        [
            "source_id",
            "source_year",
            "priority",
            "n_peptides",
            "n_pampa",
            "n_caco2",
            "shadow_groups_ge2",
            "samples_in_repeated_shadow_groups",
            "max_shadow_group_size",
            "top_shadow_sequence_v0",
            "queue_reason",
        ]
    ]
    queue["doi"] = ""
    queue["paper_title"] = ""
    queue["supplementary_table_status"] = "unknown"
    queue["compound_name_mapping_status"] = "unknown"
    queue["dominant_edit_types"] = ""
    queue["estimated_single_edit_pairs"] = ""
    queue["estimated_multi_edit_pairs"] = ""
    queue["manual_notes"] = ""
    return queue


def build_monomer_frequency(peptide_table: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for _, row in peptide_table.iterrows():
        for monomer in str(row["standardized_monomer_list"]).split(";"):
            if not monomer or monomer == "nan":
                continue
            records.append(
                {
                    "monomer": monomer,
                    "source_id": row["source_id"],
                    "peptide_id": row["peptide_id"],
                    "anchor_guess_v0": normalize_anchor(monomer),
                }
            )
    counts = pd.DataFrame(records)
    if counts.empty:
        return pd.DataFrame(columns=["monomer", "count", "source_count", "anchor_guess_v0"])
    return (
        counts.groupby("monomer")
        .agg(
            count=("peptide_id", "count"),
            source_count=("source_id", "nunique"),
            anchor_guess_v0=("anchor_guess_v0", "first"),
        )
        .reset_index()
        .sort_values(["count", "source_count"], ascending=False)
    )


def write_report(
    peptide_table: pd.DataFrame,
    summary: pd.DataFrame,
    monomer_frequency: pd.DataFrame,
) -> None:
    assay_counts = {assay: int(peptide_table[assay].notna().sum()) for assay in ASSAY_COLUMNS}
    top_sources = summary.head(15)
    top_monomers = monomer_frequency.head(25)

    report = [
        "# Initial Table Build Report",
        "",
        "Generated from CycPeptMPDB-derived public tables.",
        "",
        "## Outputs",
        "",
        f"- `{PEPTIDE_OUT.relative_to(ROOT)}`",
        f"- `{SOURCE_SUMMARY_OUT.relative_to(ROOT)}`",
        f"- `{SOURCE_QUEUE_OUT.relative_to(ROOT)}`",
        f"- `{MONOMER_COUNTS_OUT.relative_to(ROOT)}`",
        "",
        "## Dataset Size",
        "",
        f"- peptide rows: {len(peptide_table)}",
        f"- sources: {peptide_table['source_id'].nunique()}",
        f"- rows with SMILES: {int(peptide_table['has_smiles'].sum())}",
        f"- rows with HELM: {int(peptide_table['has_helm'].sum())}",
        f"- rows with monomer lists: {int(peptide_table['has_monomer_list'].sum())}",
        "",
        "## Assay Coverage",
        "",
    ]
    report.extend([f"- {assay}: {count}" for assay, count in assay_counts.items()])
    report.extend(
        [
            "",
            "## Top Sources By Repeated Rough Shadow Groups",
            "",
            top_sources[
                [
                    "source_id",
                    "n_peptides",
                    "n_pampa",
                    "n_caco2",
                    "samples_in_repeated_shadow_groups",
                    "max_shadow_group_size",
                    "priority",
                ]
            ].to_markdown(index=False),
            "",
            "## Top Monomers",
            "",
            top_monomers.to_markdown(index=False),
            "",
            "## Caveats",
            "",
            "- `canonical_shadow_sequence_v0` is only a triage normalization.",
            "- Repeated rough-shadow groups are not yet validated scaffold groups.",
            "- No edit pairs are generated in this step.",
            "- Source papers still need DOI and supplementary table checks.",
            "",
        ]
    )
    REPORT_OUT.write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    NOTE_DIR.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(PEPTIDE_RAW, low_memory=False)
    peptide_table = build_peptide_sample_table(raw)
    source_summary = build_source_summary(peptide_table)
    source_queue = build_source_queue(source_summary)
    monomer_frequency = build_monomer_frequency(peptide_table)

    peptide_table.to_csv(PEPTIDE_OUT, index=False)
    source_summary.to_csv(SOURCE_SUMMARY_OUT, index=False)
    source_queue.to_csv(SOURCE_QUEUE_OUT, index=False)
    monomer_frequency.to_csv(MONOMER_COUNTS_OUT, index=False)
    write_report(peptide_table, source_summary, monomer_frequency)

    print(f"Wrote {PEPTIDE_OUT.relative_to(ROOT)}: {len(peptide_table)} rows")
    print(f"Wrote {SOURCE_SUMMARY_OUT.relative_to(ROOT)}: {len(source_summary)} rows")
    print(f"Wrote {SOURCE_QUEUE_OUT.relative_to(ROOT)}: {len(source_queue)} rows")
    print(f"Wrote {MONOMER_COUNTS_OUT.relative_to(ROOT)}: {len(monomer_frequency)} rows")
    print(f"Wrote {REPORT_OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
