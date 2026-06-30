#!/usr/bin/env python3
"""中文说明：构建基于规则的单体 anchor、edit scope、attachment 和 payload 注释表。"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
NOTE_DIR = ROOT / "data" / "notes"
OUT_DIR = NOTE_DIR / "automatic_annotation_drafts"
FREQ = NOTE_DIR / "initial_build_outputs" / "monomer_frequency.csv"
UNIQUE = ROOT / "data" / "raw_sources" / "cycpeptmpdb" / "unique_monomer.csv"
OUT = OUT_DIR / "monomer_anchor_table_auto_v4.csv"
REVIEW_OUT = OUT_DIR / "monomer_anchor_manual_review_queue_auto_v4.csv"
REPORT = OUT_DIR / "monomer_anchor_table_auto_v4_report.md"


CANONICAL = set("ARNDCQEGHILKMFPSTWYV")

ALIASES = {
    "dL": "L",
    "K": "K",
}

PSEUDO_ANCHORS = {
    "Nle": "Leu-like",
    "Nva": "Val/Leu-like",
    "Abu": "Ala/Val-like",
    "Bal": "Gly/Ala-like",
    "Sar": "Gly-like",
    "Aib": "Ala-like",
    "Nal": "Phe-like",
    "1-Nal": "Phe-like",
    "Cha": "Phe/Leu-like",
    "Hph": "Phe-like",
    "bHph": "Phe-like",
    "Tle": "Val/Leu-like",
    "Pip": "Pro-like",
    "Hyp": "Pro-like",
    "Phg": "Phe-like",
    "Pal": "Phe-like",
    "3Pal": "Phe-like",
    "Tza": "Phe-like",
    "Sta": "Leu/Phe-like",
    "GABA": "beta/extended-backbone",
    "5-Ava": "beta/extended-backbone",
    "Aoc(2)": "beta/extended-backbone",
    "Pye": "Glu-like",
}

AROMATIC_BASES = {"F", "Y", "W", "Phe", "Tyr", "Trp"}


def base_name(monomer: str) -> str:
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


def is_d(monomer: str) -> bool:
    token = str(monomer)
    return token.startswith("d") or token.startswith("D-") or token.startswith("Me_d")


def is_n_methyl(monomer: str) -> bool:
    token = str(monomer)
    return token.startswith("me") or token.startswith("Me_") or token.startswith("N-Me")


def is_n_substituted_gly_smiles(smiles: object) -> bool:
    if pd.isna(smiles):
        return False
    text = str(smiles)
    return "N(CC=O)" in text or "N(C)CC=O" in text or "CN(CC=O)" in text


def infer_anchor(monomer: str, smiles: object = None) -> tuple[str, str, str]:
    base = base_name(monomer)
    if base in CANONICAL:
        return base, "", "rule_canonical"
    if base in PSEUDO_ANCHORS:
        return "", PSEUDO_ANCHORS[base], "rule_pseudo_anchor"
    if base.startswith("Cha") or base == "Cha":
        return "", "Phe/Leu-like", "rule_cha_pseudo_anchor"
    if base.startswith("Hyp"):
        return "", "Pro-like", "rule_hyp_pseudo_anchor"
    if base.startswith("Aib"):
        return "", "Ala-like", "rule_aib_pseudo_anchor"
    if base.startswith("Phe(") or base.startswith("dPhe("):
        return "F", "", "rule_phe_substitution"
    if base.startswith("Tyr(") or base.startswith("dTyr("):
        return "Y", "", "rule_tyr_substitution"
    if base.startswith("Ser("):
        return "S", "", "rule_ser_sidechain_substitution"
    if base.startswith("Lys("):
        return "K", "", "rule_lys_sidechain_substitution"
    if base.startswith("Gln("):
        return "Q", "", "rule_gln_sidechain_substitution"
    if base.startswith("Asp("):
        return "D", "", "rule_asp_sidechain_substitution"
    if base.startswith("Nle("):
        return "", "Leu-like", "rule_nle_substitution"
    if base.startswith("Nva("):
        return "", "Val/Leu-like", "rule_nva_substitution"
    if base.startswith("Ala("):
        return "A", "", "rule_ala_sidechain_substitution"
    if base in {"Ala(indol-2-yl)", "Me_Ala(indol-2-yl)"}:
        return "", "Trp-like", "rule_trp_like"
    if base.endswith("_Gly") or "Gly" in base:
        return "", "Gly-like", "rule_gly_substituted"
    if is_n_substituted_gly_smiles(smiles):
        return "", "Gly-like", "rule_smiles_n_substituted_gly"
    if base.startswith("Mono"):
        return "", base, "rule_mono_fallback"
    return "", base, "rule_unknown_pseudo"


def aromatic_payload(base: str) -> tuple[str, str]:
    match = re.search(r"\((.+)\)", base)
    if not match:
        return "", ""
    substituent = match.group(1)
    if substituent in {"4-CF3", "3-Cl", "3,4-diF", "4-Cl", "4-F"}:
        return "aromatic_substitution", f"aryl_{substituent}"
    return "aromatic_substitution", f"aryl_{substituent}"


def infer_default_edit(
    monomer: str, canonical_anchor: str, pseudo_anchor: str, smiles: object = None
) -> dict[str, str]:
    base = base_name(monomer)
    d_flag = is_d(monomer)
    nme_flag = is_n_methyl(monomer)
    substitution_match = re.search(r"\((.+)\)", base)

    if monomer in CANONICAL:
        return {
            "edit_scope_default": "none",
            "operation_default": "none",
            "attachment_default": "",
            "chemical_payload_type_default": "",
            "chemical_payload_template": "",
        }

    if nme_flag and substitution_match and canonical_anchor in {"F", "Y", "W", "S", "K", "Q", "D", "A"}:
        payload = substitution_match.group(1)
        return {
            "edit_scope_default": "backbone_and_R_group_edit",
            "operation_default": "N_methylation+substitution",
            "attachment_default": "backbone_N;side_chain",
            "chemical_payload_type_default": "combined_delta",
            "chemical_payload_template": f"N-H_to_N-CH3;sidechain_to_{payload}",
        }

    if d_flag and substitution_match and canonical_anchor in {"F", "Y", "W", "S", "K", "Q", "D", "A"}:
        payload = substitution_match.group(1)
        return {
            "edit_scope_default": "stereochemistry_and_R_group_edit",
            "operation_default": "D_L_inversion+substitution",
            "attachment_default": "alpha_C;side_chain",
            "chemical_payload_type_default": "combined_delta",
            "chemical_payload_template": f"L_to_D;sidechain_to_{payload}",
        }

    if nme_flag and d_flag and canonical_anchor in CANONICAL:
        return {
            "edit_scope_default": "backbone_and_stereochemistry_edit",
            "operation_default": "N_methylation+D_L_inversion",
            "attachment_default": "backbone_N;alpha_C",
            "chemical_payload_type_default": "combined_delta",
            "chemical_payload_template": "N-H_to_N-CH3;L_to_D",
        }

    if nme_flag and canonical_anchor in CANONICAL:
        return {
            "edit_scope_default": "backbone_edit",
            "operation_default": "N_methylation",
            "attachment_default": "backbone_N",
            "chemical_payload_type_default": "delta_graph",
            "chemical_payload_template": "N-H_to_N-CH3",
        }

    if d_flag and (canonical_anchor in CANONICAL or pseudo_anchor):
        return {
            "edit_scope_default": "stereochemistry_edit",
            "operation_default": "D_L_inversion",
            "attachment_default": "alpha_C",
            "chemical_payload_type_default": "chirality_change_flag",
            "chemical_payload_template": "L_to_D",
        }

    if base.startswith("Phe(") or base.startswith("Tyr("):
        operation, payload = aromatic_payload(base)
        return {
            "edit_scope_default": "R_group_edit",
            "operation_default": operation,
            "attachment_default": "aryl_ring",
            "chemical_payload_type_default": "r_group_delta",
            "chemical_payload_template": payload,
        }

    if base.startswith("Ser("):
        payload = substitution_match.group(1) if substitution_match else base
        return {
            "edit_scope_default": "R_group_edit",
            "operation_default": "serine_O_substitution",
            "attachment_default": "serine_O",
            "chemical_payload_type_default": "r_group_replacement",
            "chemical_payload_template": f"Ser_OH_to_O-{payload}",
        }

    if base.startswith("Lys("):
        payload = substitution_match.group(1) if substitution_match else base
        return {
            "edit_scope_default": "R_group_edit",
            "operation_default": "lysine_side_chain_substitution",
            "attachment_default": "lysine_epsilon_N",
            "chemical_payload_type_default": "r_group_replacement",
            "chemical_payload_template": f"Lys_sidechain_to_{payload}",
        }

    if base.startswith("Gln("):
        payload = substitution_match.group(1) if substitution_match else base
        return {
            "edit_scope_default": "R_group_edit",
            "operation_default": "glutamine_side_chain_substitution",
            "attachment_default": "glutamine_side_chain_amide",
            "chemical_payload_type_default": "r_group_replacement",
            "chemical_payload_template": f"Gln_sidechain_to_{payload}",
        }

    if base.startswith("Asp("):
        payload = substitution_match.group(1) if substitution_match else base
        return {
            "edit_scope_default": "R_group_edit",
            "operation_default": "aspartate_side_chain_derivatization",
            "attachment_default": "aspartate_side_chain_carboxyl",
            "chemical_payload_type_default": "r_group_replacement",
            "chemical_payload_template": f"Asp_sidechain_to_{payload}",
        }

    if base.startswith("Ala("):
        payload = substitution_match.group(1) if substitution_match else base
        return {
            "edit_scope_default": "R_group_edit",
            "operation_default": "alanine_side_chain_substitution",
            "attachment_default": "alanine_beta_substituent",
            "chemical_payload_type_default": "r_group_replacement",
            "chemical_payload_template": f"Ala_sidechain_to_{payload}",
        }

    if base.startswith("Nle(") or base.startswith("Nva("):
        payload = substitution_match.group(1) if substitution_match else base
        return {
            "edit_scope_default": "R_group_edit",
            "operation_default": "alkyl_side_chain_substitution",
            "attachment_default": "side_chain",
            "chemical_payload_type_default": "r_group_replacement",
            "chemical_payload_template": f"{pseudo_anchor}_to_{payload}",
        }

    if base == "Aib" or pseudo_anchor == "Ala-like":
        return {
            "edit_scope_default": "residue_replacement",
            "operation_default": "alpha_methylation_or_alpha_disubstitution",
            "attachment_default": "alpha_C",
            "chemical_payload_type_default": "full_residue_graph",
            "chemical_payload_template": f"Ala-like_to_{base}",
        }

    if pseudo_anchor in {"Leu-like", "Val/Leu-like", "Ala/Val-like", "Phe-like", "Trp-like", "Phe/Leu-like"}:
        return {
            "edit_scope_default": "residue_replacement",
            "operation_default": "hydrophobic_R_group_replacement",
            "attachment_default": "side_chain",
            "chemical_payload_type_default": "full_residue_graph",
            "chemical_payload_template": f"{canonical_anchor or pseudo_anchor}_to_{base}",
        }

    if pseudo_anchor == "Pro-like":
        operation = "hydroxyproline_or_proline_ring_substitution" if base.startswith("Hyp") else "proline_ring_or_hydroxylation_replacement"
        return {
            "edit_scope_default": "residue_replacement",
            "operation_default": operation,
            "attachment_default": "side_chain_or_ring",
            "chemical_payload_type_default": "full_residue_graph",
            "chemical_payload_template": f"Pro-like_to_{base}",
        }

    if pseudo_anchor == "Gly-like":
        return {
            "edit_scope_default": "residue_replacement",
            "operation_default": "N_substituted_glycine_or_glycine_like",
            "attachment_default": "side_chain_or_backbone_N",
            "chemical_payload_type_default": "full_residue_graph",
            "chemical_payload_template": f"Gly-like_to_{base}",
        }

    if is_n_substituted_gly_smiles(smiles):
        return {
            "edit_scope_default": "residue_replacement",
            "operation_default": "N_substituted_glycine_or_peptoid_like",
            "attachment_default": "backbone_N_or_side_chain",
            "chemical_payload_type_default": "full_residue_graph",
            "chemical_payload_template": f"Gly-like_to_{base}",
        }

    if pseudo_anchor == "beta/extended-backbone":
        return {
            "edit_scope_default": "backbone_edit",
            "operation_default": "backbone_extension_or_omega_amino_acid",
            "attachment_default": "backbone",
            "chemical_payload_type_default": "full_residue_graph",
            "chemical_payload_template": base,
        }

    if pseudo_anchor in {"Glu-like", "Leu/Phe-like"}:
        return {
            "edit_scope_default": "residue_replacement",
            "operation_default": "side_chain_or_backbone_mimetic_replacement",
            "attachment_default": "side_chain_or_backbone",
            "chemical_payload_type_default": "full_residue_graph",
            "chemical_payload_template": f"{pseudo_anchor}_to_{base}",
        }

    if base in {"ac-", "-pip"} or base.endswith("-"):
        return {
            "edit_scope_default": "terminal_or_capping_edit",
            "operation_default": "terminal_capping_or_fragment",
            "attachment_default": "terminal_or_linker",
            "chemical_payload_type_default": "full_residue_graph",
            "chemical_payload_template": base,
        }

    return {
        "edit_scope_default": "full_residue_fallback",
        "operation_default": "full_residue_replacement",
        "attachment_default": "unknown",
        "chemical_payload_type_default": "full_residue_graph",
        "chemical_payload_template": base,
    }


def manual_priority(row: pd.Series) -> str:
    if row["count"] >= 500:
        return "high"
    if row["edit_scope_default"] in {
        "full_residue_fallback",
        "backbone_and_stereochemistry_edit",
        "backbone_and_R_group_edit",
        "stereochemistry_and_R_group_edit",
    }:
        return "high"
    if row["monomer_smiles"] != row["monomer_smiles"]:
        return "high"
    if row["count"] >= 50:
        return "medium"
    return "low"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    freq = pd.read_csv(FREQ)
    unique = pd.read_csv(UNIQUE).rename(columns={"Symbol": "source_monomer_name"})
    unique["join_symbol"] = unique["source_monomer_name"]
    freq["join_symbol"] = freq["monomer"].map(lambda x: ALIASES.get(x, x))
    table = freq.merge(
        unique[["join_symbol", "SMILES", "MolWt"]],
        on="join_symbol",
        how="left",
    )
    table = table.rename(
        columns={
            "monomer": "source_monomer_name",
            "SMILES": "monomer_smiles",
            "MolWt": "molwt",
        }
    )
    anchors = table.apply(lambda row: infer_anchor(row["source_monomer_name"], row["monomer_smiles"]), axis=1)
    table["canonical_anchor"] = anchors.map(lambda x: x[0])
    table["pseudo_anchor"] = anchors.map(lambda x: x[1])
    table["anchor_rule"] = anchors.map(lambda x: x[2])
    table["standardized_name"] = table["source_monomer_name"]
    table["base_monomer_name"] = table["source_monomer_name"].map(base_name)
    table["stereochemistry_default"] = table["source_monomer_name"].map(lambda x: "D" if is_d(x) else "L")
    table["is_n_methyl"] = table["source_monomer_name"].map(is_n_methyl)

    edit_defaults = table.apply(
        lambda row: infer_default_edit(
            row["source_monomer_name"],
            row["canonical_anchor"],
            row["pseudo_anchor"],
            row["monomer_smiles"],
        ),
        axis=1,
    )
    edit_df = pd.DataFrame(edit_defaults.tolist())
    table = pd.concat([table, edit_df], axis=1)
    table["manual_curation_priority"] = table.apply(manual_priority, axis=1)
    table["manual_check"] = False
    table["quality_flag"] = "auto_v1_needs_review"
    table["notes"] = ""

    columns = [
        "source_monomer_name",
        "standardized_name",
        "base_monomer_name",
        "monomer_smiles",
        "molwt",
        "count",
        "source_count",
        "canonical_anchor",
        "pseudo_anchor",
        "anchor_rule",
        "stereochemistry_default",
        "is_n_methyl",
        "edit_scope_default",
        "operation_default",
        "attachment_default",
        "chemical_payload_type_default",
        "chemical_payload_template",
        "manual_curation_priority",
        "manual_check",
        "quality_flag",
        "notes",
    ]
    table[columns].to_csv(OUT, index=False)
    review = table[
        (table["manual_curation_priority"].eq("high"))
        | (table["monomer_smiles"].isna())
        | (table["edit_scope_default"].eq("full_residue_fallback"))
    ][columns].copy()
    review.to_csv(REVIEW_OUT, index=False)

    priority_counts = table["manual_curation_priority"].value_counts().rename_axis("priority").reset_index(name="count")
    scope_counts = table["edit_scope_default"].value_counts().rename_axis("edit_scope").reset_index(name="count")
    missing_smiles = table[table["monomer_smiles"].isna()][
        ["source_monomer_name", "count", "canonical_anchor", "pseudo_anchor", "edit_scope_default"]
    ].head(40)
    report = [
        "# Monomer Anchor Table v4 Report",
        "",
        f"Output: `{OUT.relative_to(ROOT)}`",
        "",
        "## Size",
        "",
        f"- rows: {len(table)}",
        f"- missing SMILES: {int(table['monomer_smiles'].isna().sum())}",
        f"- manual review queue rows: {len(review)}",
        "",
        "## Manual Curation Priority",
        "",
        priority_counts.to_markdown(index=False),
        "",
        "## Edit Scope Counts",
        "",
        scope_counts.to_markdown(index=False),
        "",
        "## Top Missing SMILES",
        "",
        missing_smiles.to_markdown(index=False),
        "",
        "## Notes",
        "",
        "- This is an automatic v4 table, not final human-curated chemistry.",
        "- Rules incorporate common non-natural amino acid naming patterns and monomer SMILES where available.",
        "- Use this table before expanding operation coverage beyond N-methylation and D/L inversion.",
        "- High-priority rows should be manually checked first.",
        "",
    ]
    REPORT.write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote {OUT.relative_to(ROOT)}: {len(table)} rows")
    print(f"Wrote {REVIEW_OUT.relative_to(ROOT)}: {len(review)} rows")
    print(f"Wrote {REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
