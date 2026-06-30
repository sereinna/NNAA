"""中文说明：CEDG 单体 pair 级 edit event/anchor/operation 规则，供训练导出、外部验证和候选生成复用。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable


CANONICAL = set("ARNDCQEGHILKMFPSTWYV")

ANCHOR_MAP = {
    "Abu": "A",
    "dAbu": "A",
    "Me_Abu": "A",
    "Me_dAbu": "A",
    "Nva": "V",
    "dNva": "V",
    "Me_Nva": "V",
    "Me_dNva": "V",
    "Nle": "L",
    "dNle": "L",
    "Cha": "F",
    "dCha": "F",
    "Me_Cha": "F",
    "Sar": "G",
    "Et_Gly": "G",
    "Pr_Gly": "G",
    "iBu_Gly": "G",
    "Mono115": "terminal_cap",
}

N_SUBSTITUTED_GLY = {
    "Sar": "N-Me",
    "Et_Gly": "N-Et",
    "Pr_Gly": "N-Pr",
    "iBu_Gly": "N-iBu",
}

SIDECHAIN_PAYLOAD = {
    "A": "Ala_methyl",
    "dA": "Ala_methyl",
    "meA": "Ala_methyl",
    "Me_dA": "Ala_methyl",
    "Abu": "ethyl",
    "dAbu": "ethyl",
    "Me_Abu": "ethyl",
    "Me_dAbu": "ethyl",
    "Nva": "linear_propyl",
    "dNva": "linear_propyl",
    "Me_Nva": "linear_propyl",
    "Me_dNva": "linear_propyl",
    "L": "Leu_isobutyl",
    "dL": "Leu_isobutyl",
    "Cha": "cyclohexylmethyl",
    "dCha": "cyclohexylmethyl",
    "P": "Pro_ring",
    "dP": "Pro_ring",
}

ALPHA_AMINO_ACID_LIKE = set(SIDECHAIN_PAYLOAD)
PROLINE_LIKE = {"P", "dP"}
N_METHYL_ALPHA = {"meA", "Me_dA", "Me_Abu", "Me_dAbu", "Me_Nva", "Me_dNva", "Me_Cha"}


@dataclass(frozen=True)
class PairEditAnnotation:
    anchor_for_alignment: str
    final_edit_scope: str
    edit_event_subclass: str
    attachment_type: str
    final_chemical_payload_type: str
    final_chemical_payload: str


def anchor_for_monomer(token: str) -> str:
    clean = str(token).strip().strip("[]")
    if clean in ANCHOR_MAP:
        return ANCHOR_MAP[clean]
    clean = re.sub(r"^(me|Me)_?", "", clean)
    if clean.startswith("d") and len(clean) > 1:
        clean = clean[1:]
    return clean if clean in CANONICAL else clean


def anchor_for_monomer_with_rules(token: str, rules: dict[str, dict[str, object]] | None = None) -> str:
    """Return final-table anchor when available, otherwise fall back to built-in rules."""

    if rules is not None:
        rule = rules.get(str(token), {})
        anchor = rule.get("anchor_for_alignment") or rule.get("canonical_anchor")
        if anchor is not None and str(anchor) and str(anchor).lower() != "nan":
            return str(anchor)
    return anchor_for_monomer(token)


def is_d_monomer(token: str) -> bool:
    clean = str(token)
    return clean.startswith("d") or clean.startswith("D-") or clean.startswith("Me_d")


def is_n_methyl_alpha_monomer(token: str) -> bool:
    return str(token) in N_METHYL_ALPHA or str(token).startswith("N-Me")


def _dedupe_join(parts: list[str], sep: str = "+") -> str:
    return sep.join(dict.fromkeys(part for part in parts if part))


def _state_delta_parts(original: str, modified: str) -> tuple[list[str], list[str], list[str], list[str]]:
    scopes: list[str] = []
    operations: list[str] = []
    attachments: list[str] = []
    payloads: list[str] = []
    if is_d_monomer(original) != is_d_monomer(modified):
        scopes.append("stereochemistry_edit")
        operations.append("D_L_inversion" if is_d_monomer(modified) else "D_to_L_inversion")
        attachments.append("alpha_C")
        payloads.append("L_to_D" if is_d_monomer(modified) else "D_to_L")
    if is_n_methyl_alpha_monomer(original) != is_n_methyl_alpha_monomer(modified):
        scopes.append("backbone_edit")
        operations.append("N_methylation" if is_n_methyl_alpha_monomer(modified) else "N_demethylation")
        attachments.append("backbone_N")
        payloads.append("N-H_to_N-CH3" if is_n_methyl_alpha_monomer(modified) else "N-CH3_to_N-H")
    return scopes, operations, attachments, payloads


def _annotation(
    anchor: str,
    scopes: list[str],
    operations: list[str],
    attachments: list[str],
    payloads: list[str],
    payload_type: str = "combined_delta",
) -> PairEditAnnotation:
    scope = _dedupe_join(scopes)
    operation = _dedupe_join(operations)
    attachment = _dedupe_join(attachments, sep=";")
    payload = _dedupe_join(payloads, sep=";")
    if payload_type == "combined_delta" and scope == "R_group_edit":
        payload_type = "r_group_delta"
    if payload_type == "combined_delta" and scope == "stereochemistry_edit":
        payload_type = "chirality_change_flag"
    if payload_type == "combined_delta" and scope == "backbone_edit":
        payload_type = "delta_graph"
    return PairEditAnnotation(anchor, scope, operation, attachment, payload_type, payload)


def infer_pair_edit_annotation(
    original: str,
    modified: str,
    anchor_resolver: Callable[[str], str] = anchor_for_monomer,
) -> PairEditAnnotation | None:
    """Infer a complete pair-level local edit annotation.

    The return value is intentionally complete. Callers should not append monomer-table
    fallback fields after this function succeeds, otherwise pair-specific chemistry is
    polluted by single-monomer defaults.
    """

    original = str(original)
    modified = str(modified)
    if original == modified:
        return None

    if original in N_SUBSTITUTED_GLY and modified in N_SUBSTITUTED_GLY:
        return _annotation(
            "G",
            ["backbone_and_R_group_edit"],
            ["N_substituted_glycine_substituent_edit"],
            ["backbone_N"],
            [f"{N_SUBSTITUTED_GLY[original]}_to_{N_SUBSTITUTED_GLY[modified]}"],
            "n_substituent_delta",
        )

    if original in N_SUBSTITUTED_GLY and modified in ALPHA_AMINO_ACID_LIKE:
        new_n_state = "N-Me" if is_n_methyl_alpha_monomer(modified) else "N-H"
        scopes = ["backbone_and_R_group_edit"]
        operations = ["peptoid_to_alpha_amino_acid"]
        attachments = ["backbone_N", "alpha_C", "sidechain_Cbeta"]
        payloads = [
            f"{N_SUBSTITUTED_GLY[original]}_glycine_to_{modified}_alpha_amino_acid",
            f"{N_SUBSTITUTED_GLY[original]}_to_{new_n_state}",
        ]
        return _annotation(anchor_resolver(modified), scopes, operations, attachments, payloads)

    if original in ALPHA_AMINO_ACID_LIKE and modified in N_SUBSTITUTED_GLY:
        old_n_state = "N-Me" if is_n_methyl_alpha_monomer(original) else "N-H"
        scopes = ["backbone_and_R_group_edit"]
        operations = ["alpha_amino_acid_to_peptoid"]
        attachments = ["backbone_N", "alpha_C", "sidechain_Cbeta"]
        payloads = [
            f"{original}_alpha_amino_acid_to_{N_SUBSTITUTED_GLY[modified]}_glycine",
            f"{old_n_state}_to_{N_SUBSTITUTED_GLY[modified]}",
        ]
        return _annotation(anchor_resolver(original), scopes, operations, attachments, payloads)

    if (original in PROLINE_LIKE and modified in {"meA", "Me_dA"}) or (
        modified in PROLINE_LIKE and original in {"meA", "Me_dA"}
    ):
        scopes, operations, attachments, payloads = _state_delta_parts(original, modified)
        if original in PROLINE_LIKE:
            operations.insert(0, "proline_ring_opening_to_N_methyl_alanine")
            payloads.insert(0, "Pro_ring_to_N_methyl_Ala")
        else:
            operations.insert(0, "N_methyl_alanine_to_proline_ring_closure")
            payloads.insert(0, "N_methyl_Ala_to_Pro_ring")
        scopes.insert(0, "backbone_and_R_group_edit")
        attachments = ["backbone_N", "sidechain_Cbeta"] + attachments
        return _annotation("A", scopes, operations, attachments, payloads)

    if original in ALPHA_AMINO_ACID_LIKE and modified in ALPHA_AMINO_ACID_LIKE:
        scopes, operations, attachments, payloads = _state_delta_parts(original, modified)
        old_sidechain = SIDECHAIN_PAYLOAD[original]
        new_sidechain = SIDECHAIN_PAYLOAD[modified]
        if old_sidechain != new_sidechain:
            scopes.insert(0, "R_group_edit")
            operations.insert(0, "hydrophobic_sidechain_replacement")
            attachments.insert(0, "sidechain_Cbeta")
            payloads.insert(0, f"{old_sidechain}_to_{new_sidechain}")
        if not payloads:
            scopes.append("R_group_edit")
            operations.append("monomer_replacement")
            attachments.append("sidechain_Cbeta")
            payloads.append(f"{original}_to_{modified}")
        return _annotation(anchor_resolver(original), scopes, operations, attachments, payloads)

    return None
