"""中文说明：CEDG 环肽拓扑模块，构建 residue 级特征、环状距离和拓扑图张量。"""

from __future__ import annotations

import math
from collections import Counter

import torch


RESIDUE_FEATURE_DIM = 12
TOPOLOGY_FEATURE_DIM = 24

RESIDUE_EDGE_TYPES = {
    "none": 0,
    "self": 1,
    "sequence_bond": 2,
    "head_to_tail_cyclization": 3,
    "sidechain_linker": 4,
    "disulfide": 5,
    "lactam": 6,
    "thioether": 7,
    "staple": 8,
    "contact_3d": 9,
}

POSITIVE_RESIDUES = {"K", "R", "H", "Lys", "Arg", "His", "Lys-like"}
NEGATIVE_RESIDUES = {"D", "E", "Asp", "Glu", "Asp-like", "Glu-like"}
SIDECHAIN_ATTACHMENT_RESIDUES = {
    "F",
    "W",
    "Y",
    "K",
    "S",
    "C",
    "D",
    "E",
    "H",
    "Phe",
    "Trp",
    "Tyr",
    "Lys",
    "Ser",
    "Cys",
    "Asp",
    "Glu",
    "His",
}


def cyclic_distance(i: int, j: int, length: int) -> int:
    """Return residue distance on a cyclic sequence."""

    if length <= 1:
        return 0
    distance = abs(i - j)
    return min(distance, length - distance)


def residue_charge_value(token: str) -> float:
    """Small rule-based residue charge proxy used as a topology feature."""

    if token in POSITIVE_RESIDUES:
        return 1.0
    if token in NEGATIVE_RESIDUES:
        return -1.0
    return 0.0


def is_backbone_n_methylatable(token: str) -> bool:
    """Rule proxy for whether a residue has a backbone NH available for methylation."""

    return token not in {"P", "Pro", "N-Me", "me", "Me"}


def is_sidechain_attachment_possible(token: str) -> bool:
    """Rule proxy for residues commonly supporting sidechain/R-group edits."""

    return token in SIDECHAIN_ATTACHMENT_RESIDUES or token.endswith("-like")


def is_pseudo_anchor(token: str) -> bool:
    return token.endswith("-like")


def normalize_token(token: object) -> str:
    if token is None:
        return "<UNK>"
    text = str(token).strip()
    return text if text and text.lower() != "nan" else "<UNK>"


def infer_is_cyclic(record: dict[str, object], length: int) -> bool:
    """Infer cyclic status from explicit fields when available, otherwise from SMILES/ring cue."""

    cyclization_type = normalize_token(record.get("cyclization_type"))
    if cyclization_type not in {"<UNK>", "linear", "none"}:
        return True
    parent_smiles = normalize_token(record.get("parent_smiles"))
    return bool(length >= 4 and any(char.isdigit() for char in parent_smiles))


def infer_cyclization_type(record: dict[str, object], length: int) -> str:
    """Best-effort cyclization type inference for current sparse exports."""

    explicit = normalize_token(record.get("cyclization_type"))
    if explicit != "<UNK>":
        return explicit
    if infer_is_cyclic(record, length):
        return "head_to_tail"
    return "linear"


def is_cyclization_anchor(record: dict[str, object], index: int, length: int) -> bool:
    cyclization_type = infer_cyclization_type(record, length)
    return cyclization_type == "head_to_tail" and index in {1, length} and length > 2


def is_linker_adjacent(record: dict[str, object], index: int) -> bool:
    for link in record.get("topology_links", []) or []:
        if int(link.get("site_i", 0) or 0) == index or int(link.get("site_j", 0) or 0) == index:
            return True
    return False


def is_terminal_cap_adjacent(record: dict[str, object], index: int, length: int) -> bool:
    monomers = record.get("parent_monomer_list") or []
    if not monomers:
        return False
    terminal_tokens = {"Ac", "acetyl", "NH2", "amide", "terminal_cap"}
    if index == 1:
        return any(str(monomers[0]).startswith(token) for token in terminal_tokens)
    if index == length:
        return any(str(monomers[-1]).endswith(token) for token in terminal_tokens)
    return False


def build_residue_features(tokens: list[str], record: dict[str, object]) -> list[list[float]]:
    """Build extended residue-level topology features."""

    length = max(len(tokens), 1)
    edited_sites = [
        int(event.get("site_index") or 0)
        for event in record.get("edit_set", [])
        if int(event.get("site_index") or 0) > 0
    ]
    out: list[list[float]] = []
    for index in range(1, length + 1):
        scaled = index / length
        if edited_sites:
            min_dist = min(cyclic_distance(index, site, length) for site in edited_sites)
            min_dist_scaled = min_dist / max(length // 2, 1)
        else:
            min_dist_scaled = 1.0
        token = tokens[index - 1] if index - 1 < len(tokens) else "<UNK>"
        out.append(
            [
                scaled,
                math.sin(scaled * 2.0 * math.pi),
                math.cos(scaled * 2.0 * math.pi),
                float(index in edited_sites),
                float(min_dist_scaled),
                float(is_cyclization_anchor(record, index, length)),
                float(is_linker_adjacent(record, index)),
                float(is_terminal_cap_adjacent(record, index, length)),
                float(is_backbone_n_methylatable(token)),
                float(is_sidechain_attachment_possible(token)),
                float(is_pseudo_anchor(token)),
                residue_charge_value(token),
            ]
        )
    return out


def shortest_path_distance(adj: torch.Tensor) -> torch.Tensor:
    """Dense Floyd-Warshall shortest path distance for small residue graphs."""

    length = adj.shape[0]
    inf = float(length + 1)
    dist = torch.full((length, length), inf, dtype=torch.float32)
    dist[adj > 0] = 1.0
    dist.fill_diagonal_(0.0)
    for k in range(length):
        dist = torch.minimum(dist, dist[:, k].unsqueeze(1) + dist[k, :].unsqueeze(0))
    return dist.clamp_max(inf) / max(length, 1)


def build_residue_topology(tokens: list[str], record: dict[str, object]) -> dict[str, torch.Tensor]:
    """Build residue adjacency, edge types, and shortest-path distances."""

    length = max(len(tokens), 1)
    adj = torch.eye(length, dtype=torch.float32)
    edge_type = torch.zeros((length, length), dtype=torch.long)
    edge_type.fill_(RESIDUE_EDGE_TYPES["none"])
    for i in range(length):
        edge_type[i, i] = RESIDUE_EDGE_TYPES["self"]

    for i in range(length - 1):
        adj[i, i + 1] = adj[i + 1, i] = 1.0
        edge_type[i, i + 1] = edge_type[i + 1, i] = RESIDUE_EDGE_TYPES["sequence_bond"]

    cyclization_type = infer_cyclization_type(record, length)
    if cyclization_type == "head_to_tail" and length > 2:
        adj[0, length - 1] = adj[length - 1, 0] = 1.0
        edge_type[0, length - 1] = edge_type[length - 1, 0] = RESIDUE_EDGE_TYPES["head_to_tail_cyclization"]

    for link in record.get("topology_links", []) or []:
        i = int(link.get("site_i", 0) or 0) - 1
        j = int(link.get("site_j", 0) or 0) - 1
        if 0 <= i < length and 0 <= j < length:
            link_type = normalize_token(link.get("link_type"))
            type_id = RESIDUE_EDGE_TYPES.get(link_type, RESIDUE_EDGE_TYPES["sidechain_linker"])
            adj[i, j] = adj[j, i] = 1.0
            edge_type[i, j] = edge_type[j, i] = type_id

    return {
        "residue_adj": adj,
        "residue_edge_type": edge_type,
        "residue_distance": shortest_path_distance(adj),
    }


def build_scaffold_topology_features(tokens: list[str], record: dict[str, object]) -> list[float]:
    """Build scaffold topology + coarse physicochemical descriptor vector."""

    length = len(tokens)
    edit_set = record.get("edit_set", []) or []
    edit_count = len(edit_set) or int(record.get("edit_count") or 0)
    cyclization_type = infer_cyclization_type(record, max(length, 1))
    is_cyclic = infer_is_cyclic(record, max(length, 1))
    scope_counter = Counter(str(event.get("final_edit_scope")) for event in edit_set)
    payload_counter = Counter(str(event.get("final_chemical_payload")) for event in edit_set)
    charge = sum(residue_charge_value(token) for token in tokens)
    n_methyl_sites = sum("N-H_to_N-CH3" in str(event.get("final_chemical_payload")) for event in edit_set)
    d_sites = sum("L_to_D" in str(event.get("final_chemical_payload")) for event in edit_set)
    smiles = normalize_token(record.get("parent_smiles"))
    num_rings = sum(char.isdigit() for char in smiles)

    return [
        min(length, 64) / 64.0,
        min(edit_count, 8) / 8.0,
        float(is_cyclic),
        float(cyclization_type == "head_to_tail"),
        float(cyclization_type == "sidechain_to_sidechain"),
        float(cyclization_type == "disulfide"),
        float(cyclization_type == "lactam"),
        float(cyclization_type == "thioether"),
        float(cyclization_type == "staple"),
        float(cyclization_type not in {"linear", "head_to_tail", "sidechain_to_sidechain", "disulfide", "lactam", "thioether", "staple"}),
        min(length, 64) / 64.0 if is_cyclic else 0.0,
        min(num_rings, 16) / 16.0,
        min(length, 64) / 64.0 if is_cyclic else 0.0,
        1.0 if is_cyclic else 0.0,
        float("terminal" in " ".join(map(str, record.get("parent_monomer_list") or [])).lower()),
        float(any(link for link in record.get("topology_links", []) or [])),
        min(len(record.get("topology_links", []) or []), 8) / 8.0,
        min(n_methyl_sites, 8) / 8.0,
        min(d_sites, 8) / 8.0,
        max(min(charge / 8.0, 1.0), -1.0),
        min(sum(token in {"S", "T", "Y", "K", "R", "N", "Q", "Ser", "Thr", "Tyr", "Lys", "Arg", "Asn", "Gln"} for token in tokens), 16) / 16.0,
        min(sum(token in {"D", "E", "S", "T", "Y", "N", "Q", "Asp", "Glu", "Ser", "Thr", "Tyr", "Asn", "Gln"} for token in tokens), 16) / 16.0,
        min(scope_counter.get("R_group_edit", 0) + scope_counter.get("backbone_and_R_group_edit", 0), 8) / 8.0,
        min(sum("CF3" in payload or "Ph" in payload for payload in payload_counter), 8) / 8.0,
    ]
