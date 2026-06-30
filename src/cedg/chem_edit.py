"""中文说明：CEDG attachment-aware 化学编辑图模块，构建 old/new/delta graph、atom action 和 attachment 标记。"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import torch

try:
    from rdkit import Chem
    from rdkit.Chem import rdFMCS
except ImportError:  # pragma: no cover - handled by fallback graph validity.
    Chem = None
    rdFMCS = None


ATOM_FEATURE_DIM = 10
EDGE_FEATURE_DIM = 11

ATOM_ACTIONS = {
    "none": 0,
    "core_unchanged": 1,
    "added": 2,
    "deleted": 3,
    "atom_changed": 4,
    "bond_changed": 5,
    "stereo_changed": 6,
    "charge_changed": 7,
}

ATOM_ROLES = {
    "unknown": 0,
    "backbone_N": 1,
    "alpha_C": 2,
    "carbonyl_C": 3,
    "carbonyl_O": 4,
    "sidechain": 5,
    "r_group": 6,
    "linker": 7,
    "attachment_dummy": 8,
    "attachment_atom": 9,
}

GRAPH_MODES = {
    "failed": 0,
    "mapped_delta": 1,
    "mcs_delta": 2,
    "r_group_dummy": 3,
    "stereo_flag": 4,
    "curated_fallback": 5,
    "full_residue_fallback": 6,
}


@dataclass
class EditGraph:
    old_atom_features: torch.Tensor
    old_adjacency: torch.Tensor
    old_atom_mask: torch.Tensor
    new_atom_features: torch.Tensor
    new_adjacency: torch.Tensor
    new_atom_mask: torch.Tensor
    delta_atom_features: torch.Tensor
    delta_edge_features: torch.Tensor
    delta_atom_mask: torch.Tensor
    atom_action: torch.Tensor
    atom_role: torch.Tensor
    attachment_index_old: int
    attachment_index_new: int
    attachment_index_delta: int
    anchor_index_old: int
    anchor_index_new: int
    valid: bool
    mode: str
    error: str | None = None


def edit_graph_to_dict(graph: EditGraph) -> dict[str, object]:
    """Serialize EditGraph to a torch-saveable dictionary."""

    return {
        "old_atom_features": graph.old_atom_features,
        "old_adjacency": graph.old_adjacency,
        "old_atom_mask": graph.old_atom_mask,
        "new_atom_features": graph.new_atom_features,
        "new_adjacency": graph.new_adjacency,
        "new_atom_mask": graph.new_atom_mask,
        "delta_atom_features": graph.delta_atom_features,
        "delta_edge_features": graph.delta_edge_features,
        "delta_atom_mask": graph.delta_atom_mask,
        "atom_action": graph.atom_action,
        "atom_role": graph.atom_role,
        "attachment_index_old": graph.attachment_index_old,
        "attachment_index_new": graph.attachment_index_new,
        "attachment_index_delta": graph.attachment_index_delta,
        "anchor_index_old": graph.anchor_index_old,
        "anchor_index_new": graph.anchor_index_new,
        "valid": graph.valid,
        "mode": graph.mode,
        "error": graph.error,
    }


def edit_graph_from_dict(data: dict[str, object]) -> EditGraph:
    """Load EditGraph from a dictionary produced by `edit_graph_to_dict`."""

    return EditGraph(
        old_atom_features=data["old_atom_features"],
        old_adjacency=data["old_adjacency"],
        old_atom_mask=data["old_atom_mask"],
        new_atom_features=data["new_atom_features"],
        new_adjacency=data["new_adjacency"],
        new_atom_mask=data["new_atom_mask"],
        delta_atom_features=data["delta_atom_features"],
        delta_edge_features=data["delta_edge_features"],
        delta_atom_mask=data["delta_atom_mask"],
        atom_action=data["atom_action"],
        atom_role=data["atom_role"],
        attachment_index_old=int(data["attachment_index_old"]),
        attachment_index_new=int(data["attachment_index_new"]),
        attachment_index_delta=int(data["attachment_index_delta"]),
        anchor_index_old=int(data["anchor_index_old"]),
        anchor_index_new=int(data["anchor_index_new"]),
        valid=bool(data["valid"]),
        mode=str(data["mode"]),
        error=None if data.get("error") is None else str(data.get("error")),
    )


def normalize_token(token: object) -> str:
    if token is None:
        return "<UNK>"
    text = str(token).strip()
    return text if text and text.lower() != "nan" else "<UNK>"


def atom_features(atom: object) -> list[float]:
    atomic_num = float(atom.GetAtomicNum())
    degree = float(atom.GetDegree())
    formal_charge = float(atom.GetFormalCharge())
    hybridization = str(atom.GetHybridization())
    chiral_tag = str(atom.GetChiralTag())
    return [
        min(atomic_num, 60.0) / 60.0,
        degree / 4.0,
        formal_charge / 3.0,
        float(atom.GetTotalNumHs()) / 4.0,
        float(atom.GetIsAromatic()),
        float(atom.IsInRing()),
        float(hybridization == "SP"),
        float(hybridization == "SP2"),
        float(hybridization == "SP3"),
        float(chiral_tag != "CHI_UNSPECIFIED"),
    ]


def empty_mol_graph() -> dict[str, object]:
    return {
        "atom_features": torch.zeros((1, ATOM_FEATURE_DIM), dtype=torch.float32),
        "adjacency": torch.eye(1, dtype=torch.float32),
        "atom_mask": torch.zeros(1, dtype=torch.bool),
        "valid": False,
    }


def mol_to_graph(mol: object | None) -> dict[str, object]:
    if mol is None or Chem is None or mol.GetNumAtoms() == 0:
        return empty_mol_graph()
    features = torch.tensor([atom_features(atom) for atom in mol.GetAtoms()], dtype=torch.float32)
    n_atoms = features.shape[0]
    adjacency = torch.eye(n_atoms, dtype=torch.float32)
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        adjacency[i, j] = 1.0
        adjacency[j, i] = 1.0
    degree = adjacency.sum(dim=-1).clamp_min(1.0)
    norm = degree.pow(-0.5)
    adjacency = norm.unsqueeze(1) * adjacency * norm.unsqueeze(0)
    return {
        "atom_features": features,
        "adjacency": adjacency,
        "atom_mask": torch.ones(n_atoms, dtype=torch.bool),
        "valid": True,
    }


def smiles_to_mol(smiles: object) -> object | None:
    if Chem is None or smiles is None or pd.isna(smiles):
        return None
    return Chem.MolFromSmiles(str(smiles))


def empty_edit_graph(mode: str = "failed", error: str | None = None) -> EditGraph:
    graph = empty_mol_graph()
    delta_edge = torch.zeros((1, 1, EDGE_FEATURE_DIM), dtype=torch.float32)
    return EditGraph(
        old_atom_features=graph["atom_features"],
        old_adjacency=graph["adjacency"],
        old_atom_mask=graph["atom_mask"],
        new_atom_features=graph["atom_features"],
        new_adjacency=graph["adjacency"],
        new_atom_mask=graph["atom_mask"],
        delta_atom_features=graph["atom_features"],
        delta_edge_features=delta_edge,
        delta_atom_mask=graph["atom_mask"],
        atom_action=torch.zeros(1, dtype=torch.long),
        atom_role=torch.zeros(1, dtype=torch.long),
        attachment_index_old=-1,
        attachment_index_new=-1,
        attachment_index_delta=-1,
        anchor_index_old=-1,
        anchor_index_new=-1,
        valid=False,
        mode=mode,
        error=error,
    )


def infer_atom_role(index: int, attachment_type: str, attachment_index: int = -1) -> int:
    if index == attachment_index:
        return ATOM_ROLES["attachment_atom"]
    if "backbone_N" in attachment_type:
        return ATOM_ROLES["backbone_N"]
    if "alpha_C" in attachment_type:
        return ATOM_ROLES["alpha_C"]
    if "linker" in attachment_type:
        return ATOM_ROLES["linker"]
    if "side" in attachment_type or "phenyl" in attachment_type or "ring" in attachment_type:
        return ATOM_ROLES["r_group"]
    return ATOM_ROLES["unknown"]


def make_delta_edge_features(mol: object | None, n_atoms: int, is_new: bool = True) -> torch.Tensor:
    edge = torch.zeros((n_atoms, n_atoms, EDGE_FEATURE_DIM), dtype=torch.float32)
    if mol is None or Chem is None:
        return edge
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        order = float(bond.GetBondTypeAsDouble())
        values = torch.tensor(
            [
                1.0,
                float(order == 1.0),
                float(order == 2.0),
                float(order == 3.0),
                float(bond.GetIsAromatic()),
                float(not is_new),
                float(is_new),
                0.0,
                0.0,
                0.0,
                float(bond.IsInRing()),
            ],
            dtype=torch.float32,
        )
        edge[i, j] = values
        edge[j, i] = values
    return edge


def make_edit_graph_from_mols(
    old_mol: object | None,
    new_mol: object | None,
    mode: str,
    attachment_type: str,
    atom_action_id: int,
    valid: bool = True,
) -> EditGraph:
    old_graph = mol_to_graph(old_mol)
    new_graph = mol_to_graph(new_mol)
    delta_source = new_mol if new_graph["valid"] else old_mol
    delta_graph = mol_to_graph(delta_source)
    n_delta = int(delta_graph["atom_features"].shape[0])
    attachment_index = 0 if delta_graph["valid"] else -1
    actions = torch.full((n_delta,), atom_action_id, dtype=torch.long)
    roles = torch.tensor(
        [infer_atom_role(i, attachment_type, attachment_index) for i in range(n_delta)],
        dtype=torch.long,
    )
    return EditGraph(
        old_atom_features=old_graph["atom_features"],
        old_adjacency=old_graph["adjacency"],
        old_atom_mask=old_graph["atom_mask"],
        new_atom_features=new_graph["atom_features"],
        new_adjacency=new_graph["adjacency"],
        new_atom_mask=new_graph["atom_mask"],
        delta_atom_features=delta_graph["atom_features"],
        delta_edge_features=make_delta_edge_features(delta_source, n_delta, is_new=True),
        delta_atom_mask=delta_graph["atom_mask"],
        atom_action=actions,
        atom_role=roles,
        attachment_index_old=0 if old_graph["valid"] else -1,
        attachment_index_new=0 if new_graph["valid"] else -1,
        attachment_index_delta=attachment_index,
        anchor_index_old=0 if old_graph["valid"] else -1,
        anchor_index_new=0 if new_graph["valid"] else -1,
        valid=valid and (old_graph["valid"] or new_graph["valid"] or delta_graph["valid"]),
        mode=mode,
    )


def find_strict_mcs_mapping(old_mol: object, new_mol: object, timeout: int = 10) -> dict[int, int] | None:
    if rdFMCS is None or Chem is None:
        return None
    params = rdFMCS.MCSParameters()
    params.Timeout = int(timeout)
    params.AtomCompareParameters.MatchChiralTag = True
    params.AtomCompareParameters.MatchFormalCharge = True
    params.AtomCompareParameters.MatchValences = True
    params.BondCompareParameters.RingMatchesRingOnly = True
    params.BondCompareParameters.CompleteRingsOnly = True
    params.AtomTyper = rdFMCS.AtomCompare.CompareElements
    params.BondTyper = rdFMCS.BondCompare.CompareOrderExact
    result = rdFMCS.FindMCS([old_mol, new_mol], params)
    if result.canceled or not result.smartsString:
        return None
    query = Chem.MolFromSmarts(result.smartsString)
    old_match = old_mol.GetSubstructMatch(query)
    new_match = new_mol.GetSubstructMatch(query)
    if not old_match or not new_match:
        return None
    return dict(zip(old_match, new_match))


def try_mcs_residue_delta(event: dict[str, object], monomer_smiles: dict[str, str], **kwargs: object) -> EditGraph:
    old_smiles = event.get("old_full_residue_smiles") or monomer_smiles.get(str(event.get("original_monomer")))
    new_smiles = event.get("new_full_residue_smiles") or monomer_smiles.get(str(event.get("modified_monomer")))
    old_mol = smiles_to_mol(old_smiles)
    new_mol = smiles_to_mol(new_smiles)
    if old_mol is None or new_mol is None:
        return empty_edit_graph("mcs_delta", "missing old/new residue smiles")
    mapping = find_strict_mcs_mapping(old_mol, new_mol, int(kwargs.get("mcs_timeout", 10)))
    if not mapping:
        return empty_edit_graph("mcs_delta", "mcs failed")
    attachment_type = normalize_token(event.get("attachment_type"))
    return make_edit_graph_from_mols(
        old_mol,
        new_mol,
        mode="mcs_delta",
        attachment_type=attachment_type,
        atom_action_id=ATOM_ACTIONS["atom_changed"],
    )


def try_stereo_flag_graph(event: dict[str, object], monomer_smiles: dict[str, str], **_: object) -> EditGraph:
    payload = normalize_token(event.get("final_chemical_payload"))
    if "L_to_D" not in payload and "D_to_L" not in payload:
        return empty_edit_graph("stereo_flag", "not stereochemistry-only")
    smiles = monomer_smiles.get(str(event.get("modified_monomer"))) or monomer_smiles.get(str(event.get("original_monomer")))
    mol = smiles_to_mol(smiles)
    if mol is None:
        graph = empty_edit_graph("stereo_flag", "no residue smiles for stereo flag")
        graph.valid = True
        graph.atom_action = torch.tensor([ATOM_ACTIONS["stereo_changed"]], dtype=torch.long)
        graph.atom_role = torch.tensor([ATOM_ROLES["alpha_C"]], dtype=torch.long)
        graph.delta_atom_mask = torch.ones(1, dtype=torch.bool)
        graph.old_atom_mask = torch.ones(1, dtype=torch.bool)
        graph.new_atom_mask = torch.ones(1, dtype=torch.bool)
        return graph
    return make_edit_graph_from_mols(
        mol,
        mol,
        mode="stereo_flag",
        attachment_type="alpha_C",
        atom_action_id=ATOM_ACTIONS["stereo_changed"],
    )


def try_curated_fragment_fallback(
    event: dict[str, object],
    monomer_smiles: dict[str, str],
    fragment_resolver: object | None = None,
    **_: object,
) -> EditGraph:
    if fragment_resolver is None:
        return empty_edit_graph("curated_fallback", "no fragment resolver")
    old_smiles, new_smiles = fragment_resolver(event, monomer_smiles)
    old_mol = smiles_to_mol(old_smiles) if old_smiles else None
    new_mol = smiles_to_mol(new_smiles) if new_smiles else None
    if old_mol is None and new_mol is None:
        return empty_edit_graph("curated_fallback", "no curated fragment smiles")
    action = ATOM_ACTIONS["added"] if new_mol is not None else ATOM_ACTIONS["deleted"]
    return make_edit_graph_from_mols(
        old_mol,
        new_mol,
        mode="curated_fallback",
        attachment_type=normalize_token(event.get("attachment_type")),
        atom_action_id=action,
    )


def build_attachment_aware_edit_graph(
    event: dict[str, object],
    monomer_smiles: dict[str, str],
    fragment_resolver: object | None = None,
    radius: int = 2,
    mcs_timeout: int = 10,
) -> EditGraph:
    """Build the best available attachment-aware edit graph for one edit event."""

    del radius
    for builder in (
        try_mcs_residue_delta,
        try_stereo_flag_graph,
        try_curated_fragment_fallback,
    ):
        graph = builder(
            event,
            monomer_smiles,
            fragment_resolver=fragment_resolver,
            mcs_timeout=mcs_timeout,
        )
        if graph.valid:
            return graph
    return empty_edit_graph(mode="failed", error="all builders failed")
