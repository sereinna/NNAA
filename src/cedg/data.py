"""中文说明：CEDG 第一阶段数据层，负责 clean/QC 字段拆分、PLM 序列、拓扑图、EditGraph 和候选组 batching。

Dataset, vocabulary, and batching utilities for CEDG-Set.

The input is the model-ready JSONL exported from the final split tables. This
module builds train-only vocabularies and converts parent peptide context,
site-level edit metadata, and attachment-aware local edit payloads into tensors.
The atom-level branch encodes payload old/new fragment graphs parsed from
`final_chemical_payload`, not full noncanonical-residue replacement graphs.
"""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import torch
from torch.utils.data import Dataset

from .chem_edit import (
    ATOM_ACTIONS,
    ATOM_FEATURE_DIM,
    ATOM_ROLES,
    EDGE_FEATURE_DIM,
    GRAPH_MODES,
    EditGraph,
    build_attachment_aware_edit_graph,
    edit_graph_from_dict,
    edit_graph_to_dict,
    empty_edit_graph,
)
from .plm import load_plm_embedding, plm_cache_path
from .topology import (
    RESIDUE_EDGE_TYPES,
    RESIDUE_FEATURE_DIM,
    TOPOLOGY_FEATURE_DIM,
    build_residue_features,
    build_residue_topology,
    build_scaffold_topology_features,
)


PAD = "<PAD>"
UNK = "<UNK>"
ROOT = Path(__file__).resolve().parents[2]
MONOMER_TABLE = ROOT / "data" / "final" / "annotations" / "monomer_anchor_table.csv"
PAYLOAD_FEATURE_DIM = 8

SHADOW_FIELDS = ("parent_shadow_sequence",)
EDIT_INPUT_FIELDS = (
    "anchor_for_alignment",
    "original_monomer",
    "modified_monomer",
    "edit_event_class",
    "edit_event_subclass",
    "final_edit_scope",
    "attachment_type",
    "final_chemical_payload_type",
)
EDIT_QC_FIELDS = (
    "edit_model_use_tier",
    "local_model_eligibility",
    "atom_model_eligibility",
    "quality_flag_final",
)
SAMPLE_INPUT_FIELDS = (
    "assay_type",
)
SAMPLE_QC_FIELDS = (
    "source_id",
    "source_slug",
    "sample_model_use_tier",
    "candidate_generation_strategy",
    "table_type",
)
EDIT_FIELDS = EDIT_INPUT_FIELDS
SAMPLE_FIELDS = SAMPLE_INPUT_FIELDS

CANONICAL_TO_PLM = {
    "A": "A",
    "R": "R",
    "N": "N",
    "D": "D",
    "C": "C",
    "Q": "Q",
    "E": "E",
    "G": "G",
    "H": "H",
    "I": "I",
    "L": "L",
    "K": "K",
    "M": "M",
    "F": "F",
    "P": "P",
    "S": "S",
    "T": "T",
    "W": "W",
    "Y": "Y",
    "V": "V",
    "Ala": "A",
    "Arg": "R",
    "Asn": "N",
    "Asp": "D",
    "Cys": "C",
    "Gln": "Q",
    "Glu": "E",
    "Gly": "G",
    "His": "H",
    "Ile": "I",
    "Leu": "L",
    "Lys": "K",
    "Met": "M",
    "Phe": "F",
    "Pro": "P",
    "Ser": "S",
    "Thr": "T",
    "Trp": "W",
    "Tyr": "Y",
    "Val": "V",
    "Lys-like": "K",
    "Leu-like": "L",
    "Phe-like": "F",
    "Trp-like": "W",
    "Ala-like": "A",
    "Gly-like": "G",
    "UNK": "X",
    "<UNK>": "X",
}


class Vocab:
    """Simple token vocabulary with PAD and UNK."""

    def __init__(self) -> None:
        self.token_to_id = {PAD: 0, UNK: 1}
        self.id_to_token = [PAD, UNK]

    def add(self, token: object) -> int:
        text = normalize_token(token)
        if text not in self.token_to_id:
            self.token_to_id[text] = len(self.id_to_token)
            self.id_to_token.append(text)
        return self.token_to_id[text]

    def encode(self, token: object) -> int:
        return self.token_to_id.get(normalize_token(token), self.token_to_id[UNK])

    def __len__(self) -> int:
        return len(self.id_to_token)


@dataclass
class CEDGVocabs:
    """All vocabularies needed by the current clean-input CEDG model."""

    shadow: Vocab
    payload: Vocab
    edit: dict[str, Vocab]
    sample: dict[str, Vocab]
    monomer_smiles: dict[str, str]


def normalize_token(token: object) -> str:
    if token is None:
        return UNK
    text = str(token).strip()
    return text if text and text.lower() != "nan" else UNK


def shadow_tokens(record: dict[str, object]) -> list[str]:
    text = normalize_token(record.get("parent_shadow_sequence"))
    return [part for part in text.split(".") if part]


def shadow_to_plm_sequence(tokens: list[str]) -> str:
    """Convert canonical/pseudo-canonical shadow tokens to PLM one-letter sequence."""

    return "".join(CANONICAL_TO_PLM.get(token, "X") for token in tokens)


def split_parts(value: object, separators: tuple[str, ...] = (";", "|", ",")) -> list[str]:
    text = normalize_token(value)
    for separator in separators:
        text = text.replace(separator, "|")
    return [part.strip() for part in text.split("|") if part.strip() and part.strip() != UNK]


def payload_tokens(event: dict[str, object]) -> list[str]:
    """Tokenize a local chemical delta string into compact payload tokens."""

    payload = normalize_token(event.get("final_chemical_payload"))
    payload_type = normalize_token(event.get("final_chemical_payload_type"))
    scope = normalize_token(event.get("final_edit_scope"))
    attachment = normalize_token(event.get("attachment_type") or event.get("attachment_default"))
    pieces = [f"type={payload_type}", f"scope={scope}", f"attachment={attachment}"]
    for token in split_parts(payload, separators=(";", ",")):
        pieces.append(f"delta={token}")
        if "_to_" in token:
            left, right = token.split("_to_", 1)
            pieces.append(f"from={left}")
            pieces.append(f"to={right}")
    for token in split_parts(attachment):
        pieces.append(f"attach_part={token}")
    return pieces


FRAGMENT_SMILES = {
    "H": "",
    "N-H": "",
    "N-CH3": "C",
    "CH3": "C",
    "Me": "C",
    "Et": "CC",
    "Pr": "CCC",
    "iBu": "CC(C)C",
    "F": "[F]",
    "4-F": "[F]",
    "3-Cl": "[Cl]",
    "Cl": "[Cl]",
    "3,4-diF": "[F].[F]",
    "4-NO2": "O=[N+]([O-])",
    "4-CF3": "C(F)(F)F",
    "CF3": "C(F)(F)F",
    "OMe": "OC",
    "O-tBu": "OC(C)(C)C",
    "tBu": "C(C)(C)C",
    "Bn": "Cc1ccccc1",
    "Bn(4-Cl)": "Cc1ccc(Cl)cc1",
    "3-pyridylethyl": "CCc1cccnc1",
    "PhPr": "CCCc1ccccc1",
    "Ph": "c1ccccc1",
    "phenyl": "c1ccccc1",
    "CH2Ph": "Cc1ccccc1",
    "CH2CH2Ph": "CCc1ccccc1",
    "CH2CH2CH2Ph": "CCCc1ccccc1",
    "cyclohexyl": "C1CCCCC1",
    "naphthyl": "c1ccc2ccccc2c1",
    "1-naphthyl": "c1ccc2ccccc2c1",
    "hexyl": "CCCCCC",
    "MeOEt": "COCC",
    "cHexCH2": "CC1CCCCC1",
    "betaAla": "NCCC=O",
    "NMe-GABA": "CNCCCC=O",
    "Pye": "O=C1CCC(=O)N1",
}


def load_monomer_smiles(path: Path = MONOMER_TABLE) -> dict[str, str]:
    """Load monomer SMILES for payloads that reference named fragments."""

    smiles: dict[str, str] = {}
    if not path.exists():
        return smiles
    table = pd.read_csv(path)
    for row in table.itertuples():
        value = getattr(row, "monomer_smiles")
        if pd.isna(value):
            continue
        for key in (getattr(row, "source_monomer_name"), getattr(row, "standardized_name")):
            if pd.notna(key):
                smiles[str(key)] = str(value)
    return smiles


def payload_parts(event: dict[str, object]) -> list[tuple[str, str]]:
    """Extract old/new local edit components from the curated payload string."""

    payload = normalize_token(event.get("final_chemical_payload"))
    parts: list[tuple[str, str]] = []
    for raw in payload.replace(",", "|").replace(";", "|").split("|"):
        token = raw.strip()
        if "_to_" not in token:
            continue
        left, right = token.split("_to_", 1)
        parts.append((left, right))
    return parts


def payload_flags(event: dict[str, object]) -> list[float]:
    """Encode non-graph edit details such as stereochemical direction."""

    payload = normalize_token(event.get("final_chemical_payload"))
    scope = normalize_token(event.get("final_edit_scope"))
    payload_type = normalize_token(event.get("final_chemical_payload_type"))
    attachment = normalize_token(event.get("attachment_type") or event.get("attachment_default"))
    return [
        float("L_to_D" in payload),
        float("D_to_L" in payload),
        float("N-H_to_N-CH3" in payload),
        float("N-CH3_to_N-H" in payload),
        float("N-substituent" in payload),
        float("R_group" in scope or "sidechain" in payload or "phenyl" in payload or "side" in attachment),
        float("backbone" in scope or "backbone" in attachment),
        float(payload_type == "combined_delta"),
    ]


def clean_fragment_name(name: str) -> str:
    """Remove anchor/attachment prefixes while preserving the chemical payload."""

    text = name.strip()
    if text in FRAGMENT_SMILES:
        return text
    if "substituent(" in text:
        return text[text.find("substituent(") :]
    for prefix in (
        "Phe_sidechain_",
        "Phe_phenyl_",
        "Phe_CH2Ph_",
        "Phe_NH_",
        "Ala_CH3_",
        "Ala_",
        "Ser_OH_",
        "Glu_sidechain_",
        "Gln_sidechain_",
        "Leu_isobutyl_",
        "Leu_NH_",
    ):
        if text.startswith(prefix):
            return text[len(prefix) :]
    return text


def fragment_smiles(name: str, monomer_smiles: dict[str, str]) -> str | None:
    """Map a curated payload fragment label to a small-molecule SMILES."""

    text = clean_fragment_name(name)
    if text in ("L", "D"):
        return None
    if text in FRAGMENT_SMILES:
        return FRAGMENT_SMILES[text] or None
    if text.startswith("substituent(") and text.endswith(")"):
        key = text[len("substituent(") : -1]
        return monomer_smiles.get(key)
    if text in monomer_smiles:
        return monomer_smiles[text]
    return None


def curated_fragment_smiles_pair(
    event: dict[str, object],
    monomer_smiles: dict[str, str],
) -> tuple[str | None, str | None]:
    """Resolve curated payload delta into old/new fragment SMILES fallback."""

    old_smiles_values: list[str] = []
    new_smiles_values: list[str] = []
    for old_name, new_name in payload_parts(event):
        old_smiles = fragment_smiles(old_name, monomer_smiles)
        new_smiles = fragment_smiles(new_name, monomer_smiles)
        if old_smiles is not None:
            old_smiles_values.append(old_smiles)
        if new_smiles is not None:
            new_smiles_values.append(new_smiles)
    return ".".join(old_smiles_values) or None, ".".join(new_smiles_values) or None


def load_jsonl(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            records.append(json.loads(line))
    return records


def build_vocabs(records: Iterable[dict[str, object]]) -> CEDGVocabs:
    shadow = Vocab()
    payload = Vocab()
    edit = {field: Vocab() for field in EDIT_FIELDS}
    sample = {field: Vocab() for field in SAMPLE_FIELDS}

    for record in records:
        if record["split"] != "train":
            continue
        for token in shadow_tokens(record):
            shadow.add(token)
        for field, vocab in sample.items():
            vocab.add(record.get(field))
        for event in record["edit_set"]:
            for field, vocab in edit.items():
                vocab.add(event.get(field))
            for token in payload_tokens(event):
                payload.add(token)
    return CEDGVocabs(
        shadow=shadow,
        payload=payload,
        edit=edit,
        sample=sample,
        monomer_smiles=load_monomer_smiles(),
    )


class CEDGDataset(Dataset):
    """Torch dataset for one split of CEDG-Score JSONL records."""

    def __init__(
        self,
        records: list[dict[str, object]],
        vocabs: CEDGVocabs,
        split: str,
        allow_absolute_only: bool = False,
        graph_cache_dir: Path | None = None,
        use_graph_cache: bool = True,
        plm_cache_dir: Path | None = None,
        esm_model_name: str = "esm2_t6_8M_UR50D",
        use_plm_cache: bool = False,
        preload_cache: bool = False,
    ) -> None:
        self.records = [record for record in records if record["split"] == split]
        self.vocabs = vocabs
        self.allow_absolute_only = allow_absolute_only
        self.graph_cache_dir = graph_cache_dir
        self.use_graph_cache = use_graph_cache
        self.plm_cache_dir = plm_cache_dir
        self.esm_model_name = esm_model_name
        self.use_plm_cache = use_plm_cache
        self.preload_cache = preload_cache
        self._graph_cache: dict[Path, EditGraph] = {}
        self._plm_cache: dict[Path, tuple[torch.Tensor, torch.Tensor]] = {}
        if self.graph_cache_dir is not None and self.use_graph_cache:
            self.graph_cache_dir.mkdir(parents=True, exist_ok=True)
        if self.preload_cache:
            self.preload_tensor_cache()

    def preload_tensor_cache(self) -> None:
        """Preload existing graph/PLM cache files to avoid per-batch small-file I/O."""

        for record in self.records:
            if self.use_graph_cache and self.graph_cache_dir is not None:
                for event_index, event in enumerate(list(record.get("edit_set", []) or [])):
                    cache_path = self.graph_cache_path(record, event, event_index)
                    if cache_path is not None and cache_path.exists() and cache_path not in self._graph_cache:
                        try:
                            self._graph_cache[cache_path] = edit_graph_from_dict(
                                torch.load(cache_path, map_location="cpu", weights_only=False)
                            )
                        except Exception:
                            cache_path.unlink(missing_ok=True)
            if self.use_plm_cache and self.plm_cache_dir is not None:
                tokens = shadow_tokens(record)
                plm_sequence = shadow_to_plm_sequence(tokens)
                cache_path = plm_cache_path(self.plm_cache_dir, record["sample_id"], plm_sequence, self.esm_model_name)
                if cache_path.exists() and cache_path not in self._plm_cache:
                    self._plm_cache[cache_path] = load_plm_embedding(cache_path)

    def __len__(self) -> int:
        return len(self.records)

    def graph_cache_path(self, record: dict[str, object], event: dict[str, object], event_index: int) -> Path | None:
        if self.graph_cache_dir is None or not self.use_graph_cache:
            return None
        payload = json.dumps(
            {
                "sample_id": record.get("sample_id"),
                "event_index": event_index,
                "event": event,
                "cache_version": "edit_graph_v2",
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:20]
        sample = str(record.get("sample_id", "sample")).replace("/", "_")
        return self.graph_cache_dir / f"{sample}_e{event_index:02d}_{digest}.pt"

    def build_or_load_graph(self, record: dict[str, object], event: dict[str, object], event_index: int) -> EditGraph:
        cache_path = self.graph_cache_path(record, event, event_index)
        if cache_path is not None and cache_path in self._graph_cache:
            return self._graph_cache[cache_path]
        if cache_path is not None and cache_path.exists():
            try:
                return edit_graph_from_dict(torch.load(cache_path, map_location="cpu", weights_only=False))
            except Exception:
                cache_path.unlink(missing_ok=True)
        graph = build_attachment_aware_edit_graph(
            event,
            self.vocabs.monomer_smiles,
            fragment_resolver=curated_fragment_smiles_pair,
        )
        if cache_path is not None:
            tmp_path = cache_path.with_suffix(".tmp")
            torch.save(edit_graph_to_dict(graph), tmp_path)
            tmp_path.replace(cache_path)
            if self.preload_cache:
                self._graph_cache[cache_path] = graph
        return graph

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]
        edit_set = list(record.get("edit_set", []) or [])
        if not edit_set and not self.allow_absolute_only:
            raise ValueError("CEDGScoreModel requires at least one edit event per sample.")
        tokens = shadow_tokens(record)
        plm_sequence = shadow_to_plm_sequence(tokens)
        topology = build_residue_topology(tokens, record)
        plm_cached_embedding = None
        plm_cached_mask = None
        if self.use_plm_cache and self.plm_cache_dir is not None:
            cache_path = plm_cache_path(self.plm_cache_dir, record["sample_id"], plm_sequence, self.esm_model_name)
            if cache_path in self._plm_cache:
                plm_cached_embedding, plm_cached_mask = self._plm_cache[cache_path]
            elif cache_path.exists():
                plm_cached_embedding, plm_cached_mask = load_plm_embedding(cache_path)
        return {
            "sample_id": record["sample_id"],
            "source_id": record["source_id"],
            "candidate_group_id": normalize_token(
                record.get("candidate_group_id")
                or f"{record.get('source_id')}:{record.get('assay_type')}:parent_{record.get('parent_peptide_id')}"
            ),
            "plm_sequence": plm_sequence,
            "plm_cached_embedding": plm_cached_embedding,
            "plm_cached_mask": plm_cached_mask,
            "shadow_ids": [self.vocabs.shadow.encode(token) for token in tokens],
            "residue_features": build_residue_features(tokens, record),
            "residue_topology": topology,
            "sample_cat": {
                field: vocab.encode(record.get(field))
                for field, vocab in self.vocabs.sample.items()
            },
            "edit_cat": [
                {field: self.vocabs.edit[field].encode(event.get(field)) for field in EDIT_FIELDS}
                for event in edit_set
            ],
            "payload_ids": [
                [self.vocabs.payload.encode(token) for token in payload_tokens(event)]
                for event in edit_set
            ],
            "payload_graphs": [
                self.build_or_load_graph(record, event, event_index)
                for event_index, event in enumerate(edit_set)
            ],
            "payload_features": [payload_flags(event) for event in edit_set],
            "edit_site": [int(event.get("site_index") or 0) for event in edit_set],
            "topology_features": build_scaffold_topology_features(tokens, record),
            "property_before": float(record["property_before"]),
            "property_after": float(record["property_after"]),
            "delta_property": float(record["delta_property"]),
            "direction_label": int(float(record["delta_property"]) > 0),
            "sample_weight": float(record["sample_weight"]),
            "censored_property_flag": float(bool(record["censored_property_flag"])),
            "contains_manual_curation_monomer_final": float(
                bool(record.get("contains_manual_curation_monomer_final"))
            ),
            "contains_residue_graph_replacement": float(
                bool(record.get("contains_residue_graph_replacement"))
            ),
            "contains_unknown_replacement": float(bool(record.get("contains_unknown_replacement"))),
            "edit_count": int(record["edit_count"]),
        }


def pad_2d(sequences: list[list[int]], pad_value: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max((len(seq) for seq in sequences), default=1)
    out = torch.full((len(sequences), max_len), pad_value, dtype=torch.long)
    mask = torch.zeros((len(sequences), max_len), dtype=torch.bool)
    for row, seq in enumerate(sequences):
        if not seq:
            continue
        out[row, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        mask[row, : len(seq)] = True
    return out, mask


def collate_cedg(batch: list[dict[str, object]]) -> dict[str, object]:
    """Collate variable-length shadow sequences and edit sets."""

    shadow_ids, shadow_mask = pad_2d([item["shadow_ids"] for item in batch])
    residue_feature_tensor = torch.zeros(
        (len(batch), shadow_ids.shape[1], RESIDUE_FEATURE_DIM), dtype=torch.float32
    )
    for row, item in enumerate(batch):
        features = item["residue_features"]
        if features:
            residue_feature_tensor[row, : len(features)] = torch.tensor(features, dtype=torch.float32)
    cached_embeddings = [item.get("plm_cached_embedding") for item in batch]
    plm_cached_embedding_tensor = None
    plm_cached_mask_tensor = None
    if all(embedding is not None for embedding in cached_embeddings):
        plm_dim = int(cached_embeddings[0].shape[-1])
        plm_cached_embedding_tensor = torch.zeros(
            (len(batch), shadow_ids.shape[1], plm_dim),
            dtype=torch.float32,
        )
        plm_cached_mask_tensor = torch.zeros((len(batch), shadow_ids.shape[1]), dtype=torch.bool)
        for row, item in enumerate(batch):
            embedding = item["plm_cached_embedding"]
            mask = item["plm_cached_mask"]
            length = min(int(embedding.shape[0]), shadow_ids.shape[1])
            plm_cached_embedding_tensor[row, :length] = embedding[:length]
            plm_cached_mask_tensor[row, :length] = mask[:length]
    max_edits = max(max(len(item["edit_cat"]), 1) for item in batch)
    max_payload = max(
        (len(tokens) for item in batch for tokens in item["payload_ids"]),
        default=1,
    )
    max_atoms = max(
        (
            int(tensor.shape[0])
            for item in batch
            for graph in item["payload_graphs"]
            for tensor in (
                graph.old_atom_features,
                graph.new_atom_features,
                graph.delta_atom_features,
            )
        ),
        default=1,
    )
    max_residues = shadow_ids.shape[1]

    edit_cat: dict[str, torch.Tensor] = {}
    edit_mask = torch.zeros((len(batch), max_edits), dtype=torch.bool)
    for field in EDIT_FIELDS:
        edit_cat[field] = torch.zeros((len(batch), max_edits), dtype=torch.long)
    edit_site = torch.zeros((len(batch), max_edits), dtype=torch.float32)
    payload_ids = torch.zeros((len(batch), max_edits, max_payload), dtype=torch.long)
    payload_mask = torch.zeros((len(batch), max_edits, max_payload), dtype=torch.bool)
    payload_old_atom_features = torch.zeros(
        (len(batch), max_edits, max_atoms, ATOM_FEATURE_DIM), dtype=torch.float32
    )
    payload_new_atom_features = torch.zeros_like(payload_old_atom_features)
    payload_old_adjacency = torch.zeros((len(batch), max_edits, max_atoms, max_atoms), dtype=torch.float32)
    payload_new_adjacency = torch.zeros_like(payload_old_adjacency)
    payload_delta_atom_features = torch.zeros_like(payload_old_atom_features)
    payload_delta_edge_features = torch.zeros(
        (len(batch), max_edits, max_atoms, max_atoms, EDGE_FEATURE_DIM), dtype=torch.float32
    )
    payload_old_atom_mask = torch.zeros((len(batch), max_edits, max_atoms), dtype=torch.bool)
    payload_new_atom_mask = torch.zeros_like(payload_old_atom_mask)
    payload_delta_atom_mask = torch.zeros_like(payload_old_atom_mask)
    payload_atom_action = torch.zeros((len(batch), max_edits, max_atoms), dtype=torch.long)
    payload_atom_role = torch.zeros_like(payload_atom_action)
    payload_attachment_index_old = torch.full((len(batch), max_edits), -1, dtype=torch.long)
    payload_attachment_index_new = torch.full_like(payload_attachment_index_old, -1)
    payload_attachment_index_delta = torch.full_like(payload_attachment_index_old, -1)
    payload_graph_mode = torch.zeros((len(batch), max_edits), dtype=torch.long)
    payload_graph_valid_mask = torch.zeros((len(batch), max_edits), dtype=torch.bool)
    payload_feature_tensor = torch.zeros((len(batch), max_edits, PAYLOAD_FEATURE_DIM), dtype=torch.float32)
    residue_adj = torch.zeros((len(batch), max_residues, max_residues), dtype=torch.float32)
    residue_edge_type = torch.zeros((len(batch), max_residues, max_residues), dtype=torch.long)
    residue_distance = torch.zeros((len(batch), max_residues, max_residues), dtype=torch.float32)

    for row, item in enumerate(batch):
        events = item["edit_cat"]
        sites = item["edit_site"]
        payloads = item["payload_ids"]
        payload_graphs = item["payload_graphs"]
        payload_features = item["payload_features"]
        topology = item["residue_topology"]
        n_residues = int(topology["residue_adj"].shape[0])
        residue_adj[row, :n_residues, :n_residues] = topology["residue_adj"]
        residue_edge_type[row, :n_residues, :n_residues] = topology["residue_edge_type"]
        residue_distance[row, :n_residues, :n_residues] = topology["residue_distance"]
        edit_mask[row, : len(events)] = True
        for col, event in enumerate(events):
            for field in EDIT_FIELDS:
                edit_cat[field][row, col] = event[field]
            edit_site[row, col] = float(sites[col])
            payload = payloads[col]
            payload_ids[row, col, : len(payload)] = torch.tensor(payload, dtype=torch.long)
            payload_mask[row, col, : len(payload)] = True
            payload_feature_tensor[row, col] = torch.tensor(payload_features[col], dtype=torch.float32)
            graph = payload_graphs[col]

            n_old = int(graph.old_atom_features.shape[0])
            payload_old_atom_features[row, col, :n_old] = graph.old_atom_features
            payload_old_adjacency[row, col, :n_old, :n_old] = graph.old_adjacency
            payload_old_atom_mask[row, col, :n_old] = graph.old_atom_mask

            n_new = int(graph.new_atom_features.shape[0])
            payload_new_atom_features[row, col, :n_new] = graph.new_atom_features
            payload_new_adjacency[row, col, :n_new, :n_new] = graph.new_adjacency
            payload_new_atom_mask[row, col, :n_new] = graph.new_atom_mask

            n_delta = int(graph.delta_atom_features.shape[0])
            payload_delta_atom_features[row, col, :n_delta] = graph.delta_atom_features
            payload_delta_edge_features[row, col, :n_delta, :n_delta] = graph.delta_edge_features
            payload_delta_atom_mask[row, col, :n_delta] = graph.delta_atom_mask
            payload_atom_action[row, col, :n_delta] = graph.atom_action
            payload_atom_role[row, col, :n_delta] = graph.atom_role
            payload_attachment_index_old[row, col] = graph.attachment_index_old
            payload_attachment_index_new[row, col] = graph.attachment_index_new
            payload_attachment_index_delta[row, col] = graph.attachment_index_delta
            payload_graph_mode[row, col] = GRAPH_MODES.get(graph.mode, GRAPH_MODES["failed"])
            payload_graph_valid_mask[row, col] = bool(graph.valid)

    sample_cat = {
        field: torch.tensor([item["sample_cat"][field] for item in batch], dtype=torch.long)
        for field in SAMPLE_FIELDS
    }
    numeric = torch.tensor(
        [
            [
                float(item["property_before"]),
                float(item["edit_count"]),
            ]
            for item in batch
        ],
        dtype=torch.float32,
    )
    topology_feature_tensor = torch.tensor(
        [item["topology_features"] for item in batch],
        dtype=torch.float32,
    )
    group_to_id: dict[str, int] = {}
    group_ids: list[int] = []
    for item in batch:
        group = str(item["candidate_group_id"])
        if group not in group_to_id:
            group_to_id[group] = len(group_to_id)
        group_ids.append(group_to_id[group])

    out = {
        "sample_id": [item["sample_id"] for item in batch],
        "source_id": [item["source_id"] for item in batch],
        "candidate_group_id": [item["candidate_group_id"] for item in batch],
        "candidate_group_index": torch.tensor(group_ids, dtype=torch.long),
        "shadow_ids": shadow_ids,
        "shadow_mask": shadow_mask,
        "plm_sequence": [item["plm_sequence"] for item in batch],
        "residue_features": residue_feature_tensor,
        "residue_adj": residue_adj,
        "residue_edge_type": residue_edge_type,
        "residue_distance": residue_distance,
        "sample_cat": sample_cat,
        "edit_cat": edit_cat,
        "edit_site": edit_site,
        "payload_ids": payload_ids,
        "payload_mask": payload_mask,
        "payload_old_atom_features": payload_old_atom_features,
        "payload_new_atom_features": payload_new_atom_features,
        "payload_delta_atom_features": payload_delta_atom_features,
        "payload_old_adjacency": payload_old_adjacency,
        "payload_new_adjacency": payload_new_adjacency,
        "payload_delta_edge_features": payload_delta_edge_features,
        "payload_old_atom_mask": payload_old_atom_mask,
        "payload_new_atom_mask": payload_new_atom_mask,
        "payload_delta_atom_mask": payload_delta_atom_mask,
        "payload_atom_action": payload_atom_action,
        "payload_atom_role": payload_atom_role,
        "payload_attachment_index_old": payload_attachment_index_old,
        "payload_attachment_index_new": payload_attachment_index_new,
        "payload_attachment_index_delta": payload_attachment_index_delta,
        "payload_graph_mode": payload_graph_mode,
        "payload_graph_valid_mask": payload_graph_valid_mask,
        "payload_features": payload_feature_tensor,
        "edit_mask": edit_mask,
        "numeric": numeric,
        "topology_features": topology_feature_tensor,
        "property_after": torch.tensor([item["property_after"] for item in batch], dtype=torch.float32),
        "delta_property": torch.tensor([item["delta_property"] for item in batch], dtype=torch.float32),
        "direction_label": torch.tensor([item["direction_label"] for item in batch], dtype=torch.float32),
        "sample_weight": torch.tensor([item["sample_weight"] for item in batch], dtype=torch.float32),
    }
    if plm_cached_embedding_tensor is not None and plm_cached_mask_tensor is not None:
        out["plm_cached_embedding"] = plm_cached_embedding_tensor
        out["plm_cached_mask"] = plm_cached_mask_tensor
    return out


class CEDGCollatorWithPLM:
    """Optional wrapper that adds external PLM tokenizer outputs to the regular CEDG batch."""

    def __init__(self, tokenizer: object, base_collate: object = collate_cedg, max_plm_length: int = 256) -> None:
        self.tokenizer = tokenizer
        self.base_collate = base_collate
        self.max_plm_length = max_plm_length

    def __call__(self, batch: list[dict[str, object]]) -> dict[str, object]:
        out = self.base_collate(batch)
        seqs = [str(item["plm_sequence"]) for item in batch]
        tok = self.tokenizer(
            seqs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_plm_length,
            add_special_tokens=True,
        )
        out["plm_input_ids"] = tok["input_ids"]
        out["plm_attention_mask"] = tok["attention_mask"]
        out["plm_lengths"] = torch.tensor([len(seq) for seq in seqs], dtype=torch.long)
        return out


class CEDGCollatorWithESM:
    """Wrapper that adds ESM tokens produced by an `ESMBatchConverter`."""

    def __init__(self, esm_converter: object, base_collate: object = collate_cedg) -> None:
        self.esm_converter = esm_converter
        self.base_collate = base_collate

    def __call__(self, batch: list[dict[str, object]]) -> dict[str, object]:
        out = self.base_collate(batch)
        seqs = [str(item["plm_sequence"]) for item in batch]
        out.update(self.esm_converter(seqs))
        return out
