"""中文说明：CEDG 第一阶段完整 scorer/ranker 模型，包含拓扑感知 peptide encoder、attachment-aware edit encoder 和候选排序头。

PyTorch CEDG-Set model.

This module implements the architecture described in `Untitled.md`: a peptide
context encoder, a site-conditioned atom-level edit encoder, an edit set
encoder, and prediction heads for property, delta, ranking, and uncertainty.
Each edit event uses curated delta metadata plus RDKit-derived old/new local
payload fragment graphs parsed from `final_chemical_payload`.
"""

from __future__ import annotations

import torch
from torch import nn

from .data import (
    ATOM_FEATURE_DIM,
    EDGE_FEATURE_DIM,
    PAYLOAD_FEATURE_DIM,
    RESIDUE_FEATURE_DIM,
    TOPOLOGY_FEATURE_DIM,
    CEDGVocabs,
    EDIT_INPUT_FIELDS,
    SAMPLE_INPUT_FIELDS,
)
from .chem_edit import ATOM_ACTIONS, ATOM_ROLES, GRAPH_MODES
from .topology import RESIDUE_EDGE_TYPES
from .plm import ESMResidueEncoder


def masked_mean(values: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    """Mean-pool a padded tensor along one dimension."""

    mask_f = mask.unsqueeze(-1).float()
    total = (values * mask_f).sum(dim=dim)
    denom = mask_f.sum(dim=dim).clamp_min(1.0)
    return total / denom


class MLP(nn.Module):
    """Small feed-forward block used by encoders and prediction heads."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PeptideContextEncoder(nn.Module):
    """Encode canonical shadow sequence into residue and scaffold context."""

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        hidden_dim: int,
        dropout: float,
        num_layers: int,
        num_heads: int,
        max_len: int,
        plm_dim: int = 0,
    ) -> None:
        super().__init__()
        self.plm_dim = plm_dim
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_len, emb_dim)
        self.residue_feature_proj = nn.Sequential(nn.Linear(RESIDUE_FEATURE_DIM, emb_dim), nn.GELU())
        self.plm_proj = (
            nn.Sequential(nn.Linear(plm_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
            if plm_dim > 0
            else None
        )
        self.plm_gate = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid()) if plm_dim > 0 else None
        self.edge_embedding = nn.Embedding(len(RESIDUE_EDGE_TYPES), hidden_dim, padding_idx=0)
        self.distance_proj = nn.Sequential(nn.Linear(1, hidden_dim), nn.GELU())
        self.topology_gate = nn.Sequential(nn.Linear(hidden_dim * 3, hidden_dim), nn.Sigmoid())
        self.input_proj = nn.Linear(emb_dim * 2, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        shadow_ids: torch.Tensor,
        shadow_mask: torch.Tensor,
        residue_features: torch.Tensor,
        residue_adj: torch.Tensor,
        residue_edge_type: torch.Tensor,
        residue_distance: torch.Tensor,
        plm_residue_embeddings: torch.Tensor | None = None,
        plm_residue_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len = shadow_ids.shape
        positions = torch.arange(seq_len, device=shadow_ids.device).unsqueeze(0).expand(batch_size, seq_len)
        token = self.embedding(shadow_ids) + self.position_embedding(
            positions.clamp_max(self.position_embedding.num_embeddings - 1)
        )
        topo = self.residue_feature_proj(residue_features)
        x = torch.cat([token, topo], dim=-1)
        x = self.input_proj(x)
        residue = self.encoder(x, src_key_padding_mask=~shadow_mask)
        residue = self.norm(residue)
        if self.plm_proj is not None and plm_residue_embeddings is not None:
            plm_context = self.plm_proj(plm_residue_embeddings)
            if plm_residue_mask is not None:
                plm_context = plm_context * plm_residue_mask.unsqueeze(-1).float()
            assert self.plm_gate is not None
            gate = self.plm_gate(torch.cat([residue, plm_context], dim=-1))
            residue = self.norm(residue + gate * plm_context)
        edge_context = self.edge_embedding(residue_edge_type).mean(dim=2)
        distance_context = self.distance_proj(residue_distance.mean(dim=2, keepdim=True))
        topo_gate = self.topology_gate(torch.cat([residue, edge_context, distance_context], dim=-1))
        neighbor_degree = residue_adj.sum(dim=-1).clamp_min(1.0)
        neighbor_context = torch.matmul(residue_adj, residue) / neighbor_degree.unsqueeze(-1)
        residue = self.norm(residue + topo_gate * neighbor_context)
        scaffold = masked_mean(residue, shadow_mask, dim=1)
        return residue, scaffold


class SiteConditionedPayloadEncoder(nn.Module):
    """Encode local chemical payload tokens under residue-site context."""

    def __init__(
        self,
        payload_vocab_size: int,
        emb_dim: int,
        hidden_dim: int,
        dropout: float,
        num_heads: int,
    ) -> None:
        super().__init__()
        self.payload_embedding = nn.Embedding(payload_vocab_size, emb_dim, padding_idx=0)
        self.payload_proj = nn.Linear(emb_dim, hidden_dim)
        self.site_to_query = nn.Linear(hidden_dim, hidden_dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.fusion = MLP(hidden_dim * 3, hidden_dim, hidden_dim, dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        payload_ids: torch.Tensor,
        payload_mask: torch.Tensor,
        site_context: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, max_edits, max_payload = payload_ids.shape
        flat_ids = payload_ids.view(batch_size * max_edits, max_payload)
        flat_mask = payload_mask.view(batch_size * max_edits, max_payload)
        flat_site = site_context.reshape(batch_size * max_edits, site_context.shape[-1])

        payload = self.payload_proj(self.payload_embedding(flat_ids))
        pooled = masked_mean(payload, flat_mask, dim=1)
        query = self.site_to_query(flat_site).unsqueeze(1)
        attended, _ = self.cross_attention(
            query=query,
            key=payload,
            value=payload,
            key_padding_mask=~flat_mask,
            need_weights=False,
        )
        attended = attended.squeeze(1)
        fused = self.fusion(torch.cat([flat_site, pooled, attended], dim=-1))
        return self.norm(fused).view(batch_size, max_edits, -1)


class GraphConvBlock(nn.Module):
    """Simple dense graph convolution block for small residue graphs."""

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        node_features: torch.Tensor,
        adjacency: torch.Tensor,
        site_context: torch.Tensor,
    ) -> torch.Tensor:
        updated = torch.matmul(adjacency, node_features)
        site = site_context.unsqueeze(1).expand(-1, updated.shape[1], -1)
        updated = self.linear(torch.cat([updated, site], dim=-1))
        return self.norm(node_features + self.dropout(torch.relu(updated)))


class PayloadGraphEncoder(nn.Module):
    """Encode a local edit payload fragment graph."""

    def __init__(self, hidden_dim: int, dropout: float, num_layers: int = 3) -> None:
        super().__init__()
        self.input_proj = nn.Linear(ATOM_FEATURE_DIM, hidden_dim)
        self.layers = nn.ModuleList([GraphConvBlock(hidden_dim, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        atom_features: torch.Tensor,
        adjacency: torch.Tensor,
        atom_mask: torch.Tensor,
        site_context: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, max_edits, max_atoms, _ = atom_features.shape
        flat_features = atom_features.view(batch_size * max_edits, max_atoms, -1)
        flat_adjacency = adjacency.view(batch_size * max_edits, max_atoms, max_atoms)
        flat_mask = atom_mask.view(batch_size * max_edits, max_atoms)
        flat_site = site_context.reshape(batch_size * max_edits, site_context.shape[-1])

        nodes = self.input_proj(flat_features)
        for layer in self.layers:
            nodes = layer(nodes, flat_adjacency, flat_site)
            nodes = nodes * flat_mask.unsqueeze(-1).float()
        pooled = masked_mean(nodes, flat_mask, dim=1)
        return self.norm(pooled).view(batch_size, max_edits, -1)


class SiteConditionedAtomEditEncoder(nn.Module):
    """Encode old/new payload fragment graphs as a site-conditioned edit delta."""

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.graph_encoder = PayloadGraphEncoder(hidden_dim, dropout)
        self.flag_proj = nn.Sequential(nn.Linear(PAYLOAD_FEATURE_DIM, hidden_dim), nn.GELU())
        self.delta_graph_encoder = PayloadGraphEncoder(hidden_dim, dropout)
        self.action_embedding = nn.Embedding(len(ATOM_ACTIONS), hidden_dim, padding_idx=0)
        self.role_embedding = nn.Embedding(len(ATOM_ROLES), hidden_dim, padding_idx=0)
        self.mode_embedding = nn.Embedding(len(GRAPH_MODES), hidden_dim, padding_idx=0)
        self.edge_proj = nn.Sequential(nn.Linear(EDGE_FEATURE_DIM, hidden_dim), nn.GELU())
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.fusion = MLP(hidden_dim * 9, hidden_dim, hidden_dim, dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        payload_old_atom_features: torch.Tensor,
        payload_new_atom_features: torch.Tensor,
        payload_old_adjacency: torch.Tensor,
        payload_new_adjacency: torch.Tensor,
        payload_delta_atom_features: torch.Tensor,
        payload_delta_edge_features: torch.Tensor,
        payload_old_atom_mask: torch.Tensor,
        payload_new_atom_mask: torch.Tensor,
        payload_delta_atom_mask: torch.Tensor,
        payload_atom_action: torch.Tensor,
        payload_atom_role: torch.Tensor,
        payload_graph_mode: torch.Tensor,
        payload_graph_valid_mask: torch.Tensor,
        payload_features: torch.Tensor,
        site_context: torch.Tensor,
    ) -> torch.Tensor:
        original = self.graph_encoder(
            payload_old_atom_features,
            payload_old_adjacency,
            payload_old_atom_mask,
            site_context,
        )
        modified = self.graph_encoder(
            payload_new_atom_features,
            payload_new_adjacency,
            payload_new_atom_mask,
            site_context,
        )
        delta_adjacency = (payload_delta_edge_features[..., 0] > 0).float()
        eye = torch.eye(delta_adjacency.shape[-1], device=delta_adjacency.device).view(1, 1, delta_adjacency.shape[-1], delta_adjacency.shape[-1])
        delta_adjacency = torch.maximum(delta_adjacency, eye)
        degree = delta_adjacency.sum(dim=-1).clamp_min(1.0)
        norm = degree.pow(-0.5)
        delta_adjacency = norm.unsqueeze(-1) * delta_adjacency * norm.unsqueeze(-2)
        delta_graph = self.delta_graph_encoder(
            payload_delta_atom_features,
            delta_adjacency,
            payload_delta_atom_mask,
            site_context,
        )
        action_context = masked_mean(self.action_embedding(payload_atom_action), payload_delta_atom_mask, dim=2)
        role_context = masked_mean(self.role_embedding(payload_atom_role), payload_delta_atom_mask, dim=2)
        edge_context = self.edge_proj(payload_delta_edge_features).mean(dim=(2, 3))
        mode_context = self.mode_embedding(payload_graph_mode)
        delta = (modified - original) * payload_graph_valid_mask.unsqueeze(-1).float()
        flags = self.flag_proj(payload_features)
        gated_delta = delta * self.gate(site_context)
        fused = self.fusion(
            torch.cat(
                [
                    site_context,
                    original,
                    modified,
                    gated_delta,
                    delta_graph,
                    action_context,
                    role_context,
                    edge_context + mode_context,
                    flags,
                ],
                dim=-1,
            )
        )
        return self.norm(fused)


class EditEventEncoder(nn.Module):
    """Fuse edit metadata, site position, site context, and payload encoding."""

    def __init__(
        self,
        vocabs: CEDGVocabs,
        emb_dim: int,
        hidden_dim: int,
        dropout: float,
        num_heads: int,
    ) -> None:
        super().__init__()
        self.embeddings = nn.ModuleDict(
            {
                field: nn.Embedding(len(vocabs.edit[field]), emb_dim, padding_idx=0)
                for field in EDIT_INPUT_FIELDS
            }
        )
        self.site_proj = nn.Sequential(nn.Linear(2, emb_dim), nn.GELU())
        self.payload = SiteConditionedPayloadEncoder(
            len(vocabs.payload), emb_dim, hidden_dim, dropout, num_heads
        )
        self.atom_edit = SiteConditionedAtomEditEncoder(hidden_dim, dropout)
        in_dim = emb_dim * (len(EDIT_INPUT_FIELDS) + 1) + hidden_dim * 3
        self.fusion = MLP(in_dim, hidden_dim, hidden_dim, dropout)

    def forward(
        self,
        edit_cat: dict[str, torch.Tensor],
        edit_site: torch.Tensor,
        payload_ids: torch.Tensor,
        payload_mask: torch.Tensor,
        payload_old_atom_features: torch.Tensor,
        payload_new_atom_features: torch.Tensor,
        payload_old_adjacency: torch.Tensor,
        payload_new_adjacency: torch.Tensor,
        payload_delta_atom_features: torch.Tensor,
        payload_delta_edge_features: torch.Tensor,
        payload_old_atom_mask: torch.Tensor,
        payload_new_atom_mask: torch.Tensor,
        payload_delta_atom_mask: torch.Tensor,
        payload_atom_action: torch.Tensor,
        payload_atom_role: torch.Tensor,
        payload_graph_mode: torch.Tensor,
        payload_graph_valid_mask: torch.Tensor,
        payload_features: torch.Tensor,
        edit_mask: torch.Tensor,
        residue_context: torch.Tensor,
        shadow_mask: torch.Tensor,
    ) -> torch.Tensor:
        pieces = [embedding(edit_cat[field]) for field, embedding in self.embeddings.items()]

        seq_len = shadow_mask.sum(dim=1).clamp_min(1).float()
        site_scaled = edit_site / seq_len.view(-1, 1)
        circular_site = torch.sin(site_scaled * 2.0 * torch.pi)
        pieces.append(self.site_proj(torch.stack([site_scaled, circular_site], dim=-1)))

        site_index = edit_site.long().clamp_min(1) - 1
        max_site = residue_context.shape[1] - 1
        site_index = site_index.clamp_max(max_site)
        gather_index = site_index.unsqueeze(-1).expand(-1, -1, residue_context.shape[-1])
        site_context = torch.gather(residue_context, 1, gather_index)

        payload_context = self.payload(payload_ids, payload_mask, site_context)
        atom_context = self.atom_edit(
            payload_old_atom_features,
            payload_new_atom_features,
            payload_old_adjacency,
            payload_new_adjacency,
            payload_delta_atom_features,
            payload_delta_edge_features,
            payload_old_atom_mask,
            payload_new_atom_mask,
            payload_delta_atom_mask,
            payload_atom_action,
            payload_atom_role,
            payload_graph_mode,
            payload_graph_valid_mask,
            payload_features,
            site_context,
        )
        fused = torch.cat(pieces + [site_context, payload_context, atom_context], dim=-1)
        encoded = self.fusion(fused)
        return encoded * edit_mask.unsqueeze(-1).float()


class EditSetEncoder(nn.Module):
    """Encode one or more local edits and their set-level interactions."""

    def __init__(self, hidden_dim: int, dropout: float, num_layers: int, num_heads: int) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
        self.interaction = MLP(hidden_dim * 2, hidden_dim, hidden_dim, dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, edit_embeddings: torch.Tensor, edit_mask: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(edit_embeddings, src_key_padding_mask=~edit_mask)
        encoded = encoded * edit_mask.unsqueeze(-1).float()
        mean = masked_mean(encoded, edit_mask, dim=1)
        summed = encoded.sum(dim=1)
        return self.norm(self.interaction(torch.cat([mean, summed], dim=-1)))


class PredictionHeads(nn.Module):
    """Property, delta, ranking, direction, and uncertainty heads."""

    def __init__(self, in_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.shared = MLP(in_dim, hidden_dim, hidden_dim, dropout)
        self.delta_head = nn.Linear(hidden_dim, 1)
        self.property_head = nn.Linear(hidden_dim, 1)
        self.ranking_head = nn.Linear(hidden_dim, 1)
        self.direction_head = nn.Linear(hidden_dim, 1)
        self.log_variance_head = nn.Linear(hidden_dim, 1)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        shared = self.shared(features)
        log_variance = self.log_variance_head(shared).squeeze(-1).clamp(-6.0, 4.0)
        return {
            "delta": self.delta_head(shared).squeeze(-1),
            "property_after": self.property_head(shared).squeeze(-1),
            "ranking_score": self.ranking_head(shared).squeeze(-1),
            "direction_logit": self.direction_head(shared).squeeze(-1),
            "log_variance": log_variance,
            "uncertainty": torch.exp(0.5 * log_variance),
        }


class CEDGScoreModel(nn.Module):
    """CEDG-Set scorer for delta PAMPA, property, rank, and uncertainty."""

    def __init__(
        self,
        vocabs: CEDGVocabs,
        emb_dim: int = 64,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        peptide_layers: int = 2,
        edit_layers: int = 2,
        num_heads: int = 4,
        max_len: int = 128,
        esm_model: nn.Module | None = None,
        esm_repr_layer: int | None = None,
        esm_dim: int = 0,
        freeze_esm: bool = True,
    ) -> None:
        super().__init__()
        self.esm_encoder = (
            ESMResidueEncoder(esm_model, int(esm_repr_layer), esm_dim, freeze=freeze_esm)
            if esm_model is not None and esm_dim > 0 and esm_repr_layer is not None
            else None
        )
        self.esm_dim = int(esm_dim)
        self.peptide = PeptideContextEncoder(
            len(vocabs.shadow),
            emb_dim,
            hidden_dim,
            dropout,
            peptide_layers,
            num_heads,
            max_len,
            plm_dim=self.esm_dim,
        )
        self.edit_event = EditEventEncoder(vocabs, emb_dim, hidden_dim, dropout, num_heads)
        self.edit_set = EditSetEncoder(hidden_dim, dropout, edit_layers, num_heads)
        self.sample_embeddings = nn.ModuleDict(
            {
                field: nn.Embedding(len(vocabs.sample[field]), emb_dim, padding_idx=0)
                for field in SAMPLE_INPUT_FIELDS
            }
        )
        self.numeric_proj = nn.Sequential(nn.Linear(2, emb_dim), nn.LayerNorm(emb_dim), nn.GELU())
        self.topology_proj = nn.Sequential(nn.Linear(TOPOLOGY_FEATURE_DIM, emb_dim), nn.LayerNorm(emb_dim), nn.GELU())
        head_in = hidden_dim * 2 + emb_dim * (len(SAMPLE_INPUT_FIELDS) + 2)
        self.heads = PredictionHeads(head_in, hidden_dim, dropout)

    def forward(self, batch: dict[str, object]) -> dict[str, torch.Tensor]:
        plm_residue_embeddings = None
        plm_residue_mask = None
        if self.esm_encoder is not None:
            plm_residue_embeddings, plm_residue_mask = self.esm_encoder(
                batch["plm_tokens"],
                batch["plm_lengths"],
                target_len=batch["shadow_ids"].shape[1],
            )
        elif "plm_cached_embedding" in batch:
            plm_residue_embeddings = batch["plm_cached_embedding"]
            plm_residue_mask = batch["plm_cached_mask"]
        residue_context, peptide_context = self.peptide(
            batch["shadow_ids"],
            batch["shadow_mask"],
            batch["residue_features"],
            batch["residue_adj"],
            batch["residue_edge_type"],
            batch["residue_distance"],
            plm_residue_embeddings,
            plm_residue_mask,
        )
        edit_embeddings = self.edit_event(
            batch["edit_cat"],
            batch["edit_site"],
            batch["payload_ids"],
            batch["payload_mask"],
            batch["payload_old_atom_features"],
            batch["payload_new_atom_features"],
            batch["payload_old_adjacency"],
            batch["payload_new_adjacency"],
            batch["payload_delta_atom_features"],
            batch["payload_delta_edge_features"],
            batch["payload_old_atom_mask"],
            batch["payload_new_atom_mask"],
            batch["payload_delta_atom_mask"],
            batch["payload_atom_action"],
            batch["payload_atom_role"],
            batch["payload_graph_mode"],
            batch["payload_graph_valid_mask"],
            batch["payload_features"],
            batch["edit_mask"],
            residue_context,
            batch["shadow_mask"],
        )
        edit_set = self.edit_set(edit_embeddings, batch["edit_mask"])
        sample_parts = [
            embedding(batch["sample_cat"][field])
            for field, embedding in self.sample_embeddings.items()
        ]
        sample_parts.append(self.numeric_proj(batch["numeric"]))
        sample_parts.append(self.topology_proj(batch["topology_features"]))
        features = torch.cat([peptide_context, edit_set] + sample_parts, dim=-1)
        return self.heads(features)
