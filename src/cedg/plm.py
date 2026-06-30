"""中文说明：CEDG 可选 ESM/PLM 编码模块，负责加载 ESM、构造 token batch 并返回 residue-level embedding。"""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch import nn


ESM_MODEL_NAMES = {
    "esm2_t6_8M_UR50D": "esm2_t6_8M_UR50D",
    "esm2_t36_3B_UR50D": "esm2_t36_3B_UR50D",
}

ESM_MODEL_SPECS = {
    "esm2_t6_8M_UR50D": {"layers": 6, "dim": 320},
    "esm2_t36_3B_UR50D": {"layers": 36, "dim": 2560},
}


def esm_model_spec(model_name: str) -> tuple[int, int]:
    """Return repr layer and embedding dim without loading the model when known."""

    spec = ESM_MODEL_SPECS.get(model_name)
    if spec is None:
        _, _, layer, dim = load_esm_model(model_name)
        return layer, dim
    return int(spec["layers"]), int(spec["dim"])


def load_esm_model(model_name: str = "esm2_t6_8M_UR50D") -> tuple[nn.Module, object, int, int]:
    """Load an ESM model through torch.hub and return model, alphabet, layer, dim."""

    hub_name = ESM_MODEL_NAMES.get(model_name, model_name)
    model, alphabet = torch.hub.load("facebookresearch/esm:main", hub_name)
    repr_layer = int(model.num_layers)
    embed_dim = int(model.embed_dim)
    return model, alphabet, repr_layer, embed_dim


def plm_cache_path(
    cache_dir: Path,
    sample_id: object,
    sequence: str,
    model_name: str,
) -> Path:
    """Stable cache path for frozen PLM residue embeddings."""

    payload = f"{sample_id}|{sequence}|{model_name}|plm_cache_v1"
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:20]
    sample = str(sample_id).replace("/", "_")
    return cache_dir / f"{sample}_{digest}.pt"


def save_plm_embedding(
    path: Path,
    embedding: torch.Tensor,
    mask: torch.Tensor,
    sequence: str,
    model_name: str,
) -> None:
    """Save one sample's residue-level PLM embedding."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    torch.save(
        {
            "embedding": embedding.detach().cpu(),
            "mask": mask.detach().cpu(),
            "sequence": sequence,
            "model_name": model_name,
        },
        tmp_path,
    )
    tmp_path.replace(path)


def load_plm_embedding(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Load one cached residue-level PLM embedding."""

    data = torch.load(path, map_location="cpu", weights_only=False)
    return data["embedding"], data["mask"].bool()


class ESMBatchConverter:
    """Callable collator helper that converts PLM sequences into ESM token tensors."""

    def __init__(self, alphabet: object, max_length: int = 256) -> None:
        self.converter = alphabet.get_batch_converter()
        self.max_length = max_length

    def __call__(self, sequences: list[str]) -> dict[str, torch.Tensor]:
        clipped = [seq[: self.max_length] for seq in sequences]
        labels = [f"seq_{idx}" for idx in range(len(clipped))]
        _, _, tokens = self.converter(list(zip(labels, clipped)))
        lengths = torch.tensor([len(seq) for seq in clipped], dtype=torch.long)
        return {
            "plm_tokens": tokens,
            "plm_lengths": lengths,
        }


class ESMResidueEncoder(nn.Module):
    """Frozen or fine-tunable ESM residue encoder."""

    def __init__(
        self,
        esm_model: nn.Module,
        repr_layer: int,
        output_dim: int,
        freeze: bool = True,
    ) -> None:
        super().__init__()
        self.esm = esm_model
        self.repr_layer = repr_layer
        self.output_dim = output_dim
        self.freeze = freeze
        if freeze:
            self.esm.eval()
            for param in self.esm.parameters():
                param.requires_grad = False

    def forward(
        self,
        plm_tokens: torch.Tensor,
        plm_lengths: torch.Tensor,
        target_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.freeze:
            self.esm.eval()
            with torch.no_grad():
                outputs = self.esm(plm_tokens, repr_layers=[self.repr_layer], return_contacts=False)
        else:
            outputs = self.esm(plm_tokens, repr_layers=[self.repr_layer], return_contacts=False)
        residue = outputs["representations"][self.repr_layer]
        batch_size = residue.shape[0]
        out = residue.new_zeros((batch_size, target_len, residue.shape[-1]))
        mask = torch.zeros((batch_size, target_len), dtype=torch.bool, device=residue.device)
        for row in range(batch_size):
            length = min(int(plm_lengths[row].item()), target_len, residue.shape[1] - 2)
            if length > 0:
                out[row, :length] = residue[row, 1 : 1 + length]
                mask[row, :length] = True
        return out, mask
