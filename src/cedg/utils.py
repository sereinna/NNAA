"""中文说明：CEDG 训练通用工具模块，集中放置随机种子、设备搬运等和模型无关的辅助函数。"""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set Python, NumPy, and PyTorch random seeds."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_to_device(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    """Move tensors in a nested batch dictionary to the selected device."""

    out: dict[str, object] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device)
        elif isinstance(value, dict):
            out[key] = {
                sub_key: sub_value.to(device) if isinstance(sub_value, torch.Tensor) else sub_value
                for sub_key, sub_value in value.items()
            }
        else:
            out[key] = value
    return out
