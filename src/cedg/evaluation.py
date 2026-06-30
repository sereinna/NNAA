"""中文说明：CEDG 第一阶段评估循环模块，负责模型推理、结果收集和指标汇总。"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from .metrics import cedg_regression_ranking_metrics
from .model import CEDGScoreModel
from .utils import move_to_device


@torch.no_grad()
def evaluate(model: CEDGScoreModel, loader: DataLoader, device: torch.device) -> dict[str, float]:
    """Run CEDG scorer/ranker evaluation on one dataloader."""

    model.eval()
    y_delta: list[float] = []
    p_delta: list[float] = []
    y_prop: list[float] = []
    p_prop: list[float] = []
    y_dir: list[int] = []
    p_dir: list[float] = []
    p_rank: list[float] = []
    p_uncertainty: list[float] = []
    groups: list[str] = []

    for batch in loader:
        batch = move_to_device(batch, device)
        outputs = model(batch)
        y_delta.extend(batch["delta_property"].detach().cpu().numpy().tolist())
        p_delta.extend(outputs["delta"].detach().cpu().numpy().tolist())
        y_prop.extend(batch["property_after"].detach().cpu().numpy().tolist())
        p_prop.extend(outputs["property_after"].detach().cpu().numpy().tolist())
        y_dir.extend(batch["direction_label"].detach().cpu().numpy().astype(int).tolist())
        p_dir.extend(torch.sigmoid(outputs["direction_logit"]).detach().cpu().numpy().tolist())
        p_rank.extend(outputs["ranking_score"].detach().cpu().numpy().tolist())
        p_uncertainty.extend(outputs["uncertainty"].detach().cpu().numpy().tolist())
        groups.extend([str(x) for x in batch["candidate_group_id"]])

    return cedg_regression_ranking_metrics(
        y_delta=np.asarray(y_delta),
        p_delta=np.asarray(p_delta),
        y_property=np.asarray(y_prop),
        p_property=np.asarray(p_prop),
        y_direction=np.asarray(y_dir),
        p_direction=np.asarray(p_dir),
        p_ranking=np.asarray(p_rank),
        p_uncertainty=np.asarray(p_uncertainty),
        groups=groups,
    )
