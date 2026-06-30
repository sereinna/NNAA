"""中文说明：CEDG 第一阶段训练损失模块，集中管理性质回归、异方差不确定性和候选组排序损失。"""

from __future__ import annotations

import torch
from torch import nn


def weighted_mse(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Weighted MSE normalized by total sample weight."""

    return ((pred - target).pow(2) * weight).sum() / weight.sum().clamp_min(1e-8)


def weighted_heteroscedastic_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    log_variance: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Gaussian negative log-likelihood with learned log variance."""

    raw = 0.5 * (torch.exp(-log_variance) * (pred - target).pow(2) + log_variance)
    return (raw * weight).sum() / weight.sum().clamp_min(1e-8)


def group_pairwise_ranking_loss(
    score: torch.Tensor,
    target: torch.Tensor,
    group_index: torch.Tensor,
    weight: torch.Tensor,
    margin: float = 0.05,
) -> torch.Tensor:
    """Pairwise ranking loss within candidate groups that share the same parent scaffold."""

    losses: list[torch.Tensor] = []
    for group in torch.unique(group_index):
        idx = torch.nonzero(group_index == group, as_tuple=False).flatten()
        if idx.numel() < 2:
            continue
        y = target[idx]
        s = score[idx]
        w = weight[idx]
        diff_y = y.unsqueeze(1) - y.unsqueeze(0)
        valid = diff_y > margin
        if not torch.any(valid):
            continue
        diff_s = s.unsqueeze(1) - s.unsqueeze(0)
        pair_weight = (w.unsqueeze(1) * w.unsqueeze(0)).sqrt()
        losses.append((nn.functional.softplus(-diff_s[valid]) * pair_weight[valid]).mean())
    if not losses:
        return score.sum() * 0.0
    return torch.stack(losses).mean()


def group_listwise_ranking_loss(
    score: torch.Tensor,
    target: torch.Tensor,
    group_index: torch.Tensor,
    weight: torch.Tensor,
    temperature: float = 0.2,
) -> torch.Tensor:
    """Listwise soft-label ranking loss within candidate groups."""

    losses: list[torch.Tensor] = []
    for group in torch.unique(group_index):
        idx = torch.nonzero(group_index == group, as_tuple=False).flatten()
        if idx.numel() < 2:
            continue
        y = target[idx]
        s = score[idx]
        w = weight[idx]
        target_prob = torch.softmax(y / temperature, dim=0)
        log_prob = torch.log_softmax(s, dim=0)
        losses.append((-(target_prob * log_prob) * w / w.mean().clamp_min(1e-8)).sum())
    if not losses:
        return score.sum() * 0.0
    return torch.stack(losses).mean()


def group_topk_ranking_loss(
    score: torch.Tensor,
    target: torch.Tensor,
    group_index: torch.Tensor,
    weight: torch.Tensor,
    top_fraction: float = 0.2,
    margin: float = 0.1,
) -> torch.Tensor:
    """Emphasize separating experimentally best candidates from the rest of each group."""

    losses: list[torch.Tensor] = []
    for group in torch.unique(group_index):
        idx = torch.nonzero(group_index == group, as_tuple=False).flatten()
        if idx.numel() < 3:
            continue
        y = target[idx]
        s = score[idx]
        w = weight[idx]
        top_count = max(1, int(round(float(idx.numel()) * top_fraction)))
        top_local = torch.topk(y, k=top_count).indices
        bottom_count = max(1, int(idx.numel()) - top_count)
        bottom_local = torch.topk(-y, k=bottom_count).indices
        diff_y = y[top_local].unsqueeze(1) - y[bottom_local].unsqueeze(0)
        valid = diff_y > margin
        if not torch.any(valid):
            continue
        diff_s = s[top_local].unsqueeze(1) - s[bottom_local].unsqueeze(0)
        pair_weight = (w[top_local].unsqueeze(1) * w[bottom_local].unsqueeze(0)).sqrt()
        losses.append((nn.functional.softplus(-diff_s[valid]) * pair_weight[valid]).mean())
    if not losses:
        return score.sum() * 0.0
    return torch.stack(losses).mean()


def compute_cedg_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    weights: dict[str, float] | None = None,
) -> torch.Tensor:
    """Joint CEDG scorer/ranker objective used by the first-stage training script."""

    loss_weights = {
        "delta": 1.0,
        "property": 0.2,
        "ranking_mse": 0.05,
        "pairwise": 0.6,
        "listwise": 0.4,
        "topk": 0.8,
        "direction": 0.2,
    }
    if weights is not None:
        loss_weights.update(weights)
    weight = batch["sample_weight"]
    delta_loss = weighted_heteroscedastic_mse(
        outputs["delta"], batch["delta_property"], outputs["log_variance"], weight
    )
    property_loss = weighted_mse(outputs["property_after"], batch["property_after"], weight)
    ranking_loss = weighted_mse(outputs["ranking_score"], batch["delta_property"], weight)
    group_ranking_loss = group_pairwise_ranking_loss(
        outputs["ranking_score"],
        batch["delta_property"],
        batch["candidate_group_index"],
        weight,
    )
    listwise_ranking_loss = group_listwise_ranking_loss(
        outputs["ranking_score"],
        batch["delta_property"],
        batch["candidate_group_index"],
        weight,
    )
    topk_ranking_loss = group_topk_ranking_loss(
        outputs["ranking_score"],
        batch["delta_property"],
        batch["candidate_group_index"],
        weight,
    )
    direction_loss_raw = nn.functional.binary_cross_entropy_with_logits(
        outputs["direction_logit"], batch["direction_label"], reduction="none"
    )
    direction_loss = (direction_loss_raw * weight).sum() / weight.sum().clamp_min(1e-8)
    return (
        loss_weights["delta"] * delta_loss
        + loss_weights["property"] * property_loss
        + loss_weights["ranking_mse"] * ranking_loss
        + loss_weights["pairwise"] * group_ranking_loss
        + loss_weights["listwise"] * listwise_ranking_loss
        + loss_weights["topk"] * topk_ranking_loss
        + loss_weights["direction"] * direction_loss
    )
