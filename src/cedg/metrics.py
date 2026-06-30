"""中文说明：CEDG 第一阶段评估指标模块，包含回归、方向分类、不确定性和候选组排序指标。"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, roc_auc_score


def safe_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Spearman correlation with stable NaN handling for constant arrays."""

    if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
        return float("nan")
    value = spearmanr(y_true, y_pred).statistic
    return float(value) if not math.isnan(value) else float("nan")


def safe_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    """AUROC with stable NaN handling when one class is absent."""

    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, score))


def group_pairwise_accuracy(groups: list[str], y_true: np.ndarray, score: np.ndarray) -> float:
    """Pairwise ordering accuracy within candidate groups."""

    correct = 0
    total = 0
    by_group: dict[str, list[int]] = {}
    for idx, group in enumerate(groups):
        by_group.setdefault(group, []).append(idx)
    for indices in by_group.values():
        if len(indices) < 2:
            continue
        for i, left in enumerate(indices):
            for right in indices[i + 1 :]:
                true_diff = y_true[left] - y_true[right]
                if true_diff == 0:
                    continue
                pred_diff = score[left] - score[right]
                correct += int((true_diff > 0) == (pred_diff > 0))
                total += 1
    return float(correct / total) if total else float("nan")


def group_ndcg_at_k(groups: list[str], y_true: np.ndarray, score: np.ndarray, k: int = 5) -> float:
    """NDCG@k within parent-scaffold candidate groups."""

    values: list[float] = []
    by_group: dict[str, list[int]] = {}
    for idx, group in enumerate(groups):
        by_group.setdefault(group, []).append(idx)
    for indices in by_group.values():
        if len(indices) < 2:
            continue
        top = sorted(indices, key=lambda i: score[i], reverse=True)[:k]
        ideal = sorted(indices, key=lambda i: y_true[i], reverse=True)[:k]
        min_rel = min(float(y_true[i]) for i in indices)
        gains = np.asarray([float(y_true[i]) - min_rel for i in top], dtype=float)
        ideal_gains = np.asarray([float(y_true[i]) - min_rel for i in ideal], dtype=float)
        discounts = 1.0 / np.log2(np.arange(2, len(gains) + 2))
        dcg = float(np.sum(gains * discounts))
        idcg = float(np.sum(ideal_gains * discounts))
        if idcg > 0:
            values.append(dcg / idcg)
    return float(np.mean(values)) if values else float("nan")


def cedg_regression_ranking_metrics(
    y_delta: np.ndarray,
    p_delta: np.ndarray,
    y_property: np.ndarray,
    p_property: np.ndarray,
    y_direction: np.ndarray,
    p_direction: np.ndarray,
    p_ranking: np.ndarray,
    p_uncertainty: np.ndarray,
    groups: list[str],
) -> dict[str, float]:
    """Aggregate all first-stage CEDG validation/test metrics."""

    abs_err = np.abs(y_delta - p_delta)
    uncertainty = np.maximum(p_uncertainty, 1e-8)
    z_score = abs_err / uncertainty
    return {
        "delta_mae": float(mean_absolute_error(y_delta, p_delta)),
        "delta_rmse": float(mean_squared_error(y_delta, p_delta) ** 0.5),
        "delta_spearman": safe_spearman(y_delta, p_delta),
        "ranking_spearman": safe_spearman(y_delta, p_ranking),
        "ranking_pairwise_accuracy": group_pairwise_accuracy(groups, y_delta, p_ranking),
        "ranking_ndcg_at_5": group_ndcg_at_k(groups, y_delta, p_ranking, k=5),
        "property_mae": float(mean_absolute_error(y_property, p_property)),
        "direction_accuracy": float(accuracy_score(y_direction, p_direction >= 0.5)),
        "direction_auc": safe_auc(y_direction, p_direction),
        "uncertainty_error_spearman": safe_spearman(abs_err, p_uncertainty),
        "uncertainty_abs_z_mean": float(np.mean(z_score)),
        "selective_risk_coverage_80": selective_risk(abs_err, p_uncertainty, coverage=0.8),
        "mean_uncertainty": float(np.mean(p_uncertainty)),
    }


def selective_risk(abs_error: np.ndarray, uncertainty: np.ndarray, coverage: float = 0.8) -> float:
    """Mean error after retaining the lowest-uncertainty fraction."""

    if len(abs_error) == 0:
        return float("nan")
    keep = max(1, int(round(len(abs_error) * coverage)))
    order = np.argsort(uncertainty)
    return float(np.mean(abs_error[order[:keep]]))
