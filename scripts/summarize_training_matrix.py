#!/usr/bin/env python3
"""中文说明：汇总原始、Faris-only、plus_faris 三组训练/交叉评估结果。"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "reports" / "training_matrix"


def load(path: str) -> dict[str, float]:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def metric(data: dict[str, float], key: str, prefix: str = "") -> float | None:
    return data.get(prefix + key)


def main() -> None:
    original = load("runs/cedg_set_esm_cached_full/metrics.json")
    faris_only = load("runs/cedg_set_esm_cached_faris_only_rankheavy/metrics.json")
    faris_only_on_original = load("runs/cedg_set_esm_cached_faris_only_rankheavy/original_test_metrics.json")
    plus = load("runs/cedg_set_esm_cached_plus_faris_rankheavy/metrics.json")
    plus_on_original = load("runs/cedg_set_esm_cached_plus_faris_rankheavy/original_test_metrics.json")
    original_on_faris = load("reports/external_benchmarks/2024_faris_local_neighbor_d1/metrics.json")

    rows = [
        {
            "train_setting": "original_only",
            "eval_setting": "original_test",
            "delta_mae": metric(original, "delta_mae", "test_"),
            "delta_spearman": metric(original, "delta_spearman", "test_"),
            "direction_auc": metric(original, "direction_auc", "test_"),
            "ranking_spearman": metric(original, "ranking_spearman", "test_"),
            "ranking_ndcg_at_5": metric(original, "ranking_ndcg_at_5", "test_"),
            "best_hit_at_5": None,
            "positive_hit_at_5": None,
        },
        {
            "train_setting": "original_only",
            "eval_setting": "faris_all_external",
            "delta_mae": None,
            "delta_spearman": original_on_faris.get("overall_spearman"),
            "direction_auc": None,
            "ranking_spearman": original_on_faris.get("overall_spearman"),
            "ranking_ndcg_at_5": original_on_faris.get("ndcg_at_5"),
            "best_hit_at_5": original_on_faris.get("best_hit_at_5"),
            "positive_hit_at_5": original_on_faris.get("positive_hit_at_5"),
        },
        {
            "train_setting": "faris_only",
            "eval_setting": "faris_test",
            "delta_mae": metric(faris_only, "delta_mae", "test_"),
            "delta_spearman": metric(faris_only, "delta_spearman", "test_"),
            "direction_auc": metric(faris_only, "direction_auc", "test_"),
            "ranking_spearman": metric(faris_only, "ranking_spearman", "test_"),
            "ranking_ndcg_at_5": metric(faris_only, "ranking_ndcg_at_5", "test_"),
            "best_hit_at_5": None,
            "positive_hit_at_5": None,
        },
        {
            "train_setting": "faris_only",
            "eval_setting": "original_test",
            "delta_mae": faris_only_on_original.get("delta_mae"),
            "delta_spearman": faris_only_on_original.get("delta_spearman"),
            "direction_auc": faris_only_on_original.get("direction_auc"),
            "ranking_spearman": faris_only_on_original.get("ranking_spearman"),
            "ranking_ndcg_at_5": faris_only_on_original.get("ranking_ndcg_at_5"),
            "best_hit_at_5": None,
            "positive_hit_at_5": None,
        },
        {
            "train_setting": "original_plus_faris",
            "eval_setting": "combined_test",
            "delta_mae": metric(plus, "delta_mae", "test_"),
            "delta_spearman": metric(plus, "delta_spearman", "test_"),
            "direction_auc": metric(plus, "direction_auc", "test_"),
            "ranking_spearman": metric(plus, "ranking_spearman", "test_"),
            "ranking_ndcg_at_5": metric(plus, "ranking_ndcg_at_5", "test_"),
            "best_hit_at_5": None,
            "positive_hit_at_5": None,
        },
        {
            "train_setting": "original_plus_faris",
            "eval_setting": "original_test",
            "delta_mae": plus_on_original.get("delta_mae"),
            "delta_spearman": plus_on_original.get("delta_spearman"),
            "direction_auc": plus_on_original.get("direction_auc"),
            "ranking_spearman": plus_on_original.get("ranking_spearman"),
            "ranking_ndcg_at_5": plus_on_original.get("ranking_ndcg_at_5"),
            "best_hit_at_5": None,
            "positive_hit_at_5": None,
        },
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(OUT_DIR / "training_matrix_summary.csv", index=False)
    markdown = [
        "# CEDG-Set Training Matrix Summary",
        "",
        "| Train setting | Eval setting | delta MAE | delta Spearman | direction AUC | ranking Spearman | NDCG@5 | best_hit@5 | positive_hit@5 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        markdown.append(
            "| {train_setting} | {eval_setting} | {delta_mae} | {delta_spearman} | {direction_auc} | "
            "{ranking_spearman} | {ranking_ndcg_at_5} | {best_hit_at_5} | {positive_hit_at_5} |".format(
                **{
                    key: "N/A" if value is None else f"{value:.3f}" if isinstance(value, float) else value
                    for key, value in row.items()
                }
            )
        )
    markdown.extend(
        [
            "",
            "Interpretation:",
            "",
            "- `N/A` means the metric is not produced by that evaluation protocol, not that the run is missing.",
            "- `original_only -> original_test` is still the strongest in-domain baseline.",
            "- `original_only -> faris_all_external` shows useful external ranking signal on Faris local SAR.",
            "- `faris_only` is weak despite being in-domain, indicating that 1434 Faris pairs are too small for the full architecture.",
            "- `original_plus_faris` trains successfully but does not yet outperform the original-only model; next runs should tune Faris weight, checkpoint selection, or staged fine-tuning.",
        ]
    )
    (OUT_DIR / "training_matrix_summary.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
