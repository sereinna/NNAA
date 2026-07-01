#!/usr/bin/env python3
"""中文说明：汇总原始、Faris-only、plus_faris 三组训练/交叉评估结果。"""

from __future__ import annotations

import math
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "reports" / "training_matrix"


def load(path: str) -> dict[str, float]:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def metric(data: dict[str, float], key: str) -> float | None:
    value = data.get(key)
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return float(value)


def make_row(train_setting: str, eval_setting: str, data: dict[str, float], note: str) -> dict[str, object]:
    return {
        "train_setting": train_setting,
        "eval_setting": eval_setting,
        "delta_mae": metric(data, "delta_mae"),
        "delta_spearman": metric(data, "delta_spearman"),
        "direction_auc": metric(data, "direction_auc"),
        "ranking_spearman": metric(data, "ranking_spearman"),
        "ranking_ndcg_at_5": metric(data, "ranking_ndcg_at_5"),
        "best_hit_at_5": metric(data, "best_hit_at_5"),
        "positive_hit_at_5": metric(data, "positive_hit_at_5"),
        "note": note,
    }


def fmt(value: object) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def main() -> None:
    original_on_original = load("runs/cedg_set_esm_cached_full/original_test_metrics_unified.json")
    original_on_faris = load("reports/external_benchmarks/2024_faris_local_neighbor_d1/metrics.json")
    faris_on_faris = load("runs/cedg_set_esm_cached_faris_only_rankheavy/faris_test_metrics_unified.json")
    faris_on_original = load("runs/cedg_set_esm_cached_faris_only_rankheavy/original_test_metrics_unified.json")
    plus_on_combined = load("runs/cedg_set_esm_cached_plus_faris_rankheavy/combined_test_metrics_unified.json")
    plus_on_original = load("runs/cedg_set_esm_cached_plus_faris_rankheavy/original_test_metrics_unified.json")

    rows = [
        make_row("original_only", "original_test", original_on_original, "held-out original split"),
        make_row("original_only", "faris_all_score", original_on_faris, "all Faris local SAR groups scored as literature benchmark"),
        make_row("faris_only", "faris_test", faris_on_faris, "held-out Faris split"),
        make_row("faris_only", "original_test", faris_on_original, "cross-dataset original split"),
        make_row("original_plus_faris", "combined_test", plus_on_combined, "held-out split after merging original and Faris"),
        make_row("original_plus_faris", "original_test", plus_on_original, "cross-check on original test split"),
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(OUT_DIR / "training_matrix_summary.csv", index=False)
    markdown = [
        "# CEDG-Set Training Matrix Summary",
        "",
        "| Train setting | Eval setting | delta MAE | delta Spearman | direction AUC | ranking Spearman | NDCG@5 | best_hit@5 | positive_hit@5 | Note |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        markdown.append(
            "| {train_setting} | {eval_setting} | {delta_mae} | {delta_spearman} | {direction_auc} | "
            "{ranking_spearman} | {ranking_ndcg_at_5} | {best_hit_at_5} | {positive_hit_at_5} | {note} |".format(
                **{key: fmt(value) for key, value in row.items()}
            )
        )
    markdown.extend(
        [
            "",
            "Interpretation:",
            "",
            "- Metrics are now computed with a unified protocol wherever the dataset has `candidate_group_id` and `delta_property`.",
            "- `best_hit@5` checks whether the experimentally best edit in each candidate group is recovered in model top-5.",
            "- `positive_hit@5` checks whether model top-5 contains at least one experimentally positive edit in groups with positives.",
            "- `original_only -> original_test` is still the strongest in-domain baseline.",
            "- `original_only -> faris_all_score` shows useful cross-literature ranking signal on Faris local SAR.",
            "- `faris_only` is weak despite being in-domain, indicating that 1434 Faris pairs are too small for the full architecture.",
            "- `original_plus_faris` trains successfully but does not yet outperform the original-only model; next runs should tune Faris weight, checkpoint selection, or staged fine-tuning.",
        ]
    )
    (OUT_DIR / "training_matrix_summary.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
