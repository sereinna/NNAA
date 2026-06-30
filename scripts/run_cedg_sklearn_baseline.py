#!/usr/bin/env python3
"""中文说明：在模型可读 CEDG 数据集上运行轻量 sklearn baseline，用于和神经模型对照。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.dummy import DummyRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATASET = (
    ROOT
    / "data"
    / "final"
    / "model_ready"
    / "peptide_component"
    / "cedg_score_dataset.jsonl"
)
OUT_DIR = ROOT / "data" / "final" / "model_ready" / "peptide_component" / "baseline"


def load_records(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            records.append(json.loads(line))
    return records


def add_count(features: dict[str, float], key: str, value: object, amount: float = 1.0) -> None:
    if value is None:
        return
    text = str(value)
    if not text or text == "nan":
        return
    features[f"{key}={text}"] = features.get(f"{key}={text}", 0.0) + amount


def shadow_tokens(sequence: object) -> list[str]:
    return [part for part in str(sequence).split(".") if part]


def payload_tokens(payload: object) -> list[str]:
    return [part for part in str(payload).replace(";", "_").split("_") if part]


def featurize(record: dict[str, object], feature_set: str) -> dict[str, float]:
    features: dict[str, float] = {
        "bias": 1.0,
        "property_before": float(record["property_before"]),
        "edit_count": float(record["edit_count"]),
        "censored_property_flag": float(bool(record["censored_property_flag"])),
        "contains_residue_graph_replacement": float(
            bool(record["contains_residue_graph_replacement"])
        ),
    }

    add_count(features, "assay_type", record.get("assay_type"))
    add_count(features, "table_type", record.get("table_type"))
    add_count(features, "sample_model_use_tier", record.get("sample_model_use_tier"))

    if feature_set in {"shadow_edit", "shadow_edit_source"}:
        for token in shadow_tokens(record.get("parent_shadow_sequence")):
            add_count(features, "shadow_token", token)
        tokens = shadow_tokens(record.get("parent_shadow_sequence"))
        for left, right in zip(tokens, tokens[1:]):
            add_count(features, "shadow_bigram", f"{left}.{right}")

    if feature_set in {"edit", "shadow_edit", "shadow_edit_source"}:
        edit_set = record["edit_set"]
        assert isinstance(edit_set, list)
        for edit in edit_set:
            assert isinstance(edit, dict)
            site = edit.get("site_index")
            add_count(features, "site", site)
            add_count(features, "anchor", edit.get("anchor_for_alignment"))
            add_count(features, "scope", edit.get("final_edit_scope"))
            add_count(features, "payload_type", edit.get("final_chemical_payload_type"))
            add_count(features, "payload", edit.get("final_chemical_payload"))
            add_count(features, "subclass", edit.get("edit_event_subclass"))
            add_count(features, "orig", edit.get("original_monomer"))
            add_count(features, "mod", edit.get("modified_monomer"))
            add_count(
                features,
                "site_scope",
                f"{site}:{edit.get('final_edit_scope')}",
            )
            add_count(
                features,
                "anchor_scope",
                f"{edit.get('anchor_for_alignment')}:{edit.get('final_edit_scope')}",
            )
            for token in payload_tokens(edit.get("final_chemical_payload")):
                add_count(features, "payload_token", token)

    if feature_set == "shadow_edit_source":
        add_count(features, "source_id", record.get("source_id"))
        add_count(
            features,
            "source_strategy",
            f"{record.get('source_id')}:{record.get('candidate_generation_strategy')}",
        )

    return features


def split_records(records: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    out = {"train": [], "val": [], "test": []}
    for record in records:
        out[str(record["split"])].append(record)
    return out


def arrays(records: Iterable[dict[str, object]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = list(records)
    y = np.asarray([float(row["delta_property"]) for row in rows])
    direction = np.asarray([int(float(row["delta_property"]) > 0) for row in rows])
    weights = np.asarray([float(row["sample_weight"]) for row in rows])
    return y, direction, weights


def safe_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
        return float("nan")
    value = spearmanr(y_true, y_pred).statistic
    return float(value) if not math.isnan(value) else float("nan")


def safe_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, score))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "spearman": safe_spearman(y_true, y_pred),
        "direction_accuracy_from_delta": float(accuracy_score(y_true > 0, y_pred > 0)),
        "direction_auc_from_delta": safe_auc((y_true > 0).astype(int), y_pred),
    }


def source_mean_predictions(
    train: list[dict[str, object]],
    target: list[dict[str, object]],
) -> np.ndarray:
    train_df = pd.DataFrame(
        {
            "source_id": [row["source_id"] for row in train],
            "delta_property": [row["delta_property"] for row in train],
        }
    )
    global_mean = float(train_df["delta_property"].mean())
    source_mean = train_df.groupby("source_id")["delta_property"].mean().to_dict()
    return np.asarray([source_mean.get(row["source_id"], global_mean) for row in target])


def fit_ridge(
    train: list[dict[str, object]],
    target: list[dict[str, object]],
    feature_set: str,
) -> np.ndarray:
    x_train = [featurize(row, feature_set) for row in train]
    x_target = [featurize(row, feature_set) for row in target]
    y_train, _, weights = arrays(train)
    model = make_pipeline(
        DictVectorizer(sparse=True),
        StandardScaler(with_mean=False),
        Ridge(alpha=1.0),
    )
    model.fit(x_train, y_train, ridge__sample_weight=weights)
    return model.predict(x_target)


def fit_logistic_direction(
    train: list[dict[str, object]],
    target: list[dict[str, object]],
    feature_set: str,
) -> np.ndarray:
    x_train = [featurize(row, feature_set) for row in train]
    x_target = [featurize(row, feature_set) for row in target]
    _, y_train, weights = arrays(train)
    model = make_pipeline(
        DictVectorizer(sparse=True),
        StandardScaler(with_mean=False),
        LogisticRegression(max_iter=2000, class_weight="balanced"),
    )
    model.fit(x_train, y_train, logisticregression__sample_weight=weights)
    return model.predict_proba(x_target)[:, 1]


def evaluate(records: list[dict[str, object]]) -> pd.DataFrame:
    split = split_records(records)
    train = split["train"]
    y_train, _, train_weights = arrays(train)
    results: list[dict[str, object]] = []

    dummy = DummyRegressor(strategy="mean")
    dummy.fit(np.zeros((len(y_train), 1)), y_train, sample_weight=train_weights)

    model_names = [
        "mean_delta",
        "source_mean_delta",
        "ridge_edit",
        "ridge_shadow_edit",
        "ridge_shadow_edit_source",
    ]

    predictions: dict[str, dict[str, np.ndarray]] = {name: {} for name in model_names}
    direction_scores: dict[str, dict[str, np.ndarray]] = {}
    for split_name in ("val", "test"):
        target = split[split_name]
        predictions["mean_delta"][split_name] = dummy.predict(np.zeros((len(target), 1)))
        predictions["source_mean_delta"][split_name] = source_mean_predictions(train, target)
        predictions["ridge_edit"][split_name] = fit_ridge(train, target, "edit")
        predictions["ridge_shadow_edit"][split_name] = fit_ridge(train, target, "shadow_edit")
        predictions["ridge_shadow_edit_source"][split_name] = fit_ridge(
            train, target, "shadow_edit_source"
        )
        direction_scores[split_name] = {
            "logreg_edit": fit_logistic_direction(train, target, "edit"),
            "logreg_shadow_edit": fit_logistic_direction(train, target, "shadow_edit"),
            "logreg_shadow_edit_source": fit_logistic_direction(
                train, target, "shadow_edit_source"
            ),
        }

    for model_name, by_split in predictions.items():
        for split_name, pred in by_split.items():
            y_true, direction_true, _ = arrays(split[split_name])
            metrics = regression_metrics(y_true, pred)
            row = {
                "task": "delta_regression",
                "model": model_name,
                "split": split_name,
                "rows": len(y_true),
            }
            row.update(metrics)
            results.append(row)

    for split_name, by_model in direction_scores.items():
        _, direction_true, _ = arrays(split[split_name])
        for model_name, score in by_model.items():
            pred_label = (score >= 0.5).astype(int)
            results.append(
                {
                    "task": "direction_classification",
                    "model": model_name,
                    "split": split_name,
                    "rows": len(direction_true),
                    "accuracy": float(accuracy_score(direction_true, pred_label)),
                    "auc": safe_auc(direction_true, score),
                }
            )

    return pd.DataFrame(results)


def write_report(results: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_dir / "sklearn_baseline_results.csv", index=False)
    lines = [
        "# CEDG-Score sklearn Baseline Validation",
        "",
        "Dataset: `data/final/model_ready/peptide_component/cedg_score_dataset.jsonl`",
        "",
        "## Delta Regression",
        "",
        results[results["task"].eq("delta_regression")]
        .sort_values(["split", "mae"])
        .to_markdown(index=False),
        "",
        "## Direction Classification",
        "",
        results[results["task"].eq("direction_classification")]
        .sort_values(["split", "auc"], ascending=[True, False])
        .to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        "- `mean_delta` is the minimum sanity baseline.",
        "- `source_mean_delta` tests how much source identity alone explains the split.",
        "- `ridge_edit` uses edit-set metadata without peptide shadow context.",
        "- `ridge_shadow_edit` adds parent shadow sequence token features.",
        "- `ridge_shadow_edit_source` adds source identity and should be treated as a source-bias upper sanity check, not the main model.",
        "",
    ]
    (out_dir / "sklearn_baseline_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    records = load_records(args.dataset)
    results = evaluate(records)
    write_report(results, args.out_dir)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
