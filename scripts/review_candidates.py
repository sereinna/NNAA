#!/usr/bin/env python3
"""中文说明：对候选 edit pair 应用保守自动审核规则，标记可用性和风险原因。"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

ALLOWED_SINGLE_EDIT_SCOPES = {"backbone_edit", "stereochemistry_edit"}
ALLOWED_SINGLE_OPERATIONS = {"N_methylation", "D_L_inversion"}


def is_censored(row: pd.Series) -> bool:
    return float(row["property_before"]) <= -9.99 or float(row["property_after"]) <= -9.99


def classify_pair(pair: pd.Series, events: pd.DataFrame) -> tuple[str, str, str]:
    pair_events = events[events["pair_id"].eq(pair["pair_id"])]
    if pair_events.empty:
        return "needs_manual", "missing_event", "No edit_event rows found for pair."

    if not bool(pair["same_scaffold"]):
        return "reject_auto", "not_same_scaffold", "Pair is not marked as same scaffold."

    if pair["assay_comparability"] != "same_source_same_assay":
        return (
            "needs_manual",
            "assay_comparability_uncertain",
            "Assay comparability is not same_source_same_assay.",
        )

    if int(pair["edit_count"]) != len(pair_events):
        return (
            "needs_manual",
            "edit_count_event_mismatch",
            "Pair edit_count does not match number of event rows.",
        )

    censored = is_censored(pair)

    if int(pair["edit_count"]) == 1:
        event = pair_events.iloc[0]
        if event["edit_scope"] not in ALLOWED_SINGLE_EDIT_SCOPES:
            return (
                "needs_manual",
                "unsupported_single_edit_scope",
                f"Single edit scope {event['edit_scope']} is not auto-accepted.",
            )
        if event["operation_type"] not in ALLOWED_SINGLE_OPERATIONS:
            return (
                "needs_manual",
                "unsupported_single_operation",
                f"Single edit operation {event['operation_type']} is not auto-accepted.",
            )
        if pair["evidence_level"] != "A_true_pair_candidate":
            if not censored:
                return (
                    "needs_manual",
                    "single_edit_evidence_not_A",
                    "Single edit is structurally simple but evidence level is not A candidate.",
                )
        if censored:
            return (
                "accept_auto_censored",
                "single_edit_supported_operation_censored_property",
                "Single-site supported operation accepted with censored_property_flag=True; document PAMPA -10 style labels.",
            )
        return (
            "accept_auto",
            "single_edit_same_source_non_censored_supported_operation",
            "Single-site same-source same-assay candidate with supported operation and non-censored PAMPA.",
        )

    if int(pair["edit_count"]) == 2:
        supported = pair_events["edit_scope"].isin(ALLOWED_SINGLE_EDIT_SCOPES).all() and pair_events[
            "operation_type"
        ].isin(ALLOWED_SINGLE_OPERATIONS).all()
        if supported:
            if censored:
                return (
                    "accept_auto_multisite_censored",
                    "multi_edit_supported_operation_censored_property",
                    "Multi-site supported edit set accepted with censored_property_flag=True; keep pair label at edit-set level.",
                )
            return (
                "accept_auto_multisite",
                "multi_edit_supported_operation_edit_set_label",
                "Multi-site supported edit set accepted; pair-level delta belongs to the whole edit set.",
            )
        return (
            "needs_manual",
            "multi_edit_unsupported_or_mixed",
            "Multi-edit candidate requires manual attribution review.",
        )

    return (
        "needs_manual",
        "too_many_edits",
        "More than two edits or unclear edit count.",
    )


def review_candidates(input_dir: Path, output_dir: Path) -> dict[str, int]:
    pair_path = input_dir / "edit_pair_table_2015_wang.csv"
    event_path = input_dir / "edit_event_table_2015_wang.csv"
    pairs = pd.read_csv(pair_path)
    events = pd.read_csv(event_path)

    rows = []
    for _, pair in pairs.iterrows():
        decision, reason, note = classify_pair(pair, events)
        row = pair.to_dict()
        row["review_decision"] = decision
        row["review_reason"] = reason
        row["reviewer_notes"] = note
        rows.append(row)

    reviewed = pd.DataFrame(rows)
    reviewed["censored_property_flag"] = reviewed.apply(is_censored, axis=1)
    reviewed["training_split_hint"] = reviewed["edit_count"].map(
        lambda x: "single_site" if int(x) == 1 else "multi_site"
    )
    event_review = events.merge(
        reviewed[
            [
                "pair_id",
                "review_decision",
                "review_reason",
                "reviewer_notes",
                "censored_property_flag",
                "training_split_hint",
            ]
        ],
        on="pair_id",
        how="left",
    )

    reviewed.to_csv(pair_path, index=False)
    event_review.to_csv(event_path, index=False)

    accepted_mask = reviewed["review_decision"].str.startswith("accept_auto")
    single_site = reviewed[accepted_mask & reviewed["training_split_hint"].eq("single_site")].copy()
    multi_site = reviewed[accepted_mask & reviewed["training_split_hint"].eq("multi_site")].copy()
    single_site.to_csv(output_dir / "training_single_site_pairs_2015_wang.csv", index=False)
    multi_site.to_csv(output_dir / "training_multi_site_pairs_2015_wang.csv", index=False)

    single_events = event_review[event_review["pair_id"].isin(single_site["pair_id"])].copy()
    multi_events = event_review[event_review["pair_id"].isin(multi_site["pair_id"])].copy()
    single_events.to_csv(output_dir / "training_single_site_events_2015_wang.csv", index=False)
    multi_events.to_csv(output_dir / "training_multi_site_events_2015_wang.csv", index=False)

    reason_counts = reviewed.groupby(["review_decision", "review_reason"]).size().reset_index(name="count")
    reason_counts.to_csv(output_dir / "review_decision_summary_2015_wang.csv", index=False)

    return {
        "reviewed": len(reviewed),
        "accepted": int(accepted_mask.sum()),
        "rejected": int(reviewed["review_decision"].eq("reject_auto").sum()),
        "manual": int(reviewed["review_decision"].eq("needs_manual").sum()),
        "single_site_pairs": len(single_site),
        "multi_site_pairs": len(multi_site),
        "events": len(event_review),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=ROOT / "data" / "pilot_2015_wang",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "pilot_2015_wang",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    counts = review_candidates(args.input_dir, args.output_dir)
    for key, value in counts.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
