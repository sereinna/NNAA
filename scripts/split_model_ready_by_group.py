#!/usr/bin/env python3
"""中文说明：按 candidate_group_id 对 model-ready JSONL 重新切分 train/val/test。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def split_for_group(group_id: str, seed: int, train_frac: float, val_frac: float) -> str:
    digest = hashlib.sha1(f"{seed}:{group_id}".encode("utf-8")).hexdigest()
    value = int(digest[:12], 16) / float(16**12)
    if value < train_frac:
        return "train"
    if value < train_frac + val_frac:
        return "val"
    return "test"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--out-jsonl", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--sample-model-use-tier", default=None)
    args = parser.parse_args()

    records = load_jsonl(args.input_jsonl)
    group_to_split = {
        str(record.get("candidate_group_id")): split_for_group(
            str(record.get("candidate_group_id")), args.seed, args.train_frac, args.val_frac
        )
        for record in records
    }
    split_counts: dict[str, int] = {}
    for record in records:
        split = group_to_split[str(record.get("candidate_group_id"))]
        record["split"] = split
        if args.sample_model_use_tier is not None:
            record["sample_model_use_tier"] = args.sample_model_use_tier
        split_counts[split] = split_counts.get(split, 0) + 1

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_jsonl.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {
        "input_jsonl": str(args.input_jsonl),
        "out_jsonl": str(args.out_jsonl),
        "rows": len(records),
        "group_count": len(group_to_split),
        "split_counts": split_counts,
        "seed": args.seed,
        "train_frac": args.train_frac,
        "val_frac": args.val_frac,
    }
    out_summary = args.out_summary or args.out_jsonl.with_name(args.out_jsonl.stem + "_summary.json")
    out_summary.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
