#!/usr/bin/env python3
"""中文说明：预构建 CEDG EditGraph tensor 缓存，避免训练时重复运行 RDKit/MCS 构图。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cedg.data import CEDGDataset, build_vocabs, collate_cedg, load_jsonl  # noqa: E402


DATASET = ROOT / "data" / "final" / "model_ready" / "peptide_component" / "cedg_score_dataset.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--graph-cache-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--allow-absolute-only", action="store_true")
    args = parser.parse_args()

    cache_dir = args.graph_cache_dir
    if cache_dir is None:
        cache_dir = ROOT / "data" / "final" / "cache" / "edit_graphs" / args.dataset.parent.name

    records = load_jsonl(args.dataset)
    vocabs = build_vocabs(records)
    total = 0
    for split in ("train", "val", "test"):
        dataset = CEDGDataset(
            records,
            vocabs,
            split,
            allow_absolute_only=args.allow_absolute_only,
            graph_cache_dir=cache_dir,
            use_graph_cache=True,
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_cedg,
        )
        for batch in loader:
            total += len(batch["sample_id"])
        print(f"{split}: cached {len(dataset)} samples")
    print(f"cache_dir={cache_dir}")
    print(f"total={total}")


if __name__ == "__main__":
    main()
