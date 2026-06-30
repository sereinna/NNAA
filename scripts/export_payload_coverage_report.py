#!/usr/bin/env python3
"""中文说明：导出 CEDG EditGraph 覆盖率报告，统计 graph mode、validity、attachment 和 payload 类型分布。"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cedg.data import CEDGDataset, build_vocabs, collate_cedg, load_jsonl  # noqa: E402


DATASET = ROOT / "data" / "final" / "model_ready" / "peptide_component" / "cedg_score_dataset.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "reports" / "cedg_payload_coverage")
    parser.add_argument("--graph-cache-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    records = load_jsonl(args.dataset)
    vocabs = build_vocabs(records)
    cache_dir = args.graph_cache_dir or ROOT / "data" / "final" / "cache" / "edit_graphs" / args.dataset.parent.name
    rows: list[dict[str, object]] = []
    mode_counter: Counter[str] = Counter()
    attachment_counter: Counter[str] = Counter()
    payload_type_counter: Counter[str] = Counter()
    valid_counter: Counter[str] = Counter()

    for split in ("train", "val", "test"):
        dataset = CEDGDataset(records, vocabs, split, graph_cache_dir=cache_dir, use_graph_cache=True)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_cedg)
        for batch in loader:
            for row_idx, sample_id in enumerate(batch["sample_id"]):
                for edit_idx in range(int(batch["edit_mask"][row_idx].sum().item())):
                    mode_id = int(batch["payload_graph_mode"][row_idx, edit_idx].item())
                    mode = next((key for key, value in __import__("cedg.chem_edit", fromlist=["GRAPH_MODES"]).GRAPH_MODES.items() if value == mode_id), "unknown")
                    valid = bool(batch["payload_graph_valid_mask"][row_idx, edit_idx].item())
                    rows.append(
                        {
                            "split": split,
                            "sample_id": sample_id,
                            "edit_index": edit_idx,
                            "graph_mode": mode,
                            "graph_valid": valid,
                        }
                    )
                    mode_counter[mode] += 1
                    valid_counter[str(valid)] += 1
            for item in dataset.records[:0]:
                del item
        for record in dataset.records:
            for event in record.get("edit_set", []) or []:
                attachment_counter[str(event.get("attachment_type"))] += 1
                payload_type_counter[str(event.get("final_chemical_payload_type"))] += 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out_dir / "graph_mode_by_event.csv", index=False)
    pd.DataFrame(mode_counter.items(), columns=["graph_mode", "count"]).to_csv(args.out_dir / "graph_mode_summary.csv", index=False)
    pd.DataFrame(valid_counter.items(), columns=["graph_valid", "count"]).to_csv(args.out_dir / "graph_valid_summary.csv", index=False)
    pd.DataFrame(attachment_counter.items(), columns=["attachment_type", "count"]).to_csv(args.out_dir / "attachment_summary.csv", index=False)
    pd.DataFrame(payload_type_counter.items(), columns=["payload_type", "count"]).to_csv(args.out_dir / "payload_type_summary.csv", index=False)
    print(f"wrote report to {args.out_dir}")


if __name__ == "__main__":
    main()
