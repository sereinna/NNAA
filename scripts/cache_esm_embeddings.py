#!/usr/bin/env python3
"""中文说明：预计算冻结 ESM residue embedding 缓存，避免每个训练 epoch 重复跑 ESM 前向。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cedg.data import CEDGDataset, build_vocabs, collate_cedg, load_jsonl  # noqa: E402
from cedg.plm import ESMBatchConverter, ESMResidueEncoder, load_esm_model, plm_cache_path, save_plm_embedding  # noqa: E402
from cedg.utils import move_to_device  # noqa: E402


DATASET = ROOT / "data" / "final" / "model_ready" / "peptide_component" / "cedg_score_dataset.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--esm-model", default="esm2_t6_8M_UR50D")
    parser.add_argument("--plm-cache-dir", type=Path, default=None)
    parser.add_argument("--graph-cache-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-plm-length", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    cache_dir = args.plm_cache_dir
    if cache_dir is None:
        cache_dir = ROOT / "data" / "final" / "cache" / "plm_embeddings" / args.dataset.parent.name / args.esm_model
    graph_cache_dir = args.graph_cache_dir
    if graph_cache_dir is None:
        graph_cache_dir = ROOT / "data" / "final" / "cache" / "edit_graphs" / args.dataset.parent.name

    records = load_jsonl(args.dataset)
    vocabs = build_vocabs(records)
    esm_model, alphabet, repr_layer, esm_dim = load_esm_model(args.esm_model)
    encoder = ESMResidueEncoder(esm_model, repr_layer, esm_dim, freeze=True).to(args.device)
    converter = ESMBatchConverter(alphabet, max_length=args.max_plm_length)

    total = 0
    skipped = 0
    for split in ("train", "val", "test"):
        dataset = CEDGDataset(
            records,
            vocabs,
            split,
            graph_cache_dir=graph_cache_dir,
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
            sequences = [str(seq) for seq in batch["plm_sequence"]]
            sample_ids = list(batch["sample_id"])
            missing_rows = []
            for row, (sample_id, sequence) in enumerate(zip(sample_ids, sequences)):
                path = plm_cache_path(cache_dir, sample_id, sequence, args.esm_model)
                if path.exists():
                    skipped += 1
                else:
                    missing_rows.append(row)
            if not missing_rows:
                total += len(sample_ids)
                continue
            esm_batch = converter([sequences[row] for row in missing_rows])
            esm_batch = move_to_device(esm_batch, torch.device(args.device))
            with torch.no_grad():
                embeddings, masks = encoder(
                    esm_batch["plm_tokens"],
                    esm_batch["plm_lengths"],
                    target_len=max(len(sequences[row]) for row in missing_rows),
                )
            for local_row, original_row in enumerate(missing_rows):
                sample_id = sample_ids[original_row]
                sequence = sequences[original_row]
                length = len(sequence)
                path = plm_cache_path(cache_dir, sample_id, sequence, args.esm_model)
                save_plm_embedding(
                    path,
                    embeddings[local_row, :length],
                    masks[local_row, :length],
                    sequence,
                    args.esm_model,
                )
            total += len(sample_ids)
        print(f"{split}: processed {len(dataset)} samples")
    print(f"cache_dir={cache_dir}")
    print(f"total={total} skipped_existing={skipped}")


if __name__ == "__main__":
    main()
