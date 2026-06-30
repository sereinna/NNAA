#!/usr/bin/env python3
"""中文说明：训练 CEDG 第一阶段 scorer/ranker，联合性质、Δproperty、不确定性和候选组排序目标。

Train the CEDG-Set scorer.

This script trains the architecture from `src/cedg/model.py` on the model-ready
JSONL. The model consumes parent peptide context plus a site-conditioned local
edit set with curated payload metadata and RDKit-derived old/new payload
fragment graphs. It predicts delta PAMPA, property_after, ranking score,
direction, and uncertainty.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cedg.data import CEDGCollatorWithESM, CEDGDataset, build_vocabs, collate_cedg, load_jsonl  # noqa: E402
from cedg.evaluation import evaluate  # noqa: E402
from cedg.losses import compute_cedg_loss  # noqa: E402
from cedg.model import CEDGScoreModel  # noqa: E402
from cedg.plm import ESMBatchConverter, esm_model_spec, load_esm_model  # noqa: E402
from cedg.samplers import CandidateGroupBatchSampler  # noqa: E402
from cedg.utils import move_to_device, set_seed  # noqa: E402


DATASET = ROOT / "data" / "final" / "model_ready" / "peptide_component" / "cedg_score_dataset.jsonl"
OUT_DIR = ROOT / "runs" / "cedg_score"


def loss_weights_from_args(args: argparse.Namespace) -> dict[str, float]:
    return {
        "delta": args.loss_delta,
        "property": args.loss_property,
        "ranking_mse": args.loss_ranking_mse,
        "pairwise": args.loss_pairwise,
        "listwise": args.loss_listwise,
        "topk": args.loss_topk,
        "direction": args.loss_direction,
    }


def make_loader(
    dataset: CEDGDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    group_aware: bool = False,
    seed: int = 7,
    collate_fn: object = collate_cedg,
    prefetch_factor: int = 2,
) -> DataLoader:
    loader_kwargs = {
        "num_workers": num_workers,
        "collate_fn": collate_fn,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = prefetch_factor
    if group_aware:
        sampler = CandidateGroupBatchSampler(
            [str(item["candidate_group_id"]) for item in dataset],
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
        )
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            **loader_kwargs,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        **loader_kwargs,
    )


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    records = load_jsonl(args.dataset)
    vocabs = build_vocabs(records)
    graph_cache_dir = args.graph_cache_dir
    if graph_cache_dir is None and not args.no_graph_cache:
        graph_cache_dir = ROOT / "data" / "final" / "cache" / "edit_graphs" / args.dataset.parent.name
    plm_cache_dir = args.plm_cache_dir
    if plm_cache_dir is None and args.use_esm_cache:
        plm_cache_dir = ROOT / "data" / "final" / "cache" / "plm_embeddings" / args.dataset.parent.name / args.esm_model

    train_ds = CEDGDataset(
        records,
        vocabs,
        "train",
        allow_absolute_only=args.allow_absolute_only,
        graph_cache_dir=graph_cache_dir,
        use_graph_cache=not args.no_graph_cache,
        plm_cache_dir=plm_cache_dir,
        esm_model_name=args.esm_model,
        use_plm_cache=args.use_esm_cache,
        preload_cache=args.preload_cache,
    )
    val_ds = CEDGDataset(
        records,
        vocabs,
        "val",
        allow_absolute_only=args.allow_absolute_only,
        graph_cache_dir=graph_cache_dir,
        use_graph_cache=not args.no_graph_cache,
        plm_cache_dir=plm_cache_dir,
        esm_model_name=args.esm_model,
        use_plm_cache=args.use_esm_cache,
        preload_cache=args.preload_cache,
    )
    test_ds = CEDGDataset(
        records,
        vocabs,
        "test",
        allow_absolute_only=args.allow_absolute_only,
        graph_cache_dir=graph_cache_dir,
        use_graph_cache=not args.no_graph_cache,
        plm_cache_dir=plm_cache_dir,
        esm_model_name=args.esm_model,
        use_plm_cache=args.use_esm_cache,
        preload_cache=args.preload_cache,
    )
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    esm_model = None
    esm_repr_layer = None
    esm_dim = 0
    collate_fn = collate_cedg
    if args.use_esm:
        esm_model, alphabet, esm_repr_layer, esm_dim = load_esm_model(args.esm_model)
        collate_fn = CEDGCollatorWithESM(ESMBatchConverter(alphabet, max_length=args.max_plm_length))
    elif args.use_esm_cache:
        esm_repr_layer, esm_dim = esm_model_spec(args.esm_model)

    train_loader = make_loader(
        train_ds,
        args.batch_size,
        True,
        args.num_workers,
        args.group_aware_batches,
        args.seed,
        collate_fn,
        args.prefetch_factor,
    )
    val_loader = make_loader(
        val_ds,
        args.batch_size,
        False,
        args.num_workers,
        collate_fn=collate_fn,
        prefetch_factor=args.prefetch_factor,
    )
    test_loader = make_loader(
        test_ds,
        args.batch_size,
        False,
        args.num_workers,
        collate_fn=collate_fn,
        prefetch_factor=args.prefetch_factor,
    )

    model = CEDGScoreModel(
        vocabs=vocabs,
        emb_dim=args.emb_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        peptide_layers=args.peptide_layers,
        edit_layers=args.edit_layers,
        num_heads=args.num_heads,
        max_len=args.max_len,
        esm_model=esm_model,
        esm_repr_layer=esm_repr_layer,
        esm_dim=esm_dim,
        freeze_esm=args.freeze_esm,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_weights = loss_weights_from_args(args)
    print(
        json.dumps(
            {
                "device": str(device),
                "cuda_available": torch.cuda.is_available(),
                "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "",
                "train_records": len(train_ds),
                "val_records": len(val_ds),
                "test_records": len(test_ds),
                "batch_size": args.batch_size,
                "num_workers": args.num_workers,
                "prefetch_factor": args.prefetch_factor if args.num_workers > 0 else None,
                "group_aware_batches": args.group_aware_batches,
                "graph_cache_dir": str(graph_cache_dir) if graph_cache_dir is not None else "",
                "use_esm_cache": args.use_esm_cache,
                "plm_cache_dir": str(plm_cache_dir) if plm_cache_dir is not None else "",
                "preload_cache": args.preload_cache,
                "loss_weights": loss_weights,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, object]] = []
    best_val = float("inf")
    best_path = out_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses: list[float] = []
        for batch in train_loader:
            batch = move_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch)
            loss = compute_cedg_loss(outputs, batch, weights=loss_weights)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        val_metrics = evaluate(model, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

        if val_metrics["delta_mae"] < best_val:
            best_val = val_metrics["delta_mae"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "args": vars(args),
                    "vocab_sizes": {
                        "shadow": len(vocabs.shadow),
                        "payload": len(vocabs.payload),
                        "monomer_smiles": len(vocabs.monomer_smiles),
                        "edit": {key: len(value) for key, value in vocabs.edit.items()},
                        "sample": {key: len(value) for key, value in vocabs.sample.items()},
                        "use_esm": args.use_esm,
                        "esm_model": args.esm_model if args.use_esm else None,
                    },
                },
                best_path,
            )

    model.eval()
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    val_metrics = evaluate(model, val_loader, device)
    test_metrics = evaluate(model, test_loader, device)
    final = {
        "best_val_delta_mae": best_val,
        **{f"val_{key}": value for key, value in val_metrics.items()},
        **{f"test_{key}": value for key, value in test_metrics.items()},
    }
    (out_dir / "metrics.json").write_text(json.dumps(final, indent=2, sort_keys=True), encoding="utf-8")
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
    print("FINAL", json.dumps(final, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--emb-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--peptide-layers", type=int, default=2)
    parser.add_argument("--edit-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--group-aware-batches", action="store_true")
    parser.add_argument("--allow-absolute-only", action="store_true")
    parser.add_argument("--graph-cache-dir", type=Path, default=None)
    parser.add_argument("--no-graph-cache", action="store_true")
    parser.add_argument("--use-esm", action="store_true")
    parser.add_argument("--use-esm-cache", action="store_true")
    parser.add_argument("--preload-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--loss-delta", type=float, default=1.0)
    parser.add_argument("--loss-property", type=float, default=0.2)
    parser.add_argument("--loss-ranking-mse", type=float, default=0.05)
    parser.add_argument("--loss-pairwise", type=float, default=0.6)
    parser.add_argument("--loss-listwise", type=float, default=0.4)
    parser.add_argument("--loss-topk", type=float, default=0.8)
    parser.add_argument("--loss-direction", type=float, default=0.2)
    parser.add_argument("--esm-model", default="esm2_t6_8M_UR50D")
    parser.add_argument("--freeze-esm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-plm-length", type=int, default=256)
    parser.add_argument("--plm-cache-dir", type=Path, default=None)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
