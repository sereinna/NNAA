#!/usr/bin/env python3
"""中文说明：用指定 checkpoint 在指定 model-ready 数据集 split 上做同口径评估。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cedg.data import CEDGDataset, build_vocabs, collate_cedg, load_jsonl  # noqa: E402
from cedg.evaluation import evaluate  # noqa: E402
from cedg.model import CEDGScoreModel  # noqa: E402
from cedg.plm import esm_model_spec  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vocab-dataset", type=Path, required=True)
    parser.add_argument("--eval-dataset", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--esm-model", default=None)
    parser.add_argument("--graph-cache-dir", type=Path, default=None)
    parser.add_argument("--plm-cache-dir", type=Path, default=None)
    args = parser.parse_args()

    vocab_records = load_jsonl(args.vocab_dataset)
    eval_records = load_jsonl(args.eval_dataset)
    vocabs = build_vocabs(vocab_records)
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_args = checkpoint.get("args", {})
    esm_model_name = args.esm_model or str(model_args.get("esm_model", "esm2_t6_8M_UR50D"))
    esm_repr_layer, esm_dim = esm_model_spec(esm_model_name)
    model = CEDGScoreModel(
        vocabs=vocabs,
        emb_dim=int(model_args.get("emb_dim", 64)),
        hidden_dim=int(model_args.get("hidden_dim", 128)),
        dropout=float(model_args.get("dropout", 0.1)),
        peptide_layers=int(model_args.get("peptide_layers", 2)),
        edit_layers=int(model_args.get("edit_layers", 2)),
        num_heads=int(model_args.get("num_heads", 4)),
        max_len=int(model_args.get("max_len", 128)),
        esm_model=None,
        esm_repr_layer=esm_repr_layer,
        esm_dim=esm_dim,
        freeze_esm=True,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])

    graph_cache_dir = args.graph_cache_dir
    if graph_cache_dir is None:
        graph_cache_dir = ROOT / "data" / "final" / "cache" / "edit_graphs" / args.eval_dataset.parent.name
    plm_cache_dir = args.plm_cache_dir
    if plm_cache_dir is None:
        plm_cache_dir = (
            ROOT
            / "data"
            / "final"
            / "cache"
            / "plm_embeddings"
            / args.eval_dataset.parent.name
            / esm_model_name
        )

    dataset = CEDGDataset(
        eval_records,
        vocabs,
        args.split,
        graph_cache_dir=graph_cache_dir,
        use_graph_cache=True,
        plm_cache_dir=plm_cache_dir,
        esm_model_name=esm_model_name,
        use_plm_cache=True,
        preload_cache=True,
    )
    loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "collate_fn": collate_cedg,
        "pin_memory": device.type == "cuda",
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    loader = DataLoader(dataset, **loader_kwargs)
    metrics = evaluate(model, loader, device)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
