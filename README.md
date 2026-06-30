# CEDG-Set NNAA

中文说明：这是一个面向非天然环肽透膜性优化的 CEDG-Set scorer/ranker 项目。模型以 parent peptide 和局部 edit set 为输入，联合学习 `delta_property` 回归、候选组排序、top-k recovery、方向判断和不确定性估计。

## Core Structure

- `src/cedg/`：核心模型、数据张量化、loss、ESM cache、edit graph 和候选生成模块。
- `scripts/`：数据构建、训练、评估、候选推荐和缓存脚本。
- `data/final/model_ready/peptide_component/`：原始 model-ready 训练集。
- `data/final/model_ready/peptide_component_plus_faris/`：加入 Faris/2024 local SAR 后的增强训练集。
- `data/external_benchmarks/literature_sources/2024_jacs_macrocyclic_10mer/`：Faris/2024 strict-anchor local SAR 训练数据。
- `CEDG_Set_项目汇报说明.md`：项目背景、模型设计、数据处理和实验设计说明。

## Main Training Command

```bash
conda run -n nnaa python scripts/train_cedg_score.py \
  --dataset data/final/model_ready/peptide_component_plus_faris/cedg_score_dataset.jsonl \
  --out-dir runs/cedg_set_esm_cached_plus_faris_rankheavy \
  --epochs 30 \
  --batch-size 512 \
  --emb-dim 64 \
  --hidden-dim 128 \
  --peptide-layers 2 \
  --edit-layers 2 \
  --num-heads 4 \
  --group-aware-batches \
  --use-esm-cache \
  --device cuda:0
```

## Notes

Large generated caches, raw downloads, smoke runs, and model checkpoints are intentionally excluded from Git. They can be regenerated from the included scripts and model-ready datasets.
