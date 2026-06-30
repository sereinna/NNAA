# 外部文献验证集说明

这个目录用于放置近年新文献整理出来的独立验证集，目标是评估 CEDG 模型能否在真实文献优化系列中把有效突变排进 top-k。

## 推荐输入格式

优先使用 `.jsonl`，每行一个候选 parent -> modified edit record，字段尽量和 `data/final/model_ready/peptide_component/cedg_score_dataset.jsonl` 一致。

最小必需字段：

```text
sample_id
source_id
source_slug
assay_type
parent_name
modified_name
parent_shadow_sequence
parent_monomer_list
modified_monomer_list
property_before
property_after
delta_property
edit_count
edit_set
```

强烈推荐字段：

```text
candidate_group_id
parent_peptide_id
modified_peptide_id
parent_smiles
modified_smiles
unordered_pair_id
paper_title
doi
compound_id_in_paper
```

`candidate_group_id` 应该表示同一个 parent peptide 下的一组候选优化，例如：

```text
2024_bergeron:PAMPA:parent_MortiamideD:shadow_x_x_x
```

## 评估命令

```bash
conda run -n nnaa python scripts/evaluate_external_benchmark.py \
  --checkpoint runs/cedg_set_esm_cached_full/best_model.pt \
  --external-dataset data/external_benchmarks/2024_bergeron/model_ready.jsonl \
  --out-dir reports/external_benchmarks/2024_bergeron \
  --top-ks 1,3,5,10 \
  --positive-threshold 0 \
  --device cuda:0
```

输出：

```text
scored_candidates.csv      每个候选的 pred_delta/ranking_score/uncertainty/recommendation_score
group_topk_recovery.csv    每个 parent group 的 best-hit@k 和 positive-hit@k
metrics.json               整体 Spearman、pairwise accuracy、NDCG@k、hit@k 和训练集重叠报告
```

## 关键指标

```text
best_hit_at_k
```

文献中 observed_delta 最高的候选是否进入模型 top-k。

```text
positive_hit_at_k
```

文献中任意 observed_delta > positive_threshold 的候选是否进入模型 top-k。

```text
topk_mean_delta_minus_group_mean
```

模型 top-k 的真实平均提升是否高于该 parent group 的平均水平。

```text
overlap_report
```

外部集和训练集在 sample_id、SMILES、pair key 等层面的重叠检查。若有重叠，应在论文或报告中单独说明，必要时剔除。
