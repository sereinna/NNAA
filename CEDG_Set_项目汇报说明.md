# CEDG-Set 非天然环肽透膜性优化模型项目汇报说明

## 1. 项目背景

非天然环肽具有较好的靶点覆盖能力和结构多样性，但透膜性往往难以直接预测和优化。实验上常见的优化方式是围绕一个 parent peptide 做局部结构改造，例如 D/L 构型变化、N-甲基化、疏水侧链替换、peptoid-like 单体替换或 Pro/N-methyl Ala 环化关系调整，然后比较 PAMPA 或类似透膜性指标。

本项目的目标不是只训练一个普通性质预测模型，而是构建一个面向优化建议的排序模型：给定一个肽及其可能编辑集合，模型需要判断哪些候选改造更可能提升透膜性，并输出 candidate group、top-k edit set 和候选排序结果。

## 2. 核心想法

当前模型采用 CEDG-Set 思路：把 parent peptide 表示、局部 edit event 表示和候选组排序目标结合起来。

每条训练样本表示为一个 parent peptide 到 modified peptide 的编辑关系：

- `parent_monomer_list`：改造前单体序列。
- `modified_monomer_list`：改造后单体序列。
- `edit_set`：一个或多个局部编辑事件。
- `property_before` / `property_after`：改造前后 PAMPA 数值。
- `delta_property`：改造带来的性质变化。
- `candidate_group_id`：同一个 parent 下的候选集合，用于 ranking/top-k 训练。

因此模型既学习绝对的 `delta_property`，也学习在同一个 parent 候选组中如何排序候选。

## 3. 模型框架

当前代码位于 `src/cedg/`，已经按功能模块拆分：

- `data.py`：数据读取、词表、张量化、graph/ESM cache 接入。
- `model.py`：CEDG scorer/ranker 主模型。
- `losses.py`：delta regression、property regression、pairwise/listwise/top-k ranking、direction loss。
- `chem_edit.py`：attachment-aware old/new/delta atom graph 构建。
- `plm.py`：ESM/PLM 表征加载与缓存。
- `topology.py`：环肽拓扑和 scaffold feature。
- `candidates.py`：新肽候选生成。
- `edit_rules.py`：pair-level edit event/anchor/operation 规则。

模型输入包括：

- parent peptide shadow sequence；
- residue/topology features；
- ESM embedding 或 ESM cache；
- edit event categorical fields；
- edit payload tokens；
- atom-level edit graph；
- sample-level metadata。

模型输出包括：

- `delta_property` 预测；
- `property_after` 预测；
- `ranking_score`；
- `direction_logit`；
- `uncertainty/log_variance`。

## 4. 当前训练目标

当前损失函数不只做 delta regression，而是更偏向实际推荐场景：

- delta regression：预测改造后性质变化。
- property regression：辅助预测改造后绝对性质。
- pairwise ranking：同一 parent group 内两两候选排序。
- listwise ranking：同一候选组整体排序分布。
- top-k ranking：更强调恢复实验中最优候选。
- direction loss：判断改造方向是否提升。
- uncertainty loss：估计预测不确定性。

默认训练权重已设置为 ranking-heavy：

```text
delta = 1.0
property = 0.2
ranking_mse = 0.05
pairwise = 0.6
listwise = 0.4
topk = 0.8
direction = 0.2
```

## 5. 原始数据与已有模型

原始模型训练集：

```text
data/final/model_ready/peptide_component/cedg_score_dataset.jsonl
```

当前已有正式模型：

```text
runs/cedg_set_esm_cached_full/best_model.pt
```

该模型已经使用 GPU 训练，并支持 ESM cache 与 edit graph cache。当前测试集主要指标：

| 指标 | 数值 |
|---|---:|
| test_delta_mae | 0.424 |
| test_delta_spearman | 0.692 |
| test_direction_auc | 0.819 |
| test_ranking_spearman | 0.655 |
| test_ranking_ndcg_at_5 | 0.837 |

## 6. 新增 Faris/2024 数据集

新增文献数据来自 Faris 2024 JACS macrocyclic 10-mer 数据。原始 CSV 位于：

```text
data/external_benchmarks/literature_sources/2024_jacs_macrocyclic_10mer/raw/CycPeptMPDB_Peptide_Source_2024_Faris.csv
```

该数据集和原始训练数据存在明显分布差异，因此没有简单地把一个 parent 连到所有 peptide，而是构建局部 SAR neighbor pair：

- 只连接 Hamming distance = 1 的 peptide pair；
- 每个 parent 的真实局部邻居构成一个 candidate group；
- 过滤过小 delta，当前阈值为 `min_abs_delta = 0.05`；
- 生成双向 directed pair，使模型学习改造方向。

最终新增：

```text
1434 条 Faris local SAR pair
229 个有效 candidate group
234 个 source peptide
```

当前推荐使用的 Faris model-ready 文件：

```text
data/external_benchmarks/literature_sources/2024_jacs_macrocyclic_10mer/model_ready_local_neighbor_d1.jsonl
data/external_benchmarks/literature_sources/2024_jacs_macrocyclic_10mer/model_ready_local_neighbor_d1.csv
```

CSV 版本已经导出，便于人工检查和汇报展示。

## 7. Faris edit event/anchor/operation 优化

为了让新数据对模型公平，Faris 的 edit event 不再使用粗糙的 `monomer_replacement`，而是加入 pair-level 规则。

主要优化包括：

- `Abu <-> L/Nva/Cha`：标注为疏水侧链替换，不再错误追加 N-甲基化或 D/L 变化。
- `L <-> dL`、`meA <-> Me_dA`、`P <-> dP`：标注为明确的 stereochemistry edit。
- `Et_Gly/Pr_Gly/iBu_Gly/Sar` 之间：标注为 N-substituted glycine substituent edit。
- `Me_dA <-> P/dP`：标注为 N-methyl Ala 与 Pro ring closure/opening 相关变化。
- `Et_Gly/Pr_Gly <-> Me_Abu/Me_Nva`：标注为 peptoid 与 alpha amino acid 连接方式变化，而不是普通 N-methylation。

规则代码：

```text
src/cedg/edit_rules.py
```

检查结果：

```text
Faris strict-anchor training set: 1434 rows
bad annotation count: 0
无 nan / none / natural_reference_residue
```

当前 Faris strict-anchor 训练版本遵循 final anchor table：`Nva/dNva` 作为 Val-like anchor，`anchor_for_alignment = V`；因此 `Nva -> L` 的 parent shadow 使用 `V`，edit anchor 使用 parent/original anchor `V`，不再使用早期 external pairwise override。

## 8. 增强训练集 plus_faris

已经把 Faris 1434 条样本合并进原始训练数据，形成增强训练集：

```text
data/final/model_ready/peptide_component_plus_faris/cedg_score_dataset.jsonl
data/final/model_ready/peptide_component_plus_faris/cedg_score_dataset.csv
```

数据量：

| 数据来源 | 条数 |
|---|---:|
| 原始 peptide_component | 9180 |
| Faris/2024 local SAR | 1434 |
| 合计 | 10614 |

Faris 样本按 `candidate_group_id` 分组切分，避免同一个 parent 的候选同时出现在训练和验证/测试中：

| split | Faris 条数 |
|---|---:|
| train | 1165 |
| val | 117 |
| test | 152 |

合并后总 split：

| split | 总条数 |
|---|---:|
| train | 8708 |
| val | 911 |
| test | 995 |

Faris 样本当前使用 `sample_weight = 0.7`，让新数据参与训练，同时避免单一新文献过度支配原始多文献训练分布。

## 9. 外部验证结果

使用当前旧正式模型 `runs/cedg_set_esm_cached_full/best_model.pt` 在 Faris refined v3 上评估，得到：

```text
reports/external_benchmarks/2024_faris_local_neighbor_d1_refined_v3/
```

主要指标：

| 指标 | 数值 |
|---|---:|
| n_rows | 1434 |
| n_groups | 229 |
| overall_spearman | 0.508 |
| group_pairwise_accuracy | 0.525 |
| best_hit_at_1 | 0.183 |
| best_hit_at_3 | 0.572 |
| best_hit_at_5 | 0.773 |
| positive_hit_at_1 | 0.612 |
| positive_hit_at_3 | 0.900 |
| positive_hit_at_5 | 0.967 |
| ndcg_at_5 | 0.741 |
| overlap with training | 0 |

解释：

- 旧模型对 Faris 局部 SAR 有一定排序信号，overall Spearman 约 0.51。
- top-5 恢复较好，best_hit@5 约 0.77，positive_hit@5 约 0.97。
- top-1 仍不稳定，说明模型还需要用 Faris-like 数据或更强 ranking 目标进一步训练。

## 10. 候选生成与优化建议

当前系统已经不仅能打分，还能为新 peptide 构建候选：

- 先基于 edit library 枚举 single-edit；
- 对 single-edit 打分；
- 选 single-edit top beam；
- 再组合二点或多点 edit；
- 输出 candidate group、top-k edit set、CSV/JSONL。

相关脚本：

```text
scripts/recommend_for_peptide.py
scripts/score_candidate_groups.py
```

这意味着模型原则上可以对一个新 peptide 生成优化建议，而不是只能在训练集中已有组合之间选择。不过候选空间仍由 edit library 和 monomer/edit rules 约束，不是无限穷举所有化学可能性。

## 11. 建议实验设计

为了同时说明泛化能力和加入新数据后的改进，建议做三组模型：

### A. 原始训练集模型

训练数据：

```text
data/final/model_ready/peptide_component/cedg_score_dataset.jsonl
```

用途：

- 作为主外部泛化基线；
- 评估不使用 Faris 训练时，对 Faris local SAR 的预测能力。

已有模型：

```text
runs/cedg_set_esm_cached_full/best_model.pt
```

### B. Faris-only 诊断模型

训练数据：

```text
data/external_benchmarks/literature_sources/2024_jacs_macrocyclic_10mer/model_ready_local_neighbor_d1.jsonl
```

用途：

- 判断同一模型架构是否能学习 Faris 数据内部规律；
- 作为 domain-specific upper bound/diagnostic；
- 不建议作为最终主模型，因为数据来源单一。

### C. 原始 + Faris 增强模型

训练数据：

```text
data/final/model_ready/peptide_component_plus_faris/cedg_score_dataset.jsonl
```

用途：

- 作为后续推荐系统主模型；
- 兼顾原始多文献数据和 Faris-like 新局部 SAR 数据；
- 可比较是否提升 Faris test split、原始 test split、以及候选 top-k recovery。

正式训练命令建议：

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

## 12. 当前目录整理

当前推荐关注的主线文件：

```text
src/cedg/                                      # 模型代码
scripts/                                      # 数据处理、训练、评估、推荐脚本
data/final/model_ready/peptide_component/      # 原始 model-ready 数据
data/final/model_ready/peptide_component_plus_faris/  # 加入 Faris 后的增强数据
data/external_benchmarks/literature_sources/2024_jacs_macrocyclic_10mer/  # Faris 数据
runs/cedg_set_esm_cached_full/                 # 当前旧正式模型
reports/external_benchmarks/2024_faris_local_neighbor_d1_refined_v3/  # 最新 Faris 外部评估
```

旧版本和中间结果已非破坏性归档：

```text
data/external_benchmarks/literature_sources/2024_jacs_macrocyclic_10mer/archive_intermediate/
reports/external_benchmarks/archive_old/
runs/archive_smoke/
runs/archive_legacy/
```

## 13. 当前结论

目前模型已经具备以下能力：

- 使用 ESM cache 和 graph cache 训练；
- 读取 parent peptide + edit set；
- 建模局部 edit event、anchor、operation、payload 和 atom-level delta graph；
- 做 delta/property/direction/ranking/top-k 多目标训练；
- 对新 peptide 生成候选并输出 top-k edit set；
- 将 Faris/2024 新文献数据处理成和原始训练数据同构的 1434 条局部 SAR pair；
- 生成 plus_faris 增强训练集；
- 通过 GPU smoke test 确认增强数据可以训练。

还需要继续完成：

- 正式训练 `plus_faris` 增强模型；
- 比较原始模型、Faris-only 模型、plus_faris 模型；
- 评估增强模型在原始 test split 和 Faris test split 上是否同时提升；
- 进一步扩展 edit library，使新 peptide 优化建议覆盖更多可靠的非天然单体变化；
- 若用于论文级结果，需要固定随机种子并重复 3 次以上报告均值和标准差。
