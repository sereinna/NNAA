# CEDG Model-Ready Dataset v2.0

中文说明：v2.0 是一个独立的数据版本，用于降低 `2020_Townsend` 在训练集中的支配比例，同时扩展少数来源和保留正负优化方向。

## Outputs

```text
data/final/model_ready/ori_v2_0/cedg_score_dataset.jsonl
data/final/model_ready/ori_v2_0/cedg_score_dataset.csv
data/final/model_ready/ori_v2_0_plus_faris2024/cedg_score_dataset.jsonl
data/final/model_ready/ori_v2_0_plus_faris2024/cedg_score_dataset.csv
```

## Rules

- Keep both pair directions instead of removing `A -> B` / `B -> A` reverse-direction rows.
- Increase large same-shadow group cap from 200 to 500 directional rows per group.
- Apply a source-level cap to `2020_Townsend`: maximum 3000 selected rows.
- Keep the same chemistry filters as v1: final monomer annotations, adaptive edit limit `max_edits=4`, `max_edit_fraction=0.25`.
- Split with source-stratified peptide-component balancing, so peptide components are not split across train/val/test within a source.
- Merge Faris using the existing `2024_Faris` training records from `peptide_component_plus_faris`, preserving its group split and weight.

## Counts

| dataset | rows |
|---|---:|
| `ori_v2_0` | 7252 |
| `ori_v2_0_plus_faris2024` | 8686 |

`ori_v2_0_plus_faris2024` source composition:

| source_id | rows | fraction |
|---|---:|---:|
| `2020_Townsend` | 3000 | 34.54% |
| `2021_Kelly` | 2042 | 23.51% |
| `2016_Furukawa` | 1500 | 17.27% |
| `2024_Faris` | 1434 | 16.51% |
| other ori sources | 710 | 8.17% |

## Rebuild

```bash
conda run -n nnaa python scripts/build_cedg_dataset_v2.py
```
