# Final Dataset Layout

This directory contains the current best version of the curated NNAA edit-level training export.

## Annotations

- `annotations/monomer_anchor_table.csv`: final curated monomer anchor/edit annotation table.
- `annotations/monomer_anchor_manual_review_queue.csv`: monomers still worth manual checking.
- `annotations/monomer_anchor_summary.csv`: annotation status summary.
- `annotations/peptide_sample_table.csv`: peptide-level CycPeptMPDB sample table with final shadow sequences.

## Training Tables

Each source under `training/<source_slug>/` has only two model-facing tables:

- `training_single_site.csv`: one row is one directional parent-to-modified peptide pair with exactly one accepted edit.
- `training_multi_site.csv`: one row is one directional parent-to-modified peptide pair with multiple accepted edits stored in `edit_set_json`.
- `training_summary.csv`: row counts, censored counts, and skipped/filtered group notes.

Current exported sources:

- `2015_bockus_2`
- `2015_wang`
- `2016_furukawa`
- `2013_chugai`
- `2018_naylor`
- `2020_le_roux`
- `2020_townsend`
- `2021_golosov`
- `2021_kelly`
- `2022_taechalertpaisarn`

Current total rows:

- single-site: 2548 rows, including 506 censored rows.
- multi-site: 6632 rows, including 1801 censored rows.

`PAMPA = -10.00` values are retained and marked with `censored_property_flag=True`.

Reverse-direction duplicate pairs are removed. If both directions exist for the same unordered peptide pair, the export keeps the direction with positive `delta_property` where possible, i.e. the row describing the modification direction associated with improved PAMPA. Unique one-direction candidates are retained even when their `delta_property` is negative.

The edit-count limit is adaptive by peptide length: `min(max_edits, max(2, ceil(length * max_edit_fraction)))`. The current export uses `max_edits=4` and `max_edit_fraction=0.25`, so short peptides can still keep up to 2 edits, length 9-12 peptides can keep up to 3 edits, and longer peptides are capped at 4 edits.

For large same-shadow groups, including `2016_Furukawa`, `2020_Townsend`, and `2021_Kelly`, the export keeps high-signal candidates using `large_group_top_abs_delta_filtered` with a maximum of 200 pairs per large group.

## Split Tables

Leakage-aware split exports are under `splits/`.

Recommended default:

```text
splits/peptide_component/training_single_site.csv
splits/peptide_component/training_multi_site.csv
```

These files add a `split` column and guarantee no source-specific peptide node appears in more than one split.

Current `peptide_component` split:

```text
train: 7543 rows
val:    795 rows
test:   842 rows
```

`splits/source_shadow/` is stricter by source+final-shadow group, but should be treated as a diagnostic split because some sources have too few large groups to distribute evenly.

## Model-Ready Export

The first CEDG-Set modeling step should use the CEDG-Score dataset:

```text
model_ready/peptide_component/cedg_score_dataset.jsonl
```

This JSONL unifies single-site and multi-site rows into:

```text
parent peptide P + edit_set E -> delta_property
```

It is aligned with the MVP in `Untitled.md`: start with edit scoring / delta PAMPA prediction before open-vocabulary R-group generation.

The initial sklearn baseline has been run in the `nnaa` conda environment. Results are under:

```text
model_ready/peptide_component/baseline/
```

## Reproducible Export

Use:

```bash
python scripts/export_final_training_by_source.py \
  --source-id 2015_Wang \
  --source-id '2020_Le Roux' \
  --source-id 2013_CHUGAI \
  --source-id 2022_Taechalertpaisarn \
  --source-id 2021_Golosov \
  --source-id 2015_Bockus_2 \
  --source-id 2018_Naylor \
  --assay PAMPA \
  --max-edits 4 \
  --max-edit-fraction 0.25 \
  --max-group-size 50

python scripts/export_final_training_by_source.py \
  --source-id 2016_Furukawa \
  --source-id 2020_Townsend \
  --source-id 2021_Kelly \
  --assay PAMPA \
  --max-edits 4 \
  --max-edit-fraction 0.25 \
  --max-group-size 50 \
  --large-group-strategy top-delta \
  --max-pairs-per-large-group 200

python scripts/build_final_splits.py

python scripts/export_model_ready_cedg.py --split-scheme peptide_component

conda run -n nnaa python scripts/run_cedg_sklearn_baseline.py
```
