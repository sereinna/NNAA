# CEDG-Set Modeling Plan

Date: 2026-06-29

This plan aligns the current final dataset with `Untitled.md`.

## Current Dataset Status

Recommended input tables:

```text
data/final/splits/peptide_component/training_single_site.csv
data/final/splits/peptide_component/training_multi_site.csv
```

Current rows:

```text
train: 7543
val:    795
test:   842
total: 9180
```

Leakage checks:

```text
peptide_node_split_leakage: 0
unordered_pair_duplicate_rows: 0
```

## First Modeling Target

The first model should be **CEDG-Score**, a minimal scoring version of CEDG-Set:

```text
Input:  parent peptide P + edit set E
Output: delta_property = PAMPA(P') - PAMPA(P)
```

Secondary outputs:

```text
property_after
direction label: delta_property > 0
```

Do not start with open-vocabulary R-group generation. Generation depends on a stable scorer.

## Model Input Mapping

Parent peptide context:

```text
canonical_shadow_sequence_final
parent_monomer_list
parent_smiles
assay_type
property_before
```

Edit set:

For single-site rows, the edit event comes from flat columns:

```text
site_index
anchor_for_alignment
original_monomer
modified_monomer
edit_event_class
edit_event_subclass
final_edit_scope
final_chemical_payload_type
final_chemical_payload
quality_flag_final
```

For multi-site rows, the edit events come from:

```text
edit_set_json
```

Pair-level labels and weights:

```text
property_before
property_after
delta_property
censored_property_flag
confidence_weight
sample_model_use_tier
contains_residue_graph_replacement
```

## MVP Feature Tiers

### Tier 0: Dataset Sanity Baselines

Use scikit-learn features only:

```text
parent shadow token n-grams
edit_count
site_index set
anchor tokens
edit_scope tokens
payload_type tokens
chemical_payload string tokens
source_id
censored_property_flag
```

Baselines:

```text
Ridge / ElasticNet for delta_property
HistGradientBoostingRegressor for delta_property
LogisticRegression / HistGradientBoostingClassifier for direction accuracy
```

Purpose:

```text
Check that the split, labels, and edit representation carry signal.
Estimate how much source/Townsend bias remains.
```

### Tier 1: CEDG-Score Metadata Model

Neural model without atom graph payload:

```text
peptide context encoder: token embedding over canonical shadow residues
edit metadata encoder: embeddings for site, anchor, edit_scope, payload_type, payload token
edit set encoder: mean/sum pooling + MLP
heads: delta regression, property_after regression, direction classification
```

This is the first publishable architecture ablation because it tests the CEDG-Set task framing.

### Tier 2: Site-Conditioned Payload Encoder

Replace payload string embedding with atom-level or fragment-level encoder:

```text
R-group / delta graph encoder
site-conditioned fusion
edit set encoder
prediction heads
```

This is the core CEDG-Set model in the paper.

## Main Evaluation

Use `peptide_component` split as the default development split.

Metrics:

```text
delta MAE
delta RMSE
delta Spearman
direction accuracy
direction AUROC
calibration / selective risk later
```

Report metrics both overall and stratified by:

```text
single-site vs multi-site
censored vs non-censored
source_id
edit_count
final_edit_scope
sample_model_use_tier
```

## Required Baselines

Start with:

```text
mean train delta
source mean delta
parent property_before only
edit metadata bag-of-features
parent shadow + edit metadata
```

Later baselines:

```text
whole-SMILES / SELFIES model
tokenized ncAA transformer
chemical fingerprint / graph only
nearest-neighbor SAR
RDKit descriptors + RF/XGBoost
```

## Important Current Risks

`2020_Townsend` dominates the dataset:

```text
2020_Townsend rows: 7170 / 9180
```

Therefore, source-stratified reporting is mandatory. A balanced training variant can be built later, but the current full split should be retained.

Censored PAMPA rows are retained:

```text
censored_property_flag=True
```

The first baseline can train with `confidence_weight`; later models should consider censored-aware loss or separate censored ablation.

`atom_replacement_auxiliary` rows are rare but chemically different. Report both:

```text
all rows
core_local_edit only
```

## Next Concrete Step

Build a model-ready JSONL/Parquet export from the split CSVs:

```text
data/final/model_ready/cedg_score_peptide_component.jsonl
```

Each row should contain:

```json
{
  "sample_id": "...",
  "split": "train",
  "source_id": "...",
  "parent_shadow_sequence": "...",
  "parent_monomer_list": ["..."],
  "modified_monomer_list": ["..."],
  "edit_set": [{"site_index": 1, "...": "..."}],
  "property_before": -6.2,
  "property_after": -5.4,
  "delta_property": 0.8,
  "sample_weight": 1.0
}
```

## Baseline Validation

Environment:

```text
conda env: nnaa
environment file: environment_nnaa.yml
```

Command:

```bash
conda run -n nnaa python scripts/run_cedg_sklearn_baseline.py
```

Outputs:

```text
data/final/model_ready/peptide_component/baseline/sklearn_baseline_results.csv
data/final/model_ready/peptide_component/baseline/sklearn_baseline_report.md
```

Key result:

```text
mean_delta test MAE: 0.903
ridge_edit test MAE: 0.490
ridge_edit test Spearman: 0.714
ridge_edit test direction AUC: 0.831
logreg_shadow_edit_source test direction AUC: 0.870
```

Interpretation:

```text
The edit-set metadata carries clear predictive signal for delta PAMPA.
Source mean alone performs poorly, so the result is not explained only by source identity.
Naive high-dimensional shadow sequence features overfit or degrade ridge regression, so the next model should use a controlled context encoder rather than raw sparse shadow n-grams.
```

## CEDG-Score PyTorch MVP

Implementation:

```text
src/cedg/data.py
src/cedg/model.py
scripts/train_cedg_score.py
```

Architecture:

```text
PeptideContextEncoder: shadow residue embedding + masked pooling
EditEventEncoder: site-aware categorical edit metadata encoder
EditSetEncoder: mean/sum pooling + MLP interaction term
Heads: delta_property, property_after, direction
```

Training command:

```bash
conda run -n nnaa python scripts/train_cedg_score.py \
  --epochs 30 \
  --batch-size 512 \
  --out-dir runs/cedg_score_v1 \
  --device cuda:0
```

Current v1 result:

```text
best val delta MAE: 0.349
val delta Spearman: 0.705
val direction AUC: 0.839

test delta MAE: 0.291
test delta Spearman: 0.773
test direction AUC: 0.908
```

Interpretation:

```text
The first neural CEDG-Score model improves over the sklearn sanity baseline.
The result is promising but should not yet be treated as final because it needs multi-seed validation, source-stratified metrics, and harder split evaluation.
```
