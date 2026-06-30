# CEDG-Score sklearn Baseline Validation

Dataset: `data/final/model_ready/peptide_component/cedg_score_dataset.jsonl`

## Delta Regression

| task             | model                    | split   |   rows |      mae |     rmse |   spearman |   direction_accuracy_from_delta |   direction_auc_from_delta |   accuracy |   auc |
|:-----------------|:-------------------------|:--------|-------:|---------:|---------:|-----------:|--------------------------------:|---------------------------:|-----------:|------:|
| delta_regression | ridge_edit               | test    |    842 | 0.48974  | 0.705381 |   0.714226 |                        0.699525 |                   0.830719 |        nan |   nan |
| delta_regression | ridge_shadow_edit_source | test    |    842 | 0.619191 | 1.00398  |   0.658996 |                        0.694774 |                   0.799055 |        nan |   nan |
| delta_regression | ridge_shadow_edit        | test    |    842 | 0.740304 | 1.3203   |   0.653309 |                        0.64133  |                   0.794362 |        nan |   nan |
| delta_regression | mean_delta               | test    |    842 | 0.903365 | 1.30971  | nan        |                        0.711401 |                   0.5      |        nan |   nan |
| delta_regression | source_mean_delta        | test    |    842 | 0.921229 | 1.31398  |   0.108308 |                        0.710214 |                   0.538291 |        nan |   nan |
| delta_regression | ridge_edit               | val     |    795 | 0.575268 | 0.870763 |   0.672267 |                        0.716981 |                   0.759668 |        nan |   nan |
| delta_regression | ridge_shadow_edit_source | val     |    795 | 1.07154  | 1.48001  |   0.538879 |                        0.706918 |                   0.726752 |        nan |   nan |
| delta_regression | ridge_shadow_edit        | val     |    795 | 1.07796  | 1.81248  |   0.538125 |                        0.703145 |                   0.715367 |        nan |   nan |
| delta_regression | mean_delta               | val     |    795 | 1.12353  | 1.5803   | nan        |                        0.654088 |                   0.5      |        nan |   nan |
| delta_regression | source_mean_delta        | val     |    795 | 1.13965  | 1.5934   |   0.046273 |                        0.655346 |                   0.534818 |        nan |   nan |

## Direction Classification

| task                     | model                     | split   |   rows |   mae |   rmse |   spearman |   direction_accuracy_from_delta |   direction_auc_from_delta |   accuracy |      auc |
|:-------------------------|:--------------------------|:--------|-------:|------:|-------:|-----------:|--------------------------------:|---------------------------:|-----------:|---------:|
| direction_classification | logreg_shadow_edit_source | test    |    842 |   nan |    nan |        nan |                             nan |                        nan |   0.787411 | 0.86981  |
| direction_classification | logreg_shadow_edit        | test    |    842 |   nan |    nan |        nan |                             nan |                        nan |   0.786223 | 0.868162 |
| direction_classification | logreg_edit               | test    |    842 |   nan |    nan |        nan |                             nan |                        nan |   0.769596 | 0.85695  |
| direction_classification | logreg_edit               | val     |    795 |   nan |    nan |        nan |                             nan |                        nan |   0.777358 | 0.832486 |
| direction_classification | logreg_shadow_edit_source | val     |    795 |   nan |    nan |        nan |                             nan |                        nan |   0.773585 | 0.822122 |
| direction_classification | logreg_shadow_edit        | val     |    795 |   nan |    nan |        nan |                             nan |                        nan |   0.74717  | 0.796507 |

## Interpretation

- `mean_delta` is the minimum sanity baseline.
- `source_mean_delta` tests how much source identity alone explains the split.
- `ridge_edit` uses edit-set metadata without peptide shadow context.
- `ridge_shadow_edit` adds parent shadow sequence token features.
- `ridge_shadow_edit_source` adds source identity and should be treated as a source-bias upper sanity check, not the main model.
