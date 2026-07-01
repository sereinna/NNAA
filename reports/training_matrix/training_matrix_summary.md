# CEDG-Set Training Matrix Summary

| Train setting | Eval setting | delta MAE | delta Spearman | direction AUC | ranking Spearman | NDCG@5 | best_hit@5 | positive_hit@5 | Note |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| original_only | original_test | 0.424 | 0.692 | 0.819 | 0.655 | 0.839 | 0.852 | 0.992 | held-out original split |
| original_only | faris_all_score | 1.372 | 0.509 | 0.757 | 0.510 | 0.746 | 0.773 | 0.967 | all Faris local SAR groups scored as literature benchmark |
| faris_only | faris_test | 0.999 | 0.370 | 0.686 | 0.283 | 0.763 | 0.826 | 0.957 | held-out Faris split |
| faris_only | original_test | 0.726 | 0.387 | 0.704 | 0.255 | 0.778 | 0.793 | 0.992 | cross-dataset original split |
| original_plus_faris | combined_test | 0.701 | 0.575 | 0.765 | 0.547 | 0.816 | 0.835 | 0.993 | held-out split after merging original and Faris |
| original_plus_faris | original_test | 0.666 | 0.627 | 0.785 | 0.588 | 0.832 | 0.852 | 1.000 | cross-check on original test split |

Interpretation:

- Metrics are now computed with a unified protocol wherever the dataset has `candidate_group_id` and `delta_property`.
- `best_hit@5` checks whether the experimentally best edit in each candidate group is recovered in model top-5.
- `positive_hit@5` checks whether model top-5 contains at least one experimentally positive edit in groups with positives.
- `original_only -> original_test` is still the strongest in-domain baseline.
- `original_only -> faris_all_score` shows useful cross-literature ranking signal on Faris local SAR.
- `faris_only` is weak despite being in-domain, indicating that 1434 Faris pairs are too small for the full architecture.
- `original_plus_faris` trains successfully but does not yet outperform the original-only model; next runs should tune Faris weight, checkpoint selection, or staged fine-tuning.
