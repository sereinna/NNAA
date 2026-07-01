# CEDG-Set Training Matrix Summary

| Train setting | Eval setting | delta MAE | delta Spearman | direction AUC | ranking Spearman | NDCG@5 | best_hit@5 | positive_hit@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| original_only | original_test | 0.424 | 0.692 | 0.819 | 0.655 | 0.837 |  |  |
| original_only | faris_all_external |  | 0.508 |  | 0.508 | 0.745 | 0.773 | 0.967 |
| faris_only | faris_test | 0.999 | 0.368 | 0.683 | 0.281 | 0.739 |  |  |
| faris_only | original_test | 0.726 | 0.387 | 0.704 | 0.255 | 0.778 |  |  |
| original_plus_faris | combined_test | 0.701 | 0.575 | 0.765 | 0.547 | 0.820 |  |  |
| original_plus_faris | original_test | 0.666 | 0.627 | 0.785 | 0.588 | 0.832 |  |  |

Interpretation:

- `original_only -> original_test` is still the strongest in-domain baseline.
- `original_only -> faris_all_external` shows useful external ranking signal on Faris local SAR.
- `faris_only` is weak despite being in-domain, indicating that 1434 Faris pairs are too small for the full architecture.
- `original_plus_faris` trains successfully but does not yet outperform the original-only model; next runs should tune Faris weight, checkpoint selection, or staged fine-tuning.
