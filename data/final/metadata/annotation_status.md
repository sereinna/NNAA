# Final Annotation Status

Date: 2026-06-29

The active curated annotation files are stored under `data/final/annotations/`.

## Active Files

```text
data/final/annotations/monomer_anchor_table.csv
data/final/annotations/monomer_anchor_manual_review_queue.csv
data/final/annotations/monomer_anchor_summary.csv
data/final/annotations/peptide_sample_table.csv
data/final/metadata/edit_event_table_schema.csv
```

## Active Monomer Annotation Summary

```text
monomer_anchor_table.csv rows: 312
monomer_anchor_manual_review_queue.csv rows: 93
peptide_sample_table.csv rows: 7451
```

`quality_flag_final` distribution:

```text
local_interpretable_edit:        196
unknown_replacement:              76
natural_or_anchor_equivalent:     20
residue_graph_replacement:        14
terminal_or_capping_context:       6
```

`model_use_tier_default` distribution:

```text
core_local_edit:                 190
atom_unknown_auxiliary:           68
reference_or_no_edit:             20
atom_replacement_auxiliary:       14
exclude_until_curated:             8
terminal_context_only:             6
local_candidate_needs_review:      6
```

Manual curation status:

```text
requires_manual_curation_final=False: 219
requires_manual_curation_final=True:   93
```

