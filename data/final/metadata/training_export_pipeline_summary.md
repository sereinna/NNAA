# Final Training Export Pipeline Summary

Date: 2026-06-29

## Active Inputs

```text
data/final/annotations/peptide_sample_table.csv
data/final/annotations/monomer_anchor_table.csv
```

## Script

```text
scripts/export_final_training_by_source.py
```

The export uses final monomer annotation fields:

```text
anchor_for_alignment
final_edit_scope
final_chemical_payload_type
final_chemical_payload
model_use_tier_default
use_for_local_edit_model
use_for_atom_edit_model
quality_flag_final
requires_manual_curation_final
```

## Output Format

Each source exports only two model-facing training tables plus a small summary:

```text
data/final/training/<source_slug>/training_single_site.csv
data/final/training/<source_slug>/training_multi_site.csv
data/final/training/<source_slug>/training_summary.csv
```

Single-site rows are flat. Multi-site rows store all site edits in `edit_set_json` and keep the property delta at pair level.

## Current Source Exports

| source_id | single-site rows | multi-site rows | note |
|---|---:|---:|---|
| 2013_CHUGAI | 74 | 8 | small repeated-shadow groups; no large-group sampling needed |
| 2015_Wang | 24 | 32 | reverse-direction duplicates removed |
| 2020_Le Roux | 95 | 165 | reverse-direction duplicates removed |
| 2022_Taechalertpaisarn | 34 | 9 | reverse-direction duplicates removed |
| 2021_Golosov | 9 | 13 | reverse-direction duplicates removed |
| 2015_Bockus_2 | 7 | 0 | stricter filtering |
| 2018_Naylor | 17 | 5 | reverse-direction duplicates removed |
| 2016_Furukawa | 43 | 557 | large groups sampled with top-|delta| strategy after reverse-direction deduplication |
| 2020_Townsend | 2118 | 5052 | large groups sampled with top-|delta| strategy after reverse-direction deduplication |
| 2021_Kelly | 127 | 791 | large groups sampled with top-|delta| strategy after reverse-direction deduplication |

Totals:

```text
single-site rows: 2548
multi-site rows: 6632
single-site censored rows: 506
multi-site censored rows: 1801
reverse-direction duplicate candidates removed before export: 50173
```

`PAMPA = -10.00` values are retained with `censored_property_flag=True`.

Reverse-direction duplicate pairs are removed before writing final tables. If both directions exist for the same unordered peptide pair, the kept row preferentially has positive `delta_property`, so it describes the modification direction associated with improved PAMPA. Unique one-direction candidates are retained even if their `delta_property` is negative.

Edit labels are pair-specific, not copied blindly from the modified monomer default. The exporter compares original and modified monomer state and combines backbone N-methylation and stereochemistry deltas when both changed. For example, `dL -> meL` is labeled as `backbone_and_stereochemistry_edit` with `N-H_to_N-CH3;D_to_L`, while `dL -> Me_dL` remains only `N-H_to_N-CH3` because both sides are D-Leu. For different-anchor edits, default monomer payloads are stripped of state terms before adding the actual pair-level state change, preventing duplicated labels such as `L_to_D;L_to_D`.

The edit-count limit is adaptive by peptide length:

```text
allowed_edits = min(max_edits, max(2, ceil(length * max_edit_fraction)))
```

The current export uses:

```text
--max-edits 4
--max-edit-fraction 0.25
```

This keeps up to 2 edits for short peptides, up to 3 edits for length 9-12 peptides, and caps longer peptides at 4 edits.

## Why Counts Changed

The current export uses the curated monomer table rather than only hard-coded operations such as:

```text
N_methylation
D_L_inversion
```

It includes local edit types where final annotation says they are eligible, and excludes rows marked as unknown, terminal-only, reference/no-edit, or not curated enough. This makes the final export chemically cleaner even when some source-level counts become smaller.

The initial clean export did not include `2020_Townsend`, `2021_Kelly`, or `2013_CHUGAI`. `2020_Townsend` and `2021_Kelly` contain very large same-shadow groups, so direct all-vs-all expansion would create a large number of highly related pairs. `2013_CHUGAI` has many PAMPA rows but comparatively small repeated-shadow groups, so it was lower priority. The current export includes all three using reverse-direction deduplication, adaptive edit-count limits, and top-|delta| sampling for large groups.

## Large Source Handling

`2016_Furukawa`, `2020_Townsend`, and `2021_Kelly` are handled with controlled large-group sampling.

`2016_Furukawa` final shadow grouping creates three large groups:

```text
A.G.L.L.P.G: 244
F.G.L.L.P.G: 344
L.G.L.L.P.G: 100
```

The current export uses:

```text
--large-group-strategy top-delta
--max-pairs-per-large-group 200
```

This keeps high-|delta| unique unordered peptide pairs from each large group after choosing a single direction per pair. The selected set is censored-rich because many largest changes involve `PAMPA = -10.00`.

`2020_Townsend` has 13 large same-shadow groups sampled with the same rule. `2021_Kelly` has 3 large same-shadow groups sampled with the same rule.
