# Initial Table Build Report

Generated from CycPeptMPDB-derived public tables.

## Outputs

- `data/processed/peptide_sample_table.csv`
- `data/processed/source_summary.csv`
- `data/source_paper_queue.csv`
- `data/processed/monomer_frequency.csv`

## Dataset Size

- peptide rows: 7451
- sources: 47
- rows with SMILES: 7451
- rows with HELM: 7451
- rows with monomer lists: 7451

## Assay Coverage

- PAMPA: 6941
- Caco2: 649
- MDCK: 40
- RRCK: 186

## Top Sources By Repeated Rough Shadow Groups

| source_id              |   n_peptides |   n_pampa |   n_caco2 |   samples_in_repeated_shadow_groups |   max_shadow_group_size | priority   |
|:-----------------------|-------------:|----------:|----------:|------------------------------------:|------------------------:|:-----------|
| 2020_Townsend          |         3086 |      3086 |         0 |                                3082 |                     641 | A          |
| 2021_Kelly             |         1519 |      1519 |         0 |                                1400 |                    1100 | A          |
| 2016_Furukawa          |          688 |       688 |         8 |                                 306 |                      17 | B          |
| 2013_CHUGAI            |          878 |       878 |         0 |                                  56 |                       3 | C          |
| 2018_CHUGAI            |          374 |         0 |       374 |                                  52 |                       4 | C          |
| 2015_Wang              |           62 |        62 |        62 |                                  44 |                      19 | A          |
| 2018_Naylor            |           81 |        72 |         0 |                                  40 |                       8 | B          |
| 2021_Golosov           |           27 |        23 |         0 |                                  22 |                       7 | B          |
| 2022_Bhardwaj          |          136 |       133 |        40 |                                  20 |                       2 | C          |
| 2022_Taechalertpaisarn |           52 |        52 |         5 |                                  20 |                       4 | B          |
| 2020_Le Roux           |           47 |        47 |        39 |                                  20 |                       9 | B          |
| 2015_Hewitt            |           18 |         0 |        18 |                                  17 |                      17 | B          |
| 2015_Bockus_2          |           17 |        17 |        17 |                                  12 |                       4 | B          |
| 2016_Hickey            |           18 |        18 |         0 |                                  10 |                       6 | D          |
| 2015_Marelli           |           10 |        10 |        10 |                                  10 |                       6 | D          |

## Top Monomers

| monomer        |   count |   source_count | anchor_guess_v0   |
|:---------------|--------:|---------------:|:------------------|
| meL            |    6920 |             30 | L                 |
| L              |    5691 |             35 | L                 |
| dP             |    5016 |             28 | P                 |
| P              |    4915 |             26 | P                 |
| dL             |    4283 |             21 | L                 |
| F              |    3986 |             27 | F                 |
| Me_dL          |    3915 |             21 | L                 |
| T              |    2687 |             12 | T                 |
| meA            |    2382 |             17 | A                 |
| A              |    1914 |             29 | A                 |
| meF            |    1880 |             16 | F                 |
| ac-            |    1519 |              1 | ac-               |
| Bn_Gly         |    1492 |              4 | Bn_Gly            |
| Me_dA          |    1341 |             14 | A                 |
| bHph           |     996 |              1 | bHph              |
| dA             |     971 |             20 | A                 |
| -pip           |     812 |              2 | -pip              |
| Nle            |     754 |              2 | Nle               |
| Sar            |     728 |             19 | Sar               |
| D              |     721 |              4 | D                 |
| Pr_Gly         |     569 |              4 | Pr_Gly            |
| I              |     493 |             12 | I                 |
| meV            |     446 |             14 | V                 |
| V              |     406 |             15 | V                 |
| Asp_piperidide |     343 |              2 | Asp_piperidide    |

## Caveats

- `canonical_shadow_sequence_v0` is only a triage normalization.
- Repeated rough-shadow groups are not yet validated scaffold groups.
- No edit pairs are generated in this step.
- Source papers still need DOI and supplementary table checks.
