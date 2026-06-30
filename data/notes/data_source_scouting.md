# Data Source Scouting Notes

Date: 2026-06-29

## Main conclusion

Start from CycPeptMPDB-derived public CSV files, but do not treat the whole database as edit-level data. Use it as the peptide sample backbone, then mine matched edit pairs source by source.

The highest-priority sources for CEDG-style edit pairs are:

1. `2020_Townsend`
2. `2021_Kelly`
3. `2015_Wang`
4. `2016_Furukawa`
5. `2020_Le Roux`
6. `2022_Taechalertpaisarn`

`2013_CHUGAI` and `2018_CHUGAI` are large but need manual inspection because a simple shadow-sequence grouping did not reveal as many repeated scaffold groups as expected.

## Public data assets

- CycPeptMPDB web database: http://cycpeptmpdb.com/
- CycPeptMPDB paper: https://doi.org/10.1021/acs.jcim.2c01573
- CycPeptMP processed data and monomer table: https://github.com/akiyamalab/cycpeptmp
- BenchmarkCycPeptMP baselines: https://github.com/Gobliu/BenchmarkCycPeptMP

Local quick check of `CycPeptMPDB_Peptide_All.csv` from the CycPeptMP repository:

```text
rows: 7451
columns: 244
PAMPA non-null: 6941
Caco2 non-null: 649
MDCK non-null: 40
RRCK non-null: 186
```

The same repository README reports deduplicated counts:

```text
all: 7451 -> 7337
PAMPA: 6941 -> 6889
```

## Source-level rough prioritization

This table was estimated from the public CSV using a coarse "shadow sequence" normalization that strips obvious `d` and `me` prefixes. It is only a triage signal, not final scaffold grouping.

| Source | Rows | PAMPA | Caco2 | Samples in repeated rough-shadow groups | Max rough-shadow group | Priority |
|---|---:|---:|---:|---:|---:|---|
| 2020_Townsend | 3086 | 3086 | 0 | 3060 | 184 | A |
| 2021_Kelly | 1519 | 1519 | 0 | 1388 | 186 | A |
| 2016_Furukawa | 688 | 688 | 8 | 306 | 13 | B+ |
| 2018_CHUGAI | 374 | 0 | 374 | 49 | 3 | B |
| 2015_Wang | 62 | 62 | 62 | 41 | 9 | A |
| 2013_CHUGAI | 878 | 878 | 0 | 30 | 2 | B |
| 2022_Taechalertpaisarn | 52 | 52 | 5 | 20 | 4 | B |
| 2021_Golosov | 27 | 23 | 0 | 20 | 7 | B |
| 2020_Le Roux | 47 | 47 | 39 | 17 | 7 | B |
| 2015_Hewitt | 18 | 0 | 18 | 16 | 8 | B |
| 2015_Bockus_2 | 17 | 17 | 17 | 12 | 4 | B |
| 2016_Hickey | 18 | 18 | 0 | 10 | 3 | C |
| 2020_Barlow | 26 | 26 | 0 | 9 | 6 | C |
| 2006_Rezai_1 | 10 | 10 | 0 | 9 | 9 | C |
| 2018_Naylor | 81 | 72 | 0 | 8 | 8 | B |

## Recommended immediate workflow

1. Download the CycPeptMPDB peptide CSV and monomer table into `data/raw_sources/cycpeptmpdb/`.
2. Build `peptide_sample_table.csv` directly from the public columns:
   - `CycPeptMPDB_ID`
   - `Source`
   - `Year`
   - `Original_Name_in_Source_Literature`
   - `SMILES`
   - `HELM`
   - `Sequence`
   - `Monomer_Length`
   - `Molecule_Shape`
   - `Permeability`
   - `PAMPA`
   - `Caco2`
   - `MDCK`
   - `RRCK`
3. For `2020_Townsend` and `2021_Kelly`, manually inspect the original source papers/tables or supplementary data before automatic pair generation.
4. Build `monomer_anchor_table` from `monomer_table.csv` plus manual review of top-frequency monomers.
5. Generate a conservative first edit-pair set only when:
   - same `Source`;
   - same assay column;
   - same rough scaffold group after manual check;
   - 1 or 2 changed monomer sites;
   - monomer changes can be mapped to anchor/edit_scope/payload or explicit fallback.

## Risks

- CycPeptMPDB has enough peptide-level rows but not all rows are matched edit pairs.
- Simple shadow grouping can overmerge positional isomers or undermerge modified monomers.
- Some large sources may be combinatorial libraries rather than clean medicinal-chemistry parent-to-child pairs.
- Sampled negatives should not be labeled as true negatives unless there is observed worse experimental evidence.

## Auxiliary resources

- CyclicPepedia can help with cyclic peptide structure/sequence conversion and monomer reference coverage.
- NORINE can help normalize natural nonribosomal peptide monomers.
- HELM monomer resources are useful for attachment point definitions.
- SWEMacrocycleDB is outside the first peptide-focused MVP but may be useful later for semipeptidic/nonpeptidic macrocycle permeability comparison.

## Expanded Data Source Map

### Tier A: primary permeability data

These are the only sources that should enter the first CEDG permeability MVP as primary labels.

| Source | Role | Use |
|---|---|---|
| CycPeptMPDB | main cyclic peptide permeability database | peptide_sample_table backbone |
| CycPeptMP GitHub | processed CycPeptMPDB CSV and monomer table | raw table, monomer correspondence, baseline reproduction |
| BenchmarkCycPeptMP | method benchmark and scaffold/random split reference | baseline/split comparison |
| CycPeptMPDB source papers such as 2020_Townsend, 2021_Kelly, 2015_Wang | same-source SAR series | edit_pair_table and edit_event_table mining |

Important distinction: CycPeptMPDB is peptide-level data. CEDG needs source-wise matched edit pairs. The primary curation task is not simply downloading CycPeptMPDB, but turning selected same-source SAR series into parent/modified pairs.

### Tier B: auxiliary peptide property labels

These are useful, but should not be mixed with passive membrane permeability regression.

| Dataset family | Examples | Possible use | Main caution |
|---|---|---|---|
| CPP / uptake | CPPsite 2.0, CellPPD | weak cell-entry or uptake auxiliary labels | uptake is not passive permeability |
| Hemolysis | Hemolytik, HemoPI/HemoPI2, DBAASP, DRAMP | toxicity head, safety reranking | species, concentration and assay conditions differ |
| Antimicrobial peptides | DBAASP, DRAMP, CAMP_R4, APD, dbAMP | broad peptide pretraining and safety/activity context | antimicrobial activity is target/assay dependent |
| Therapeutic peptide databases | SATPdb, THPdb, PepTherDia | clinical modification vocabulary and case studies | sparse and heterogeneous labels |

Recommended policy: keep these datasets in separate tables and use them only for auxiliary pretraining, multitask heads, or qualitative case studies. Do not compute delta_property across these sources.

### Tier C: monomer, topology and structure references

These are high-value for representation standardization.

| Resource | Use |
|---|---|
| SwissSidechain | non-natural sidechain structures, D/L forms, SMILES/MOL2/PDB parameters |
| HELMMonomerSets | HELM monomer vocabulary and attachment points |
| NORINE | nonribosomal peptide monomer and natural product scaffold vocabulary |
| CyclicPepedia | cyclic peptide structure/sequence conversion and broad cyclic peptide annotations |
| CyBase, ConoServer, CPDB | cyclic/disulfide-rich/PTM peptide topology examples |
| MacroConf | 3D conformer benchmark for later 3D-risk module |

Recommended policy: use these to improve `monomer_anchor_table`, `chemical_payload`, topology annotation and validation scripts. They should not define permeability labels.

### Tier D: out-of-scope or later-stage resources

| Resource | Reason to delay |
|---|---|
| SWEMacrocycleDB | permeability labels are useful, but many molecules are nonpeptidic/semipeptidic; good external comparison, not first peptide MVP |
| ChEMBL ADME assays | broad and heterogeneous; curation burden is high |
| TDC ADME | small-molecule benchmark, not cyclic peptide edit data |
| SmProt / large unlabeled peptide corpora | useful only for pretraining experiments |

## What "complete" means for this project

The dataset inventory is complete enough when every candidate source has one of these statuses:

```text
primary_permeability
source_paper_for_pair_mining
auxiliary_property
monomer_structure_reference
topology_structure_reference
external_validation
out_of_scope
```

The current `data/source_inventory.csv` uses priority/confidence notes rather than a dedicated status column. Add a status column before building ingestion scripts.

## Recommended next curation targets

1. Freeze the main passive-permeability scope:
   - first target: PAMPA only;
   - secondary target: Caco2 only after PAMPA pipeline works;
   - do not mix CPP uptake or hemolysis with permeability.

2. Download primary raw assets:
   - CycPeptMPDB peptide CSV;
   - CycPeptMPDB monomer table;
   - CycPeptMPDB unique monomer table;
   - BenchmarkCycPeptMP split files if baseline reproduction is planned.

3. Build `source_paper_queue.csv` for the top CycPeptMPDB sources:
   - `2020_Townsend`;
   - `2021_Kelly`;
   - `2015_Wang`;
   - `2016_Furukawa`;
   - `2020_Le Roux`;
   - `2022_Taechalertpaisarn`;
   - `2021_Golosov`;
   - `2015_Bockus_2`;
   - `2015_Hewitt`.

4. For each source paper, record:
   - DOI;
   - supplementary table availability;
   - whether source compound names in CycPeptMPDB match source tables;
   - whether the series has a clear parent scaffold;
   - dominant edit types;
   - assay type and units;
   - estimated single-edit and multi-edit pair count.

5. Build the first `monomer_anchor_table` from:
   - CycPeptMPDB `monomer_table.csv`;
   - CycPeptMPDB `unique_monomer.csv`;
   - SwissSidechain;
   - HELMMonomerSets;
   - manual top-100 monomer review.

## Expanded Risks

- Many peptide datasets report activity, uptake or toxicity rather than passive permeability. These are not interchangeable.
- CPP datasets often mix linear/cyclic, modified/unmodified and assay-dependent uptake data.
- Hemolysis datasets require concentration-aware labels; binary hemolytic/non-hemolytic labels can be misleading.
- AMP datasets are large but mostly not medicinal chemistry matched-pair datasets.
- Therapeutic peptide datasets are useful for vocabulary and examples but are too sparse for CEDG training.
- Monomer libraries provide structures, not property labels.
- ChEMBL-like ADME extraction could become a separate project because unit and protocol harmonization are heavy.
