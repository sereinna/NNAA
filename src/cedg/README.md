# CEDG-Set PyTorch Scorer

中文说明：本目录实现 CEDG 第一阶段完整 scorer/ranker；开放式 R-group / ΔG 生成模块不在本阶段实现。

This package implements the current neural CEDG-Set scorer:

```text
ESM/canonical shadow peptide context + topology-aware residue graph + site-conditioned edit set
-> delta PAMPA / property_after / ranking_score / direction / uncertainty
```

## Architecture

- `data.py`: clean model inputs, QC metadata separation, JSONL loading, PLM shadow sequence construction, vocabulary construction, EditGraph tensorization, batching, topology features, and candidate group ids.
- `topology.py`: residue-level topology features, cyclic distance, residue topology graph, edge types, and scaffold descriptors.
- `chem_edit.py`: attachment-aware old/new/delta edit graph objects, atom actions, atom roles, graph modes, MCS/full-residue/stereo/curated fallback builders.
- `PeptideContextEncoder`: Transformer encoder over canonical shadow residues.
- optional ESM branch: `--use-esm` loads ESM through `torch.hub`, projects residue
  embeddings, and fuses them with learned residue/topology context.
- topology-aware residue and scaffold features: cyclic residue position, cyclic distance to edited sites, edit count, ring/cyclization cues, residue topology adjacency, edge type, and shortest-path distance.
- `SiteConditionedPayloadEncoder`: encodes curated local chemical payload tokens under the selected residue-site context.
- `SiteConditionedAtomEditEncoder`: encodes RDKit-derived old/new payload fragment graphs parsed from each edit event, with residue-site context injected during graph message passing.
- `EditEventEncoder`: fuses site, anchor, edit scope, operation class, attachment type, payload type, payload tokens, atom-level graph delta, and residue context.
- `EditSetEncoder`: Transformer/set encoder for one or more simultaneous edits.
- `PredictionHeads`: delta, property, ranking, direction, and uncertainty heads.
- `losses.py`: weighted regression, heteroscedastic delta loss, and candidate-group pairwise ranking loss.
- `losses.py`: weighted regression, heteroscedastic delta loss, candidate-group pairwise ranking loss, and listwise ranking loss.
- `metrics.py`: MAE/RMSE/Spearman, direction AUROC/accuracy, uncertainty-error correlation, selective risk, uncertainty z-score, and group ranking metrics.
- `evaluation.py`: model evaluation loop and metric aggregation.
- `samplers.py`: candidate group-aware batch sampler.
- `utils.py`: seed and device helpers.

The model now uses two chemical branches:

- curated delta metadata, such as `N-H_to_N-CH3` or `D_to_L`;
- attachment-aware local payload fragment graphs, such as `N-H -> N-CH3`,
  `phenyl_para H -> F`, or `Ser_OH H -> tBu`.

This is still a scorer, not a generator. The atom-level branch now exposes an
attachment-aware `EditGraph` interface with old/new/delta tensors, atom action,
atom role, and graph mode. It can use full residue MCS or stereo-flag paths when
fields are available, and falls back to curated fragment parsing for current
exports.

## Train

Prebuild edit graph tensors once before long GPU runs:

```bash
conda run -n nnaa python scripts/cache_cedg_graphs.py \
  --batch-size 512 \
  --num-workers 0
```

The default cache location is:

```text
data/final/cache/edit_graphs/peptide_component
```

Training uses this cache by default. Disable it only for debugging with
`--no-graph-cache`.

For frozen ESM, precompute residue embeddings once:

```bash
conda run -n nnaa python scripts/cache_esm_embeddings.py \
  --batch-size 512 \
  --device cuda:0
```

Then train from cached ESM embeddings without running ESM forward every epoch:

```bash
conda run -n nnaa python scripts/train_cedg_score.py \
  --epochs 30 \
  --batch-size 512 \
  --out-dir runs/cedg_set_esm_cached_full \
  --device cuda:0 \
  --group-aware-batches \
  --use-esm-cache
```

```bash
conda run -n nnaa python scripts/train_cedg_score.py \
  --epochs 30 \
  --batch-size 384 \
  --out-dir runs/cedg_set_payload_v1 \
  --device cuda:0 \
  --group-aware-batches \
  --use-esm \
  --freeze-esm
```

The default ESM model is `esm2_t6_8M_UR50D`, loaded through `torch.hub`. Use
`--esm-model esm2_t36_3B_UR50D` only when you intentionally want the much larger
3B-parameter branch. `--freeze-esm` is enabled by default; pass `--no-freeze-esm`
to fine-tune ESM.

## Candidate Recommendations

After training, score candidate groups and export parent-level top-k edit-set
recommendations:

```bash
conda run -n nnaa python scripts/score_candidate_groups.py \
  --checkpoint runs/cedg_set_esm_cached_full/best_model.pt \
  --parent-peptide-id 1392 \
  --top-k 10 \
  --out-csv runs/cedg_set_esm_cached_full/topk_parent_1392.csv \
  --out-jsonl runs/cedg_set_esm_cached_full/topk_parent_1392.jsonl \
  --device cuda:0
```

The recommendation score is currently:

```text
ranking_score - 0.1 * uncertainty
```

Use `scripts/export_payload_coverage_report.py` to inspect graph-mode coverage,
validity, attachment distribution, and payload type distribution.

Use `--group-aware-batches` to pack same-parent candidate groups into batches
for stronger ranking supervision. The default training path rejects `K=0`
samples; use `--allow-absolute-only` only when absolute-only rows are explicitly
encoded and intended.

## Notes

This version targets the full first-stage scorer/ranker in `Untitled.md`, while
intentionally excluding the second-stage open-vocabulary generator. Candidate
ranking supports `candidate_group_id` and uses groupwise pairwise ranking loss,
with a pointwise `delta_property` proxy retained as an auxiliary objective.
