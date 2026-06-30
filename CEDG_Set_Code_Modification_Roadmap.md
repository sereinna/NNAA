# CEDG-Set 代码修改总路线图

> 目的：把当前的 CEDG-Score 第一阶段 MVP 升级为更接近论文规划中的 **PLM + attachment-aware atom-level ΔG graph + topology-aware cyclic peptide context** 版本。  
> 重点：本文只总结代码侧需要修改的部分；数据源、数据收集和标签构造由你自己负责，但我会列出模型接口所需要的数据字段。

---

## 0. 当前代码定位与目标状态

### 0.1 当前代码大致状态

当前实现已经有以下模块：

```text
data.py
  JSONL loading
  train-only vocabs
  parent shadow sequence tensorization
  edit metadata tensorization
  curated payload tokenization
  old/new payload fragment graph parsing
  compact topology_features / residue_features
  collate_cedg

model.py
  PeptideContextEncoder = learned embedding + position embedding + TransformerEncoder
  SiteConditionedPayloadEncoder = payload token cross-attention
  PayloadGraphEncoder = dense adjacency graph conv
  SiteConditionedAtomEditEncoder = old/new graph encoder + modified - original
  EditEventEncoder
  EditSetEncoder
  PredictionHeads

losses.py
  weighted MSE
  heteroscedastic delta loss
  group pairwise ranking loss

metrics.py
  regression metrics
  ranking pairwise accuracy / NDCG@5
  direction accuracy / AUROC
  uncertainty-error Spearman

evaluation.py
  model inference and metric aggregation
```

当前最大问题不是“不能跑”，而是几个关键 claim 还没有代码支撑：

```text
1. PLM branch 目前不是 PLM，而是从零训练的 Transformer。
2. payload graph 目前主要是 curated fragment graph，不是真正自动化 attachment-aware ΔG graph。
3. topology 目前是轻量手工特征，不是真正 residue topology graph / macrocycle topology encoder。
4. ranking loss 已有，但 group-aware batching / listwise ranking / top-k recommendation pipeline 仍需要增强。
5. uncertainty 目前是 heteroscedastic σ，不等于完整 OOD / calibration / selective prediction 模块。
```

### 0.2 目标状态

建议目标架构改为：

```text
canonical shadow sequence
    ↓
PLM / ESM encoder
    ↓
residue PLM embeddings H_plm

residue topology features + residue topology graph
    ↓
TopologyAwareResidueEncoder
    ↓
residue context H

for each edit e_k:
    site_k selects h_site_k
    edit metadata embedding
    payload token embedding
    attachment-aware old/new/full/delta graph encoding
    site-conditioned atom-level ΔG graph encoder
    ↓
    edit embedding u_k

edit set {u_1, ..., u_K}
    ↓
EditSetEncoder / interaction network
    ↓
z_E

h_P + z_E + clean sample context
    ↓
property / delta / ranking / direction / uncertainty / optional auxiliary heads
```

---

## 1. 总优先级

| 优先级 | 修改主题 | 是否影响论文主 claim | 主要文件 |
|---|---|---:|---|
| P0 | 拆分 clean input 与 QC/leakage metadata | 高 | `data.py`, `model.py` |
| P0 | 支持 K=0 / absolute-only 或显式禁止并报错 | 中 | `data.py`, `model.py`, `losses.py` |
| P1 | 引入 PLM peptide encoder | 很高 | `data.py`, `model.py`, train script |
| P1 | PLM collator、token alignment、freeze/unfreeze 策略 | 很高 | `data.py`, `model.py`, train script |
| P2 | 新增 `chem_edit.py` 自动 ΔG graph extraction | 很高 | `chem_edit.py`, `data.py` |
| P2 | attachment-aware graph tensorization | 很高 | `data.py`, `model.py` |
| P2 | DeltaGraphEncoder / edge-aware GNN | 很高 | `model.py` |
| P3 | topology features 扩展 | 高 | `data.py`, `model.py` |
| P3 | residue topology graph / cyclic distance | 高 | `data.py`, `model.py` |
| P4 | group-aware sampler / listwise ranking | 高 | `samplers.py`, `losses.py`, train script |
| P4 | uncertainty calibration / OOD metrics | 中高 | `metrics.py`, `evaluation.py`, `losses.py` |
| P5 | ablation configs / reports / tests / README | 高 | `configs/`, `tests/`, `README.md` |

---

## 2. 需要新增或重构的文件

建议最终文件结构：

```text
cedg/
  __init__.py
  data.py
  chem_edit.py              # 新增：自动 attachment-aware ΔG graph extraction
  topology.py               # 可选新增：topology 特征和 residue graph 构造
  model.py
  losses.py
  metrics.py
  evaluation.py
  samplers.py               # 新增：candidate group-aware sampler
  utils.py

scripts/
  train_cedg_score.py
  evaluate_cedg_score.py
  export_payload_coverage_report.py
  run_ablation.py

configs/
  cedg_mvp.yaml
  cedg_plm_frozen.yaml
  cedg_plm_topology.yaml
  cedg_delta_graph.yaml
  cedg_full_model.yaml
  ablation_no_plm.yaml
  ablation_no_topology.yaml
  ablation_no_payload_graph.yaml
  ablation_no_edit_scope.yaml
  ablation_concat_fusion.yaml

tests/
  test_data_collate.py
  test_plm_alignment.py
  test_chem_edit.py
  test_topology.py
  test_model_forward.py
  test_losses.py
```

---

# Part A. `data.py` 修改清单

---

## A1. 拆分模型输入字段与 QC / leakage 字段

当前 `EDIT_FIELDS` 和 `SAMPLE_FIELDS` 中有一些字段更像数据质量、来源或 curation 信息。它们可以用于过滤、加权、分层分析，但不建议作为主模型输入，否则容易在 random split 或同源数据中造成 source artifact leakage。

### 建议修改

把原来的：

```python
EDIT_FIELDS = (...)
SAMPLE_FIELDS = (...)
```

改成：

```python
EDIT_INPUT_FIELDS = (
    "anchor_for_alignment",
    "original_monomer",
    "modified_monomer",
    "edit_event_class",
    "edit_event_subclass",
    "final_edit_scope",
    "attachment_type",
    "final_chemical_payload_type",
)

EDIT_QC_FIELDS = (
    "edit_model_use_tier",
    "local_model_eligibility",
    "atom_model_eligibility",
    "quality_flag_final",
)

SAMPLE_INPUT_FIELDS = (
    "assay_type",
)

SAMPLE_QC_FIELDS = (
    "source_id",
    "source_slug",
    "sample_model_use_tier",
    "candidate_generation_strategy",
    "table_type",
)
```

### 模型输入建议

主论文结果使用：

```text
EDIT_INPUT_FIELDS + SAMPLE_INPUT_FIELDS + numeric scientific features
```

不要把这些作为主模型输入：

```text
source_id
source_slug
quality_flag_final
edit_model_use_tier
sample_model_use_tier
local_model_eligibility
atom_model_eligibility
candidate_generation_strategy
```

这些字段可以用于：

```text
sample_weight
filtering
stratified metrics
coverage report
ablation: with_metadata vs clean_input
```

---

## A2. 增加 PLM sequence 构造

当前 `shadow_ids` 进入 learned embedding。PLM 需要一字母 amino acid sequence。

### 新增 mapping

```python
CANONICAL_TO_PLM = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C",
    "Gln": "Q", "Glu": "E", "Gly": "G", "His": "H", "Ile": "I",
    "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F", "Pro": "P",
    "Ser": "S", "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V",

    # pseudo-anchor fallback
    "Lys-like": "K",
    "Leu-like": "L",
    "Phe-like": "F",
    "Trp-like": "W",
    "Ala-like": "A",
    "Gly-like": "G",
    "UNK": "X",
    "<UNK>": "X",
}


def shadow_to_plm_sequence(tokens: list[str]) -> str:
    return "".join(CANONICAL_TO_PLM.get(token, "X") for token in tokens)
```

### 修改 `CEDGDataset.__getitem__`

```python
tokens = shadow_tokens(record)
return {
    ...,
    "plm_sequence": shadow_to_plm_sequence(tokens),
    "shadow_ids": [self.vocabs.shadow.encode(token) for token in tokens],
    ...,
}
```

### 注意

不要把 `4-F-Phe`、`N-Me-Trp`、`D-Phe` 直接加进 PLM tokenizer。PLM branch 只看 canonical shadow；非天然信息继续走 `chemical_payload` / `edit event` branch。

---

## A3. 增加 PLM collator wrapper

不要在 `CEDGDataset` 里做 PLM tokenizer。建议保留 `collate_cedg`，新增 wrapper：

```python
class CEDGCollatorWithPLM:
    def __init__(self, tokenizer, base_collate=collate_cedg, max_plm_length: int = 256):
        self.tokenizer = tokenizer
        self.base_collate = base_collate
        self.max_plm_length = max_plm_length

    def __call__(self, batch):
        out = self.base_collate(batch)
        seqs = [item["plm_sequence"] for item in batch]
        tok = self.tokenizer(
            seqs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_plm_length,
            add_special_tokens=True,
        )
        out["plm_input_ids"] = tok["input_ids"]
        out["plm_attention_mask"] = tok["attention_mask"]
        out["plm_lengths"] = torch.tensor([len(s) for s in seqs], dtype=torch.long)
        return out
```

---

## A4. 扩展 residue-level features

当前 `RESIDUE_FEATURE_DIM = 4`，只含：

```text
position_norm
sin_pos
cos_pos
is_edited_site
```

建议扩展到 12 或 16 维，例如：

```python
RESIDUE_FEATURE_DIM = 12
```

建议特征：

```text
position_norm
sin_pos
cos_pos
is_edited_site
min_cyclic_distance_to_any_edit
is_cyclization_anchor
is_linker_adjacent
is_terminal_cap_adjacent
is_backbone_N_methylatable
is_sidechain_attachment_possible
is_pseudo_anchor
residue_charge_class_or_scaled_charge
```

### 建议 helper

```python
def cyclic_distance(i: int, j: int, length: int) -> int:
    d = abs(i - j)
    return min(d, length - d)
```

### 修改 `residue_features(record)`

```python
def residue_features(record: dict[str, object]) -> list[list[float]]:
    tokens = shadow_tokens(record)
    length = max(len(tokens), 1)
    edited_sites = [
        int(event.get("site_index") or 0)
        for event in record.get("edit_set", [])
        if int(event.get("site_index") or 0) > 0
    ]

    out = []
    for index in range(1, length + 1):
        scaled = index / length
        if edited_sites:
            min_dist = min(cyclic_distance(index, s, length) for s in edited_sites)
            min_dist = min_dist / max(length // 2, 1)
        else:
            min_dist = 1.0

        token = tokens[index - 1] if index - 1 < len(tokens) else "UNK"
        out.append([
            scaled,
            math.sin(scaled * 2.0 * math.pi),
            math.cos(scaled * 2.0 * math.pi),
            float(index in edited_sites),
            float(min_dist),
            float(is_cyclization_anchor(record, index)),
            float(is_linker_adjacent(record, index)),
            float(is_terminal_cap_adjacent(record, index)),
            float(is_backbone_n_methylatable(token)),
            float(is_sidechain_attachment_possible(token)),
            float(is_pseudo_anchor(token)),
            residue_charge_value(token),
        ])
    return out
```

需要新增的 helper 可以先用规则占位：

```python
def is_cyclization_anchor(record, index): return False

def is_linker_adjacent(record, index): return False

def is_terminal_cap_adjacent(record, index): return False

def is_backbone_n_methylatable(token): return token not in {"Pro", "N-Me"}

def is_sidechain_attachment_possible(token): return token in {"Phe", "Trp", "Tyr", "Lys", "Ser", "Cys", "Asp", "Glu"}

def is_pseudo_anchor(token): return token.endswith("-like")

def residue_charge_value(token): ...
```

---

## A5. 扩展 scaffold-level topology features

当前 `topology_features()` 只有 4 维：

```text
length_norm
edit_count_norm
coarse cyclized cue
multi flag
```

建议新增：

```python
TOPOLOGY_FEATURE_DIM = 24  # 或 32
```

建议字段：

```text
length_norm
edit_count_norm
is_cyclic
cyclization_type_head_to_tail
cyclization_type_sidechain_to_sidechain
cyclization_type_disulfide
cyclization_type_lactam
cyclization_type_thioether
cyclization_type_staple
cyclization_type_unknown
macrocycle_ring_size_norm
num_rings_norm
largest_ring_size_norm
ring_residue_fraction
has_terminal_cap
has_linker
linker_length_norm
num_backbone_N_methylation_sites
num_D_residue_sites
net_charge_norm
global_hbd_norm
global_hba_norm
global_tpsa_norm
global_clogp_norm
```

其中 RDKit-derived descriptors 可以单独作为 `global_physchem_features`，不要全部叫 topology。论文中可写：

```text
topology descriptors + global physicochemical descriptors
```

---

## A6. 新增 residue topology graph tensorization

这是 topology-aware cyclic peptide model 的关键。

### 新增 edge type vocabulary

```python
RESIDUE_EDGE_TYPES = {
    "none": 0,
    "self": 1,
    "sequence_bond": 2,
    "head_to_tail_cyclization": 3,
    "sidechain_linker": 4,
    "disulfide": 5,
    "lactam": 6,
    "thioether": 7,
    "staple": 8,
    "contact_3d": 9,
}
```

### 新增函数

```python
def residue_topology(record: dict[str, object]) -> dict[str, torch.Tensor]:
    tokens = shadow_tokens(record)
    length = max(len(tokens), 1)

    adj = torch.eye(length, dtype=torch.float32)
    edge_type = torch.zeros((length, length), dtype=torch.long)
    edge_type.fill_(RESIDUE_EDGE_TYPES["none"])
    for i in range(length):
        edge_type[i, i] = RESIDUE_EDGE_TYPES["self"]

    # sequence edges
    for i in range(length - 1):
        adj[i, i + 1] = adj[i + 1, i] = 1.0
        edge_type[i, i + 1] = edge_type[i + 1, i] = RESIDUE_EDGE_TYPES["sequence_bond"]

    # head-to-tail cyclic edge
    cyclization_type = normalize_token(record.get("cyclization_type"))
    if cyclization_type == "head_to_tail" and length > 2:
        adj[0, length - 1] = adj[length - 1, 0] = 1.0
        edge_type[0, length - 1] = edge_type[length - 1, 0] = RESIDUE_EDGE_TYPES["head_to_tail_cyclization"]

    # sidechain / linker edges from record["topology_links"] if provided
    for link in record.get("topology_links", []) or []:
        i = int(link["site_i"]) - 1
        j = int(link["site_j"]) - 1
        if 0 <= i < length and 0 <= j < length:
            typ = RESIDUE_EDGE_TYPES.get(str(link.get("link_type")), RESIDUE_EDGE_TYPES["sidechain_linker"])
            adj[i, j] = adj[j, i] = 1.0
            edge_type[i, j] = edge_type[j, i] = typ

    distance = shortest_path_distance(adj)
    return {
        "residue_adj": adj,
        "residue_edge_type": edge_type,
        "residue_distance": distance,
    }
```

### `collate_cedg` 增加输出

```text
residue_adj:        [B, L, L]
residue_edge_type:  [B, L, L]
residue_distance:   [B, L, L]
```

---

## A7. 用 `chem_edit.py` 替代当前 curated payload graph 主路径

当前 `payload_delta_graphs(event, monomer_smiles)` 返回：

```text
old_graph, new_graph
```

建议改为：

```python
from .chem_edit import build_attachment_aware_edit_graph

"payload_graphs": [
    build_attachment_aware_edit_graph(event, self.vocabs.monomer_smiles)
    for event in edit_set
]
```

同时保留旧函数，但改名：

```python
payload_delta_graphs_curated_fallback()
```

---

## A8. 修改 graph collate 输出

理想 `EditGraph` 不只有 old/new graph，还要有 delta graph、atom action、atom role、edge feature、attachment index、graph mode。

### `collate_cedg` 新增 tensor

```text
payload_old_atom_features        [B, K, A_old, F_atom]
payload_new_atom_features        [B, K, A_new, F_atom]
payload_delta_atom_features      [B, K, A_delta, F_atom]

payload_old_adjacency            [B, K, A_old, A_old]
payload_new_adjacency            [B, K, A_new, A_new]
payload_delta_edge_features      [B, K, A_delta, A_delta, F_edge]

payload_old_atom_mask            [B, K, A_old]
payload_new_atom_mask            [B, K, A_new]
payload_delta_atom_mask          [B, K, A_delta]

payload_atom_action              [B, K, A_delta]
payload_atom_role                [B, K, A_delta]
payload_attachment_index_old     [B, K]
payload_attachment_index_new     [B, K]
payload_attachment_index_delta   [B, K]
payload_graph_mode               [B, K]
payload_graph_valid_mask         [B, K]
```

### 为什么要这样改

只用 `modified - original` 会让模型知道“变化了”，但不知道：

```text
哪个 atom 是 added
哪个 atom 是 deleted
哪个 atom 是 core
哪个 bond 是 attachment bond
哪个 atom 是 backbone_N / alpha_C / sidechain atom
这个 edit 是 MCS mapping 得到的，还是 curated fallback
```

这些信息应该显式进入模型。

---

## A9. K=0 / absolute-only 样本处理

当前 `collate_cedg` 用：

```python
max_edits = max(len(item["edit_cat"]) for item in batch)
```

如果 batch 中存在 `edit_set=[]`，模型后续 attention 可能出错。

### 两种处理方式

#### 方案 1：主训练禁止 K=0

在 `CEDGDataset.__getitem__` 或数据导出时检查：

```python
if len(edit_set) == 0 and not allow_absolute_only:
    raise ValueError("CEDGScoreModel requires at least one edit event per sample.")
```

#### 方案 2：支持 null edit

新增：

```text
edit_scope = null_edit
operation = none
attachment = none
chemical_payload_type = none
payload graph = empty graph
edit_mask = True for one pseudo edit
label_mask_delta = False if no property_before
label_mask_property = True
```

同时 `losses.py` 使用 label mask。

---

# Part B. 新增 `chem_edit.py`：自动 attachment-aware ΔG graph

---

## B1. 为什么必须新增 `chem_edit.py`

当前 `data.py` 里有：

```text
FRAGMENT_SMILES
payload_parts()
fragment_smiles()
payload_delta_graphs()
```

这说明当前 graph 来源主要是 curated string → hand-written fragment SMILES。理想版本应支持：

```text
old residue graph
new residue graph
atom mapping old -> new
core / added / deleted / changed atom mask
attachment atom / attachment bond
local delta graph
full residue graph fallback
```

这些逻辑太复杂，不应该继续塞在 `data.py`。建议单独放进 `chem_edit.py`。

---

## B2. 新增 dataclass

```python
from dataclasses import dataclass
import torch

@dataclass
class EditGraph:
    old_atom_features: torch.Tensor
    old_adjacency: torch.Tensor
    old_atom_mask: torch.Tensor

    new_atom_features: torch.Tensor
    new_adjacency: torch.Tensor
    new_atom_mask: torch.Tensor

    delta_atom_features: torch.Tensor
    delta_edge_features: torch.Tensor
    delta_atom_mask: torch.Tensor

    atom_action: torch.Tensor
    atom_role: torch.Tensor

    attachment_index_old: int
    attachment_index_new: int
    attachment_index_delta: int

    anchor_index_old: int
    anchor_index_new: int

    valid: bool
    mode: str
    error: str | None = None
```

### atom_action taxonomy

```python
ATOM_ACTIONS = {
    "none": 0,
    "core_unchanged": 1,
    "added": 2,
    "deleted": 3,
    "atom_changed": 4,
    "bond_changed": 5,
    "stereo_changed": 6,
    "charge_changed": 7,
}
```

### atom_role taxonomy

```python
ATOM_ROLES = {
    "unknown": 0,
    "backbone_N": 1,
    "alpha_C": 2,
    "carbonyl_C": 3,
    "carbonyl_O": 4,
    "sidechain": 5,
    "r_group": 6,
    "linker": 7,
    "attachment_dummy": 8,
    "attachment_atom": 9,
}
```

### edge feature taxonomy

```python
EDGE_FEATURES = [
    "bond_exists",
    "bond_order_single",
    "bond_order_double",
    "bond_order_triple",
    "bond_aromatic",
    "old_only",
    "new_only",
    "common",
    "bond_order_changed",
    "is_attachment_bond",
    "is_ring_bond",
]
```

---

## B3. 主入口函数

```python
def build_attachment_aware_edit_graph(
    event: dict[str, object],
    monomer_smiles: dict[str, str],
    radius: int = 2,
    mcs_timeout: int = 10,
) -> EditGraph:
    """
    Priority:
    1. explicit atom-mapped old/new residue graph
    2. old/new full residue graph + MCS mapping
    3. explicit r_old_smiles / r_new_smiles with dummy attachment atom
    4. chirality-only flag graph
    5. curated payload fallback
    """
    for builder in (
        try_atom_mapped_residue_delta,
        try_mcs_residue_delta,
        try_r_group_dummy_attachment,
        try_stereo_flag_graph,
        try_curated_fragment_fallback,
    ):
        graph = builder(event, monomer_smiles, radius=radius, mcs_timeout=mcs_timeout)
        if graph.valid:
            return graph
    return empty_edit_graph(mode="failed", error="all builders failed")
```

---

## B4. Builder 1：atom-mapped residue delta

如果数据能提供 atom-mapped old/new SMILES，优先用它。

### 输入字段建议

```text
old_residue_atom_mapped_smiles
new_residue_atom_mapped_smiles
old_full_residue_smiles
new_full_residue_smiles
attachment_atom_map_num
anchor_atom_map_num
```

### 处理逻辑

```text
1. parse old/new mol
2. read atom map numbers
3. map old atom -> new atom by map number
4. detect added / deleted / changed atoms
5. detect changed bonds
6. expand around affected atoms by radius
7. build delta graph with atom_action and atom_role
8. locate attachment atom
```

---

## B5. Builder 2：MCS residue delta

当没有 atom map，但有 old/new full residue graph 时，用 MCS 自动找 common core。

### 伪代码

```python
def find_strict_mcs_mapping(old_mol, new_mol, timeout: int = 10):
    from rdkit.Chem import rdFMCS

    params = rdFMCS.MCSParameters()
    params.Timeout = timeout
    params.AtomCompareParameters.MatchChiralTag = True
    params.AtomCompareParameters.MatchFormalCharge = True
    params.AtomCompareParameters.MatchValences = True
    params.BondCompareParameters.RingMatchesRingOnly = True
    params.BondCompareParameters.CompleteRingsOnly = True
    params.AtomTyper = rdFMCS.AtomCompare.CompareElements
    params.BondTyper = rdFMCS.BondCompare.CompareOrderExact

    result = rdFMCS.FindMCS([old_mol, new_mol], params)
    if result.canceled or not result.smartsString:
        return None

    query = Chem.MolFromSmarts(result.smartsString)
    old_match = old_mol.GetSubstructMatch(query)
    new_match = new_mol.GetSubstructMatch(query)
    if not old_match or not new_match:
        return None

    return dict(zip(old_match, new_match))
```

### 注意事项

MCS 对以下情况可能不稳定：

```text
对称芳环
重复取代基
长链 linker
large scaffold rearrangement
stereochemistry-only edit
ring opening / ring closure
```

所以需要：

```text
attachment_type constraint
anchor atom constraint
fallback mode tracking
coverage report
manual inspect for failure modes
```

---

## B6. Builder 3：R-group dummy attachment graph

适合：

```text
Phe para-H -> para-F
Phe para-H -> para-CF3
Ser OH-H -> O-tBu
Lys sidechain NH2 -> guanidino
```

理想表示：

```text
old R: [*:1][H]
new R: [*:1]F

old R: [*:1][H]
new R: [*:1]C(F)(F)F
```

### 必须显式保存

```text
dummy attachment atom [*:1]
attachment_index
attachment_bond_type
attachment_type_id
r_old_graph
r_new_graph
```

---

## B7. Builder 4：stereochemistry-only graph

适合：

```text
L_to_D
D_to_L
```

不要伪造 atom deletion/addition。建议：

```text
mode = stereo_flag
old/new graph = alpha carbon local graph if available
atom_action = stereo_changed on alpha_C
payload_features includes L_to_D / D_to_L
```

如果没有 full residue graph，则用 empty graph + explicit flag。

---

## B8. Builder 5：curated fallback

保留当前 `FRAGMENT_SMILES` 方案作为 fallback。

```python
def try_curated_fragment_fallback(event, monomer_smiles, **kwargs) -> EditGraph:
    old_graph, new_graph = payload_delta_graphs_curated_fallback(event, monomer_smiles)
    return make_edit_graph_from_old_new_fragments(
        old_graph,
        new_graph,
        mode="curated_fallback",
    )
```

论文中要区分：

```text
mapped_delta
mcs_delta
r_group_dummy
full_residue_fallback
stereo_flag
curated_fallback
failed
```

---

## B9. Payload coverage report

新增脚本：

```text
scripts/export_payload_coverage_report.py
```

输出：

```text
n_events
n_valid_graph
coverage_by_mode
coverage_by_edit_scope
coverage_by_operation
coverage_by_attachment
failed_examples_top50
fallback_rate
stereo_flag_rate
full_residue_fallback_rate
curated_fallback_rate
```

这是论文里支撑 “atom-level ΔG graph” claim 的关键补充材料。

---

# Part C. `model.py` 修改清单

---

## C1. 把当前 `PeptideContextEncoder` 改名保留为 ablation

当前：

```python
class PeptideContextEncoder(nn.Module):
    ...
```

建议改名：

```python
class LearnedPeptideContextEncoder(nn.Module):
    """Ablation encoder: learned shadow embedding + Transformer."""
```

这样可以做：

```text
No PLM / learned sequence encoder only
```

---

## C2. 新增 `PLMPeptideContextEncoder`

### 建议接口

```python
class PLMPeptideContextEncoder(nn.Module):
    def __init__(
        self,
        plm_name: str,
        hidden_dim: int,
        residue_feature_dim: int,
        dropout: float,
        freeze_plm: bool = True,
        unfreeze_last_n_layers: int = 0,
        use_topology_encoder: bool = True,
        num_residue_edge_types: int = 10,
    ) -> None:
        ...

    def forward(
        self,
        plm_input_ids: torch.Tensor,
        plm_attention_mask: torch.Tensor,
        plm_lengths: torch.Tensor,
        residue_features: torch.Tensor,
        shadow_mask: torch.Tensor,
        residue_adj: torch.Tensor | None = None,
        residue_edge_type: torch.Tensor | None = None,
        residue_distance: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ...
```

### 内部结构

```text
AutoModel.from_pretrained(plm_name)
PLM output token_hidden
special-token alignment
plm_proj
residue_feature_proj
fusion MLP
optional ResidueTopologyEncoder
masked pool -> scaffold embedding
```

### 关键：PLM residue alignment

通常需要跳过第一个 special token：

```python
for b in range(batch_size):
    L = int(plm_lengths[b].item())
    L = min(L, max_len)
    plm_residue[b, :L] = token_hidden[b, 1 : 1 + L]
```

需要在 `test_plm_alignment.py` 中测试：

```text
sequence length = L
residue_context shape = [B, L_pad, H]
shadow_mask 与 plm_lengths 对齐
special token 不进入 residue index
```

---

## C3. 新增 `ResidueTopologyEncoder`

```python
class ResidueTopologyEncoder(nn.Module):
    def __init__(self, hidden_dim: int, num_edge_types: int, dropout: float, num_layers: int = 2):
        super().__init__()
        self.edge_embedding = nn.Embedding(num_edge_types, hidden_dim)
        self.layers = nn.ModuleList([
            ResidueTopologyBlock(hidden_dim, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, residue, residue_adj, residue_edge_type, mask):
        for layer in self.layers:
            residue = layer(residue, residue_adj, residue_edge_type, mask)
        return residue
```

### 简单 dense message passing block

```python
class ResidueTopologyBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.edge_embedding = None  # 或由外部传入
        self.message = nn.Linear(hidden_dim * 3, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, residue, residue_adj, edge_emb, mask):
        B, L, H = residue.shape
        src = residue.unsqueeze(2).expand(B, L, L, H)
        dst = residue.unsqueeze(1).expand(B, L, L, H)
        msg = self.message(torch.cat([src, dst, edge_emb], dim=-1))
        msg = msg * residue_adj.unsqueeze(-1)
        deg = residue_adj.sum(dim=-1, keepdim=True).clamp_min(1.0)
        update = msg.sum(dim=2) / deg
        out = self.norm(residue + self.dropout(torch.relu(update)))
        return out * mask.unsqueeze(-1).float()
```

### 作用

这个模块让模型显式知道：

```text
residue i 和 residue j 是否相邻
是否通过 head-to-tail cyclization 相连
是否通过 disulfide/lactam/thioether/staple 相连
site 间 cyclic distance
```

---

## C4. 修改 `CEDGScoreModel.__init__`

新增配置：

```python
class CEDGScoreModel(nn.Module):
    def __init__(
        self,
        vocabs: CEDGVocabs,
        emb_dim: int = 64,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        peptide_encoder_type: str = "plm",  # "plm" | "learned"
        plm_name: str = "facebook/esm2_t12_35M_UR50D",
        freeze_plm: bool = True,
        unfreeze_last_n_layers: int = 0,
        use_topology_graph: bool = True,
        use_delta_graph: bool = True,
        ...,
    ):
        ...
```

选择 encoder：

```python
if peptide_encoder_type == "plm":
    self.peptide = PLMPeptideContextEncoder(...)
elif peptide_encoder_type == "learned":
    self.peptide = LearnedPeptideContextEncoder(...)
else:
    raise ValueError(f"Unknown peptide_encoder_type: {peptide_encoder_type}")
```

---

## C5. 修改 `CEDGScoreModel.forward`

PLM path：

```python
if self.peptide_encoder_type == "plm":
    residue_context, peptide_context = self.peptide(
        batch["plm_input_ids"],
        batch["plm_attention_mask"],
        batch["plm_lengths"],
        batch["residue_features"],
        batch["shadow_mask"],
        batch.get("residue_adj"),
        batch.get("residue_edge_type"),
        batch.get("residue_distance"),
    )
else:
    residue_context, peptide_context = self.peptide(
        batch["shadow_ids"],
        batch["shadow_mask"],
        batch["residue_features"],
    )
```

---

## C6. 修改 `SiteConditionedAtomEditEncoder`

当前逻辑：

```text
z_old = graph_encoder(old)
z_new = graph_encoder(new)
delta = z_new - z_old
gated_delta = delta * gate(site_context)
```

建议升级为：

```text
z_old = old_encoder(old_graph, site_context)
z_new = new_encoder(new_graph, site_context)
z_delta = delta_encoder(delta_graph, atom_action, atom_role, edge_features, site_context)
z_attach = attachment_pool(delta_nodes, attachment_index)
z_change = z_new - z_old
fusion([site_context, z_old, z_new, z_change, z_delta, z_attach, flags, graph_mode_emb])
```

### 新 forward signature

```python
def forward(
    self,
    payload_old_atom_features,
    payload_new_atom_features,
    payload_delta_atom_features,
    payload_old_adjacency,
    payload_new_adjacency,
    payload_delta_edge_features,
    payload_old_atom_mask,
    payload_new_atom_mask,
    payload_delta_atom_mask,
    payload_atom_action,
    payload_atom_role,
    payload_attachment_index_delta,
    payload_graph_mode,
    payload_graph_valid_mask,
    payload_features,
    site_context,
) -> torch.Tensor:
    ...
```

---

## C7. 新增 `DeltaGraphEncoder`

```python
class DeltaGraphEncoder(nn.Module):
    def __init__(
        self,
        atom_feature_dim: int,
        edge_feature_dim: int,
        hidden_dim: int,
        num_atom_actions: int,
        num_atom_roles: int,
        dropout: float,
        num_layers: int = 3,
    ):
        super().__init__()
        self.atom_proj = nn.Linear(atom_feature_dim, hidden_dim)
        self.action_emb = nn.Embedding(num_atom_actions, hidden_dim)
        self.role_emb = nn.Embedding(num_atom_roles, hidden_dim)
        self.edge_proj = nn.Linear(edge_feature_dim, hidden_dim)
        self.layers = nn.ModuleList([...])
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, atom_features, edge_features, atom_mask, atom_action, atom_role, site_context):
        ...
```

---

## C8. Edge-aware graph message passing

当前 `GraphConvBlock` 只用 adjacency，没有 edge type。建议新增：

```python
class EdgeAwareGraphBlock(nn.Module):
    def __init__(self, hidden_dim: int, edge_dim: int, dropout: float):
        super().__init__()
        self.message = nn.Linear(hidden_dim * 2 + edge_dim + hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, node, edge, node_mask, site_context):
        B, A, H = node.shape
        src = node.unsqueeze(2).expand(B, A, A, H)
        dst = node.unsqueeze(1).expand(B, A, A, H)
        site = site_context[:, None, None, :].expand(B, A, A, H)
        msg = self.message(torch.cat([src, dst, edge, site], dim=-1))
        pair_mask = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
        msg = msg * pair_mask.unsqueeze(-1).float()
        deg = pair_mask.sum(dim=2, keepdim=True).clamp_min(1).float()
        update = msg.sum(dim=2) / deg
        return self.norm(node + self.dropout(torch.relu(update)))
```

---

## C9. 修改 `EditEventEncoder.site_proj`

当前：

```python
self.site_proj = nn.Sequential(nn.Linear(2, emb_dim), nn.GELU())
```

只用了：

```text
site_scaled
sin(site_scaled)
```

建议改成 6-8 维：

```text
site_scaled
site_sin
site_cos
site_degree
site_is_cyclization_anchor
site_min_distance_to_other_edit
site_min_cyclic_distance_to_linker
site_payload_graph_valid
```

修改：

```python
self.site_proj = nn.Sequential(
    nn.Linear(SITE_FEATURE_DIM, emb_dim),
    nn.LayerNorm(emb_dim),
    nn.GELU(),
)
```

---

## C10. 修改 `EditSetEncoder` 支持 pairwise interaction bias

当前 `EditSetEncoder` 是 transformer + mean/sum。建议保留，但加可选 pairwise features：

```text
site distance between edits
same operation flag
same attachment flag
same edit_scope flag
cyclic distance bias
```

第一版可以先加 additive interaction MLP：

```python
single = masked_mean(encoded, edit_mask, dim=1)
pairwise = compute_pairwise_edit_interaction(encoded, edit_site, edit_mask, residue_distance)
z_E = MLP([single, pairwise, encoded.sum(dim=1)])
```

这样能更明确地建模多点 edit 协同/冲突。

---

## C11. Prediction heads 修改

当前输出保留：

```text
delta
property_after
ranking_score
direction_logit
log_variance
uncertainty
```

建议新增可选辅助头：

```text
operation_aux_head
edit_scope_aux_head
payload_type_aux_head
graph_mode_aux_head
site_effect_aux_head
```

这些辅助头不是必须进主结果，但能帮助表示学习，也能做诊断。

---

# Part D. `losses.py` 修改清单

---

## D1. 使用 loss config dataclass

当前 loss 权重写死：

```python
return delta_loss + 0.3 * property_loss + 0.1 * ranking_loss + ...
```

建议：

```python
@dataclass
class CEDGLossConfig:
    delta_weight: float = 1.0
    property_weight: float = 0.3
    ranking_pointwise_weight: float = 0.1
    ranking_pairwise_weight: float = 0.35
    ranking_listwise_weight: float = 0.0
    direction_weight: float = 0.2
    calibration_weight: float = 0.0
    aux_weight: float = 0.0
```

---

## D2. 增加 label mask

支持 absolute-only / censored / weak-label 时需要：

```text
property_label_mask
delta_label_mask
direction_label_mask
ranking_label_mask
```

修改：

```python
property_loss = masked_weighted_mse(..., mask=batch["property_label_mask"])
delta_loss = masked_heteroscedastic_mse(..., mask=batch["delta_label_mask"])
```

---

## D3. 增加 listwise ranking loss

pairwise loss 保留，但 candidate ranking 推荐加 listwise softmax：

```python
def group_listwise_softmax_loss(score, target, group_index, weight, temperature=1.0):
    losses = []
    for group in torch.unique(group_index):
        idx = torch.nonzero(group_index == group, as_tuple=False).flatten()
        if idx.numel() < 2:
            continue
        y = target[idx]
        s = score[idx] / temperature
        y_prob = torch.softmax(y, dim=0)
        log_p = torch.log_softmax(s, dim=0)
        losses.append(-(y_prob * log_p).sum())
    if not losses:
        return score.sum() * 0.0
    return torch.stack(losses).mean()
```

---

## D4. Ranking batch 问题

当前 `group_pairwise_ranking_loss` 只在一个 batch 内找同组候选。如果 DataLoader 随机打散，很多 group 在 batch 内只有一个样本，ranking loss 会经常为 0。

必须配合：

```text
CandidateGroupBatchSampler
```

或者在 collate 前保证同一 `candidate_group_id` 的多个候选进入同一 batch。

---

## D5. Uncertainty / calibration loss

当前 heteroscedastic NLL 可以保留。可以新增：

```text
calibration regularization
variance floor penalty
selective risk objective
OOD auxiliary classification loss, if OOD labels exist
```

不要在没有 OOD labels 或 hard split evidence 时强称 OOD uncertainty。

---

# Part E. `metrics.py` 修改清单

---

## E1. Ranking metrics 扩展

当前已有：

```text
ranking_pairwise_accuracy
ranking_ndcg_at_5
```

建议新增：

```text
hit@1
hit@3
hit@5
mean_reciprocal_rank
candidate_top1_delta_regret
candidate_topk_enrichment
spearman_by_group_mean
ndcg@1 / ndcg@3 / ndcg@10
```

### top-k regret

```python
def topk_regret(groups, y_true, score, k=1):
    # ideal best y - best y among predicted top-k
    ...
```

---

## E2. Uncertainty metrics 扩展

当前只有：

```text
uncertainty_error_spearman
mean_uncertainty
```

建议新增：

```text
negative_log_likelihood
calibration_curve_bins
expected_calibration_error_regression
coverage@1sigma
coverage@2sigma
selective_risk_auc
risk_at_coverage_80
risk_at_coverage_50
OOD AUROC, if OOD labels exist
```

---

## E3. Split-specific metrics

新增函数：

```python
def metrics_by_group(metadata_field: list[str], ...):
    ...
```

输出：

```text
metrics_by_edit_scope
metrics_by_operation
metrics_by_payload_type
metrics_by_graph_mode
metrics_by_scaffold_split
metrics_by_rgroup_seen_unseen
metrics_by_payload_coverage_mode
```

这对论文非常重要，因为模型可能总体不错，但在 `full_residue_fallback` 或 `stereo_flag` 上性能很差。

---

# Part F. `evaluation.py` 修改清单

---

## F1. 保存 prediction table

当前 `evaluate()` 只返回 metrics。建议新增：

```python
@torch.no_grad()
def predict(model, loader, device) -> pd.DataFrame:
    ...
```

保存字段：

```text
sample_id
source_id
candidate_group_id
edit_scope
operation
attachment
payload_type
graph_mode
property_after_true
property_after_pred
delta_true
delta_pred
ranking_score
direction_prob
uncertainty
split
scaffold_id
rgroup_split_group
```

---

## F2. Bootstrap confidence interval

新增：

```python
def bootstrap_metrics(pred_df, metric_fn, n_boot=1000, group_col=None):
    ...
```

论文中报告：

```text
MAE ± CI
Spearman ± CI
NDCG@5 ± CI
```

---

# Part G. 新增 `samplers.py`

---

## G1. CandidateGroupBatchSampler

目的：让同一 `candidate_group_id` 的多个候选进入同一 batch，否则 groupwise ranking loss 经常没有有效 pair。

```python
class CandidateGroupBatchSampler(torch.utils.data.Sampler[list[int]]):
    def __init__(self, dataset, batch_size: int, shuffle: bool = True, drop_last: bool = False):
        self.group_to_indices = build_group_index(dataset)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self):
        groups = list(self.group_to_indices)
        if self.shuffle:
            random.shuffle(groups)
        batch = []
        for group in groups:
            indices = self.group_to_indices[group]
            if self.shuffle:
                random.shuffle(indices)
            for idx in indices:
                batch.append(idx)
            if len(batch) >= self.batch_size:
                yield batch[: self.batch_size]
                batch = batch[self.batch_size :]
        if batch and not self.drop_last:
            yield batch
```

更强版本：每个 batch 包含若干完整 candidate groups，必要时动态 batch size。

---

# Part H. `utils.py` 修改清单

---

建议新增：

```python
def masked_softmax(logits, mask, dim=-1): ...

def masked_sum(values, mask, dim): ...

def masked_mean(values, mask, dim): ...  # 可从 model.py 移到 utils.py

def safe_to_device(obj, device): ...

def detach_to_numpy(tensor): ...

def seed_worker(worker_id): ...

def count_parameters(model): ...

def freeze_module(module): ...

def unfreeze_last_n_transformer_layers(plm, n): ...
```

---

# Part I. 训练脚本 / config 修改清单

---

## I1. 新增训练参数

```text
--peptide-encoder-type plm|learned
--plm-name facebook/esm2_t12_35M_UR50D
--freeze-plm true|false
--unfreeze-last-n-layers 0|2|4
--plm-lr 1e-5
--base-lr 1e-4
--use-topology-graph true|false
--use-delta-graph true|false
--use-clean-input-only true|false
--ranking-loss pairwise|listwise|both
--group-aware-batching true|false
--loss-config configs/loss.yaml
--save-predictions true
--save-payload-coverage true
```

---

## I2. Optimizer param groups

```python
plm_params = []
base_params = []
for name, p in model.named_parameters():
    if not p.requires_grad:
        continue
    if name.startswith("peptide.plm"):
        plm_params.append(p)
    else:
        base_params.append(p)

optimizer = torch.optim.AdamW([
    {"params": base_params, "lr": args.base_lr, "weight_decay": args.weight_decay},
    {"params": plm_params, "lr": args.plm_lr, "weight_decay": args.weight_decay},
])
```

---

## I3. 推荐训练模式

```text
mode 0: smoke test
    learned encoder, no PLM, no delta graph, tiny batch

mode 1: PLM frozen
    frozen PLM + existing payload graph

mode 2: PLM frozen + topology graph
    add residue topology encoder

mode 3: PLM frozen + topology graph + attachment-aware ΔG graph
    full first-stage model

mode 4: PLM last layers unfrozen
    unfreeze last 2 or 4 layers only, smaller LR
```

---

# Part J. README / package documentation 修改

---

## J1. `README.md` 应更新 claim

当前 README 需要拆成：

```text
Current MVP
Target full CEDG-Score
Not implemented yet
```

### 修改后建议写法

```text
This package implements CEDG-Score, a library-constrained context-conditioned edit scorer/ranker for cyclic peptide permeability optimization.

The full model supports:
- PLM-based canonical shadow sequence encoding;
- topology-aware residue graph fusion for cyclic peptides;
- site-conditioned local chemical edit encoding;
- attachment-aware atom-level edit difference graphs when old/new residue graphs or R-group graphs are available;
- fallback to curated payload fragment graphs for ambiguous edits.

This package does not directly implement open-vocabulary R-group generation unless the generator module is explicitly enabled.
```

---

## J2. `__init__.py` 更新

```python
from .data import CEDGDataset, CEDGCollatorWithPLM, build_vocabs, collate_cedg
from .chem_edit import build_attachment_aware_edit_graph, EditGraph
from .model import CEDGScoreModel
from .losses import compute_cedg_loss
from .evaluation import evaluate, predict
```

---

# Part K. Tests 修改清单

---

## K1. `test_data_collate.py`

必须测试：

```text
K=1 edit
K>1 edit set
K=0 null edit or explicit failure
empty graph
invalid SMILES
unknown payload token
unseen payload maps to UNK
batch with mixed sequence length
batch with mixed edit count
residue topology tensors pad correctly
payload delta tensors pad correctly
```

---

## K2. `test_plm_alignment.py`

必须测试：

```text
plm_sequence length == number of shadow tokens
PLM special tokens are excluded from residue embeddings
site_index selects correct residue embedding
padding positions are masked
pseudo-anchor maps to valid one-letter fallback
```

---

## K3. `test_chem_edit.py`

必须测试：

```text
H_to_F R-group edit
N-H_to_N-CH3 backbone edit
L_to_D stereo edit
full_residue_graph fallback
invalid SMILES fallback
symmetric aromatic MCS case
attachment dummy exists
atom_action contains added/deleted/core
atom_role contains attachment atom
```

---

## K4. `test_topology.py`

必须测试：

```text
linear peptide sequence edges
head-to-tail cyclic edge
sidechain linker edge
cyclic shortest path distance
site distance features
residue edge type padding
```

---

## K5. `test_model_forward.py`

必须测试：

```text
learned encoder forward
PLM frozen forward, if transformers available
use_topology_graph true/false
use_delta_graph true/false
batch with K=1
batch with K>1
outputs contain delta/property/ranking/direction/log_variance/uncertainty
loss backward works
```

---

# Part L. Ablation configs

---

你至少需要这些 ablation 配置：

```text
A0 current MVP
A1 learned shadow Transformer only
A2 PLM only, no chemical payload
A3 chemical payload only, no PLM context
A4 PLM + payload tokens, no atom graph
A5 PLM + curated fragment graph
A6 PLM + attachment-aware ΔG graph
A7 no topology
A8 topology features only, no topology graph
A9 topology graph full
A10 concat fusion instead of site-conditioned atom encoding
A11 no edit_scope
A12 no operation
A13 no attachment_type
A14 single-edit additive pooling vs EditSetEncoder
A15 clean-input-only vs metadata-included
```

---

# Part M. 论文 claim 与代码支持关系

---

## M1. 当前代码能支持的 claim

```text
library-constrained context-conditioned edit scorer/ranker
parent shadow sequence + edit set structured representation
site-conditioned payload token and fragment graph encoding
property / delta / ranking / uncertainty multi-head prediction
```

---

## M2. 加 PLM 后能支持的 claim

```text
pretrained PLM-based peptide scaffold context encoder
canonical shadow sequence provides scaffold context
noncanonical chemistry is modeled separately through edit payload branch
```

前提：

```text
PLMPeptideContextEncoder implemented
PLM tokenizer/collator implemented
PLM alignment tested
PLM ablation included
```

---

## M3. 加 attachment-aware ΔG graph 后能支持的 claim

```text
attachment-aware atom-level edit difference graph representation
local old/new residue or R-group graphs are compared by atom mapping or MCS
added/deleted/changed atoms and attachment atoms are explicitly encoded
fallback to full residue graph or curated fragment graph for ambiguous edits
```

前提：

```text
chem_edit.py implemented
EditGraph dataclass implemented
collate supports delta graph tensors
DeltaGraphEncoder implemented
payload coverage report included
```

---

## M4. 加 topology graph 后能支持的 claim

```text
topology-aware cyclic peptide context encoder
residue-level graph captures sequence edges, cyclization edges, and linker edges
cyclic site distances are used for edit context and edit-set interactions
```

前提：

```text
residue_topology() implemented
residue_adj / edge_type / distance tensors implemented
ResidueTopologyEncoder implemented
site features include cyclic distance
ablation no-topology included
```

---

## M5. 暂时不能强 claim 的内容

除非额外实现，否则不要写：

```text
open-vocabulary R-group generation
fully automated ncAA decomposition for all residues
3D conformational risk modeling
validated OOD uncertainty
universal permeability predictor across assay types
```

可以写成：

```text
The scorer is designed to be compatible with future local R-group / ΔG generation.
```

---

# Part N. 推荐实施顺序

---

## Round 0：安全修补

```text
[ ] 拆分 EDIT_INPUT_FIELDS / EDIT_QC_FIELDS
[ ] 拆分 SAMPLE_INPUT_FIELDS / SAMPLE_QC_FIELDS
[ ] 主模型默认 clean-input-only
[ ] K=0 样本显式处理
[ ] group-aware sampler 初版
[ ] README claim 修正
```

---

## Round 1：PLM 最小闭环

```text
[ ] shadow_to_plm_sequence()
[ ] CEDGCollatorWithPLM
[ ] LearnedPeptideContextEncoder 重命名保留
[ ] PLMPeptideContextEncoder
[ ] CEDGScoreModel 支持 peptide_encoder_type
[ ] train script 增加 plm args
[ ] optimizer param groups
[ ] test_plm_alignment.py
[ ] ablation: learned vs frozen PLM
```

完成后可合理表述：

```text
The peptide scaffold is encoded by a pretrained PLM using the canonical shadow sequence.
```

---

## Round 2：Topology-aware context

```text
[ ] RESIDUE_FEATURE_DIM 扩展到 12+
[ ] TOPOLOGY_FEATURE_DIM 扩展到 24+
[ ] residue_topology()
[ ] collate residue_adj / residue_edge_type / residue_distance
[ ] ResidueTopologyEncoder
[ ] EditEventEncoder site feature 扩展
[ ] ablation: no topology / feature-only / graph topology
[ ] test_topology.py
```

完成后可合理表述：

```text
The PLM residue embeddings are fused with cyclic topology-aware residue graph features.
```

---

## Round 3：Attachment-aware ΔG graph

```text
[ ] 新建 chem_edit.py
[ ] EditGraph dataclass
[ ] atom-mapped residue delta builder
[ ] MCS residue delta builder
[ ] R-group dummy attachment builder
[ ] stereo flag graph builder
[ ] curated fallback builder
[ ] payload coverage report
[ ] collate delta graph tensors
[ ] DeltaGraphEncoder
[ ] EdgeAwareGraphBlock
[ ] SiteConditionedAtomEditEncoder forward signature 更新
[ ] test_chem_edit.py
[ ] ablation: curated graph vs attachment-aware ΔG graph
```

完成后可合理表述：

```text
Local edits are represented as attachment-aware atom-level edit difference graphs.
```

---

## Round 4：Ranking 和 uncertainty 完整化

```text
[ ] CandidateGroupBatchSampler
[ ] listwise ranking loss
[ ] hit@k / regret / enrichment metrics
[ ] prediction table export
[ ] uncertainty calibration metrics
[ ] selective risk metrics
[ ] split-specific metrics
[ ] bootstrap CI
```

完成后可合理表述：

```text
The model supports candidate edit ranking under parent-scaffold candidate groups.
```

---

## Round 5：论文复现实验包

```text
[ ] configs for all ablations
[ ] scripts/run_ablation.py
[ ] scripts/export_payload_coverage_report.py
[ ] scripts/evaluate_cedg_score.py
[ ] prediction CSV export
[ ] model card / README
[ ] unit tests
[ ] seed control
[ ] environment file
```

---

# Part O. 最终 acceptance checklist

---

## O1. PLM checklist

```text
[ ] `plm_sequence` exists in each dataset item
[ ] tokenizer output exists in batch
[ ] PLM special token alignment tested
[ ] frozen PLM forward works
[ ] unfreeze_last_n_layers works
[ ] optimizer has separate PLM lr
[ ] ablation: learned encoder vs PLM
```

---

## O2. ΔG graph checklist

```text
[ ] old/new residue or R-group graph can be parsed
[ ] atom mapping or MCS path exists
[ ] attachment dummy / attachment atom index exists
[ ] atom_action exists
[ ] atom_role exists
[ ] edge features exist
[ ] full residue fallback exists
[ ] stereo-only edit does not fake atom graph change
[ ] curated fallback retained
[ ] payload coverage report generated
[ ] graph mode enters model or evaluation metadata
```

---

## O3. Topology checklist

```text
[ ] topology_features > 4 dims
[ ] residue_features > 4 dims
[ ] residue_adj exists
[ ] residue_edge_type exists
[ ] residue_distance exists
[ ] head-to-tail cyclic edge represented
[ ] sidechain/linker edge represented when available
[ ] site features include sin and cos
[ ] site features include cyclic distance to other edits
[ ] topology ablation exists
```

---

## O4. Ranking checklist

```text
[ ] candidate_group_id preserved
[ ] group-aware sampler implemented
[ ] pairwise ranking loss has nonzero valid pairs
[ ] listwise ranking optional
[ ] hit@k / NDCG@k / regret metrics implemented
[ ] prediction table supports case study
```

---

## O5. Publication readiness checklist

```text
[ ] clean-input-only main result
[ ] metadata-included only as ablation or diagnostic
[ ] random split only sanity check
[ ] scaffold split result
[ ] R-group split result
[ ] monomer family split result
[ ] source/assay split or external validation if available
[ ] PLM-only baseline
[ ] chemistry-only baseline
[ ] whole-SMILES / fingerprint baseline
[ ] tokenized ncAA baseline
[ ] no-topology ablation
[ ] no-payload graph ablation
[ ] no-edit_scope ablation
[ ] concat vs site-conditioned fusion ablation
[ ] payload coverage report
[ ] top-k recommendation case study
[ ] uncertainty calibration report
```

---

# Part P. 最小可修改版本：如果你只想先快速升级

---

如果只做最少但最关键的改动，建议按下面 6 个 PR 做：

## PR 1：clean input + K=0 handling

```text
files:
  data.py
  model.py
  losses.py

changes:
  split input/QC fields
  default clean-input-only
  explicit K=0 handling
```

## PR 2：PLM branch

```text
files:
  data.py
  model.py
  train_cedg_score.py
  tests/test_plm_alignment.py

changes:
  plm_sequence
  CEDGCollatorWithPLM
  PLMPeptideContextEncoder
  optimizer param groups
```

## PR 3：Topology graph

```text
files:
  data.py or topology.py
  model.py
  tests/test_topology.py

changes:
  residue_topology
  residue_adj / edge_type / distance collate
  ResidueTopologyEncoder
```

## PR 4：Attachment-aware ΔG graph

```text
files:
  chem_edit.py
  data.py
  model.py
  tests/test_chem_edit.py

changes:
  EditGraph
  MCS / atom-map / r-group dummy / stereo / curated fallback
  DeltaGraphEncoder
```

## PR 5：Ranking and metrics

```text
files:
  samplers.py
  losses.py
  metrics.py
  evaluation.py

changes:
  group-aware sampler
  listwise ranking loss
  hit@k / regret / enrichment
  prediction export
```

## PR 6：Reproducibility and ablation

```text
files:
  configs/
  scripts/
  README.md
  tests/

changes:
  ablation configs
  payload coverage report
  experiment runner
  README claim update
```

---

# Part Q. 一句话总结

你的代码下一步不是简单“加深模型”，而是补齐三条主干：

```text
1. PLM：让 parent scaffold context 真正来自 pretrained peptide/protein language model。
2. ΔG graph：让 chemical payload 从 curated fragment graph 升级为 attachment-aware atom-level edit difference graph。
3. Topology：让 cyclic peptide 不只是 sequence，而是带环化/linker/cyclic distance 的 residue topology graph。
```

这三条改完后，CEDG-Set 的论文表述才会从：

```text
一个结构化 edit-set scorer MVP
```

升级为：

```text
PLM-conditioned, topology-aware, attachment-aware atom-level local chemical edit scorer/ranker for cyclic peptide optimization
```

