nijuede# Untitled

# CEDG\-Set：面向非天然环肽的上下文条件局部化学编辑框架

## 0\. 一句话概括

本工作不是直接从零生成整条非天然环肽，而是把非天然环肽优化转化为一个更贴近药化迭代的问题：**给定一个已有 parent cyclic peptide scaffold，模型学习应该在哪些位点做哪些局部非天然化学编辑，并预测这些编辑组合对透膜性、溶血、溶解度等性质的影响。**

核心思想是：

```text
Parent peptide scaffold P
+ local chemical edit set E = {e_1, e_2, ..., e_K}
→ modified peptide P'
→ property(P') 或 Δproperty(P → P')
```

其中每个局部编辑事件写作：

```text
e_k = (
  site_k,
  anchor_k,
  edit_scope_k,
  operation_k,
  attachment_k,
  chemical_payload_k
)
```

- `site_k`：第几个残基位置被修改；

- `anchor_k`：该位点对应的 canonical 或 pseudo\-canonical 参照残基，例如 Phe、Trp、Lys\-like；

- `edit_scope_k`：修改发生在哪个结构层级，例如 R\-group、backbone、stereochemistry、linker；

- `operation_k`：具体编辑机制，例如 aromatic substitution、N\-methylation、D/L inversion；

- `attachment_k`：局部化学 payload 接回残基的位置，例如 phenyl\_para、backbone\_N、alpha\_C；

- `chemical_payload_k`：真正的原子级化学变化，例如 H→F、N\-H→N\-CH3、L→D、R\_old→R\_new。

因此，这个工作既不是“把每个非天然氨基酸当作一个新 token”，也不是“把整条环肽当成普通小分子 SMILES”。它采用**混合粒度表示**：整体环肽在 residue level 建模，局部非天然编辑在 atom level 建模。

---

## 1\. 研究背景：为什么需要这个问题定义

非天然环肽 / macrocyclic peptide 是药物发现中很重要的一类分子。它们可以覆盖传统小分子难以覆盖的靶点，同时相比蛋白或抗体又更容易进行化学修饰。但非天然环肽优化有一个核心困难：

> 性质**不是由某一个非天然残基本身决定的**，而是由“局部化学修改 × 肽链上下文 × 环化拓扑 × 目标性质”共同决定的。
> 
> 

例如：

- 同样的 N\-methylation，放在一个暴露的 backbone amide 上可能提升透膜性，但放在关键 hydrogen bond 位点可能破坏构象或结合；

- 同样的 D\-Phe，放在不同 turn 或不同环化拓扑中可能稳定构象，也可能破坏 scaffold；

- 同样的 para\-F 或 para\-CF3，放在一个外向芳香侧链上可能增强膜相互作用，但放在内向疏水核心或结合界面上可能产生不同效果；

- Lys→Orn、Dab、Dap 这类侧链长度和电荷调节，不只是“换成另一个氨基酸”，而是改变 side\-chain reach、charge distribution 和 local polarity。

所以，如果模型只看到：

```text
Phe -> 4-F-Phe
Trp -> N-Me-Trp
L-Phe -> D-Phe
```

它学到的很可能只是“非天然残基名字”或“monomer token”的统计相关性，而不是可迁移的药化规律。

本工作想解决的问题是：

```text
在给定 peptide scaffold 和目标性质时，
某个位点上的某个局部化学编辑到底会产生什么影响？
```

进一步地，当模型能够判断候选编辑后，再扩展到：

```text
在指定 site / anchor / operation 下，能否生成新的 R-group 或 ΔG chemical payload？
```

---

## 2\. 这项工作到底要做什么

### 2\.1 工作目标

本项目目标是建立一个 **Context\-conditioned Edit Difference Graph Set framework，简称 CEDG\-Set**。

它包含三层目标：

### 目标 A：构建 edit\-level 非天然环肽数据集

把原始数据从普通的：

```text
peptide → property
```

整理成：

```text
parent peptide P
+ edit set E
→ modified peptide P'
→ property(P') / Δproperty(P → P') / ranking label
```

这样数据就不仅能训练普通性质预测器，还能训练局部编辑效果模型和候选推荐模型。

### 目标 B：提出 CEDG\-Set 评分模型

模型输入一个 parent peptide 和一个或多个候选局部编辑事件，输出：

```text
predicted property(P')
predicted Δproperty(P → P')
rank score(P, E)
uncertainty / OOD risk
```

其中 peptide scaffold 用 PLM / peptide context encoder 编码，局部 edit 用 atom\-level ΔG / R\-group graph 编码，再通过 site\-conditioned atom\-level fusion 学习上下文与化学 payload 的交互。

### 目标 C：扩展到局部 R\-group / ΔG 生成

在 scorer 稳定后，不再局限于已有非天然氨基酸库，而是在给定：

```text
P, site, anchor, edit_scope, operation, target property
```

的条件下，生成新的 attachment\-aware R\-group 或 ΔG graph，并通过 scorer、uncertainty、合成可行性和轻量 3D 风险进行 reranking。

---

## 3\. 工作整体架构

整体工作分成四个层级：数据层、表示层、模型层和推荐/生成层。

```text
Raw peptide / cyclic peptide data
        ↓
Data construction layer
  peptide_sample_table
  monomer_anchor_table
  edit_pair_table
  edit_event_table
  candidate_edit_table
        ↓
Representation layer
  parent peptide scaffold P
  canonical shadow sequence
  topology features
  edit set E = {e_k}
  atom-level chemical_payload graph
        ↓
Model layer: CEDG-Set
  peptide context encoder
  site-conditioned atom-level edit encoder
  edit set encoder
  prediction / ranking / uncertainty heads
        ↓
Application layer
  library-constrained edit ranking
  candidate edit recommendation
  open-vocabulary R-group / ΔG generation
  uncertainty / synthesis / 3D reranking
```

### 3\.1 第一阶段：库约束 edit scoring / ranking

第一阶段先不要求模型发明新的 R\-group，而是在已有非天然残基 / R\-group / modification library 里做选择。

任务形式：

```text
Input:
  parent peptide scaffold P
  target property y
  candidate edit set E or candidate edit library L

Output:
  score(P, E, y)
  property(P')
  Δproperty(P → P')
  top-k recommended edits
  uncertainty
```

这一阶段可以形成第一篇论文的主体，因为它已经包含：

- edit\-level benchmark；

- mixed\-granularity representation；

- context\-conditioned edit scoring model；

- hard split evaluation；

- top\-k medicinal chemistry recommendation。

### 3\.2 第二阶段：开放式 R\-group / ΔG generation

第二阶段在第一阶段 scorer 的基础上做生成。

生成对象不是整条 peptide，而是：

```text
attachment-aware local chemical payload
= R-group graph / ΔG graph / modified residue fragment
```

生成流程：

```text
P, site, anchor, edit_scope, operation, target
        ↓
conditional R-group / ΔG generator
        ↓
validity + attachment check
        ↓
CEDG-Set scorer
        ↓
uncertainty / synthesis / 3D risk reranking
        ↓
top-k proposed local edits
```

---

## 4\. 输入与输出定义

### 4\.1 基本输入

模型的主输入不是一串文字，而是结构化对象：

```text
Input = {
  parent_peptide: P,
  edit_set: E = {e_1, ..., e_K},
  target_property: y,
  optional_constraints: C
}
```

其中 parent peptide 包含：

```text
canonical_shadow_sequence
topology_features
cyclization_type
site_index / residue position features
optional global physicochemical descriptors
```

edit set 中每个 edit event 包含：

```text
site_k
anchor_k
edit_scope_k
operation_k
attachment_k
chemical_payload_k
```

### 4\.2 基本输出

模型输出：

```text
property_pred       # modified peptide 的预测性质
Δproperty_pred      # parent -> modified 的性质变化
rank_score          # 用于 candidate ranking
uncertainty         # OOD 或不确定性风险
optional_auxiliary  # site / operation / payload 相关辅助输出
```

### 4\.3 单点与多点的统一形式

`K=1` 是单点编辑：

```text
E = {e_1}
```

`K>1` 是多个 site 同时修改：

```text
E = {e_1, e_2, ..., e_K}
```

这不是多个任务目标，而是一个 modified peptide 上的多个局部编辑事件。模型预测的是整个 edit set 共同作用后的性质。

---

## 5\. 核心表示：edit set 与 edit event

### 5\.1 为什么要用 edit set

真实药化优化中，经常不是只改一个位点。例如：

```text
site 2: Phe -> 4-F-Phe
site 5: Trp -> N-Me-Trp
```

这两个修改同时发生时，性质标签属于整个 modified peptide，而不是分别属于两个 edit。因此不能错误地拆成：

```text
e1 是 positive
e2 是 positive
```

正确表示是：

```text
P = parent peptide
E = {e1, e2}
label = property(P') or Δproperty(P → P')
```

模型要学习：

```text
Score(P, E)
```

而不是只学习：

```text
Score(P, e1) 和 Score(P, e2)
```

### 5\.2 edit event 的六元组

每个局部编辑事件定义为：

```text
e_k = (
  site_k,
  anchor_k,
  edit_scope_k,
  operation_k,
  attachment_k,
  chemical_payload_k
)
```

这六个字段的分工如下：

|字段|含义|例子|
|---|---|---|
|`site_k`|被修改的 residue 位置|site 2, site 5|
|`anchor_k`|该位置的 canonical / pseudo\-canonical 参照残基|Phe, Trp, Lys\-like|
|`edit_scope_k`|修改发生在哪个结构层级|R\_group\_edit, backbone\_edit|
|`operation_k`|具体编辑机制|aromatic\_substitution, N\_methylation|
|`attachment_k`|payload 接回哪里|phenyl\_para, backbone\_N|
|`chemical_payload_k`|原子级局部变化|H\_to\_F, NH\_to\_NCH3|

### 5\.3 edit\_scope：显式区分“换残基”和“换 R”

`edit_scope` 是本项目最新设计中非常关键的字段。它用于避免模型被误解成“只是换非天然氨基酸 token”。

建议第一版保留：

|edit\_scope|说明|例子|
|---|---|---|
|`R_group_edit`|侧链或取代基层面的局部替换|Phe para\-H → para\-F|
|`backbone_edit`|主链局部化学修改|backbone N\-H → N\-CH3|
|`stereochemistry_edit`|手性变化|L\-Phe → D\-Phe|
|`residue_replacement`|整个 monomer / residue 替换|Leu → Nle, Phe → β\-homo\-Phe|
|`linker_or_topology_edit`|环化方式、linker 或 staple 变化|disulfide → thioether|
|`terminal_or_capping_edit`|N/C 端修饰|N\-acetylation, C\-amidation|
|`handle_or_labeling_edit`|引入 handle 或 probe|azide, alkyne|
|`full_residue_fallback`|无法可靠拆分局部差异时的兜底|rare ncAA|

### 5\.4 chemical\_payload：真正保留全原子信息的地方

`chemical_payload` 不是自然语言，也不是一个非天然氨基酸名字，而是局部化学对象。它可以是：

|payload 类型|内容|例子|
|---|---|---|
|`r_group_delta`|R\-group 的差异|H → F, H → CF3|
|`r_group_replacement`|R\_old 与 R\_new 两个局部图|phenyl\-H → phenyl\-F|
|`delta_graph`|相对于 anchor 的新增、删除、替换原子/键|N\-H → N\-CH3|
|`chirality_change_flag`|手性变化|L\_to\_D|
|`linker_delta`|linker 局部变化|disulfide → thioether|
|`full_residue_graph`|拆不清时保存完整 modified residue graph|rare ncAA|

---

## 6\. 具体例子：不同 edit\_scope 如何表示

### 6\.1 R\-group edit：Phe → 4\-F\-Phe

表面看是一个非天然残基替换：

```text
site 2: Phe -> 4-F-Phe
```

但模型内部应表示为 R\-group 局部编辑：

```text
e_1 = (
  site = 2,
  anchor = Phe,
  edit_scope = R_group_edit,
  operation = aromatic_substitution,
  attachment = phenyl_para,
  chemical_payload = r_group_delta(H -> F)
)
```

也就是说，模型不是只看到 `4-F-Phe` 这个 monomer 名字，而是看到：

```text
在 Phe 的 phenyl para 位，把 H 换成 F。
```

### 6\.2 Backbone edit：Trp → N\-Me\-Trp

表面看是：

```text
site 5: Trp -> N-Me-Trp
```

模型内部表示为：

```text
e_2 = (
  site = 5,
  anchor = Trp,
  edit_scope = backbone_edit,
  operation = N_methylation,
  attachment = backbone_N,
  chemical_payload = delta_graph(N-H -> N-CH3)
)
```

这能让模型学习 N\-methylation 对 HBD、局部构象和透膜性的影响，而不是把 N\-Me\-Trp 当作一个孤立 token。

### 6\.3 Stereochemistry edit：L\-Phe → D\-Phe

```text
e_3 = (
  site = 2,
  anchor = Phe,
  edit_scope = stereochemistry_edit,
  operation = D_L_inversion,
  attachment = alpha_C,
  chemical_payload = chirality_change_flag(L_to_D)
)
```

这里 ΔG 不一定是一个复杂图，可以是 chirality flag \+ stereochemistry metadata。

### 6\.4 多位点 edit set

如果一个 modified peptide 同时有两个位点变化：

```text
site 2: Phe -> 4-F-Phe
site 5: Trp -> N-Me-Trp
```

则：

```text
E = {e_1, e_2}
```

其中：

```text
e_1 = site 2, R_group_edit, aromatic_substitution, phenyl_para, H -> F
e_2 = site 5, backbone_edit, N_methylation, backbone_N, N-H -> N-CH3
```

模型读入：

```text
Input:
  P = parent peptide
  E = [e_1, e_2]

Output:
  property(P') 或 Δproperty(P → P')
```

性质标签属于整个 edit set，不能把 e1 和 e2 分别当作独立 positive。

---

## 7\. 模型整体架构

CEDG\-Set 模型由四个核心模块构成：

```text
1. Peptide Context Encoder
2. Site-conditioned Atom-level Edit Encoder
3. Edit Set Encoder
4. Prediction / Ranking / Uncertainty Heads
```

整体信息流：

```text
Parent peptide P
  → canonical shadow sequence
  → PLM / peptide encoder
  → residue embeddings H = {h_1, ..., h_L}
  → global scaffold embedding h_P

For each edit e_k:
  site_k selects h_site_k
  edit metadata → z_meta_k
  chemical_payload_k → atom-level graph tokens A_k
  h_site_k conditions atom-level chemical encoding
  → edit embedding u_k

Edit set E:
  {u_1, ..., u_K}
  → Edit Set Encoder
  → z_E

Prediction:
  h_P + z_E
  → property / Δproperty / ranking / uncertainty
```

---

## 8\. 模型模块一：Peptide Context Encoder

### 8\.1 输入

Peptide context encoder 的输入是 parent peptide 的 residue\-level 表示：

```text
canonical_shadow_sequence
topology features
cyclization type
ring distance / circular position
optional global descriptors
```

其中 canonical shadow sequence 是把非天然残基映射到 canonical 或 pseudo\-canonical anchor 后得到的序列，例如：

```text
cyclo-[Leu, 4-F-Phe, Gly, N-Me-Trp, Val]
→ shadow: [Leu, Phe, Gly, Trp, Val]
```

### 8\.2 输出

```text
H = {h_1, h_2, ..., h_L}
```

其中 `h_i` 是第 i 个 residue 的上下文向量。

同时得到全局 scaffold embedding：

```text
h_P = Pool(H, topology)
```

### 8\.3 为什么 shadow 不会丢失非天然信息

canonical shadow 只负责让 PLM 理解 peptide scaffold 和位点上下文。真实非天然化学信息不放在 shadow 里，而是放在每个 edit event 的 `chemical_payload` 中。

因此：

```text
PLM branch: context
chemical branch: noncanonical chemistry
fusion branch: context × chemistry interaction
```

---

## 9\. 模型模块二：Site\-conditioned Atom\-level Edit Encoder

### 9\.1 这个模块解决什么问题

普通做法可能是：

```text
h_site = PLM(P)[site]
z_chem = GNN(chemical_payload)
u = MLP([h_site, z_chem])
```

这种简单拼接的问题是：化学 payload 在进入模型前已经被压成一个整体向量，site context 没有参与原子级编码。

本工作更核心的设计是：

> 让当前位点上下文 `h_site` 去条件化 atom\-level chemical payload 的编码过程。
> 
> 

也就是说，同一个 `H -> F` 或 `N-H -> N-CH3`，放在不同 site 上时，化学图会被不同 site context 重新解释。

### 9\.2 输入

对每个 edit event：

```text
h_site_k        # 来自 PLM 的位点上下文
anchor_k        # Phe / Trp / Lys-like 等
edit_scope_k    # R_group_edit / backbone_edit 等
operation_k     # aromatic_substitution / N_methylation 等
attachment_k    # phenyl_para / backbone_N 等
chemical_payload_k  # atom-level graph or flag
```

### 9\.3 输出

```text
u_k = context-specific edit embedding
```

`u_k` 表示：

```text
在 parent peptide 的 site_k 上，执行这个 chemical edit 的上下文化表示。
```

---

## 10\. 模型模块三：Edit Set Encoder

### 10\.1 为什么需要 Edit Set Encoder

如果只处理单点编辑，`u_edit` 直接进入 prediction head 即可。

但真实数据中可能出现多个位点同时变化：

```text
E = {e_1, e_2, ..., e_K}
```

这时模型需要学习多个 edit 之间的协同或冲突。

例如：

```text
site 2: Phe -> 4-F-Phe
site 5: Trp -> N-Me-Trp
```

两个修改可能共同提升透膜性，也可能一个提升透膜性、另一个破坏构象。

### 10\.2 输入和输出

输入：

```text
U = {u_1, u_2, ..., u_K}
```

输出：

```text
z_E = edit set representation
```

### 10\.3 实现方式

第一版可以用：

```text
masked mean/sum pooling + MLP
```

增强版可以用：

```text
Set Transformer
pairwise interaction network
attention pooling
```

建议形式：

```text
Score(P, E)
= Σ single_effect(P, e_k)
  + interaction_effect(P, {e_1, ..., e_K})
  + global_context_effect(P)
```

## 11\. 模型模块四：Prediction / Ranking / Uncertainty Heads

最终模型输出多种任务头：

|Head|输入|输出|用途|
|---|---|---|---|
|property head|`h_P + z_E`|`property(P')`|预测 modified peptide 的绝对性质|
|delta head|`h_P + z_E`|`Δproperty(P → P')`|学习编辑前后变化|
|ranking head|`h_P + z_E`|`rank_score`|在候选 edit set 中排序|
|uncertainty head|`h_P + z_E`|`σ` 或 risk score|识别 OOD scaffold / R\-group|
|auxiliary heads|`u_k`|operation/site/payload tasks|辅助学习表示|

完整预测：

```text
property_pred, delta_pred, rank_score, uncertainty
= Heads(h_P, z_E)
```

## 12\. 训练方式：为什么可以整体一起训练

训练样本形式：

```text
sample = {
  parent peptide P,
  edit set E,
  property_after,
  optional property_before,
  assay_type,
  evidence_level,
  confidence_weight
}
```

模型预测：

```text
property_pred = Model(P, E)
delta_pred = Model(P, E) - property_before 或独立 delta head
```

损失函数：

```text
L = λ_property * L_property
  + λ_delta * L_delta
  + λ_rank * L_ranking
  + λ_uncertainty * L_uncertainty
  + λ_aux * L_auxiliary
```

关键点：

> `site`、`edit_scope`、`operation`、`chemical_payload` 是结构化输入，不是模型先硬预测出来的离散决策。因此整个模型从 `P + E` 到最终 property / rank score 是一条可微路径，可以端到端训练。
> 
> 

对于多位点样本：

```text
P + {e_1, e_2} → property(P')
```

损失从最终 property 反传到：

```text
peptide encoder
site-conditioned atom-level edit encoder
edit set encoder
prediction heads
```

因此，多位点修改不是另一个任务，而是 `K>1` 的同一个模型输入。

---

## 13\. Candidate ranking：如何训练“哪个位点该改，怎么改”

在推荐时，模型需要从候选中选择：

```text
which site?
which edit_scope?
which operation?
which R-group / ΔG?
```

不建议把它们训练成完全分离的三个模型。更合理的是构造候选集合，统一打分：

```text
candidates = {
  (site 1, R_group_edit, aromatic_substitution, para-F),
  (site 1, R_group_edit, aromatic_substitution, para-CF3),
  (site 2, backbone_edit, N_methylation, N-CH3),
  (site 5, stereochemistry_edit, D_L_inversion, L_to_D),
  ...
}
```

模型对每个 candidate edit 或 candidate edit set 计算：

```text
Score(P, E_candidate)
```

训练时让实验中表现更好的 candidate 排在前面。

### 14\.1 负样本构造

对 observed positive edit，可以构造对比候选：

```text
same payload, different site
same site, different R-group
same site, different operation
same scaffold, historically worse edit
heuristic candidate
```

这样模型就能学习：

- 哪个 site 更适合某类化学变化；

- 某个 site 更适合 R\-group edit 还是 backbone edit；

- 同一类 operation 下哪个 payload 更有利；

- 多个 site 同时改时是否有协同。

---

## 15\. 开放式 R\-group / ΔG 生成模块

生成模块不是第一版必须完成的全部，但它是这个工作进一步做大的关键。

### 15\.1 生成对象

生成器不生成整条 peptide，而是生成局部 payload：

```text
chemical_payload_k
= R-group graph / R_delta / ΔG graph / modified residue fragment
```

生成条件：

```text
condition = {
  parent peptide context,
  site,
  anchor,
  edit_scope,
  operation,
  attachment,
  target property,
  constraints
}
```

### 15\.2 生成流程

```text
P, site, anchor, edit_scope, operation, target
        ↓
fragment-action / motif-based generator
        ↓
chemical_payload candidates
        ↓
attachment / valence / charge validity check
        ↓
CEDG-Set scorer
        ↓
uncertainty + synthesis + 3D risk reranking
        ↓
top-k local edit proposals
```

### 15\.3 为什么不是 whole peptide generation

因为本工作的核心是局部可解释药化编辑。whole peptide generation 会把 scaffold、site、operation 和 payload 混在一起，难以回答：

```text
到底是哪一个位点的哪一种化学变化带来了性质改善？
```

局部 payload generation 更符合本工作的定位。

---

## 16\. 数据是否能支撑这个模型

### 16\.1 单点数据可以直接支撑

如果样本是：

```text
parent P
site 2: Phe -> 4-F-Phe
property_before
property_after
```

那么可以构造：

```text
E = {e_1}
```

用于训练：

```text
property prediction
Δproperty prediction
candidate ranking
```

### 16\.2 多点数据需要 pair\-level \+ event\-level 结构

如果样本是：

```text
site 2: Phe -> 4-F-Phe
site 5: Trp -> N-Me-Trp
```

必须记录为：

```text
edit_pair_table: 一行，表示 parent-modified pair
edit_event_table: 两行，分别表示两个 site-level edit
```

模型读入：

```text
P + E = {e_1, e_2}
```

不要拆成两个独立 positive。

### 16\.3 如果只有单点数据怎么办

如果早期数据中大部分都是单点 edit，则模型先退化为：

```text
K = 1 的 CEDG-Set
```

也就是 CEDG\-Score。此时仍然可以发表，因为单点 edit scoring / ranking 已经能验证核心表示：

```text
site context × edit_scope × atom-level payload
```

当后续收集到更多多点 pair，再训练 Edit Set Encoder 的 interaction term。

---

## 17\. 评估设计

### 17\.1 主要任务

|任务|说明|指标|
|---|---|---|
|absolute property prediction|预测 modified peptide 的性质|MAE, RMSE, Spearman, AUROC|
|Δproperty prediction|预测 parent→modified 的变化|MAE, direction accuracy, Spearman|
|edit ranking|同一 scaffold 下候选排序|NDCG@k, hit@k, pairwise accuracy|
|site/edit recommendation|推荐 top\-k site \+ edit|top\-k enrichment, hit rate|
|uncertainty / OOD|判断未见 scaffold/R\-group 风险|ECE, selective risk, OOD AUROC|

### 17\.2 Hard split

Random split 只能作为 sanity check，主结果应强调：

```text
unseen R-group split
unseen monomer family split
unseen scaffold split
external source / assay split
```

这些 split 能证明模型不是在记忆常见 monomer 或 scaffold。

### 17\.3 Baseline

必须比较：

```text
canonical shadow + PLM only
chemical fingerprint / graph only
whole-SMILES / SELFIES model
tokenized ncAA transformer
full residue graph instead of ΔG
nearest-neighbor SAR
RDKit descriptors + RF/XGBoost
```

关键消融：

```text
No PLM context
No chemical payload
No edit_scope
No operation
No topology
concat vs site-conditioned fusion
single edit additive vs edit set interaction
```

---

## 18\. 这项工作的创新点

### 创新点 1：任务定义

将非天然环肽优化从 full\-sequence / full\-molecule generation 改写为：

```text
context-conditioned local chemical edit set learning
基于上下文条件的局部化学编辑集学习
```

也就是学习已有 scaffold 上一个或多个局部编辑事件的效果。

### 创新点 2：表示体系

提出：

```text
canonical shadow sequence + edit_scope + atom-level chemical_payload
```

其中 canonical shadow 负责 PLM 上下文，chemical payload 负责真实非天然化学差异。

### 创新点 3：site\-conditioned atom\-level edit encoder

让 site context 参与 atom\-level payload 编码，而不是简单拼接 PLM embedding 和化学图 embedding。

### 创新点 4：edit set 支持多位点修改

单点修改是 `K=1`，多点修改是 `K>1`。模型用 edit set encoder 学习多个 site edits 的协同或冲突。

### 创新点 5：可自然扩展到 R\-group / ΔG generation

生成对象是局部 attachment\-aware chemical payload，而不是整条 peptide。

---

## 19\. 第一版 MVP

第一版不需要一开始做全功能平台。建议最小闭环是：

```text
数据：permeability-focused cyclic peptide edit-level dataset
表示：canonical shadow + edit_scope + chemical_payload
模型：CEDG-Set with K=1 first, K>1 if available
任务：property prediction + Δproperty + candidate ranking
评估：random + scaffold + R-group split
baseline：PLM-only / chemistry-only / whole-SMILES / fingerprint / token ncAA
```

成功标准：

```text
1. 能稳定构造 parent peptide + edit event / edit set；
2. 能区分 R_group_edit、backbone_edit、stereochemistry_edit 等 edit_scope；
3. 能保留 atom-level R_delta / ΔG payload；
4. CEDG-Set 在 hard split 下优于关键 baseline；
5. top-k recommendation case study 有药化可解释性。
```

---

## 20\. 简短表述

本工作提出一个面向非天然环肽优化的 CEDG\-Set 框架。它不直接从零生成整条环肽，而是把药化优化表示为 parent peptide 上的一个或多个局部化学编辑事件。每个编辑事件由 `site、anchor、edit_scope、operation、attachment、chemical_payload` 定义，其中 peptide scaffold 用 canonical shadow sequence 和 PLM 建模，局部非天然编辑用 atom\-level R\-group / ΔG graph 表示。模型通过 site\-conditioned atom\-level edit encoder 学习“某个位点上下文如何影响某个局部化学 payload 的效果”，再通过 edit set encoder 支持多个位点同时修改，最终预测 modified peptide 的性质、Δproperty、候选排序和不确定性。第一阶段做库约束 edit scoring / ranking，第二阶段扩展到 attachment\-aware R\-group / ΔG 生成。

---

# CEDG\-Set 数据收集与构建指南

## 0\. 数据集目标

本数据集不是普通的：

```text
peptide → property
```

而是要构建：

```text
parent peptide scaffold P
+ edit set E = {e_1, ..., e_K}
→ modified peptide P'
→ property(P') / Δproperty(P → P') / ranking label
```

核心目标是支撑 CEDG\-Set 学习：

```text
Score(P, E)
```

其中每个 edit event 是：

```text
e_k = (
  site_k,
  anchor_k,
  edit_scope_k,
  operation_k,
  attachment_k,
  chemical_payload_k
)
```

### 0\.1 第一版聚焦范围

建议第一版聚焦：

```text
主性质：permeability / membrane permeability
主对象：cyclic peptide / macrocyclic peptide
辅助对象：short modified peptide / CPP / solubility / hemolysis data
主编辑：R-group edit、N-methylation、D/L inversion、side-chain replacement、backbone extension、linker edit
```

### 0\.2 数据构建的核心原则

1. **先保留原始信息，再做标准化。** 不要一开始就覆盖掉 raw sequence / SMILES / HELM / monomer list。

2. **pair\-level 和 event\-level 必须分开。** 一个 parent\-modified pair 可以对应多个 site\-level edits。

3. **多位点修改不能拆成多个独立 positive edit。** 标签属于整个 edit set。

4. **R\-group edit 不能只保存 ncAA 名字。** 至少要保存 anchor、attachment 和 R\_delta / R\_old / R\_new。

5. **canonical shadow 只给 peptide context encoder 用。** 真实非天然化学差异保存在 chemical\_payload 中。

6. **不同 assay 不要直接相减。** PAMPA、Caco\-2、MDCK、cell uptake 等要分组处理。

---

## 1\. 数据构建整体流程

```text
Step 1. 数据源盘点
        ↓
Step 2. 建 peptide_sample_table，保存原始 peptide 样本
        ↓
Step 3. 建 monomer_anchor_table，标准化 monomer / ncAA / modification
        ↓
Step 4. 生成 canonical shadow sequence 和 topology annotation
        ↓
Step 5. 做 scaffold grouping
        ↓
Step 6. 构建 parent-modified edit_pair_table
        ↓
Step 7. 对每个 changed site 生成 edit_event_table
        ↓
Step 8. 抽取 chemical_payload：R_delta / ΔG / full residue graph / chirality flag
        ↓
Step 9. 构造 candidate_edit_table，用于 ranking 和 negative sampling
        ↓
Step 10. 质量控制、证据分层、hard split
        ↓
Step 11. 导出模型训练样本
```

---

## 2\. 需要建立的核心数据表

建议至少建立七张表：

|表|一行代表什么|作用|
|---|---|---|
|`source_inventory.csv`|一个数据源|记录数据来源、assay、字段可用性|
|`peptide_sample_table.csv`|一个 peptide 样本|保存原始结构和性质标签|
|`monomer_anchor_table.csv`|一个 monomer / ncAA / modification|映射 anchor、edit\_scope、operation、attachment|
|`scaffold_table.csv`|一个 scaffold group|用于 matched pair、ranking 和 scaffold split|
|`edit_pair_table.csv`|一个 parent\-modified pair|保存 pair\-level 性质变化和证据等级|
|`edit_event_table.csv`|一个 pair 中的一个 site\-level edit|模型真正读取的 edit event|
|`candidate_edit_table.csv`|一个候选 edit 或 edit set|用于 ranking、site selection 和 negative sampling|

最关键的是这三张：

```text
monomer_anchor_table.csv
edit_pair_table.csv
edit_event_table.csv
```

如果要训练 ranking，还必须有：

```text
candidate_edit_table.csv
```

---

## 3\. source\_inventory：先盘点数据源

### 3\.1 为什么需要 source\_inventory

不要一开始就写复杂解析器。先回答：

```text
哪些数据源有 permeability？
哪些数据源有 cyclic peptide？
哪些有 noncanonical monomer annotation？
哪些有 same-scaffold SAR series？
哪些能构建 parent-modified pair？
assay 是否可比？
```

### 3\.2 推荐字段

```text
source_id
source_name
paper_or_database
url_or_reference
molecule_type
cyclic_or_linear
number_of_peptides
property_name
assay_type
property_unit
has_sequence
has_smiles
has_helm
has_monomer_list
has_noncanonical_residue
has_sar_series
has_same_scaffold_edits
can_construct_matched_pairs
confidence_note
notes
```

### 3\.3 数据源优先级

第一优先级：

```text
cyclic peptide permeability 数据
PAMPA / Caco-2 / MDCK 等同类 assay 的 SAR series
有 monomer list / HELM / SMILES 的 cyclic peptide 数据
```

第二优先级：

```text
chemically modified peptide property data
short peptide permeability / CPP / uptake data
hemolysis / solubility / stability data
```

第三优先级：

```text
ncAA library / building block catalog
3D conformer resources
synthetic feasibility resources
```

---

## 4\. peptide\_sample\_table：保存原始 peptide 样本

### 4\.1 一行代表什么

一行代表一个原始 peptide 或 cyclic peptide 样本，不管它现在是否能形成 pair。

### 4\.2 推荐字段

```text
peptide_id
source_id
peptide_name
sequence_raw
smiles_raw
helm_raw
monomer_list_raw
molecule_type
cyclization_type
is_cyclic
length
property_name
property_value
property_unit
assay_type
assay_condition
property_label_raw
source_reference
quality_flag
manual_check
notes
```

### 4\.3 后续标准化字段

```text
standardized_monomer_list
canonical_shadow_sequence
pseudo_shadow_sequence
topology_type
cyclization_annotation
net_charge_class
scaffold_id
standardized_property_value
label_type
```

### 4\.4 例子

原始样本：

```text
peptide_id = P001
sequence_raw = cyclo-[Leu-Phe-Gly-Trp-Val]
monomer_list_raw = Leu;Phe;Gly;Trp;Val
property_name = PAMPA permeability
property_value = -6.2
assay_type = PAMPA
```

标准化后：

```text
canonical_shadow_sequence = LFGWV
is_cyclic = true
cyclization_type = head_to_tail
scaffold_id = scaf_001
```

---

## 5\. monomer\_anchor\_table：最重要的数据资产

### 5\.1 这张表的作用

`monomer_anchor_table` 把原始 monomer / ncAA / modified residue 名称映射成：

```text
canonical_anchor / pseudo_anchor
edit_scope_default
operation_default
attachment_points
chemical structure information
```

如果这张表不稳定，后续的 anchor、edit\_scope、payload 都会漂。

### 5\.2 推荐字段

```text
monomer_id
source_monomer_name
standardized_name
monomer_smiles
helm_symbol
canonical_anchor
pseudo_anchor
anchor_confidence
edit_scope_default
operation_default
attachment_points
sidechain_exit_atom
backbone_attachment_N
backbone_attachment_C
stereochemistry
charge
full_residue_smiles
full_residue_graph_id
manual_check
quality_flag
notes
```

### 5\.3 常见记录示例

|source\_monomer\_name|canonical\_anchor|pseudo\_anchor|edit\_scope\_default|operation\_default|attachment\_points|说明|
|---|---|---|---|---|---|---|
|Phe|Phe||none|none|backbone\_N, backbone\_C, side\_chain|标准残基|
|4\-F\-Phe|Phe||R\_group\_edit|aromatic\_substitution|phenyl\_para|芳环 para 位取代|
|4\-Cl\-Phe|Phe||R\_group\_edit|aromatic\_substitution|phenyl\_para|与 4\-F\-Phe 共享 anchor|
|D\-Phe|Phe||stereochemistry\_edit|D\_L\_inversion|alpha\_C|手性变化|
|N\-Me\-Trp|Trp||backbone\_edit|N\_methylation|backbone\_N|主链 N 甲基化|
|Orn||Lys\-like|R\_group\_edit|side\_chain\_length\_change|side\_chain|Lys\-like 侧链缩短|
|Nle||Leu\-like|R\_group\_edit|hydrophobic\_R\_group\_replacement|side\_chain|Leu\-like 疏水侧链替换|
|β\-homo\-Phe||Phe\-like|residue\_replacement|beta\_gamma\_amino\_acid|backbone, side\_chain|主链骨架改变|

### 5\.4 人工审核优先级

第一版必须人工检查：

```text
top 100 高频 monomer
所有 N-methyl residues
所有 D/L stereochemistry variants
所有 halogenated aromatic residues
Orn / Dab / Dap / Nle 等常见 pseudo-anchor residues
所有 linker / cyclization-related monomers
```

---

## 6\. scaffold\_table：定义 scaffold group

### 6\.1 为什么需要 scaffold grouping

scaffold group 用于：

```text
构建 matched pair / SAR series
构造 candidate ranking group
做 scaffold split
避免 random split 虚高
```

### 6\.2 推荐字段

```text
scaffold_id
representative_peptide_id
canonical_shadow_sequence
topology_type
cyclization_type
length
fixed_motif
net_charge_class
number_of_variants
source_ids
quality_flag
notes
```

### 6\.3 初版 scaffold 判断规则

可以根据：

```text
canonical shadow sequence similarity
sequence length
cyclization type
fixed motif / conserved positions
terminal caps
net charge class
source / series annotation
```

第一版可以规则 \+ 人工校验，不必一开始追求完全自动。

---

## 7\. edit\_pair\_table：parent\-modified pair

### 7\.1 一行代表什么

一行代表一个 parent peptide 和 modified peptide 的 pair。

它保存的是 pair\-level 标签：

```text
property_before
property_after
Δproperty
edit_count
attribution_level
evidence_level
confidence_weight
```

### 7\.2 推荐字段

```text
pair_id
source_id
parent_peptide_id
modified_peptide_id
scaffold_id
same_scaffold
assay_type
assay_comparability
property_before
property_after
delta_property
label_type
edit_count
attribution_level
evidence_level
confidence_weight
quality_flag
manual_check
exclude_reason
notes
```

### 7\.3 attribution\_level

```text
single_edit             # 只有一个 site-level edit
multi_edit_known        # 多个位点修改，每个 event 都清楚
multi_edit_partial      # 多个位点修改，部分 event 不确定
multi_edit_uncertain    # 多个位点修改，但无法可靠归因
pseudo_edit             # 没有明确 parent，只能从 modified peptide 构造 shadow edit
absolute_only           # 只能用于 absolute property prediction
```

### 7\.4 evidence\_level

|等级|定义|用途|权重建议|
|---|---|---|---|
|`A_true_pair`|同 scaffold、同 assay、parent\-modified 关系明确|主训练、主评估、Δproperty|1\.0|
|`B_matched_pair`|scaffold 高度相似，assay 可比，少量 edit|ranking、contrastive、辅助训练|0\.7|
|`C_cross_study_pseudo_pair`|跨文献或实验条件不同，趋势可参考|弱监督、预训练|0\.3|
|`D_heuristic`|规则构造，不是实验 pair|规则先验、negative sampling|0\.1|

### 7\.5 单点 pair 例子

```text
pair_id = pair_001
parent_peptide_id = P0
modified_peptide_id = P1
scaffold_id = scaf_001
property_before = -6.2
property_after = -5.4
delta_property = +0.8
edit_count = 1
attribution_level = single_edit
evidence_level = A_true_pair
```

### 7\.6 多点 pair 例子

```text
pair_id = pair_004
parent_peptide_id = P0
modified_peptide_id = P4
scaffold_id = scaf_001
property_before = -6.2
property_after = -4.8
delta_property = +1.4
edit_count = 2
attribution_level = multi_edit_known
evidence_level = A_true_pair
```

注意：这个 pair 会在 `edit_event_table` 中对应两行 event。

---

## 8\. edit\_event\_table：模型真正读取的 edit event

### 8\.1 一行代表什么

一行代表一个 pair 中的一个 site\-level local edit。

如果一个 pair 有两个 site 被修改，则：

```text
edit_pair_table: 1 行
edit_event_table: 2 行
```

### 8\.2 推荐字段

```text
pair_id
event_id
event_order
site_index
anchor_residue
original_residue_or_monomer
modified_residue_or_monomer
edit_scope
operation_type
attachment_type
chemical_payload_type
delta_graph_id
delta_graph_smiles
r_old_smiles
r_new_smiles
r_delta_smiles
full_residue_graph_id
full_residue_smiles
chirality_change
linker_old_smiles
linker_new_smiles
linker_delta_smiles
event_confidence
quality_flag
manual_check
notes
```

### 8\.3 edit\_scope taxonomy

|edit\_scope|说明|示例|
|---|---|---|
|`R_group_edit`|侧链或取代基变化|Phe → 4\-F\-Phe|
|`backbone_edit`|主链局部化学变化|Trp → N\-Me\-Trp|
|`stereochemistry_edit`|手性变化|L\-Phe → D\-Phe|
|`residue_replacement`|整个 monomer 替换|Leu → Nle|
|`linker_or_topology_edit`|环化方式或 linker 改变|disulfide → thioether|
|`terminal_or_capping_edit`|端基变化|N\-acetylation|
|`handle_or_labeling_edit`|引入 handle 或 label|azide handle|
|`full_residue_fallback`|无法拆分局部差异|rare ncAA|

### 8\.4 operation\_type 第一版

```text
N_methylation
alpha_methylation
D_L_inversion
aromatic_substitution
halogenation
hydrophobic_R_group_replacement
polar_R_group_replacement
side_chain_length_change
charge_tuning
backbone_extension
beta_gamma_amino_acid
peptoid_like
macrocycle_linker_edit
terminal_capping
handle_introduction
full_residue_replacement
other
```

---

## 9\. chemical\_payload：如何体现 R\-group 和全原子信息

### 9\.1 payload 类型

|chemical\_payload\_type|内容|示例|
|---|---|---|
|`r_group_delta`|R\-group 差异|H → F, H → CF3|
|`r_group_replacement`|R\_old 和 R\_new 两个局部图|phenyl\-H → phenyl\-F|
|`delta_graph`|新增 / 删除 / 替换原子和键|N\-H → N\-CH3|
|`chirality_change_flag`|手性变化|L\_to\_D|
|`linker_delta`|linker 局部变化|disulfide → thioether|
|`full_residue_graph`|完整 modified residue 图|rare ncAA|

### 9\.2 R\-group edit 例子：Phe → 4\-F\-Phe

原始变化：

```text
site 2: Phe -> 4-F-Phe
```

event 记录：



```text
pair_id = pair_001
event_id = event_001
site_index = 2
anchor_residue = Phe
original_residue_or_monomer = Phe
modified_residue_or_monomer = 4-F-Phe
edit_scope = R_group_edit
operation_type = aromatic_substitution
attachment_type = phenyl_para
chemical_payload_type = r_group_delta
r_old_smiles = [H]
r_new_smiles = [F]
r_delta_smiles = H_to_F
```

```text
pair_id = pair_001  
event_id = event_001  
site_index = 2  
anchor_residue = 苯丙氨酸  
original_residue_or_monomer = 苯丙氨酸  
modified_residue_or_monomer = 4-氟苯丙氨酸  
edit_scope = R基团编辑  
operation_type = 芳香取代  
attachment_type = 苯环对位  
chemical_payload_type = R基团差异  
r_old_smiles = [H]  
r_new_smiles = [F]  
r_delta_smiles = H_to_F
```

### 9\.3 Backbone edit 例子：Trp → N\-Me\-Trp

```text
pair_id = pair_002
event_id = event_001
site_index = 5
anchor_residue = Trp
original_residue_or_monomer = Trp
modified_residue_or_monomer = N-Me-Trp
edit_scope = backbone_edit
operation_type = N_methylation
attachment_type = backbone_N
chemical_payload_type = delta_graph
delta_graph_smiles = N-H_to_N-CH3
```

### 9\.4 Stereochemistry edit 例子：L\-Phe → D\-Phe

```text
pair_id = pair_003
event_id = event_001
site_index = 2
anchor_residue = Phe
original_residue_or_monomer = L-Phe
modified_residue_or_monomer = D-Phe
edit_scope = stereochemistry_edit
operation_type = D_L_inversion
attachment_type = alpha_C
chemical_payload_type = chirality_change_flag
chirality_change = L_to_D
```

### 9\.5 多点 edit set 例子

pair\-level：

```text
pair_id = pair_004
parent_peptide_id = P0
modified_peptide_id = P4
property_before = -6.2
property_after = -4.8
delta_property = +1.4
edit_count = 2
attribution_level = multi_edit_known
```

event\-level：

```text
pair_004, event_001, site=2, anchor=Phe,
edit_scope=R_group_edit,
operation=aromatic_substitution,
attachment=phenyl_para,
payload=H_to_F

pair_004, event_002, site=5, anchor=Trp,
edit_scope=backbone_edit,
operation=N_methylation,
attachment=backbone_N,
payload=N-H_to_N-CH3
```

模型训练样本：

```text
Input:
  parent P0
  E = [event_001, event_002]

Label:
  Δproperty = +1.4
```

---

## 10\. candidate\_edit\_table：用于 ranking 和 negative sampling

### 10\.1 为什么需要 candidate\_edit\_table

如果只有 observed edit，模型只能学习：

```text
这个已发生 edit 的性质是多少
```

但很难学习：

```text
为什么 site 2 比 site 5 更适合改？
为什么 para-F 比 para-CF3 更好？
为什么 N-methylation 比 D/L inversion 更适合这个位置？
```

所以需要构造 candidate group，让模型在同一个 parent scaffold 下进行排序。

### 10\.2 推荐字段

```text
candidate_id
candidate_group_id
parent_peptide_id
scaffold_id
target_property
site_index
anchor_residue
edit_scope
operation_type
attachment_type
chemical_payload_type
delta_graph_id
r_old_smiles
r_new_smiles
r_delta_smiles
candidate_source
candidate_label
observed_pair_id
observed_event_id
negative_sampling_strategy
candidate_confidence
quality_flag
notes
```

### 10\.3 candidate\_source

```text
observed_positive
observed_negative
same_site_different_R
same_payload_different_site
same_site_different_operation
library_candidate
heuristic_candidate
generated_candidate
```

### 10\.4 例子

对同一个 parent peptide `P0` 构造候选集合：

```text
candidate_group_id = group_001
parent_peptide_id = P0
target_property = permeability improvement
```

候选记录：

```text
cand_001: site 2, Phe, R_group_edit, aromatic_substitution, para-F, observed_positive, label=1
cand_002: site 2, Phe, R_group_edit, aromatic_substitution, para-CF3, same_site_different_R, label=0/unknown
cand_003: site 5, Trp, backbone_edit, N_methylation, N-CH3, observed_positive, label=1
cand_004: site 3, Gly, stereochemistry_edit, D_L_inversion, L_to_D, heuristic_candidate, label=0/unknown
```

训练时可以用：

```text
pairwise ranking loss
listwise softmax loss
NDCG-oriented loss
```

---

## 11\. canonical shadow sequence 的构建

### 11\.1 作用

canonical shadow sequence 只用于 peptide context encoder，不承载真实非天然化学差异。

例子：

```text
4-F-Phe -> Phe
N-Me-Trp -> Trp
D-Phe -> Phe
Orn -> Lys-like
Nle -> Leu-like
β-homo-Phe -> Phe-like
```

### 11\.2 字段

在 `peptide_sample_table` 中保存：

```text
canonical_shadow_sequence
pseudo_shadow_sequence
shadow_mapping_confidence
shadow_mapping_notes
```

### 11\.3 注意

不能只保存 shadow，否则会丢失非天然信息。必须同步保存：

```text
edit_event_table
chemical_payload fields
monomer_anchor_table
```

---

## 12\. 数据标签和 assay 处理

### 12\.1 label\_type

```text
absolute_value
binary_high_low
within_study_rank
pairwise_preference
delta_property
weak_trend
```

### 12\.2 assay 分组原则

不要把所有 permeability assay 直接混合回归。

建议：

```text
PAMPA: 第一主任务，优先用于 passive permeability
Caco-2 / MDCK / RRCK: 单独 assay group，可作为外部或辅助任务
cell uptake / cellular activity: 弱标签，不直接当 permeability gold label
```

### 12\.3 delta\_property 的计算条件

只有满足以下条件才建议计算强 `Δproperty`：

```text
same scaffold 或高度 matched scaffold
same assay type
property unit 可比
实验条件相近
parent 和 modified 关系明确
```

否则只用于：

```text
within-study rank
pairwise preference
weak trend
absolute property
```

---

## 13\. 质量控制规则

必须排除或单独标注：

```text
结构不完整
无法确定 modified site
SMILES / HELM / sequence 互相矛盾
assay 不可比却直接计算 Δproperty
多点修改无法归因
环化方式不清楚
只有 cell activity 没有 permeability
monomer 无法可靠映射且未标记 fallback
```

建议所有表加入：

```text
quality_flag
manual_check
exclude_reason
confidence_weight
notes
```

### 13\.1 confidence\_weight 建议

```text
A_true_pair: 1.0
B_matched_pair: 0.7
C_cross_study_pseudo_pair: 0.3
D_heuristic: 0.1
full_residue_fallback: 根据情况降低
multi_edit_uncertain: 不进入主训练
```

---

## 14\. 数据划分策略

### 14\.1 必须有的 split

```text
random split
scaffold split
R-group split
monomer family split
source / assay split
```

### 14\.2 每种 split 证明什么

|Split|目的|
|---|---|
|random split|检查模型流程是否能拟合基本数据|
|scaffold split|测试能否泛化到新 peptide scaffold|
|R\-group split|测试能否泛化到未见 chemical payload|
|monomer family split|测试是否只是记住 ncAA family|
|source / assay split|测试是否依赖某个数据源 artifact|

### 14\.3 split group 字段

建议在表中加入：

```text
split_group_scaffold
split_group_rgroup
split_group_monomer_family
split_group_operation
split_group_source
```

---

## 15\. 从数据表导出模型样本

### 15\.1 单点样本

```python
sample = {
    "parent_shadow_sequence": "LFGWV",
    "topology_features": {...},
    "edit_set": [
        {
            "site_index": 2,
            "anchor": "Phe",
            "edit_scope": "R_group_edit",
            "operation": "aromatic_substitution",
            "attachment": "phenyl_para",
            "chemical_payload_type": "r_group_delta",
            "r_delta": "H_to_F",
        }
    ],
    "property_before": -6.2,
    "property_after": -5.4,
    "delta_property": +0.8,
    "evidence_level": "A_true_pair",
    "confidence_weight": 1.0,
}
```

### 15\.2 多点样本

```python
sample = {
    "parent_shadow_sequence": "LFGWV",
    "topology_features": {...},
    "edit_set": [
        {
            "site_index": 2,
            "anchor": "Phe",
            "edit_scope": "R_group_edit",
            "operation": "aromatic_substitution",
            "attachment": "phenyl_para",
            "chemical_payload_type": "r_group_delta",
            "r_delta": "H_to_F",
        },
        {
            "site_index": 5,
            "anchor": "Trp",
            "edit_scope": "backbone_edit",
            "operation": "N_methylation",
            "attachment": "backbone_N",
            "chemical_payload_type": "delta_graph",
            "delta_graph": "N-H_to_N-CH3",
        },
    ],
    "property_before": -6.2,
    "property_after": -4.8,
    "delta_property": +1.4,
    "evidence_level": "A_true_pair",
    "confidence_weight": 1.0,
}
```

### 15\.3 absolute\-only 样本

如果没有 parent：

```python
sample = {
    "modified_shadow_sequence": "LFGWV",
    "pseudo_edit_set": [...],
    "property_after": -5.4,
    "label_type": "absolute_value",
    "attribution_level": "absolute_only",
    "confidence_weight": 0.5,
}
```

这类样本可用于 property pretraining，但不用于强 Δproperty 主结果。

---

## 16\. 数据构建优先级

### 16\.1 第一阶段：能跑通 MVP

目标：

```text
500-2000 peptide samples
100-300 manually checked edit events
20-100 high-confidence edit pairs / edit sets
覆盖 top 100 高频 monomers
能完成 random / scaffold / R-group split
```

必须完成：

```text
source_inventory
peptide_sample_table
monomer_anchor_table
edit_pair_table
edit_event_table
```

### 16\.2 第二阶段：支持 ranking

新增：

```text
candidate_edit_table
ranking groups
negative sampling strategies
observed positive / sampled negative labels
```

### 16\.3 第三阶段：支持生成

新增：

```text
fragment / R-group library
attachment validity rules
valence / charge filters
commercial availability / synthesis feasibility fields
generated_candidate records
```

---

## 17\. 前四周执行计划

### Week 1：数据源盘点与原始样本表

```text
[ ] 建 source_inventory.csv
[ ] 确定主性质：优先 PAMPA / permeability
[ ] 整理主数据源 raw files
[ ] 建 peptide_sample_table.csv
[ ] 保留 raw sequence / SMILES / HELM / monomer list
```

### Week 2：monomer\_anchor\_table v0

```text
[ ] 统计高频 monomer
[ ] 人工审核 top 50-100 monomers
[ ] 定义 canonical_anchor / pseudo_anchor
[ ] 标注 edit_scope_default / operation_default / attachment_points
[ ] 生成 canonical shadow sequence v0
```

### Week 3：pair 与 event 构建

```text
[ ] 做 scaffold grouping v0
[ ] 找 same-scaffold pairs
[ ] 建 edit_pair_table.csv
[ ] 对 changed site 生成 edit_event_table.csv
[ ] 处理 single_edit 和 multi_edit_known
[ ] 标记 multi_edit_uncertain / absolute_only
```

### Week 4：payload、candidate 和 split

```text
[ ] 抽取 R_delta / ΔG / chirality flag
[ ] 人工校验典型 payload
[ ] 建 candidate_edit_table v0
[ ] 构造 random / scaffold / R-group split
[ ] 导出 CEDG-Set training samples
[ ] 跑一个最小 baseline 检查数据是否可用
```

## 18\. 数据集最终产出

第一版数据集应至少能导出：

```text
CEDG_property_dataset.pt
CEDG_delta_dataset.pt
CEDG_ranking_dataset.pt
CEDG_split_random.json
CEDG_split_scaffold.json
CEDG_split_rgroup.json
CEDG_annotation_report.md
CEDG_quality_report.md
```

并能回答：

```text
有多少 peptide？
有多少 noncanonical monomer？
有多少 edit pair？
有多少 edit event？
有多少 single-edit / multi-edit-known？
每种 edit_scope 有多少样本？
每种 operation 有多少样本？
每种 payload 类型有多少样本？
R-group split 和 scaffold split 下训练/测试是否足够？
```

---

## 19\. 简短表述

为了训练 CEDG\-Set，我们不把原始数据整理成普通 peptide\-property 表，而是整理成 parent peptide、modified peptide、edit pair 和 edit event 四个层级。`edit_pair_table` 记录一个 parent\-modified pair 的整体性质变化；`edit_event_table` 记录这个 pair 中每个 site 的具体局部化学编辑，包括 site、anchor、edit\_scope、operation、attachment 和 chemical\_payload。这样单点修改就是一个 event，多点修改就是一个 edit set；多点 pair 的标签属于整个 edit set，不会被错误拆成多个独立 positive。R\-group edit 会显式保存 R\_delta 或 R\_old/R\_new，而不是只保存 4\-F\-Phe 这种非天然氨基酸名字，从而支撑模型学习上下文条件下的局部原子级化学编辑效果。

> (注：内容由 AI 生成，请谨慎参考）
