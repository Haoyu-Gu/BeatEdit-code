# 方法论：基于 Levenshtein Transformer 的符号音乐修复

> 本文档聚焦方法的创新点与设计合理性，面向论文写作 / 答辩使用。
> 更新: 2026-02-19 | 版本: 3.0 (新增 §9 声部协调编辑 + §10 对比实验设计)

---

## 1. 研究动机

### 1.1 为什么做符号音乐修复（Symbolic Music Inpainting）

符号音乐修复是指：给定一段 MIDI 序列，其中某一区域被遮盖（例如 4~8 小节的旋律缺失），模型需要生成与上下文风格连贯、结构合理的补全片段。

这个任务在实际音乐创作中有直接价值：
- **辅助作曲**：作曲者写好首尾，由模型填充中间过渡段
- **音乐修复**：恢复损坏或缺失的乐谱片段
- **条件生成**：在给定旋律框架下生成伴奏

与自由生成（unconditional generation）不同，修复任务的核心难点在于**双向一致性**——生成片段必须同时与左侧上文和右侧下文衔接。这对纯自回归（AR）模型构成天然挑战，因为 AR 只能看到左侧上文。

### 1.2 现有方法的局限

| 方法类别 | 代表工作 | 局限 |
|---------|---------|------|
| 自回归（AR） | Music Transformer, MuseNet | 单向生成，无法利用右侧上文；修复需额外 infilling trick |
| 掩码语言模型（MLM） | MusicBERT, Mask-Predict | 一次预测所有位置，无法处理**长度不确定**的修复（原始序列与目标长度不同） |
| 扩散模型 | DiffuSeq, DDPM for music | 计算量大，迭代步数多，符号离散域需额外量化 |
| Seq2Seq | Encoder-Decoder | 需要将修复建模为翻译问题，上下文利用不自然 |

**核心矛盾**：音乐修复的目标序列**长度不确定**。掩盖 4 小节后，目标区域可能是 50 个 token，也可能是 200 个——这取决于该区域的音乐密度。固定长度的 BERT 式 mask-predict 无法处理这种情况。

### 1.3 为什么选择 Levenshtein Transformer

Levenshtein Transformer（Gu et al., 2019）原本是机器翻译中的非自回归（NAR）方法，其核心思路是将序列生成分解为三种编辑操作：

$$\text{Delete} \rightarrow \text{Insert Placeholders} \rightarrow \text{Fill Tokens}$$

这种 **"编辑式生成"** 天然适配音乐修复场景：

1. **长度自适应**：Insertion Head 预测每个间隙插入多少占位符，自动推断目标长度，不需要预先指定
2. **双向上下文**：Encoder-only 架构天然利用全局注意力，左右上下文同时可见
3. **迭代纠错**：多轮 delete→insert→fill 逐步逼近目标，而非一次赌博式预测所有 token
4. **上下文保护**：通过受约束的 Levenshtein DP，可以冻结上下文区域，只编辑掩码区

**我们的贡献**：将 Levenshtein Transformer 从 NLP 翻译任务迁移到符号音乐修复，并针对音乐领域的特殊性进行了系统性适配。

---

## 2. 方法概览

整体流程如下图所示：

```
完整 MIDI 序列
      ↓
[上文] ████遮盖区域████ [下文]        ← 创建修复对（beat-boundary masking）
      ↓
[上文] [PLH] [下文]                  ← 初始状态：1 个占位符
      ↓
┌──────────────────────────────────┐
│  迭代循环 (最多 10 轮)             │
│  ① Delete Head: 删除错误 token    │
│  ② Insert Head: 在间隙插入 PLH    │
│  ③ Token Head: 用真实 token 填充   │
│  收敛条件: 无任何编辑操作 → 停止    │
└──────────────────────────────────┘
      ↓
[上文] [生成的修复片段] [下文]         ← 最终输出
```

**核心创新点概括**：

| 创新 | 描述 |
|------|------|
| **带约束的 Levenshtein DP** | 仅在掩码区域内计算编辑距离，上下文区域冻结不参与 DP |
| **Beat-boundary masking** | 按音乐节拍边界掩盖，而非随机 token 位置 |
| **Per-token editable flags** | 用布尔标志替代脆弱的索引边界追踪，推理时鲁棒维护可编辑区域 |
| **2×2 编码方案实验设计** | 系统对比 relative/absolute × unbundled/bundled 四种音乐表示 |
| **Music BERT 预训练初始化** | 利用音乐领域 MLM 预训练的编码器权重热启动 |
| **Track-aware Attention Bias** | 可学习的声部关系偏置注入 attention (§9.2) |
| **条件式声部修复** | 支持给定一个声部修复另一个声部 (§9.3) |
| **系统性对比实验** | 3 个 baseline + 5 项消融 + 多维度评估指标 (§10) |

---

## 3. 音乐表示：四种编码方案的设计

### 3.1 设计动机

符号音乐（MIDI）本质上是**二维钢琴卷**（pitch × time），需要序列化为一维 token 序列才能送入 Transformer。序列化方式直接决定：
- 词汇表大小（影响预测难度）
- 序列长度（影响注意力计算量）
- 是否具有平移不变性（影响泛化能力）

我们设计了 **2×2 正交实验**，在两个独立维度上各取两种方案：

|  | Unbundled（分离式） | Bundled（捆绑式） |
|--|---------------------|-------------------|
| **Relative（相对位置）** | Scheme A (vocab=186) | Scheme C (vocab=7146) |
| **Absolute（绝对位置）** | Scheme B (vocab=187) | Scheme D (vocab=7146) |

### 3.2 Pitch 编码基础：Ternary Patch

所有方案共享相同的**音符状态编码**。钢琴卷中每个 2×2 patch 被编码为 4 位三进制数：

```
patch = [onset_ch0, onset_ch1, sustain_ch0, sustain_ch1]
每位取值 {0, 1, 2}:  0=无音符, 1=延续, 2=起音+延续
patch_value = d[0]×27 + d[1]×9 + d[2]×3 + d[3]×1  ∈ [0, 80]
```

共 $3^4 = 81$ 种 patch 值，这构成音符内容的基本表达单元。

### 3.3 维度一：Relative vs Absolute 位置编码

**Relative（相对位置，Scheme A/C）**：
```
token[i] 的位置 = token[i-1] 的位置 + relative_distance[i]
```
- 优势：**平移不变性**——将乐句整体移调后，相对编码不变，有助于泛化
- 优势：稀疏乐段（音符间距大）的位置值更小，分布更集中
- 劣势：解码需要**累积求和**，单个位置错误会级联传播

**Absolute（绝对位置，Scheme B/D）**：
```
token[i] 的位置 = 直接的 pitch_index ∈ [0, 87]
```
- 优势：**每个 token 独立可解**，无误差传播风险
- 优势：更适合并行解码
- 劣势：丧失平移不变性，移调后编码完全不同

**实验假设**：Relative 编码的平移不变性能提升泛化能力，但累积误差可能在迭代修复中被放大；Absolute 编码虽丧失不变性但对迭代修复更鲁棒。

### 3.4 维度二：Unbundled vs Bundled token 风格

**Unbundled（分离式，Scheme A/B）**：
每个音符用**两个 token** 表示——位置 token + 内容 token：
```
[81 + position, patch_value]    → 2 tokens/note
词汇表: 81(patch) + 88(position) + 特殊token ≈ 186
```

**Bundled（捆绑式，Scheme C/D）**：
每个音符用**一个 token** 表示——位置和内容合并编码：
```
bundled = position × 81 + patch_value  → 1 token/note
词汇表: 88 × 81 + 特殊token = 7146
```

**核心权衡**：

| 指标 | Unbundled (A/B) | Bundled (C/D) |
|------|-----------------|---------------|
| 词汇量 | ~186 (小) | ~7146 (大) |
| 序列长度 | 2×音符数 (长) | 1×音符数 (短) |
| Token 预测难度 | 低 (186-class) | 高 (7146-class) |
| 上下文窗口利用 | 较差 (序列长) | 较好 (序列短) |
| Embedding 参数量 | 95 KB | 3.6 MB |

**实验假设**：Bundled 的序列更短，Transformer 能看到更大的音乐上下文范围；但词汇表从 186 爆炸到 7146，Token Head 的分类难度增大约 38 倍。

### 3.5 具体编码示例

假设某 beat 有 3 个音符在 pitch [10, 35, 60]，patch 值为 [5, 23, 41]：

| 方案 | 编码 | Token 数 |
|------|------|---------|
| A (rel/unbundled) | `[91, 5, 106, 23, 106, 41, END]` | 7 |
| B (abs/unbundled) | `[TRACK0, 91, 5, 116, 23, 141, 41]` | 7 |
| C (rel/bundled) | `[SPLIT_0, 815, 2048, 2066]` | 4 |
| D (abs/bundled) | `[SPLIT_0, 815, 2858, 4901]` | 4 |

Bundled 编码节省约 **43%** 的 token 数。

### 3.6 双轨表示

输入为双轨钢琴卷（旋律 Track 0 + 伴奏 Track 1）。每个 beat 内两个声部交替出现：

```
[BAR] [SPLIT_0] melody_notes... [SPLIT_1] accomp_notes... [BAR] ...
```

SPLIT_0/SPLIT_1（或 TRACK0_START/TRACK1_START）标记声部切换，使模型能区分旋律与伴奏进行独立建模。

---

## 4. 模型架构：三头共享编码器

### 4.1 为什么用 Encoder-only 而非 Encoder-Decoder

原始 LevT (Gu et al., 2019) 采用 Encoder-Decoder 架构，因为翻译任务中源语言和目标语言是不同的序列。但音乐修复的本质是**在同一个序列上做局部编辑**——上下文和待修复区域共享同一个 token 空间、同一套注意力机制。

Encoder-only 的优势：
1. **与 BERT 预训练一致**：直接复用 Music BERT 的编码器权重，无需设计 cross-attention 初始化
2. **全局注意力**：每个位置可同时看到左右上下文，天然适合修复场景
3. **参数高效**：相比 Encoder-Decoder 减少约 40% 参数量（无 Decoder + cross-attention）
4. **简洁性**：三头直接从共享 hidden states 分支，不需要 Decoder 的自回归掩码

### 4.2 架构细节

```
Input → Token Embedding → Sinusoidal PE → TransformerEncoder (8L, 512H, 8Attn, 2048FFN)
                                                ↙          ↓          ↘
                                           Del Head    Ins Head    Tok Head
                                           (B,L,2)   (B,L+1,21)  (B,L,V)
```

| 组件 | 规格 | 设计理由 |
|------|------|---------|
| Hidden dim | 512 | 与 Music BERT 对齐，便于权重迁移 |
| Layers | 8 | 参数量 ~32.8M，在 2×4090 上可训练 |
| Attention heads | 8 | 每头 64 维，多头覆盖不同范围的音乐依赖 |
| FFN dim | 2048 | 4× hidden，标准 Transformer 配置 |
| Activation | GELU | 比 ReLU 更平滑，BERT 标准选择 |
| Norm | Pre-LayerNorm | 训练更稳定 |
| PE | Sinusoidal | 支持变长序列，不需要学习 |

### 4.3 三个预测头的设计

**Deletion Head — "哪些 token 是错的？"**
```
LayerNorm(512) → Dropout → Linear(512, 2)
```
- 每个 token 二分类：KEEP(0) / DELETE(1)
- 设计最简——删除是最简单的判断，不需要复杂结构

**Insertion Head — "每个间隙该插入多少占位符？"**
```
[h_{i-1}; h_i] → LayerNorm(1024) → Dropout → Linear(1024, 21)
```
- **关键设计**：对相邻 hidden states 做拼接（concatenation），构造 "间隙表示"
- 长度为 L 的序列有 L+1 个间隙（含首尾），输出 `(B, L+1, 21)`
- 预测范围 0~20：每个间隙最多插入 20 个 PLH

**为什么拼接相邻表示？** 间隙的"应该插入多少"取决于两侧内容的不连续程度。相邻 token 的 hidden state 拼接直接编码了这种局部不连续信息，比仅用单侧或全局池化更精准。

**Token Head — "占位符应该填什么？"**
```
Linear(512, 512) → GELU → LayerNorm → Linear(512, V)
```
- 在含 PLH 的序列上运行，为每个位置（特别是 PLH 位置）预测目标 token
- 使用两层 MLP（而非单层 Linear），因为 token 预测是三个子任务中最难的

### 4.4 两次前向传播

训练时 `compute_loss` 需要两次独立的 encoder forward：

| Forward | 输入序列 | 输出 | 为什么 |
|---------|---------|------|--------|
| Pass 1 | $z$（中间状态） | del_logits + ins_logits | 在"当前状态"上判断删改 |
| Pass 2 | $z_{tok}$（$z$ + 插入 PLH 后） | tok_logits | 在"插入 PLH 后的状态"上预测填充 |

这两次 forward **不可合并**，因为 $z_{tok}$ 比 $z$ 更长（插入了额外的 PLH token），是一个完全不同的序列。这是 Levenshtein 框架的固有开销，但换来的是长度自适应的能力。

### 4.5 Music BERT 预训练初始化

我们为四种编码方案各自预训练了 Music BERT（MLM 目标），然后将 embedding 层和 encoder 层的权重迁移到 LevT 的对应模块。三个预测头随机初始化（Xavier uniform）。

**迁移策略的合理性**：
- BERT 已学会了音乐 token 之间的局部依赖关系（和弦结构、节奏模式等）
- LevT 的 encoder 与 BERT 架构完全一致（同一个 TransformerEncoder），权重可直接复用
- 三个头是全新任务，随机初始化不影响收敛（从实验看，头的 loss 在前 3 个 epoch 快速下降）

---

## 5. 训练策略

### 5.1 带约束的 Levenshtein DP（核心创新）

标准 Levenshtein 距离计算的是将序列 A 变为序列 B 所需的最少编辑操作数。**我们的关键修改**：只在掩码区域 $[m_s, m_e)$ 内计算 DP，上下文区域的标签被强制设为"不操作"。

```
完整序列:  [context_left | mask_region | context_right]
DP 范围:                  |←── 仅此区域 ──→|
context 标签:  del=0, ins=0（冻结）
```

具体流程：
1. 提取中间状态 $z$ 的掩码区域 $z[m_s:m_e]$，以及目标序列 $y$ 的对应区域
2. 仅在这两个子序列之间做标准 Levenshtein DP
3. 回溯 DP 路径，提取三类标签：
   - `del_labels[i]` ∈ {0, 1}：是否删除
   - `ins_labels[i]` ∈ [0, 20]：间隙 $i$ 处插入 PLH 数
   - `tok_labels`：PLH 位置对应的目标 token
4. 上下文位置的标签设为 ignore_index=-100（不参与 loss 计算）

**为什么不在全序列上做 DP？** 如果允许 DP 在上下文区域操作，它可能会"发现"通过修改上下文来减少总编辑距离的路径，这违反了修复任务"上下文不可变"的约束。受约束 DP 保证了训练标签与推理行为的一致性。

### 5.2 中间状态采样（训练数据增强）

在实际推理中，模型看到的输入是"修复进行到一半"的中间状态——有些 token 已正确、有些还有错误。训练时需要模拟这种中间状态来生成训练对。

我们的采样策略对掩码区域的每个 token 独立执行：
- **30% 概率删除**：模拟之前迭代中多余的 token
- **20% 概率替换**为随机 token：模拟之前迭代中预测错误的 token
- **50% 概率保留**正确 token：提供部分正确信息作为锚点

**设计考量**：
- 50% 保留率确保模型能利用已有的正确 token（而非从零开始）
- 30% 删除率模拟"序列中有冗余 token 需要清理"的场景
- 20% 替换率模拟"有错误 token 需要先删除再重新生成"的场景
- 这三种破坏覆盖了实际迭代修复中可能出现的所有中间状态类型

### 5.3 Beat-boundary Masking（音乐先验）

创建训练对时的掩盖策略：

1. 将完整序列按 **beat 边界** 解析为结构化的 beat 列表
2. 随机选择**连续的若干 beat** 作为掩码区域（12.5%~50% 的 beat 数量）
3. 至少保留首尾各 1 个 beat 作为上下文

**为什么按 beat 边界而非随机 token 位置？**

音乐具有层级结构：音符 → beat → 小节 → 乐段。随机遮盖 token 可能在一个 beat 中间截断，产生不自然的边界（如一个和弦被切成一半）。按 beat 边界掩盖确保：
- 上下文的 beat 结构完整，为模型提供干净的音乐先验
- 目标区域的 beat 结构完整，训练标签语义明确
- 与实际应用场景对齐（用户通常按小节或乐句单位指定修复区域）

### 5.4 损失函数

$$\mathcal{L} = \mathcal{L}_{del} + \mathcal{L}_{ins} + \mathcal{L}_{tok}$$

| 损失项 | 公式 | 说明 |
|--------|------|------|
| $\mathcal{L}_{del}$ | `CE(del_logits, del_labels, ignore=-100)` | 二分类 CE，上下文位置忽略 |
| $\mathcal{L}_{ins}$ | `CE(ins_logits, ins_labels, ignore=-100)` | 21 分类 CE，仅掩码区间隙参与 |
| $\mathcal{L}_{tok}$ | `CE(tok_logits[PLH], tok_targets, ε=0.1)` | 仅 PLH 位置，label smoothing 缓解过拟合 |

三项等权重（$w_{del}=w_{ins}=w_{tok}=1.0$）。从实验数据看：
- Del/Ins loss 在前 5 个 epoch 快速降至极低值（< 0.1），说明删除/插入是相对简单的子任务
- Tok loss 占最终 total loss 的 **95%+**，说明 token 预测是真正的难点
- 未来可考虑动态权重调整，将更多梯度分配给 tok head

---

## 6. 迭代推理：可编辑标志机制

### 6.1 三阶段循环

```python
初始状态: [上文(frozen)] + [PLH] + [下文(frozen)]

for step in 1..max_iter:
    ① Delete: 对 editable 区域的 token，概率 > 0.5 则删除
    ② Insert: 对 editable 间隙，预测插入 PLH 数量
    ③ Fill:   对所有 PLH 位置，预测目标 token

    if 无删除 + 无插入 + 无填充 → 收敛，停止
```

### 6.2 Per-token Editable Flags（核心创新）

这是我们对原始 LevT 推理流程的重要改进。原始 LevT 用 index-based 的 mask_start/mask_end 追踪可编辑区域边界，但这种方式在迭代修复中极其脆弱：

**问题**：每次删除或插入都会改变序列长度，导致索引移位。例如：
- 在 position 5 删除 3 个 token 后，原来 mask_end=20 应该变成 17
- 在 position 10 插入 5 个 PLH 后，mask_end 又要变成 22
- 多次操作叠加，索引追踪极易出错

**我们的方案**：为每个 token 维护布尔标志 `editable[i]`：
- `True`：可编辑（属于掩码区域或新插入的 token）
- `False`：冻结（属于上下文区域，不可删除/修改）

**规则定义**：
| 操作 | editable 更新规则 |
|------|------------------|
| 删除 token $i$ | 移除 `editable[i]`，剩余标志不变 |
| 在间隙 $i$ 插入 PLH | 新 PLH 的 `editable = True` |
| 填充 PLH → real token | 保持 `editable = True`（仍可在后续迭代中被删除） |
| 上下文 token | 始终 `editable = False`，不可删除 |

**间隙可插入性判断**：
```
gap[i] 可插入 ⟺ editable[i-1] = True 或 editable[i] = True
```
即：只有在可编辑区域的边界处或内部才允许插入。这防止了模型在上下文中间"凭空"插入 token。

**为什么这个设计更优？**
1. **与序列长度变化解耦**：标志跟着 token 走，增删操作自然保持一致
2. **支持非连续可编辑区域**：理论上可处理多个分离的掩码区域
3. **实现简洁**：Python list 的增删操作天然维护 editable flags 的对应关系

### 6.3 首次迭代的 Insertion Seed

特殊情况：首次迭代时掩码区域可能只有 1 个 PLH，此时 `editable = [False, ..., True, ..., False]`。为确保第一次能插入足够多的 PLH，在掩码起始位置设置 `insertion_seed`——即使该间隙两侧有冻结 token，也允许插入。

这是一个工程细节但对正确性至关重要：没有 insertion seed，模型第一轮只能在 1 个 PLH 处操作，无法扩展到目标长度。

### 6.4 收敛与终止

收敛条件简洁而有效：
- 删除 0 个 token **且** 插入 0 个 PLH **且** 填充 0 个 PLH → 停止
- 或达到 max_iter（默认 10 轮）

从迭代修复的语义来看，"无任何操作"意味着模型认为当前序列已经是最优状态。实际实验中，大多数样本在 3~5 轮内收敛。

---

## 7. 设计决策分析

### 7.1 为什么非自回归（NAR）优于自回归（AR）？

| 维度 | AR | NAR (LevT) |
|------|-----|-----------|
| 上下文利用 | 仅左侧 | 双向全局 |
| 生成长度 | 需预先确定或用 EOS 终止 | 由 Ins Head 动态推断 |
| 推理复杂度 | $O(n)$ 序列解码 | $O(k)$ 迭代（$k$ ≪ $n$） |
| 误差累积 | 严重（每步依赖前一步） | 迭代纠错减轻 |
| 多样性 | 需要 beam search 或采样 | 天然支持温度/top-k |

### 7.2 Encoder-only vs Encoder-Decoder

原始 LevT 论文使用 Encoder-Decoder，因为翻译中 source ≠ target。但音乐修复中：
- Source（含掩码的序列）和 target（完整序列）共享同一 token 空间
- 上下文 token 在 source 和 target 中完全相同
- 不需要 cross-attention 来"翻译"

因此 Encoder-only 更自然、更高效。若实验证明容量不足，可回退到 Encoder-Decoder（ENHANCEMENT E4）。

### 7.3 训练-推理一致性考量

| 环节 | 训练 | 推理 | 一致性 |
|------|------|------|-------|
| 上下文保护 | 软约束（loss mask, ignore=-100） | 硬约束（editable flags 冻结） | 存在 gap |
| 中间状态来源 | 随机采样（30/20/50） | 模型自身前一轮输出 | 存在 gap |
| 操作序列 | DP 最优路径 | 模型贪心预测 | 基本一致 |

**承认的局限**：训练时的软约束（通过 loss mask 忽略上下文梯度）与推理时的硬约束（直接冻结 token）之间存在不匹配。这是未来优化方向之一（ENHANCEMENT E5：训练时也施加硬约束）。

### 7.4 四方案对比的实验设计合理性

2×2 正交实验设计使我们能**独立评估**两个设计维度的影响：
- 固定 token 风格，比较 A vs B（relative vs absolute，unbundled）
- 固定 token 风格，比较 C vs D（relative vs absolute，bundled）
- 固定位置编码，比较 A vs C（unbundled vs bundled，relative）
- 固定位置编码，比较 B vs D（unbundled vs bundled，absolute）

四种方案共享完全相同的模型架构、训练超参数、数据集和评估流程，唯一变量是编码方案。这确保了对比的公平性。

---

## 8. 与现有工作的关系

### 8.1 与原始 LevT (Gu et al., 2019) 的区别

| 维度 | 原始 LevT | 本工作 |
|------|----------|--------|
| 任务 | 机器翻译 | 音乐修复（条件生成） |
| 架构 | Encoder-Decoder | Encoder-only |
| 编辑范围 | 全序列可编辑 | 受约束（仅掩码区域可编辑） |
| 边界追踪 | 无（全序列可编辑） | Per-token editable flags |
| 预训练 | 无 | Music BERT 初始化 |
| 输入表示 | BPE subword | 音乐领域定制编码（4 种方案） |

### 8.2 与 Mask-Predict (Ghazvininejad et al., 2019) 的区别

Mask-Predict 也是 NAR 迭代式生成，但：
- 假设目标长度已知（从 source 长度预测），**不支持可变长度**
- 每轮只替换低置信 token，**不支持删除和插入**
- 本工作的 delete→insert→fill 三步法更灵活

### 8.3 在音乐 AI 领域的定位

| 方法 | 生成方式 | 长度处理 | 上下文利用 |
|------|---------|---------|-----------|
| Music Transformer (Huang et al.) | AR | 固定或 EOS | 仅左侧 |
| MuseNet (OpenAI) | AR | 固定 | 仅左侧 |
| MusicBERT + Mask-Predict | NAR | 固定 | 双向 |
| **LevT Music Inpainting (Ours)** | **NAR 迭代** | **自适应** | **双向** |

本工作的独特之处在于：**首次在符号音乐领域实现长度自适应的双向修复**。

---

## 9. 声部协调编辑 (Track-coordinated Editing)

### 9.1 动机：声部关系的显式建模

双轨音乐的 token 混在同一序列中，Transformer 的 self-attention 对所有 token 一视同仁。但音乐中存在两种本质不同的依赖关系：

- **同声部内依赖**：旋律线条的流畅连贯、伴奏型的一致延续
- **跨声部间依赖**：和声兼容（不出现不协调音程）、节奏互补（旋律密时伴奏疏）

标准 Transformer 必须从数据中隐式学习这两种关系的区别。但音乐编码本身已包含声部信息（SPLIT_0/SPLIT_1 标记），我们可以将这种结构先验显式注入 attention 机制。

### 9.2 Track-aware Attention Bias (T1.1)

**方法**：在 attention score 上注入可学习的声部关系偏置。

$$\text{attn}(h, i, j) \mathrel{+}= \text{track\_bias}[h, \text{track}(i), \text{track}(j)]$$

每个 token 被分配 `track_id ∈ {0=结构, 1=旋律, 2=伴奏}`。8 个 attention head 各自学习一个 3×3 偏置矩阵，共 72 个参数（+0.0002%）。

**实现方式**：通过 PyTorch `TransformerEncoder` 的 `mask` 参数注入 additive float tensor `(B×nH, L, L)`，无需修改 attention 层代码。

**设计选择**：

| 备选方案 | 未选原因 |
|----------|---------|
| Track embedding 加到 hidden state | 永久改变表示空间；bias 只调节"谁关注谁"，更精确 |
| 单一全局 bias（非 per-head） | 限制表达力；不同 head 可学不同声部交互模式 |
| 固定正/负 bias | 丧失灵活性；让模型自己学什么关系有利 |

**初始化全零** → 训练初期等同无 bias，不破坏预训练权重。旧 checkpoint 可通过 `strict=False` 加载，track_bias 从零开始训练。

**贡献定位**：声部标记已存在于 BEAT 编码中（SPLIT_0/SPLIT_1）。T1.1 的贡献是**归纳偏置的注入**——将已有结构信息显式注入 attention 机制。类似 positional encoding 将位置信息显式注入，虽然位置信息本身已隐含在 token 顺序中。

### 9.3 条件式声部修复 (T1.2 Conditional Track Inpainting)

**动机**：标准训练中两个声部同时被 corrupt，模型只学会"从噪声中同时恢复两轨"。但实际音乐创作中更常见的场景是：**旋律已知，生成配套伴奏**（或反之）。

**方法**：训练时随机混合三种 masking 模式：

| 模式 | 比例 | 行为 |
|------|------|------|
| `both` | 60% | 两个声部都 corrupt（原始行为） |
| `melody_only` | 20% | 只 corrupt 旋律，伴奏完整保留为上下文 |
| `accomp_only` | 20% | 只 corrupt 伴奏，旋律完整保留为上下文 |

**核心机制**：

1. `token_editable` 标记 mask 区域内哪些 token 可编辑（目标声部=True, 其余=False）
2. Frozen token 在中间状态采样时始终保持原值——不删除、不替换
3. Levenshtein DP 自然为 frozen token 生成 `del=0, ins=0` 标签（因为它们在中间状态和目标中完全相同）
4. **约束完全通过数据构造实现，不需要修改 loss 函数或模型架构**

**推理端适配**：新增 `inpaint_track_conditional()` 方法。与标准 `inpaint()` 的关键区别：
- 输入是**完整序列**（不删除 mask 区域），frozen 声部保留作为上下文
- `editable` 标记仅对目标声部的 token 设为 True
- 不需要 `insertion_seed`（mask 区域内已有 editable token）
- 复用已有的 del→ins→fill 迭代循环，editable 约束机制天然适用

**获得的新能力**：同一个模型，三种使用方式——双轨修复、给定旋律生成伴奏、给定伴奏修复旋律。

### 9.4 各编码方案的声部识别

| Scheme | 声部标记机制 | get_track_ids 策略 |
|--------|------------|-------------------|
| A (rel/unbundled) | 无显式标记 | beat 交替：偶数=旋律, 奇数=伴奏 |
| B (abs/unbundled) | TRACK0_START / TRACK1_START | 遇到标记切换状态 |
| C (rel/bundled) | SPLIT_0 / SPLIT_1 | 遇到标记切换 + EMPTY 交替 |
| D (abs/bundled) | SPLIT_0 / SPLIT_1 | 同 Scheme C |

输出统一为 `{0=结构, 1=旋律, 2=伴奏}`，上层的 attention bias 和 conditional masking 代码完全一致。

---

## 10. 对比实验设计

### 10.1 Baseline 方法

所有方法使用**完全相同的确定性 mask 位置**（中间 30% beats，seed=42），确保公平对比。

| Baseline | 方法 | 回答的问题 |
|----------|------|-----------|
| **Random** | 从训练集 unigram 分布采样 | 模型是否真的学到了音乐规律？ |
| **Copy-Context** | 复制相邻 beats 填充 | 模型是否超越"复制粘贴"？音乐有重复性，这个 baseline 不弱 |
| **BERT Mask-Predict** | CMLM 迭代解码（并行填充，线性衰减 remask） | 迭代编辑（LevT）vs 并行填充（CMLM），非自回归框架内部对比 |
| **Vanilla LevT** | 去掉受约束 DP，全序列可编辑 | 受约束 DP 是否必要？ |

**BERT Mask-Predict 的解码算法**（Ghazvininejad et al., 2019 CMLM）：
1. 初始：context_left + [MASK]×gt_length + context_right
2. 每轮 t (共 T=10 轮)：BERT forward → argmax 预测 + confidence (max softmax prob)
3. 保留 confidence 最高的预测，重新 mask 最低 confidence 的 $\lfloor n \times \frac{T-t-1}{T} \rfloor$ 个位置
4. 最后一轮全部 argmax，不再 remask

### 10.2 评估指标体系

| 维度 | 指标 | 意义 |
|------|------|------|
| 序列精度 | Token Accuracy | 逐位置精确匹配率 |
| 序列精度 | Normalized Edit Distance | 编辑距离 / 目标长度，越小越好 |
| 音高 | Pitch Accuracy | 仅比较 token 的音高分量 (`token // 81`) |
| 节奏 | Pattern Accuracy | 仅比较 token 的节奏分量 (`token % 81`) |
| 节奏 | Rhythm Accuracy | 同 pattern，但忽略音高的独立评估 |
| 帧级 | Framewise F1 | 解码为钢琴卷后逐帧 precision/recall/F1，更接近听感 |
| 结构 | Length Accuracy | 生成长度 / 目标长度 |
| 结构 | Note Density Ratio | 修复区域音符密度 / 上下文音符密度 |
| 效率 | Average Iterations | 收敛轮数（仅 LevT） |

**Pitch vs Rhythm 分离评估**：将音符的两个正交维度（音高 + 节奏）独立评估，诊断模型在哪个维度更强/更弱。

**Framewise F1**：将 token 序列解码回 `(88, T)` 二值钢琴卷矩阵（88 钢琴键 × T 时间帧），在物理时间-音高空间做帧级评估。这比 token 级指标更接近人耳听感，因为一个"错误"的 token 在钢琴卷上可能只偏差 1-2 个半音（仍然可接受），但 token accuracy 会判为完全错误。

### 10.3 消融实验设计

| 消融项 | 对比设置 | 验证的假设 |
|--------|---------|-----------|
| 有/无 Track-aware bias | T1.1 模型 vs 原始模型 | 声部关系偏置是否提升质量 |
| 有/无 Conditional Inpainting | T1.2 模型 vs 仅 T1.1 | 条件训练是否提升单轨修复 |
| Beat-boundary vs Random span | 修改 masking 策略 | 音乐先验是否优于通用策略 |
| BERT 初始化 vs 随机初始化 | 从零训练 | 预训练权重的价值 |
| 受约束 vs 全序列 DP | Vanilla LevT baseline | 约束是否必要 |

---

## 11. 局限性与未来方向

| 局限 | 分析 | 状态 |
|------|------|------|
| 训练-推理不匹配 | 训练用随机采样，推理用模型自身输出 | 未解决 (E2: DAgger-style) |
| 固定破坏分布 | 30/20/50 可能非最优 | 未解决 (E3: Curriculum Learning) |
| 上下文软约束 | Loss mask ≠ 硬冻结 | 未解决 (E5: 训练时硬约束) |
| max_insert=20 偏大 | 实际分布可能集中在 0-10 | 未解决 (E7: 数据驱动上界) |
| ~~仅全 beat 遮盖~~ | ~~无法处理单轨修复~~ | **已解决** (§9.3 T1.2) |
| 声部标记不统一 | 4 scheme 各自实现 get_track_ids | 待讨论 |

---

## References

1. Gu, J., Wang, C., & Zhao, J. (2019). Levenshtein Transformer. *NeurIPS*.
2. Ghazvininejad, M., Levy, O., Liu, Y., & Zettlemoyer, L. (2019). Mask-Predict: Parallel Decoding of Conditional Masked Language Models. *EMNLP*.
3. Huang, C. A., et al. (2019). Music Transformer: Generating Music with Long-Term Structure. *ICLR*.
4. Vaswani, A., et al. (2017). Attention Is All You Need. *NeurIPS*.
5. Devlin, J., et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers. *NAACL*.
