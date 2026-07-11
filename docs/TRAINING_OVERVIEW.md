# 钢琴音乐纠错模型训练总结

> 最后更新：2026-02-22

本文档汇总四种编码方案（A / B / C / D）的完整两阶段训练结果，构成 2×2 消融实验：
1. **BERT MLM 预训练**：学习音乐 token 的上下文表示
2. **GECToR 纠错训练**：基于预训练 BERT 的序列标注纠错模型

---

## 1. 编码方案概览 (2×2 消融实验)

```
┌──────────┬──────────────────────┬──────────────────────┐
│          │ 分离编码 (2tok/note) │ 捆绑编码 (1tok/note) │
├──────────┼──────────────────────┼──────────────────────┤
│ 绝对位置 │ Scheme A (no_pair)   │ Scheme D (abs_bundled)│
├──────────┼──────────────────────┼──────────────────────┤
│ 相对位置 │ Scheme B (rel_pair)  │ Scheme C (with_pair) │
└──────────┴──────────────────────┴──────────────────────┘
```

| 特征 | Scheme A | Scheme B | Scheme C | Scheme D |
|------|---------|---------|---------|---------|
| 位置编码 | 绝对位置 | 相对位置 | 相对位置 | 绝对位置 |
| Token 格式 | pos + val 分离 | pos + val 分离 | bundled (rel_pos×81+val) | bundled (abs_pos×81+val) |
| 每音符 Token 数 | 2 | 2 | 1 | 1 |
| BERT 词表大小 | 186 | 185 | 7145 | 7145 |
| GECToR 标签空间 | 350 | 350 | 14258 | 14258 |

四种方案共享：
- 三进制 patch 编码 (patch_h=1, patch_w=4, 81 种 pattern, 88 键)
- 序列格式：[BOS][TIME_SIG][BPM][BAR][拍交错]...[EOS]
- Beat 级高低声部交错：[高beat0][低beat0][高beat1][低beat1]...

---

## 2. BERT MLM 预训练

### 2.1 模型与训练配置

| 配置 | Scheme A | Scheme B | Scheme C | Scheme D |
|------|---------|---------|---------|---------|
| 模型 | BertForMaskedLM | BertForMaskedLM | BertForMaskedLM | BertForMaskedLM |
| hidden / layers / heads | 512 / 8 / 8 | 512 / 8 / 8 | 512 / 8 / 8 | 512 / 8 / 8 |
| intermediate | 2048 | 2048 | 2048 | 2048 |
| 总参数量 | 26.6M | ~26.6M | 30.2M | 30.2M |
| 词表大小 | 186 | 185 | 7145 | 7145 |
| max_seq_length | 2048 | 2048 | 2048 | 2048 |
| batch_size (per GPU) | 32 | 32 | 32 | 16 |
| gradient_accumulation | 4 | 4 | 4 | 4 |
| 有效 batch_size | 256 | 256 | 256 | 256 |
| 学习率 | 1e-4 | 1e-4 | 1e-4 | 1e-4 |
| 调度器 | cosine + 10% warmup | cosine + 10% warmup | cosine + 10% warmup | cosine + 10% warmup |
| weight_decay | 0.01 | 0.01 | 0.01 | 0.01 |
| Epochs | 30 | 32 | 30 | 30 |
| GPU | 2× 24GB | 2× 24GB | 2× 24GB | 4× 24GB |
| 训练数据 | 192,788 npz (95/5 split) | 192,788 npz | 192,788 npz | 192,788 npz |

### 2.2 训练结果

| 指标 | Scheme A | Scheme B | Scheme C | Scheme D |
|------|---------|---------|---------|---------|
| **最佳 Eval Loss** | 0.2823 | **0.2242** | 0.4490 | 0.5017 |
| 最佳 Eval PPL | 1.33 | **1.25** | 1.57 | 1.65 |
| 最佳 Step | 20,000 | 40,000 | 36,000 | 19,000 |
| 最终 Train Loss | 0.3265 | 0.2568 | 0.4910 | 0.6035 |
| 训练时长 | 8.6h | 27.3h | 8.3h | 4.6h |
| 吞吐量 | 14,188 tok/s | 14,045 tok/s | 8,927 tok/s | — |

**注意**：MLM loss 不可跨编码方案直接比较——Scheme C/D 的词表是 A/B 的 ~38 倍，预测空间大得多，loss 自然偏高。同为 bundled 编码的 C 和 D 之间可以对比：D (绝对位置, 0.5017) 显著差于 C (相对位置, 0.4490)。

### 2.3 收敛特征

| 阶段 | Scheme A | Scheme B | Scheme C | Scheme D |
|------|---------|---------|---------|---------|
| 突破瓶颈 Epoch | 8 | 7 | 10 | 未完全突破 |
| 快速收敛区间 | epoch 8-10 | epoch 7-10 | epoch 10-14 | epoch 20-26 |
| 收敛后 Loss 区间 | 0.28-0.33 | 0.22-0.26 | 0.45-0.49 | 0.50-0.60 |

四种方案都呈现 **慢启动 → 快速收敛 → 精调平稳** 的三阶段训练模式。Scheme D 收敛最慢且最终 loss 最高，表明绝对位置捆绑编码对 MLM 任务最困难。

### 2.4 Checkpoint 路径

| Scheme | Best Model 路径 |
|--------|---------------|
| A | `music_bert_no_pair/checkpoints/music_bert_no_pair/best_model/` |
| B | `music_bert/checkpoints/music_bert_no_pair_related/best_model/` |
| C | `music_bert_with_pair/checkpoints/music_bert_with_pair/best_model/` |
| D | `music_bert_absolute_bundled/checkpoints/music_bert_absolute_bundled/best_model/` |

---

## 3. GECToR 纠错训练

GECToR 训练分两个阶段：
- **Stage I**：合成错误数据训练，先冻结 BERT 2 epochs 再解冻
- **Stage III**：混入 25% 干净数据微调（clean_ratio=0.25），3 epochs

### 3.1 模型与训练配置

| 配置 | Scheme A | Scheme B | Scheme C | Scheme D |
|------|---------|---------|---------|---------|
| 标签空间 | 350 | 350 | 14258 | 14258 |
| 总参数量 | 26.5M | 26.5M | 37.2M | 37.2M |
| Stage I epochs | 20 | early stop@9 | 18 | 20 |
| Stage I batch_size | 32 | 32 | 16* | 16* |
| Stage III epochs | 3 | 3 | 3 | 3 |
| Stage III batch_size | 32 | 32 | 16 | 16 |
| Stage III lr | 5e-6 | 5e-6 | 5e-6 | 5e-6 |
| gradient_accumulation | 2 | 2 | 4 | 4 |
| clean_ratio | 0.25 | 0.25 | 0.25 | 0.25 |
| freeze_epochs (Stage I) | 2 | 2 | 2 | 2 |
| BERT unfreeze lr | 1e-5 | 1e-5 | 1e-5 | 1e-5 |
| head lr (Stage I) | 1e-4 | 1e-4 | 1e-4 | 1e-4 |
| GPU | 2× 24GB | 2× 24GB | 2× 24GB | 4× 24GB |

*Scheme C/D 标签空间为 14258，tag logits [batch, 2048, 14258] 占用大量显存，batch_size=32 会 OOM。

### 3.2 Stage I 训练结果

| 指标 | Scheme A | Scheme B | Scheme C | Scheme D |
|------|---------|---------|---------|---------|
| **最佳 Edit F1** | 0.8434 | 0.8630 | **0.8741** | 0.8428 |
| 最佳 F1 Step | 20,000 | 21,000 | 24,000 | 13,000 |
| 最佳 Eval Loss | 0.1478 | 0.1437 | 0.2385 | 0.2431 |
| 最佳 Loss Step | 25,000 | 20,000 | 22,000 | 11,000 |
| Precision@best_F1 | 0.8034 | 0.8217 | 0.8608 | 0.8069 |
| Recall@best_F1 | 0.8876 | 0.9087 | 0.8878 | 0.8820 |
| 实际训练 Epochs | 20 | 9 (early stop) | 18 | 20 |
| 训练时长 | 9.3h | 10.7h | 10.9h | 5.7h |

### 3.3 Stage III 训练结果

| 指标 | Scheme A | Scheme B | Scheme C | Scheme D |
|------|---------|---------|---------|---------|
| **最佳 Eval Loss** | 0.1465 | **0.1445** | 0.2401 | 0.2436 |
| 最佳 Loss Step | 3,000 | 2,000 | 4,000 | 2,000 |
| 最佳 Edit F1 | 0.8548 | 0.8617 | **0.8733** | 0.8431 |
| 最终 Edit F1 | 0.8437 | 0.8599 | 0.8717 | 0.8414 |
| 最终 Precision | 0.8023 | 0.8172 | 0.8558 | 0.8047 |
| 最终 Recall | 0.8895 | 0.9072 | 0.8881 | 0.8817 |
| 训练时长 | 1.4h | 1.5h | 1.7h | 0.9h |

### 3.4 GECToR 最终对比（Stage III 最佳结果）

| 指标 | Scheme A | Scheme B | Scheme C | Scheme D |
|------|---------|---------|---------|---------|
| **Edit F1** | 0.8548 | 0.8617 | **0.8733** | 0.8431 |
| Precision | 0.8354 | 0.8189 | **0.8600** | 0.8060 |
| Recall | 0.8752 | **0.9097** | 0.8869 | 0.8839 |
| Eval Loss | 0.1526 | 0.1445 | 0.2419 | 0.2437 |

**关键发现**：

1. **Scheme C (相对位置×捆绑) 纠错 F1 最高 (0.8733)**，尽管其 BERT MLM loss 最高——bundled 编码虽然 MLM 预测更难，但包含的信息更紧凑，下游纠错任务反而受益。
2. **Scheme B (相对位置×分离) 召回率最高 (0.9097)**，能检出更多错误，但精确度偏低。
3. **相对位置编码 > 绝对位置编码**：C > D（0.8733 vs 0.8431），B > A（0.8617 vs 0.8548）。在捆绑编码上差距更大（+3.0% vs +0.7%）。
4. **捆绑编码 + 相对位置是最优组合**，但**捆绑编码 + 绝对位置是最差组合** (D=0.8431)，甚至略低于分离+绝对 (A=0.8548)。
5. Stage III 对 Stage I 提升有限（0-1.1% F1），合成数据 Stage I 已基本收敛。

### 3.5 2×2 消融实验结论

```
Edit F1 排名: C (0.8733) > B (0.8617) > A (0.8548) > D (0.8431)

              分离编码        捆绑编码
绝对位置    A: 0.8548       D: 0.8431
相对位置    B: 0.8617       C: 0.8733

位置编码效应:  相对 > 绝对 (平均 +1.9%)
编码格式效应:  取决于位置编码类型
  - 相对位置下: 捆绑 > 分离 (+1.2%)
  - 绝对位置下: 分离 > 捆绑 (+1.2%)
交互效应:     相对+捆绑 最优, 绝对+捆绑 最差
```

### 3.6 GECToR Checkpoint 路径

| Scheme | Stage III Best Model |
|--------|---------------------|
| A | `correction_no_pair/checkpoints/gector_no_pair/best_model/` |
| B | `correction/checkpoints/gector/best_model/` |
| C | `correction_with_pair/checkpoints/gector_with_pair/best_model/` |
| D | `correction_absolute_bundled/checkpoints/gector_absolute_bundled/best_model/` |

---

## 4. 训练时间总结

| 阶段 | Scheme A | Scheme B | Scheme C | Scheme D |
|------|---------|---------|---------|---------|
| BERT 预训练 | 8.6h | 27.3h | 8.3h | 4.6h |
| GECToR Stage I | 9.3h | 10.7h | 10.9h | 5.7h |
| GECToR Stage III | 1.4h | 1.5h | 1.7h | 0.9h |
| **合计** | **19.3h** | **39.5h** | **20.9h** | **11.2h** |

Scheme D 使用 4 GPU 因此训练更快。

---

## 5. 代码目录

所有代码位于 `src/` 下：

| 模型 | Scheme A | Scheme B | Scheme C | Scheme D |
|------|---------|---------|---------|---------|
| BERT | `music_bert_no_pair/` | `music_bert/` | `music_bert_with_pair/` | `music_bert_absolute_bundled/` |
| GECToR | `correction_no_pair/` | `correction/` | `correction_with_pair/` | `correction_absolute_bundled/` |

---

## 6. 训练过程中遇到的问题与解决方案

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| BERT no_pair batch=64 OOM | 2×24GB 显存不足 | batch_size 64→32 |
| GECToR with_pair BERT 解冻 OOM | tag logits [32,2048,14258]≈3.5GB | batch_size 32→16, grad_accum 2→4 |
| with_pair label_extractor AssertionError | 空 beat 被 perturb_insert 错误添加音符 | perturbation.py 跳过空 beat |
| DDP unused parameter 警告 | BertModel pooler 不参与梯度计算 | `add_pooling_layer=False` |
| Checkpoint 加载缺失 pooler keys | best_model 保存时无 pooler | `strict=False` |
| nohup+conda 日志不刷新 | Python stdout 缓冲 | 通过 TensorBoard events 监控 |
| GECToR B 提前结束 | early stopping (patience=3) | 正常行为，非错误 |

---

## 7. FELIX 两阶段编辑系统

FELIX 风格的编辑模型：**Tagger**（标注编辑操作）→ **Inserter**（MLM 填充插入位置）。
用于钢琴伴奏的生成与再编辑。

### 7.1 系统架构

```
输入序列 → [Tagger] → 编辑标签 (KEEP/DELETE/REPLACE/APPEND_1..8)
                ↓
        骨架序列 (删除+插入MASK)
                ↓
         [Inserter] → 填充 MASK → 输出序列
```

- **Tagger**：TransformerEncoder (hidden=512, layers=8, heads=8, pre-norm), 11 类标签
- **Inserter**：TransformerEncoder + MLM 预测头, 词表大小同对应编码方案
- 扰动策略：4 级（L1: 5-15%, L2: 15-40%, L3: 40-70%, L4: 100% 清除），仅修改伴奏(Track 1)

### 7.2 模型参数量

| 组件 | Scheme A/B (分离) | Scheme C/D (捆绑) |
|------|-------------------|-------------------|
| Tagger | ~25.3M | ~28.9M |
| Inserter | ~25.7M | ~32.8M |

### 7.3 训练配置

| 配置 | Tagger | Inserter |
|------|--------|----------|
| Epochs | C: 30, A/B/D: 18 | 30 |
| batch_size | C: 32, A/B/D: 48 | C/D: 32, A/B: 48 |
| gradient_accumulation | C: 2, A/B/D: 2 | C/D: 3, A/B: 2 |
| 有效 batch_size | C: 64, A/B/D: 96 | C/D: 96, A/B: 96 |
| 学习率 | 1e-4 | 1e-4 |
| 调度器 | cosine + 10% warmup | cosine + 10% warmup |
| GPU | 单卡 24GB | 单卡 24GB |

### 7.4 Tagger 训练结果（全部完成）

| 指标 | Scheme A | Scheme B | Scheme C | Scheme D |
|------|---------|---------|---------|---------|
| **最佳 macro F1** | 0.4778 | **0.6050** | 0.4674 | 0.4474 |
| val_acc | 97.87% | **97.85%** | 96.67% | — |
| 训练 Epochs | 18 | 18 | 30 | 18 |
| 每 Epoch 时间 | ~58min | ~59min | ~28min* | ~60min |

*Scheme C 的 Tagger 训练在 2 GPU 上进行过，每 epoch 较快。

**Tagger 排名：B (0.6050) >> A (0.4778) > C (0.4674) > D (0.4474)**

方案 B 大幅领先，APPEND 类标签准确率普遍更高（APPEND_2=60.6%, APPEND_4=57.4%），说明相对位置+分离编码对编辑操作预测最有利。

### 7.5 Inserter 训练结果

| 指标 | Scheme A | Scheme B | Scheme C | Scheme D |
|------|---------|---------|---------|---------|
| **最佳 top1 acc** | ✅ 完成 | ✅ 完成 | **0.4297** | ✅ 完成 |
| 训练进度 | **30/30 ✅** | **30/30 ✅** | **30/30 ✅** | **30/30 ✅** |

> 注：四方案 Inserter 全部训练完成（2026-02-19）。

### 7.6 初步发现

1. **方案 B 在 Tagger 和 Inserter 组件级指标上均表现最强**（但端到端评估中 B 最差，级联错误导致悖论，详见 RESULTS_SUMMARY.md）
2. **方案 D 收敛最慢**（Inserter ep7 仅 top1=0.08），与 BERT MLM 预训练中 D 表现最差一致
3. 相对位置编码对编辑任务的优势更加显著（Tagger F1: B=0.6050 vs A=0.4778）

### 7.7 代码目录

| 方案 | 目录 |
|------|------|
| A (no_pair) | `FELIX_no_pair/` |
| B (no_pair_related) | `FELIX_no_pair_related/` |
| C (with_pair) | `FELIX/` |
| D (absolute_bundled) | `FELIX_absolute_bundled/` |

### 7.8 Checkpoint 路径

| 方案 | Tagger | Inserter |
|------|--------|----------|
| A | `FELIX_no_pair/checkpoints/tagger/tagger_best.pt` | `FELIX_no_pair/checkpoints/inserter/inserter_best.pt` |
| B | `FELIX_no_pair_related/checkpoints/tagger/tagger_best.pt` | `FELIX_no_pair_related/checkpoints/inserter/inserter_best.pt` |
| C | `FELIX/checkpoints/tagger/tagger_best.pt` | `FELIX/checkpoints/inserter/inserter_best.pt` |
| D | `FELIX_absolute_bundled/checkpoints/tagger/tagger_best.pt` | `FELIX_absolute_bundled/checkpoints/inserter/inserter_best.pt` |

### 7.9 训练中遇到的问题

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| CUBLAS_STATUS_EXECUTION_FAILED | Inserter batch_size=48 + 捆绑编码(vocab=7145) 超出 24GB | batch_size 48→32, grad_accum 2→3 |
| mask_positions 越界 CUDA assert | 序列截断后 mask 位置超过 max_seq_len | dataset.py 添加 `pos < max_seq_len` 边界检查 |
| Inserter A/B 显存利用不足 (13GB) | 分离编码模型较小 (25.7M) | batch_size 32→48, grad_accum 3→2 |
| 服务器崩溃 (4卡满载) | 4×3090 瞬时功耗尖峰超 PSU 容量 | 监控功耗，错峰调度 |
| checkpoint weights_only 加载失败 | PyTorch 安全限制 numpy scalar | `torch.load(..., weights_only=False)` |

---

## 8. 训练时间总结（含 FELIX）

| 阶段 | Scheme A | Scheme B | Scheme C | Scheme D |
|------|---------|---------|---------|---------|
| BERT 预训练 | 8.6h | 27.3h | 8.3h | 4.6h |
| GECToR Stage I | 9.3h | 10.7h | 10.9h | 5.7h |
| GECToR Stage III | 1.4h | 1.5h | 1.7h | 0.9h |
| FELIX Tagger | 17.4h | 17.7h | 14.0h | 18.0h |
| FELIX Inserter | ~30h | ~30h | 30.0h | ~30h |
| **已完成合计** | **~67h** | **~87h** | **64.9h** | **~59h** |

---

## 9. 下一步计划

1. ~~Scheme D (absolute_bundled) — BERT + GECToR 训练~~ ✅ 已完成
2. ~~四种方案 FELIX Tagger 训练~~ ✅ 已完成
3. ~~FELIX Inserter 四方案训练~~ ✅ 全部完成
4. ~~FELIX 推理 pipeline 评估（Tagger→Inserter 端到端）~~ ✅ 已完成
5. ~~四种方案的完整对比评估~~ ✅ 已完成
6. ~~结合 LLaMA 生成模型的端到端评估~~ ✅ 已完成
7. LevT Inpainting（另一台服务器，进行中）
