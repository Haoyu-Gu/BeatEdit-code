# Training Overview

> Last updated: 2026-02-22

This document collects the complete training results for the four encoding schemes (A / B / C / D), which together form the 2x2 ablation. Training proceeds in two stages:
1. **BERT MLM pre-training**: learn contextual representations of music tokens
2. **SeqTag correction training**: a GECToR-style sequence-tagging correction model built on the pre-trained BERT

The TagFill (FELIX-style tagger + inserter) system is covered in Section 7.

---

## 1. Encoding Schemes (2x2 ablation)

```
┌──────────┬──────────────────────┬──────────────────────┐
│          │ Separated (2tok/note)│ Bundled (1tok/note)  │
├──────────┼──────────────────────┼──────────────────────┤
│ Absolute │ Scheme A (no_pair)   │ Scheme D (abs_bundled)│
├──────────┼──────────────────────┼──────────────────────┤
│ Relative │ Scheme B (rel_pair)  │ Scheme C (with_pair) │
└──────────┴──────────────────────┴──────────────────────┘
```

| Feature | Scheme A | Scheme B | Scheme C | Scheme D |
|------|---------|---------|---------|---------|
| Position encoding | Absolute | Relative | Relative | Absolute |
| Token format | pos + val separated | pos + val separated | bundled (rel_pos x 81 + val) | bundled (abs_pos x 81 + val) |
| Tokens per note | 2 | 2 | 1 | 1 |
| BERT vocab size | 186 | 185 | 7145 | 7145 |
| SeqTag label space | 350 | 350 | 14258 | 14258 |

All four schemes share:
- Ternary patch encoding (patch_h=1, patch_w=4, 81 patterns, 88 keys). Ternary digits are `0` = silent, `1` = onset, `2` = sustain continuation, so a quarter note is pattern `53` = (1,2,2,2) and a pure continuation beat is `80` = (2,2,2,2).
- Sequence format: `[BOS][TIME_SIG][BPM][BAR][interleaved beats]...[EOS]`
- Beat-level interleaving of the high and low voices: `[high beat0][low beat0][high beat1][low beat1]...`

---

## 2. BERT MLM Pre-training

### 2.1 Model and Training Configuration

| Config | Scheme A | Scheme B | Scheme C | Scheme D |
|------|---------|---------|---------|---------|
| Model | BertForMaskedLM | BertForMaskedLM | BertForMaskedLM | BertForMaskedLM |
| hidden / layers / heads | 512 / 8 / 8 | 512 / 8 / 8 | 512 / 8 / 8 | 512 / 8 / 8 |
| intermediate | 2048 | 2048 | 2048 | 2048 |
| Total parameters | 26.6M | ~26.6M | 30.2M | 30.2M |
| Vocab size | 186 | 185 | 7145 | 7145 |
| max_seq_length | 2048 | 2048 | 2048 | 2048 |
| batch_size (per GPU) | 32 | 32 | 32 | 16 |
| gradient_accumulation | 4 | 4 | 4 | 4 |
| Effective batch_size | 256 | 256 | 256 | 256 |
| Learning rate | 1e-4 | 1e-4 | 1e-4 | 1e-4 |
| Scheduler | cosine + 10% warmup | cosine + 10% warmup | cosine + 10% warmup | cosine + 10% warmup |
| weight_decay | 0.01 | 0.01 | 0.01 | 0.01 |
| Epochs | 30 | 32 | 30 | 30 |
| GPU | 2x 24GB | 2x 24GB | 2x 24GB | 4x 24GB |
| Training data | 192,788 npz (95/5 split) | 192,788 npz | 192,788 npz | 192,788 npz |

The per-GPU batch sizes above are the ones used for the reported runs; the effective batch size is 256 in every case, which is what `src/pretraining/scheme_*/config.py` ships as the default (`batch_size=64`, `gradient_accumulation_steps=4`).

**Model-size overrides.** The pre-training config reads a set of environment variables in its `__post_init__`, so the backbone can be resized without editing any file — useful for pilot runs on small GPUs: `BEATEDIT_LAYERS`, `BEATEDIT_HIDDEN`, `BEATEDIT_HEADS`, `BEATEDIT_FFN`, `BEATEDIT_EPOCHS`, `BEATEDIT_BATCH`, `BEATEDIT_DATA_DIR`.

### 2.2 Training Results

| Metric | Scheme A | Scheme B | Scheme C | Scheme D |
|------|---------|---------|---------|---------|
| **Best eval loss** | 0.2823 | **0.2242** | 0.4490 | 0.5017 |
| Best eval PPL | 1.33 | **1.25** | 1.57 | 1.65 |
| Best step | 20,000 | 40,000 | 36,000 | 19,000 |
| Final train loss | 0.3265 | 0.2568 | 0.4910 | 0.6035 |
| Training time | 8.6h | 27.3h | 8.3h | 4.6h |
| Throughput | 14,188 tok/s | 14,045 tok/s | 8,927 tok/s | — |

**Note**: MLM loss is not directly comparable across encoding schemes — the vocabulary of Schemes C/D is ~38x that of A/B, so the prediction space is far larger and the loss is naturally higher. C and D, both bundled, can be compared with each other: D (absolute position, 0.5017) is clearly worse than C (relative position, 0.4490).

### 2.3 Convergence Behaviour

| Phase | Scheme A | Scheme B | Scheme C | Scheme D |
|------|---------|---------|---------|---------|
| Breakthrough epoch | 8 | 7 | 10 | never fully broke through |
| Fast-convergence window | epoch 8-10 | epoch 7-10 | epoch 10-14 | epoch 20-26 |
| Post-convergence loss range | 0.28-0.33 | 0.22-0.26 | 0.45-0.49 | 0.50-0.60 |

All four schemes show the same three-phase pattern: **slow start -> fast convergence -> flat fine-tuning**. Scheme D converges slowest and ends with the highest loss, showing that absolute-position bundled encoding is the hardest for the MLM objective.

### 2.4 Checkpoint Paths

| Scheme | Best model path |
|--------|---------------|
| A | `checkpoints/bert/scheme_A/best_model/` |
| B | `checkpoints/bert/scheme_B/best_model/` |
| C | `checkpoints/bert/scheme_C/best_model/` |
| D | `checkpoints/bert/scheme_D/best_model/` |

---

## 3. SeqTag Correction Training

SeqTag training runs in two stages:
- **Stage I**: train on synthetic error data, with the BERT encoder frozen for the first 2 epochs and unfrozen afterwards
- **Stage III**: fine-tune with 25% clean data mixed in (`clean_ratio=0.25`), 3 epochs

### 3.1 Model and Training Configuration

| Config | Scheme A | Scheme B | Scheme C | Scheme D |
|------|---------|---------|---------|---------|
| Label space | 350 | 350 | 14258 | 14258 |
| Total parameters | 26.5M | 26.5M | 37.2M | 37.2M |
| Stage I epochs | 20 | early stop @ 9 | 18 | 20 |
| Stage I batch_size | 32 | 32 | 16* | 16* |
| Stage III epochs | 3 | 3 | 3 | 3 |
| Stage III batch_size | 32 | 32 | 16 | 16 |
| Stage III lr | 5e-6 | 5e-6 | 5e-6 | 5e-6 |
| gradient_accumulation | 2 | 2 | 4 | 4 |
| Effective batch_size | 64 | 64 | 64 | 64 |
| clean_ratio | 0.25 | 0.25 | 0.25 | 0.25 |
| freeze_epochs (Stage I) | 2 | 2 | 2 | 2 |
| BERT unfreeze lr | 1e-5 | 1e-5 | 1e-5 | 1e-5 |
| Head lr (Stage I) | 1e-4 | 1e-4 | 1e-4 | 1e-4 |
| GPU | 2x 24GB | 2x 24GB | 2x 24GB | 4x 24GB |

*Schemes C/D have a label space of 14258, so the tag logits `[batch, 2048, 14258]` consume a great deal of GPU memory; `batch_size=32` runs out of memory.

### 3.2 Stage I Results

| Metric | Scheme A | Scheme B | Scheme C | Scheme D |
|------|---------|---------|---------|---------|
| **Best edit F1** | 0.8434 | 0.8630 | **0.8741** | 0.8428 |
| Best-F1 step | 20,000 | 21,000 | 24,000 | 13,000 |
| Best eval loss | 0.1478 | 0.1437 | 0.2385 | 0.2431 |
| Best-loss step | 25,000 | 20,000 | 22,000 | 11,000 |
| Precision @ best F1 | 0.8034 | 0.8217 | 0.8608 | 0.8069 |
| Recall @ best F1 | 0.8876 | 0.9087 | 0.8878 | 0.8820 |
| Epochs actually trained | 20 | 9 (early stop) | 18 | 20 |
| Training time | 9.3h | 10.7h | 10.9h | 5.7h |

### 3.3 Stage III Results

| Metric | Scheme A | Scheme B | Scheme C | Scheme D |
|------|---------|---------|---------|---------|
| **Best eval loss** | 0.1465 | **0.1445** | 0.2401 | 0.2436 |
| Best-loss step | 3,000 | 2,000 | 4,000 | 2,000 |
| Best edit F1 | 0.8548 | 0.8617 | **0.8733** | 0.8431 |
| Final edit F1 | 0.8437 | 0.8599 | 0.8717 | 0.8414 |
| Final precision | 0.8023 | 0.8172 | 0.8558 | 0.8047 |
| Final recall | 0.8895 | 0.9072 | 0.8881 | 0.8817 |
| Training time | 1.4h | 1.5h | 1.7h | 0.9h |

### 3.4 SeqTag Final Comparison (best Stage III results)

| Metric | Scheme A | Scheme B | Scheme C | Scheme D |
|------|---------|---------|---------|---------|
| **Edit F1** | 0.8548 | 0.8617 | **0.8733** | 0.8431 |
| Precision | 0.8354 | 0.8189 | **0.8600** | 0.8060 |
| Recall | 0.8752 | **0.9097** | 0.8869 | 0.8839 |
| Eval loss | 0.1526 | 0.1445 | 0.2419 | 0.2437 |

**Key findings:**

1. **Scheme C (relative x bundled) has the highest correction F1 (0.8733)**, even though its BERT MLM loss is the highest — bundled encoding makes MLM prediction harder, but the information it packs is more compact, which benefits the downstream correction task.
2. **Scheme B (relative x separated) has the highest recall (0.9097)**: it detects more errors, but at lower precision.
3. **Relative position beats absolute position**: C > D (0.8733 vs. 0.8431) and B > A (0.8617 vs. 0.8548). The gap is larger under bundled encoding (+3.0% vs. +0.7%).
4. **Bundled + relative is the best combination**, while **bundled + absolute is the worst** (D = 0.8431), even slightly below separated + absolute (A = 0.8548).
5. Stage III adds little over Stage I (0-1.1% F1); Stage I on synthetic data has largely converged already.

### 3.5 2x2 Ablation Conclusions

```
Edit F1 ranking: C (0.8733) > B (0.8617) > A (0.8548) > D (0.8431)

              Separated       Bundled
Absolute    A: 0.8548       D: 0.8431
Relative    B: 0.8617       C: 0.8733

Position-encoding effect:  relative > absolute (avg. +1.9%)
Token-format effect:       depends on the position encoding
  - Under relative position: bundled > separated (+1.2%)
  - Under absolute position: separated > bundled (+1.2%)
Interaction:               relative + bundled is best, absolute + bundled is worst
```

### 3.6 SeqTag Checkpoint Paths

| Scheme | Stage III best model |
|--------|---------------------|
| A | `checkpoints/seqtag/scheme_A/best_model/` |
| B | `checkpoints/seqtag/scheme_B/best_model/` |
| C | `checkpoints/seqtag/scheme_C/best_model/` |
| D | `checkpoints/seqtag/scheme_D/best_model/` |

---

## 4. Training-Time Summary

| Stage | Scheme A | Scheme B | Scheme C | Scheme D |
|------|---------|---------|---------|---------|
| BERT pre-training | 8.6h | 27.3h | 8.3h | 4.6h |
| SeqTag Stage I | 9.3h | 10.7h | 10.9h | 5.7h |
| SeqTag Stage III | 1.4h | 1.5h | 1.7h | 0.9h |
| **Total** | **19.3h** | **39.5h** | **20.9h** | **11.2h** |

Scheme D used 4 GPUs, hence the faster training.

---

## 5. Code Layout

All code lives under `src/`, organized by method and then by scheme:

| Model | Scheme A | Scheme B | Scheme C | Scheme D |
|------|---------|---------|---------|---------|
| Beat Encoding | `src/encoding/scheme_A/` | `src/encoding/scheme_B/` | `src/encoding/scheme_C/` | `src/encoding/scheme_D/` |
| BERT pre-training | `src/pretraining/scheme_A/` | `src/pretraining/scheme_B/` | `src/pretraining/scheme_C/` | `src/pretraining/scheme_D/` |
| SeqTag | `src/seqtag/scheme_A/` | `src/seqtag/scheme_B/` | `src/seqtag/scheme_C/` | `src/seqtag/scheme_D/` |
| TagFill | `src/tagfill/scheme_A/` | `src/tagfill/scheme_B/` | `src/tagfill/scheme_C/` | `src/tagfill/scheme_D/` |

IterEdit (Levenshtein Transformer) is scheme-independent and lives in `src/iteredit/` (`models/`, `data/`, `training/`, `inference/`, `configs/`, `evaluation/`).

Checkpoints follow the matching convention `checkpoints/{bert,seqtag,tagfill,iteredit}/scheme_X/`.

---

## 6. Problems Encountered During Training

| Problem | Cause | Solution |
|------|------|---------|
| BERT no_pair OOM at batch=64 | 2x 24GB is not enough memory | batch_size 64 -> 32 |
| SeqTag with_pair OOM when unfreezing BERT | tag logits `[32, 2048, 14258]` ~= 3.5GB | batch_size 32 -> 16, grad_accum 2 -> 4 |
| with_pair label_extractor AssertionError | `perturb_insert` wrongly added notes to empty beats | `perturbation.py` now skips empty beats |
| DDP unused-parameter warning | The BertModel pooler receives no gradient | `add_pooling_layer=False` |
| Missing pooler keys when loading a checkpoint | `best_model` was saved without a pooler | `strict=False` |
| nohup + conda logs not flushing | Python stdout buffering | Monitor via TensorBoard event files instead |
| SeqTag B finished early | Early stopping (patience=3) | Expected behaviour, not a bug |

---

## 7. TagFill: Two-Stage Editing System

The FELIX-style editing model: a **Tagger** (predicts edit operations) followed by an **Inserter** (fills MASK slots with an MLM head). It is used for generating and re-editing piano accompaniment.

### 7.1 System Architecture

```
input sequence → [Tagger] → edit labels (KEEP/DELETE/REPLACE/APPEND_1..8)
                     ↓
             skeleton sequence (deletions applied + MASK inserted)
                     ↓
              [Inserter] → fill MASK → output sequence
```

- **Tagger**: TransformerEncoder (hidden=512, layers=8, heads=8, pre-norm), 11 label classes
- **Inserter**: TransformerEncoder + MLM prediction head; vocab size matches the corresponding encoding scheme
- Perturbation strategy: 4 levels (L1: 5-15%, L2: 15-40%, L3: 40-70%, L4: 100% wipe), applied only to the accompaniment (Track 1)

### 7.2 Parameter Counts

| Component | Schemes A/B (separated) | Schemes C/D (bundled) |
|------|-------------------|-------------------|
| Tagger | ~25.3M | ~28.9M |
| Inserter | ~25.7M | ~32.8M |

### 7.3 Training Configuration

Identical for all four schemes, so that the effective batch size is the same in every cell of the 2x2 design:

| Config | Tagger | Inserter |
|------|--------|----------|
| Epochs | 30 | 30 |
| batch_size | 32 | 32 |
| gradient_accumulation | 3 | 3 |
| Effective batch_size | 96 | 96 |
| Learning rate | 1e-4 | 1e-4 |
| Scheduler | cosine + 10% warmup | cosine + 10% warmup |
| weight_decay | 0.01 | 0.01 |
| Perturbation level weights (L1-L4) | 30 / 30 / 25 / 15 | 30 / 30 / 25 / 15 |
| GPU | 1x 24GB | 1x 24GB |

At inference the Inserter uses the **linear** acceptance schedule of paper Eq. 9: over T rounds, round t accepts the `floor(|M| / (T - t + 1))` most confident MASK predictions. The default is `inserter_steps` (T) = 2; T = 1 falls back to a single-pass decode.

### 7.4 Tagger Results (all schemes complete)

| Metric | Scheme A | Scheme B | Scheme C | Scheme D |
|------|---------|---------|---------|---------|
| **Best macro F1** | 0.4778 | **0.6050** | 0.4674 | 0.4474 |
| val_acc | 97.87% | **97.85%** | 96.67% | — |
| Epochs completed | 18 | 18 | 30 | 18 |
| Time per epoch | ~58min | ~59min | ~28min* | ~60min |

*The Scheme C Tagger was trained on 2 GPUs for part of its run, hence the faster epochs. The configured budget is 30 epochs (Section 7.3); the A/B/D Tagger runs were stopped at 18 epochs.

**Tagger ranking: B (0.6050) >> A (0.4778) > C (0.4674) > D (0.4474)**

Scheme B leads by a wide margin and has generally higher accuracy on the APPEND labels (APPEND_2 = 60.6%, APPEND_4 = 57.4%), showing that relative position + separated encoding is the most favourable for predicting edit operations.

### 7.5 Inserter Results

| Metric | Scheme A | Scheme B | Scheme C | Scheme D |
|------|---------|---------|---------|---------|
| **Best top-1 acc** | done | done | **0.4297** | done |
| Training progress | **30/30** | **30/30** | **30/30** | **30/30** |

> Note: the Inserter finished training for all four schemes (2026-02-19).

### 7.6 Preliminary Findings

1. **Scheme B is the strongest on the component-level metrics of both Tagger and Inserter** (yet the worst end-to-end; the paradox is caused by cascaded errors — see RESULTS_SUMMARY.md).
2. **Scheme D converges slowest** (Inserter reached only top-1 = 0.08 at epoch 7), consistent with D being the weakest in BERT MLM pre-training.
3. The advantage of relative position encoding is even more pronounced on the editing task (Tagger F1: B = 0.6050 vs. A = 0.4778).

### 7.7 Code Layout

| Scheme | Directory |
|------|------|
| A (no_pair) | `src/tagfill/scheme_A/` |
| B (no_pair_related) | `src/tagfill/scheme_B/` |
| C (with_pair) | `src/tagfill/scheme_C/` |
| D (absolute_bundled) | `src/tagfill/scheme_D/` |

### 7.8 Checkpoint Paths

| Scheme | Tagger | Inserter |
|------|--------|----------|
| A | `checkpoints/tagfill/scheme_A/tagger/tagger_best.pt` | `checkpoints/tagfill/scheme_A/inserter/inserter_best.pt` |
| B | `checkpoints/tagfill/scheme_B/tagger/tagger_best.pt` | `checkpoints/tagfill/scheme_B/inserter/inserter_best.pt` |
| C | `checkpoints/tagfill/scheme_C/tagger/tagger_best.pt` | `checkpoints/tagfill/scheme_C/inserter/inserter_best.pt` |
| D | `checkpoints/tagfill/scheme_D/tagger/tagger_best.pt` | `checkpoints/tagfill/scheme_D/inserter/inserter_best.pt` |

### 7.9 Problems Encountered During Training

| Problem | Cause | Solution |
|------|------|---------|
| CUBLAS_STATUS_EXECUTION_FAILED | Inserter batch_size=48 with bundled encoding (vocab=7145) exceeds 24GB | batch_size 48 -> 32, grad_accum 2 -> 3 |
| mask_positions out-of-range CUDA assert | Mask positions exceeded `max_seq_len` after sequence truncation | Added a `pos < max_seq_len` bounds check in `dataset.py` |
| Inserter A/B under-used GPU memory (13GB) | The separated-encoding models are smaller (25.7M) | Kept batch_size 32 / grad_accum 3 (effective 96) for all schemes anyway, so that the effective batch is identical across the 2x2 cells |
| Server crash (4 GPUs at full load) | Transient power draw of 4x 3090 exceeded PSU capacity | Monitor power draw, stagger job scheduling |
| checkpoint `weights_only` load failure | PyTorch security restriction on numpy scalars | `torch.load(..., weights_only=False)` |

---

## 8. Training-Time Summary (including TagFill)

| Stage | Scheme A | Scheme B | Scheme C | Scheme D |
|------|---------|---------|---------|---------|
| BERT pre-training | 8.6h | 27.3h | 8.3h | 4.6h |
| SeqTag Stage I | 9.3h | 10.7h | 10.9h | 5.7h |
| SeqTag Stage III | 1.4h | 1.5h | 1.7h | 0.9h |
| TagFill Tagger | 17.4h | 17.7h | 14.0h | 18.0h |
| TagFill Inserter | ~30h | ~30h | 30.0h | ~30h |
| **Completed total** | **~67h** | **~87h** | **64.9h** | **~59h** |

---

## 9. Status

1. Scheme D (absolute_bundled) — BERT + SeqTag training — done
2. TagFill Tagger for all four schemes — done
3. TagFill Inserter for all four schemes — done
4. TagFill inference pipeline evaluation (Tagger -> Inserter, end-to-end) — done
5. Full comparison across all four schemes — done
6. End-to-end evaluation against the LLaMA generation baseline — done
7. IterEdit (Levenshtein Transformer) inpainting and editing — done; results are in `results/completion/` and `results/editing/`
