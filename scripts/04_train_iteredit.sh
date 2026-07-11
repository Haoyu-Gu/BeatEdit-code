#!/bin/bash
# ============================================================
# BeatEdit Step 4: IterEdit Training (§3.3, Table 14)
# ============================================================
# Architecture: 8-layer Transformer encoder, 512 hidden, 8 heads
#   - Deletion head:  Linear(512, 2)
#   - Insertion head: Linear(1024, 21)  — predicts 0-20 placeholders
#   - Token head:     Linear(512, V)
#
# Training: AdamW lr=3×10⁻⁴, cosine + 10% warmup, 30 epochs
# Loss: L_del + L_ins + L_tok (all cross-entropy, label smoothing ε=0.1)
# Batch: 64 effective (32 × 2 gradient accumulation)
#
# Multi-level perturbation strategy (L1:30%, L2:30%, L3:25%, L4:15%)
# with intermediate-state sampling for exposure bias reduction.
#
# Inference: iterative delete→insert→fill, max 10 iterations (completion)
#            or 3 iterations (editing), deletion threshold 0.5
# ============================================================
set -e

DATA_DIR="${DATA_DIR:?Set DATA_DIR to your preprocessed npz directory}"
BERT_CKPT="${BERT_CKPT:?Set BERT_CKPT to pretrained BERT checkpoint path}"
OUTPUT_BASE="${OUTPUT_BASE:-./checkpoints/iteredit}"

echo "=== IterEdit Training ==="

# Resolve paths before cd'ing into the source tree, so relative values
# (including the defaults below) stay anchored at the repository root.
abspath() { case "$1" in /*) printf '%s' "$1" ;; *) printf '%s' "$PWD/$1" ;; esac; }
OUTPUT_BASE="$(abspath "$OUTPUT_BASE")"
DATA_DIR="$(abspath "$DATA_DIR")"
BERT_CKPT="$(abspath "$BERT_CKPT")"

cd "src/iteredit"

# Main training (inpainting mode — used for completion task)
echo "--- Training: Inpainting mode ---"
accelerate launch training/train.py \
    --data_dir "$DATA_DIR" \
    --pretrained_bert "$BERT_CKPT" \
    --output_dir "$OUTPUT_BASE/inpainting" \
    --epochs "${BEATEDIT_EPOCHS:-30}" \
    --batch_size "${BEATEDIT_BATCH:-32}" \
    --gradient_accumulation 2 \
    --lr 3e-4 \
    --weight_decay 0.01 \
    --warmup_ratio 0.10 \
    --label_smoothing 0.1 \
    --mask_ratio_min 0.125 \
    --mask_ratio_max 0.5

# Editing mode training (used for accompaniment editing task)
echo "--- Training: Editing mode ---"
accelerate launch training/train_editing.py \
    --data_dir "$DATA_DIR" \
    --pretrained_bert "$BERT_CKPT" \
    --output_dir "$OUTPUT_BASE/editing" \
    --epochs "${BEATEDIT_EPOCHS:-30}" \
    --batch_size "${BEATEDIT_BATCH:-32}" \
    --gradient_accumulation 2 \
    --lr 3e-4

echo "=== IterEdit training complete ==="
