#!/bin/bash
# ============================================================
# BeatEdit Step 5: TagFill Training (§3.4, Table 13)
# ============================================================
# Two-stage architecture:
#   Stage 1 — Tagger:   8-layer Transformer, Linear(512, 11), Focal Loss (γ=2.0)
#   Stage 2 — Inserter: 8-layer Transformer, Linear(512, V), MaskGiT (T=2 passes)
#
# Training (both stages):
#   AdamW lr=1e-4, cosine + 10% warmup, 30 epochs
#   Batch: 96 effective (32 × 3 gradient accumulation)
#   Multi-level perturbation: L1:30%, L2:30%, L3:25%, L4:15%
#
# The inserter is initialized from the BERT pre-trained checkpoint.
# ============================================================
set -e

DATA_DIR="${DATA_DIR:?Set DATA_DIR to your preprocessed npz directory}"
BERT_CKPT="${BERT_CKPT:?Set BERT_CKPT to pretrained BERT checkpoint path}"
OUTPUT_BASE="${OUTPUT_BASE:-./checkpoints/tagfill}"
SCHEME="${SCHEME:-A}"  # A, B, C, or D

echo "=== TagFill Training: Scheme $SCHEME ==="

cd "src/tagfill/scheme_$SCHEME"

# Stage 1: Train Tagger (with BERT pre-training init)
echo "--- Stage 1: Tagger (BERT-init + Focal Loss, γ=2.0) ---"
accelerate launch training/train_tagger.py \
    --data_dir "$DATA_DIR" \
    --pretrained_bert "$BERT_CKPT" \
    --output_dir "$OUTPUT_BASE/scheme_$SCHEME/tagger" \
    --epochs "${BEATEDIT_EPOCHS:-30}" \
    --batch_size "${BEATEDIT_BATCH:-32}" \
    --gradient_accumulation 3 \
    --lr 1e-4 \
    --weight_decay 0.01 \
    --warmup_ratio 0.10 \
    --focal_gamma 2.0 \
    --level_weights "30,30,25,15"

# Stage 2: Train Inserter (initialized from BERT)
echo "--- Stage 2: Inserter (MLM on [MASK] positions) ---"
accelerate launch training/train_inserter.py \
    --data_dir "$DATA_DIR" \
    --pretrained_bert "$BERT_CKPT" \
    --output_dir "$OUTPUT_BASE/scheme_$SCHEME/inserter" \
    --epochs "${BEATEDIT_EPOCHS:-30}" \
    --batch_size "${BEATEDIT_BATCH:-32}" \
    --gradient_accumulation 3 \
    --lr 1e-4 \
    --weight_decay 0.01 \
    --warmup_ratio 0.10 \
    --level_weights "30,30,25,15"

echo "=== TagFill Scheme $SCHEME training complete ==="

# To train all 4 schemes:
# for s in A B C D; do SCHEME=$s bash scripts/05_train_tagfill.sh; done
