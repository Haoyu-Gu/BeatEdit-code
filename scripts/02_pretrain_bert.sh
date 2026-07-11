#!/bin/bash
# ============================================================
# BeatEdit Step 2: Music BERT MLM Pre-training (§3.1, Table 11)
# ============================================================
# Architecture: 8 layers, 512 hidden, 8 heads, ~26.6M params
# Training:     AdamW lr=1e-4, cosine + 10% warmup, 30 epochs
# Masking:      15% of music tokens (80% [MASK], 10% random, 10% keep)
# Batch:        256 effective (64 × 4 gradient accumulation)
#
# Train one model per encoding scheme (A/B/C/D).
# Paper Appendix C: Scheme B achieves lowest val loss (0.224, ppl 1.25).
# Training time: 4.6h (Scheme D) to 27.3h (Scheme B) on 2×GPU (24GB VRAM).
# ============================================================
set -e

DATA_DIR="${DATA_DIR:?Set DATA_DIR to your preprocessed npz directory}"
OUTPUT_BASE="${OUTPUT_BASE:-./checkpoints/bert}"
SCHEME="${SCHEME:-A}"  # A, B, C, or D

echo "=== BERT MLM Pre-training: Scheme $SCHEME ==="

# Resolve paths before cd'ing into the source tree, so relative values
# (including the defaults below) stay anchored at the repository root.
abspath() { case "$1" in /*) printf '%s' "$1" ;; *) printf '%s' "$PWD/$1" ;; esac; }
OUTPUT_BASE="$(abspath "$OUTPUT_BASE")"
DATA_DIR="$(abspath "$DATA_DIR")"

cd "src/pretraining/scheme_$SCHEME"

# Paper defaults; override with the BEATEDIT_* environment variables
# (e.g. BEATEDIT_EPOCHS=1 BEATEDIT_LAYERS=4 for a quick pilot run).
EPOCHS="${BEATEDIT_EPOCHS:-30}"
BATCH="${BEATEDIT_BATCH:-64}"
LR="${BEATEDIT_LR:-1e-4}"

accelerate launch train_mlm.py \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_BASE/scheme_$SCHEME" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH" \
    --lr "$LR" \
    --tb_log_dir "$OUTPUT_BASE/scheme_$SCHEME/logs"

echo "=== BERT Scheme $SCHEME pre-training complete ==="
echo "Checkpoint saved to: $OUTPUT_BASE/scheme_$SCHEME"

# To train all 4 schemes:
# for s in A B C D; do SCHEME=$s bash scripts/02_pretrain_bert.sh; done
