#!/bin/bash
# ============================================================
# BeatEdit Step 3: SeqTag Training (§3.2, Table 12)
# ============================================================
# Two-stage training:
#   Stage I  — 2 epochs frozen BERT (lr=1e-3), then 18 epochs full (BERT lr=1e-5, head lr=1e-4)
#   Stage III — 3 epochs clean mixing (lr=5e-6, clean_ratio=0.25)
#
# Architecture: BERT encoder + error_detector (Linear 512→2) + tag_predictor (Linear 512→350)
# Loss: L_tag (weighted CE, KEEP down-weighted 0.15) + λ·L_detect (BCE, λ=0.5)
# Batch: 64 effective / fp16
# Inference: up to 3 iterations, KEEP bias β=0.3, error threshold θ=0.5
# ============================================================
set -e

DATA_DIR="${DATA_DIR:?Set DATA_DIR to your preprocessed npz directory}"
BERT_CKPT="${BERT_CKPT:?Set BERT_CKPT to pretrained BERT checkpoint path}"
OUTPUT_BASE="${OUTPUT_BASE:-./checkpoints/seqtag}"
SCHEME="${SCHEME:-A}"  # A, B, C, or D

echo "=== SeqTag Training: Scheme $SCHEME ==="

cd "src/seqtag/scheme_$SCHEME"

# Stage I: Synthetic error training (freeze BERT for 2 epochs, then fine-tune)
echo "--- Stage I: Synthetic error training ---"
S1_OUT="$OUTPUT_BASE/scheme_$SCHEME/stage1"
S3_OUT="$OUTPUT_BASE/scheme_$SCHEME/stage3"

accelerate launch train_gector.py \
    --stage 1 \
    --data_dir "$DATA_DIR" \
    --bert_checkpoint "$BERT_CKPT" \
    --output_dir "$S1_OUT" \
    --epochs "${BEATEDIT_EPOCHS:-20}" \
    --batch_size "${BEATEDIT_BATCH:-32}" \
    --freeze_epochs 2 \
    --cold_lr 1e-3 \
    --lr_bert 1e-5 \
    --lr_head 1e-4 \
    --weight_decay 0.01 \
    --warmup_ratio 0.10 \
    --keep_weight 0.15 \
    --lambda_detect 0.5

# train_gector.py saves <output_dir>/{best_model,final_model}/model.pt; prefer the best.
if [ -f "$S1_OUT/best_model/model.pt" ]; then
    S1_CKPT="$S1_OUT/best_model"
else
    S1_CKPT="$S1_OUT/final_model"
fi
[ -f "$S1_CKPT/model.pt" ] || { echo "Stage I produced no checkpoint in $S1_OUT" >&2; exit 1; }
echo "Stage I checkpoint: $S1_CKPT"

# Stage III: Clean mixing fine-tuning
echo "--- Stage III: Clean mixing ---"
accelerate launch train_gector.py \
    --stage 3 \
    --data_dir "$DATA_DIR" \
    --checkpoint "$S1_CKPT" \
    --output_dir "$S3_OUT" \
    --epochs "${BEATEDIT_STAGE3_EPOCHS:-3}" \
    --batch_size "${BEATEDIT_BATCH:-32}" \
    --lr 5e-6 \
    --clean_ratio 0.25

echo "=== SeqTag Scheme $SCHEME training complete ==="
echo "Checkpoint: $S3_OUT"

# To train all 4 schemes:
# for s in A B C D; do SCHEME=$s bash scripts/03_train_seqtag.sh; done
