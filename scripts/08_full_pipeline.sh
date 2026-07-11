#!/usr/bin/env bash
# End-to-end pipeline for one encoding scheme:
#   BERT pre-training -> SeqTag -> TagFill -> (Scheme D only) IterEdit -> evaluation.
#
# Usage:
#   SCHEME=A DATA_DIR=/path/to/data bash scripts/08_full_pipeline.sh
#
# Model-size knobs apply to every stage, e.g. a quick pilot run:
#   SCHEME=A DATA_DIR=... BEATEDIT_LAYERS=4 BEATEDIT_HIDDEN=256 BEATEDIT_EPOCHS=3 \
#       bash scripts/08_full_pipeline.sh
set -e

SCHEME="${SCHEME:?Set SCHEME to A, B, C or D}"
DATA_DIR="${DATA_DIR:?Set DATA_DIR to the preprocessed npz directory}"
export BEATEDIT_DATA_DIR="$DATA_DIR"

BERT_CKPT="${BERT_CKPT:-checkpoints/bert/scheme_${SCHEME}/best.pt}"

echo "==> [1/5] BERT pre-training (scheme ${SCHEME})"
SCHEME="$SCHEME" DATA_DIR="$DATA_DIR" bash scripts/02_pretrain_bert.sh

echo "==> [2/5] SeqTag"
SCHEME="$SCHEME" DATA_DIR="$DATA_DIR" BERT_CKPT="$BERT_CKPT" bash scripts/03_train_seqtag.sh

echo "==> [3/5] TagFill"
SCHEME="$SCHEME" DATA_DIR="$DATA_DIR" BERT_CKPT="$BERT_CKPT" bash scripts/05_train_tagfill.sh

if [ "$SCHEME" = "D" ]; then
    echo "==> [4/5] IterEdit (uses the Scheme D backbone)"
    DATA_DIR="$DATA_DIR" BERT_CKPT="$BERT_CKPT" bash scripts/04_train_iteredit.sh
else
    echo "==> [4/5] IterEdit skipped (train it from Scheme D: SCHEME=D ... 04_train_iteredit.sh)"
fi

echo "==> [5/5] Evaluation + tables"
bash scripts/06_evaluate_all.sh
bash scripts/07_generate_tables.sh

echo "Pipeline finished for scheme ${SCHEME}."
