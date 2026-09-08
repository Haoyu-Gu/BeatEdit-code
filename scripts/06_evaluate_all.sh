#!/bin/bash
# Evaluate existing predictions for all three tasks, then summarize them.
# Predictions: PREDICTIONS_DIR/{task}/{method}/{scheme}/*.json
# Test cases: TEST_DATA_DIR/{task}/{scheme}/*.json
# This step computes metrics; inference and test-case construction run separately.
# FMD uses full-sequence embeddings; other main metrics use perturbed beats.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
RESULTS_DIR="${RESULTS_DIR:-$REPO_DIR/results}"
TEST_DATA_DIR="${TEST_DATA_DIR:-$REPO_DIR/evaluation/test_data}"
PREDICTIONS_DIR="${PREDICTIONS_DIR:-$REPO_DIR/evaluation/predictions}"
SCHEMES="${SCHEMES:-A,B,C,D}"
DEVICE="${DEVICE:-cuda}"
for task in correction editing inpainting; do
    if [ "$task" = inpainting ]; then
        methods="${INPAINTING_METHODS:-no_edit,copy_ctx,cmlm,felix,levt_inpainting}"
    else
        methods="${EDITING_METHODS:-no_edit,copy_ctx,cmlm,felix,gector,levt_editing}"
    fi
    "$PYTHON" "$REPO_DIR/evaluation/evaluate.py" \
        --task "$task" --schemes "$SCHEMES" --methods "$methods" \
        --test_data_dir "$TEST_DATA_DIR/$task" --predictions_dir "$PREDICTIONS_DIR" \
        --output_dir "$RESULTS_DIR" --scope perturbed_only \
        --per_sample --n_bootstrap 10000 --compute_fmd --device "$DEVICE" --strict
    "$PYTHON" "$REPO_DIR/evaluation/statistical_tests.py" \
        --task "$task" --schemes "$SCHEMES" --methods "$methods" --all_pairs \
        --from_results --results_dir "$RESULTS_DIR" --n_bootstrap 10000 \
        --output "$RESULTS_DIR/significance/${task}_pairwise.json"
done
PYTHON="$PYTHON" bash "$REPO_DIR/scripts/07_generate_tables.sh" "$RESULTS_DIR"
