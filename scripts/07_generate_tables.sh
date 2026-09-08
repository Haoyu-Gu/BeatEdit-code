#!/bin/bash
# Generate per-task Markdown summaries from evaluation outputs.
# Usage: bash scripts/07_generate_tables.sh [results_dir] [output_dir]
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${1:-$REPO_DIR/results}"
OUTPUT_DIR="${2:-$RESULTS_DIR/tables}"
PYTHON="${PYTHON:-python3}"

# Validate every input before writing any summaries.
for task in correction editing inpainting; do
    "$PYTHON" "$REPO_DIR/evaluation/summarize.py" \
        --task "$task" --results_dir "$RESULTS_DIR" --check_only
done
for task in correction editing inpainting; do
    "$PYTHON" "$REPO_DIR/evaluation/summarize.py" \
        --task "$task" --results_dir "$RESULTS_DIR" \
        --output "$OUTPUT_DIR/SUMMARY_${task}_perturbed_only.md" --show_ci
done
