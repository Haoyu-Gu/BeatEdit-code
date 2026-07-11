#!/bin/bash
# ============================================================
# BeatEdit Step 7: Generate Paper Tables from Results
# ============================================================
# Uses master_statistics.json to regenerate all paper tables.
# Can be run without GPU — only needs pre-computed result JSONs.
# ============================================================
set -e

echo "=== Generating Paper Tables ==="

cd evaluation

python summarize.py \
    --results_dir ../results \
    --master_stats ../results/master_statistics.json \
    --output_dir ../results/tables

echo ""
echo "Generated tables correspond to:"
echo "  Table 3:  Cross-encoding comparison"
echo "  Table 4:  Main results (best scheme per method)"
echo "  Table 5a: Encoding × Method (beat exact match)"
echo "  Table 5b: Difficulty level breakdown"
echo "  Table 8:  Inference latency"
echo "  Table 16: LLaMA baseline results"
echo "  Table 17: Diffusion baseline results"
echo "  Table 19: Full per-scheme results"
echo "  Table 20: FMD full results"
echo "  Table 23: Pairwise bootstrap comparisons"
echo "  Table 24: Two-way ANOVA"
echo ""
echo "=== Done ==="
