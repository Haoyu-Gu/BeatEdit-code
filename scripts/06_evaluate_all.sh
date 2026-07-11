#!/bin/bash
# ============================================================
# BeatEdit Step 6: Run All Evaluations (§4)
# ============================================================
# Evaluation framework:
#   3 tasks × 4 encodings × 200 samples, seed 42
#   95% bootstrap CIs (B=10,000 resamples)
#
# Metrics (computed on perturbed beats only):
#   - beat_exact_match (↑): fraction of perturbed beats matching ground truth
#   - note_f1 (↑):          note-level F1 score
#   - MPE (↓):              mean pitch error in semitones
#   - FMD (↓):              Fréchet Music Distance (BERT embeddings)
# ============================================================
set -e

echo "=== Running Evaluations ==="

cd evaluation

# Main evaluation
python evaluate.py \
    --results_dir ../results \
    --output_dir ../results \
    --n_samples 200 \
    --seed 42

# Statistical significance tests
echo "--- Statistical Tests ---"
python statistical_tests.py \
    --results_dir ../results \
    --output_dir ../results/significance \
    --n_bootstrap 10000

# ANOVA (Table 24: encoding × method interaction)
echo "--- ANOVA Analysis ---"
python anova_and_pairwise.py \
    --results_dir ../results \
    --output_dir ../results/significance

# Generate paper-ready tables
echo "--- Generating Summary Tables ---"
python summarize.py \
    --results_dir ../results \
    --output_dir ../results

echo "=== Evaluation complete ==="
echo "Results saved to: results/"
echo "Master statistics: results/master_statistics.json"
