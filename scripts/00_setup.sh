#!/bin/bash
# ============================================================
# BeatEdit Step 0: Environment Setup
# ============================================================
set -e

echo "=== BeatEdit Environment Setup ==="

# Create conda environment (optional)
# conda create -n beatedit python=3.10 -y
# conda activate beatedit

# Install dependencies
pip install -r requirements.txt

# Verify GPU
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"

# Create checkpoint directories
mkdir -p checkpoints/{bert,seqtag,iteredit,tagfill}/{scheme_A,scheme_B,scheme_C,scheme_D}

echo "=== Setup complete ==="
