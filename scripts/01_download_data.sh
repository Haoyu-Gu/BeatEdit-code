#!/bin/bash
# ============================================================
# BeatEdit Step 1: Download Datasets
# ============================================================
# Paper uses two datasets:
#   1. MuseScore Collection [19] — 192,788 piano pieces (main dataset)
#   2. Lakh MIDI Dataset [30]   — 108K multi-track pieces (§4.6)
#
# After downloading, convert to npz format using the encoding scripts.
# ============================================================
set -e

DATA_DIR="${DATA_DIR:-./data}"
mkdir -p "$DATA_DIR"

echo "=== Dataset Download Guide ==="
echo ""
echo "1. MuseScore Collection (MusicScore: https://arxiv.org/abs/2406.11462)"
echo "   - 192,788 piano pieces, 95%/5% train/test split, seed 42"
echo "   - Download from: https://huggingface.co/datasets/m-a-p/MusicScore"
echo "   - Place MIDI/MusicXML files under: $DATA_DIR/musescore/"
echo ""
echo "2. Lakh MIDI Dataset (for multi-track experiments, §4.6)"
echo "   - 108,055 pieces, 105 General MIDI programs"
echo "   - Download from: https://colinraffel.com/projects/lmd/"
echo "   - Place under: $DATA_DIR/lakh_midi/"
echo ""
echo "After downloading, preprocess with the encoding scripts:"
echo "  cd src/encoding/scheme_A && python PianoDataset.py --data_dir $DATA_DIR/musescore/"
echo ""
echo "=== Notes ==="
echo "- Paper Table 11: 192,788 pieces (183K train / 9.6K test)"
echo "- Sequences exceeding 2,048 tokens are cropped at bar boundaries"
echo "- Data augmentation: random transposition ±5 semitones (70% probability)"
