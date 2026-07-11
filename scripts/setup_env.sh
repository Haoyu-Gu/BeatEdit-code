#!/usr/bin/env bash
# One-shot environment setup: virtualenv + dependencies + smoke test.
#
# Usage:
#   bash scripts/setup_env.sh            # create ./.venv and install
#   VENV_DIR=~/envs/beatedit bash scripts/setup_env.sh
set -e

VENV_DIR="${VENV_DIR:-.venv}"
PYTHON="${PYTHON:-python3}"

echo "==> Creating virtualenv at ${VENV_DIR}"
"$PYTHON" -m venv "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "==> Installing dependencies"
pip install --upgrade pip
pip install -r requirements.txt

echo "==> Smoke test: encoding round-trip (all 4 schemes)"
python tools/encoding_demo.py

echo "==> Smoke test: decode-filter-reencode validity (Appendix E/F)"
python evaluation/verify_filter_roundtrip.py --n 50 --scheme B

echo
echo "Setup complete. Activate with:  source ${VENV_DIR}/bin/activate"
