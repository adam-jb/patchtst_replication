#!/bin/bash
# Run all supervised PatchTST experiments on ETTh1
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

source .venv/bin/activate
export PYTHONUNBUFFERED=1

for T in 96 192 336 720; do
    echo "============================================"
    echo "Starting supervised training: pred_len=$T"
    echo "============================================"
    python src/train_supervised.py \
        --data_path data/ETTh1.csv \
        --pred_len $T \
        --epochs 100 \
        --patience 20
    echo ""
done

echo "All supervised runs complete!"
