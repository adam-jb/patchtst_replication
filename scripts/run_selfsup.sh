#!/bin/bash
# Run self-supervised pretraining + fine-tuning on ETTh1
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

source .venv/bin/activate
export PYTHONUNBUFFERED=1

# Pretrain once (pred_len doesn't matter for pretraining, but needed for data loading)
echo "============================================"
echo "Starting self-supervised pretraining"
echo "============================================"
python src/train_selfsup.py \
    --data_path data/ETTh1.csv \
    --pred_len 96 \
    --mode both

# Fine-tune for remaining horizons (reuse pretrained checkpoint)
for T in 192 336 720; do
    echo "============================================"
    echo "Fine-tuning + linear probe: pred_len=$T"
    echo "============================================"
    python src/train_selfsup.py \
        --data_path data/ETTh1.csv \
        --pred_len $T \
        --skip_pretrain \
        --mode both
done

echo "All self-supervised runs complete!"
