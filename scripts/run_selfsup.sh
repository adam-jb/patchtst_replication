#!/bin/bash
# Run self-supervised pretraining + fine-tuning
# Usage: bash scripts/run_selfsup.sh [--dataset etth1|weather]
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

source .venv/bin/activate
export PYTHONUNBUFFERED=1

# Parse --dataset arg (default: etth1)
DATASET="${2:-etth1}"
for arg in "$@"; do
    case $arg in --dataset) DATASET="$2"; shift 2;; --dataset=*) DATASET="${arg#*=}";; esac
done

# Set dataset-specific config
case "$DATASET" in
    etth1)
        DATA_PATH="data/ETTh1.csv"
        EXTRA_ARGS="--n_channels 7"
        ;;
    weather)
        DATA_PATH="data/weather.csv"
        EXTRA_ARGS="--d_model 128 --nhead 16 --dim_feedforward 256 --n_channels 21"
        ;;
    *)
        echo "Unknown dataset: $DATASET (use etth1 or weather)"
        exit 1
        ;;
esac

echo "Dataset: $DATASET ($DATA_PATH)"
echo "Extra args: $EXTRA_ARGS"
echo ""

# Pretrain once (T=96 for data loading — pred_len doesn't affect pretraining)
echo "============================================"
echo "Self-supervised pretraining: $DATASET"
echo "============================================"
python src/train_selfsup.py \
    --data_path "$DATA_PATH" \
    --pred_len 96 \
    --mode both \
    $EXTRA_ARGS

# Fine-tune for remaining horizons (reuse pretrained checkpoint)
for T in 192 336 720; do
    echo "============================================"
    echo "Fine-tuning + linear probe: $DATASET pred_len=$T"
    echo "============================================"
    python src/train_selfsup.py \
        --data_path "$DATA_PATH" \
        --pred_len $T \
        --skip_pretrain \
        --mode both \
        $EXTRA_ARGS
done

echo "All self-supervised runs complete for $DATASET!"
