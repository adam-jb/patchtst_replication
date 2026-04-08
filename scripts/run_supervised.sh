#!/bin/bash
# Run all supervised PatchTST experiments
# Usage: bash scripts/run_supervised.sh [--dataset etth1|weather]
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
        EXTRA_ARGS=""
        ;;
    weather)
        DATA_PATH="data/weather.csv"
        EXTRA_ARGS="--d_model 128 --nhead 16 --dim_feedforward 256"
        ;;
    *)
        echo "Unknown dataset: $DATASET (use etth1 or weather)"
        exit 1
        ;;
esac

echo "Dataset: $DATASET ($DATA_PATH)"
echo "Extra args: $EXTRA_ARGS"
echo ""

for T in 96 192 336 720; do
    echo "============================================"
    echo "Supervised training: $DATASET pred_len=$T"
    echo "============================================"
    python src/train_supervised.py \
        --data_path "$DATA_PATH" \
        --pred_len $T \
        --epochs 100 \
        --patience 20 \
        $EXTRA_ARGS
    echo ""
done

echo "All supervised runs complete for $DATASET!"
