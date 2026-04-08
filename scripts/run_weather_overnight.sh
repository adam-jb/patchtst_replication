#!/bin/bash
# Weather self-supervised overnight run: pretrain → linear probe → reports → fine-tune → final reports
# Usage: bash scripts/run_weather_overnight.sh
# Estimated runtime: ~8-12 hours on M1 Pro MPS
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

source .venv/bin/activate
export PYTHONUNBUFFERED=1

DATA_PATH="data/weather.csv"
WEATHER_ARGS="--d_model 128 --nhead 16 --dim_feedforward 256 --n_channels 21"
HORIZONS=(96 192 336 720)

START_TIME=$(date +%s)
echo "========================================================"
echo "Weather Self-Supervised Overnight Run"
echo "Started: $(date)"
echo "========================================================"

# ──────────────────────────────────────────────────────────────
# Phase 1: Pretraining (masked patch prediction)
# L=512, P=12, S=12 (non-overlapping), batch=128, 100 epochs max
# ──────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  PHASE 1: Self-Supervised Pretraining               ║"
echo "╚══════════════════════════════════════════════════════╝"

python src/train_selfsup.py \
    --data_path "$DATA_PATH" \
    --pred_len 96 \
    --mode linear_probe \
    $WEATHER_ARGS

PHASE1_TIME=$(date +%s)
echo ""
echo "Phase 1 complete (pretrain + LP T=96): $(( (PHASE1_TIME - START_TIME) / 60 )) minutes"

# ──────────────────────────────────────────────────────────────
# Phase 2: Linear probe for remaining horizons (reuse checkpoint)
# ──────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  PHASE 2: Linear Probe — T=192, 336, 720            ║"
echo "╚══════════════════════════════════════════════════════╝"

for T in 192 336 720; do
    echo ""
    echo "--- Linear Probe: pred_len=$T ---"
    python src/train_selfsup.py \
        --data_path "$DATA_PATH" \
        --pred_len $T \
        --skip_pretrain \
        --mode linear_probe \
        $WEATHER_ARGS
done

PHASE2_TIME=$(date +%s)
echo ""
echo "Phase 2 complete (LP all horizons): $(( (PHASE2_TIME - START_TIME) / 60 )) minutes"

# ──────────────────────────────────────────────────────────────
# Phase 3: Generate interim HTML reports (LP results)
# ──────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  PHASE 3: Generating Interim Reports                 ║"
echo "╚══════════════════════════════════════════════════════╝"

python src/visualize.py --dataset weather --mode selfsup
echo "Interim reports generated (linear probe only, fine-tune pending)"

# ──────────────────────────────────────────────────────────────
# Phase 4: Fine-tuning for all horizons (reuse checkpoint)
# 10 frozen epochs + 20 unfrozen epochs per horizon
# ──────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  PHASE 4: Fine-tuning — All Horizons                 ║"
echo "╚══════════════════════════════════════════════════════╝"

for T in "${HORIZONS[@]}"; do
    echo ""
    echo "--- Fine-tuning: pred_len=$T ---"
    python src/train_selfsup.py \
        --data_path "$DATA_PATH" \
        --pred_len $T \
        --skip_pretrain \
        --mode finetune \
        $WEATHER_ARGS
done

PHASE4_TIME=$(date +%s)
echo ""
echo "Phase 4 complete (FT all horizons): $(( (PHASE4_TIME - PHASE2_TIME) / 60 )) minutes"

# ──────────────────────────────────────────────────────────────
# Phase 5: Generate final HTML reports (LP + FT)
# ──────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  PHASE 5: Generating Final Reports                   ║"
echo "╚══════════════════════════════════════════════════════╝"

python src/visualize.py --dataset weather --mode selfsup
echo ""

END_TIME=$(date +%s)
TOTAL_MIN=$(( (END_TIME - START_TIME) / 60 ))
TOTAL_HR=$(( TOTAL_MIN / 60 ))
REMAIN_MIN=$(( TOTAL_MIN % 60 ))

echo "========================================================"
echo "ALL DONE"
echo "Total runtime: ${TOTAL_HR}h ${REMAIN_MIN}m"
echo "Finished: $(date)"
echo ""
echo "Results:"
echo "  results/weather_selfsup_results.html"
echo "  results/weather_selfsup_linear_probe_T{96,192,336,720}.json"
echo "  results/weather_selfsup_finetune_T{96,192,336,720}.json"
echo "  results/weather_pretrain_history.json"
echo "========================================================"
