#!/bin/bash
# ============================================================================
# MUSE Stage 3: Synergistic Tuning
# ============================================================================
# End-to-end co-optimization with full parameter unfreezing.
# Focus: Gradient orthogonality — Semantic (ITC) and Structure (Topo) jointly
#        guide Pixel Reconstruction with differential learning rates.
#
# Prerequisites: Stage 2 checkpoint must exist.
#
# Usage:
#   bash tools/train_stage3.sh                    # Default: MUSE-1B
#   bash tools/train_stage3.sh muse_3b            # MUSE-3B variant
# ============================================================================

set -e

# --- Configuration ---
MODEL_SIZE="${1:-muse_1b}"
CONFIG="configs/${MODEL_SIZE}/stage3.yaml"
OUTPUT_DIR="outputs/${MODEL_SIZE}_stage3"
NUM_GPUS="${NUM_GPUS:-8}"
PORT="${MASTER_PORT:-10083}"
BATCH_SIZE="${BATCH_SIZE:-32}"       # Reduced batch size for Stage 3
NUM_WORKERS="${NUM_WORKERS:-16}"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

echo "============================================"
echo "  MUSE Stage 3: Synergistic Tuning"
echo "  Model: ${MODEL_SIZE}"
echo "  Config: ${CONFIG}"
echo "  Output: ${OUTPUT_DIR}"
echo "  GPUs: ${NUM_GPUS}"
echo "  Batch size per GPU: ${BATCH_SIZE}"
echo "============================================"

WANDB_MODE=offline PYTHONUNBUFFERED=1 accelerate launch \
    --num_processes=${NUM_GPUS} \
    --main_process_port=${PORT} \
    scripts/train_stage3.py \
    config=${CONFIG} \
    experiment.project="${MODEL_SIZE}" \
    experiment.name="${MODEL_SIZE}_stage3" \
    experiment.output_dir="${OUTPUT_DIR}" \
    training.per_gpu_batch_size=${BATCH_SIZE} \
    dataset.params.num_workers_per_gpu=${NUM_WORKERS} \
    2>&1 | tee "${OUTPUT_DIR}/train.log"
