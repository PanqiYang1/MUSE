#!/bin/bash
# ============================================================================
# MUSE Evaluation
# ============================================================================
# Evaluate a trained MUSE model on reconstruction metrics (rFID, PSNR, SSIM, LPIPS).
#
# Usage:
#   bash tools/evaluate.sh <config> <checkpoint> [output_dir]
#
# Example:
#   bash tools/evaluate.sh \
#       configs/muse_1b/stage3.yaml \
#       outputs/muse_1b_stage3/checkpoint-50000/ema_model/pytorch_model.bin \
#       outputs/eval_muse_1b_stage3
# ============================================================================

set -e

# --- Arguments ---
CONFIG="${1:?Error: config path required (e.g., configs/muse_1b/stage3.yaml)}"
CHECKPOINT="${2:?Error: checkpoint path required}"
OUTPUT_DIR="${3:-outputs/evaluation}"
GPU="${CUDA_DEVICE:-0}"

mkdir -p "${OUTPUT_DIR}"

echo "============================================"
echo "  MUSE Evaluation"
echo "  Config: ${CONFIG}"
echo "  Checkpoint: ${CHECKPOINT}"
echo "  Output: ${OUTPUT_DIR}"
echo "  GPU: ${GPU}"
echo "============================================"

CUDA_VISIBLE_DEVICES=${GPU} python scripts/evaluate.py \
    --config "${CONFIG}" \
    experiment.output_dir="${OUTPUT_DIR}" \
    experiment.init_weight="${CHECKPOINT}"
