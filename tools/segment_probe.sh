#!/bin/bash
# ==============================================================================
# MUSE Segmentation Linear Probe on ADE20K
# ==============================================================================
#
# Evaluates spatial feature quality by training a lightweight BN+Conv1x1
# segmentation head on frozen backbone features.
#
# Usage:
#   # MUSE mode (8 GPUs)
#   bash tools/segment_probe.sh muse \
#       --config configs/muse_1b/stage1.yaml \
#       --checkpoint /path/to/checkpoint.bin \
#       --train-url /path/to/ade20k-train-{000000..000020}.tar \
#       --val-url /path/to/ade20k-validation-{000000..000002}.tar
#
#   # Baseline mode (compare with raw ViT)
#   bash tools/segment_probe.sh baseline \
#       --config configs/muse_1b/stage1.yaml \
#       --train-url /path/to/ade20k-train-{000000..000020}.tar \
#       --val-url /path/to/ade20k-validation-{000000..000002}.tar
#
# ==============================================================================

set -e

MODE="${1:-muse}"
shift || true

# Default values
CONFIG="configs/muse_1b/stage1.yaml"
CHECKPOINT=""
TRAIN_URL=""
VAL_URL=""
BATCH_SIZE=8
EPOCHS=10
LR=0.01
NUM_GPUS=8
MASTER_PORT=29604

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)       CONFIG="$2";      shift 2 ;;
        --checkpoint)   CHECKPOINT="$2";  shift 2 ;;
        --train-url)    TRAIN_URL="$2";   shift 2 ;;
        --val-url)      VAL_URL="$2";     shift 2 ;;
        --batch-size)   BATCH_SIZE="$2";  shift 2 ;;
        --epochs)       EPOCHS="$2";      shift 2 ;;
        --lr)           LR="$2";          shift 2 ;;
        --num-gpus)     NUM_GPUS="$2";    shift 2 ;;
        --master-port)  MASTER_PORT="$2"; shift 2 ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "============================================"
echo "  MUSE Segmentation Linear Probe (ADE20K)"
echo "============================================"
echo "Mode:        ${MODE}"
echo "Config:      ${CONFIG}"
echo "Checkpoint:  ${CHECKPOINT}"
echo "Train URL:   ${TRAIN_URL}"
echo "Val URL:     ${VAL_URL}"
echo "Batch Size:  ${BATCH_SIZE}"
echo "Epochs:      ${EPOCHS}"
echo "LR:          ${LR}"
echo "GPUs:        ${NUM_GPUS}"
echo "============================================"

CMD="torchrun --nproc_per_node=${NUM_GPUS} --master_port=${MASTER_PORT} \
    scripts/segment_probe.py \
    --config ${CONFIG} \
    --mode ${MODE} \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --lr ${LR}"

if [ -n "${CHECKPOINT}" ]; then
    CMD="${CMD} --checkpoint ${CHECKPOINT}"
fi

if [ -n "${TRAIN_URL}" ]; then
    CMD="${CMD} --train_url ${TRAIN_URL}"
fi

if [ -n "${VAL_URL}" ]; then
    CMD="${CMD} --val_url ${VAL_URL}"
fi

eval ${CMD}

echo "============================================"
echo "  Done."
echo "============================================"
