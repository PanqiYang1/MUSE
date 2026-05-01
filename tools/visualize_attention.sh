#!/bin/bash
# ==============================================================================
# MUSE Attention Map Visualization
# ==============================================================================
#
# This script visualizes the topology attention maps from MUSE Synergistic Blocks.
#
# Mode 1: Single image visualization
#   bash tools/visualize_attention.sh single \
#       --config configs/muse_1b/stage1.yaml \
#       --checkpoint /path/to/checkpoint.bin \
#       --image /path/to/image.jpg \
#       --output ./attention_vis
#
# Mode 2: Batch visualization (eval dataset)
#   bash tools/visualize_attention.sh batch \
#       --config configs/muse_1b/stage1.yaml \
#       --checkpoint /path/to/checkpoint.bin \
#       --output ./attention_vis \
#       --max-images 500
#
# ==============================================================================

set -e

MODE="${1:-single}"
shift || true

# Default values
CONFIG="configs/muse_1b/stage1.yaml"
CHECKPOINT=""
IMAGE=""
OUTPUT_DIR="./attention_vis"
MAX_IMAGES=500
BATCH_SIZE=8
NUM_WORKERS=2
TEMPERATURE=0.1

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)      CONFIG="$2";      shift 2 ;;
        --checkpoint)  CHECKPOINT="$2";  shift 2 ;;
        --image)       IMAGE="$2";       shift 2 ;;
        --output)      OUTPUT_DIR="$2";  shift 2 ;;
        --max-images)  MAX_IMAGES="$2";  shift 2 ;;
        --batch-size)  BATCH_SIZE="$2";  shift 2 ;;
        --num-workers) NUM_WORKERS="$2"; shift 2 ;;
        --temperature) TEMPERATURE="$2"; shift 2 ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "============================================"
echo "  MUSE Attention Visualization"
echo "============================================"
echo "Mode:        ${MODE}"
echo "Config:      ${CONFIG}"
echo "Checkpoint:  ${CHECKPOINT}"
echo "Output Dir:  ${OUTPUT_DIR}"

if [ "${MODE}" = "single" ]; then
    if [ -z "${IMAGE}" ]; then
        echo "ERROR: --image is required for single mode."
        exit 1
    fi
    echo "Image:       ${IMAGE}"
    echo "============================================"

    python scripts/visualize_attention.py \
        config="${CONFIG}" \
        checkpoint_path="${CHECKPOINT}" \
        img_path="${IMAGE}" \
        output_dir="${OUTPUT_DIR}" \
        temperature="${TEMPERATURE}"

elif [ "${MODE}" = "batch" ]; then
    echo "Max Images:  ${MAX_IMAGES}"
    echo "Batch Size:  ${BATCH_SIZE}"
    echo "Temperature: ${TEMPERATURE}"
    echo "============================================"

    python scripts/visualize_attention.py \
        config="${CONFIG}" \
        checkpoint_path="${CHECKPOINT}" \
        output_dir="${OUTPUT_DIR}" \
        max_images="${MAX_IMAGES}" \
        batch_size="${BATCH_SIZE}" \
        num_workers="${NUM_WORKERS}" \
        temperature="${TEMPERATURE}"

else
    echo "ERROR: Unknown mode '${MODE}'. Use 'single' or 'batch'."
    exit 1
fi

echo "============================================"
echo "  Done. Results saved to: ${OUTPUT_DIR}"
echo "============================================"
