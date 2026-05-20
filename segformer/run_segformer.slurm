#!/usr/bin/env bash
#SBATCH --job-name=segformer_train_and_profile
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=segformer/logs/%x_%j.out
#SBATCH --error=segformer/logs/%x_%j.err
#SBATCH --export=NONE

set -euo pipefail

PROJECT_ROOT="${HOME}/Thesis"
SIF="${PROJECT_ROOT}/tell_seg.sif"

# ============================================================
# CONFIGURATION: Select which mode to run
# ============================================================
# Set MODE to one of: standard, profile
#   standard      - Train SegFormer model (SegFormer B0)
#   profile       - Train + profile with PyTorch profiler

MODE="profile"  # Change this to switch modes

# ============================================================
# END CONFIGURATION
# ============================================================

echo "============================================================"
echo "SegFormer Training - SegFormer B0 + BCEWithLogitsLoss"
echo "Dataset:    bing_1k"
echo "Mode:       ${MODE}"
echo "Hostname:   $(hostname)"
echo "Job ID:     ${SLURM_JOB_ID}"
echo "Start time: $(date)"
echo "============================================================"

# Pre-flight checks
if [ ! -d "${PROJECT_ROOT}" ]; then
    echo "ERROR: Project root not found: ${PROJECT_ROOT}"
    exit 1
fi

if [ ! -f "${SIF}" ]; then
    echo "ERROR: Container not found: ${SIF}"
    exit 1
fi

if [ ! -d "${PROJECT_ROOT}/bing_1k/train" ]; then
    echo "ERROR: Dataset not found: ${PROJECT_ROOT}/bing_1k/train"
    exit 1
fi

# Create logs/checkpoint directories if needed
mkdir -p "${PROJECT_ROOT}/segformer/logs"
mkdir -p "${PROJECT_ROOT}/checkpoints_segformer"

# Load Apptainer module
module load tools/Apptainer/1.4.0

cd "${PROJECT_ROOT}"

export PYTHONUNBUFFERED=1

# ============================================================
# MODE 1: STANDARD TRAINING
# Train SegFormer model without profiling
# ============================================================
if [ "${MODE}" == "standard" ]; then
    echo "Running STANDARD segformer training..."
    apptainer exec --nv \
        --bind "${HOME}/Thesis:/workspace" \
        "${HOME}/Thesis/tell_seg.sif" \
        python -u /workspace/segformer/train.py \
        2>&1 | tee "${PROJECT_ROOT}/segformer/logs/segformer_train_$(date +%Y%m%d_%H%M%S).log"

# ============================================================
# MODE 2: TRAINING WITH PROFILING
# Train + PyTorch profiler for performance analysis
# ============================================================
elif [ "${MODE}" == "profile" ]; then
    echo "Running segformer training WITH PROFILING..."
    apptainer exec --nv \
        --bind "${HOME}/Thesis:/workspace" \
        --env SEGFORMER_PROFILE=1 \
        "${HOME}/Thesis/tell_seg.sif" \
        python -u /workspace/segformer/train.py \
        2>&1 | tee "${PROJECT_ROOT}/segformer/logs/segformer_profile_$(date +%Y%m%d_%H%M%S).log"

else
    echo "ERROR: Unknown mode '${MODE}'"
    echo "Valid modes: standard, profile"
    exit 1
fi

echo "============================================================"
echo "Done."
echo "End time: $(date)"
echo "============================================================"
