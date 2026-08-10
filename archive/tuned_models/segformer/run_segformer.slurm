#!/usr/bin/env bash
#SBATCH --job-name=segformer_tuned
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=segformer/logs/%x_%j.out
#SBATCH --error=segformer/logs/%x_%j.err
#SBATCH --export=NONE
#SBATCH --qos=besteffort

set -euo pipefail

PROJECT_ROOT="${HOME}/Thesis"
SIF="${PROJECT_ROOT}/tell_seg.sif"

# ============================================================
# CONFIGURATION: Select which mode to run
# ============================================================
# Set MODE to one of: standard, tune_and_profile
#   standard         — Train segformer model (SegFormer + MiT-B0)
#   tune_and_profile — Tune with Optuna + Profile the best model

MODE="tune_and_profile"  # Change this to switch modes

# ============================================================
# END CONFIGURATION
# ============================================================

echo "============================================================"
echo "SegFormer Training (tuned_models) — SegFormer + MiT-B0 + Focal Loss"
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
# Train segformer model without profiling
# ============================================================
if [ "${MODE}" == "standard" ]; then
    echo "Running STANDARD segformer (tuned_models) training..."
    apptainer exec --nv \
        --bind "${HOME}/Thesis:/workspace" \
        --env PYTHONPATH="/workspace/.pylibs" \
        "${SIF}" \
        python -u /workspace/archive/tuned_models/segformer/train.py \
        2>&1 | tee "${PROJECT_ROOT}/segformer/logs/segformer_tuned_train_$(date +%Y%m%d_%H%M%S).log"

# ============================================================
# MODE 2: TUNING AND PROFILING
# Optuna Tuning + PyTorch profiler for performance analysis
# ============================================================
elif [ "${MODE}" == "tune_and_profile" ]; then
    echo "Running segformer (tuned_models) OPTUNA TUNING and PROFILING..."
    apptainer exec --nv \
        --bind "${HOME}/Thesis:/workspace" \
        --env PYTHONPATH="/workspace/.pylibs" \
        "${SIF}" \
        python -u /workspace/archive/tuned_models/segformer/train.py --tune --tune_trials 10 --profile \
        2>&1 | tee "${PROJECT_ROOT}/segformer/logs/segformer_tuned_profile_$(date +%Y%m%d_%H%M%S).log"

else
    echo "ERROR: Unknown mode '${MODE}'"
    echo "Valid modes: standard, tune_and_profile"
    exit 1
fi

echo "============================================================"
echo "Done."
echo "End time: $(date)"
echo "============================================================"
