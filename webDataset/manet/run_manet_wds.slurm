#!/usr/bin/env bash
#SBATCH --job-name=manet_wds_train_and_profile
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=webDataset/manet/logs/%x_%j.out
#SBATCH --error=webDataset/manet/logs/%x_%j.err
#SBATCH --export=NONE

set -euo pipefail

PROJECT_ROOT="${HOME}/Thesis"
SIF="${PROJECT_ROOT}/tell_seg.sif"

# ============================================================
# CONFIGURATION: Select which mode to run
# ============================================================
# Set MODE to one of: standard, tune_and_profile
#   standard         - Train MA-Net model using WebDataset
#   tune_and_profile - Tune with Optuna + Profile the best model

MODE="tune_and_profile"  # Change this to switch modes

# ============================================================
# END CONFIGURATION
# ============================================================

echo "============================================================"
echo "MA-Net WebDataset Training - MA-Net + EfficientNet-B3 + Focal Loss"
# We retrieve number of shards to verify
echo "Shards path: ${PROJECT_ROOT}/webDataset/shards"
echo "Mode:        ${MODE}"
echo "Hostname:    $(hostname)"
echo "Job ID:      ${SLURM_JOB_ID}"
echo "Start time:  $(date)"
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

if [ ! -d "${PROJECT_ROOT}/webDataset/shards" ]; then
    echo "ERROR: WebDataset shards not found: ${PROJECT_ROOT}/webDataset/shards"
    exit 1
fi

# Create logs/checkpoint directories if needed
mkdir -p "${PROJECT_ROOT}/webDataset/manet/logs"
mkdir -p "${PROJECT_ROOT}/checkpoints_baseline_wds"

# Load Apptainer module
module load tools/Apptainer/1.4.0

cd "${PROJECT_ROOT}"

export PYTHONUNBUFFERED=1

# ============================================================
# MODE 1: STANDARD TRAINING
# ============================================================
if [ "${MODE}" == "standard" ]; then
    echo "Running STANDARD MA-Net WebDataset training..."
    apptainer exec --nv \
        --bind "${HOME}/Thesis:/workspace" \
        --env PYTHONPATH="/workspace/.pylibs" \
        "${HOME}/Thesis/tell_seg.sif" \
        python -u /workspace/webDataset/manet/train.py \
        2>&1 | tee "${PROJECT_ROOT}/webDataset/manet/logs/manet_train_$(date +%Y%m%d_%H%M%S).log"

# ============================================================
# MODE 2: TUNING AND PROFILING
# ============================================================
elif [ "${MODE}" == "tune_and_profile" ]; then
    echo "Running MA-Net WebDataset OPTUNA TUNING and PROFILING..."
    apptainer exec --nv \
        --bind "${HOME}/Thesis:/workspace" \
        --env PYTHONPATH="/workspace/.pylibs" \
        "${HOME}/Thesis/tell_seg.sif" \
        python -u /workspace/webDataset/manet/train.py --tune --tune_trials 10 --profile \
        2>&1 | tee "${PROJECT_ROOT}/webDataset/manet/logs/manet_profile_$(date +%Y%m%d_%H%M%S).log"

else
    echo "ERROR: Unknown mode '${MODE}'"
    echo "Valid modes: standard, tune_and_profile"
    exit 1
fi

echo "============================================================"
echo "Done."
echo "End time: $(date)"
echo "============================================================"
