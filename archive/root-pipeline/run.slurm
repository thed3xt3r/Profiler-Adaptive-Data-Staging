#!/usr/bin/env bash
#SBATCH --job-name=tell_training
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=archive/root-pipeline/logs/%x_%j.out
#SBATCH --error=archive/root-pipeline/logs/%x_%j.err
#SBATCH --export=NONE

set -euo pipefail

PROJECT_ROOT="${HOME}/Thesis"
DATA_ROOT="${PROJECT_ROOT}/bing_1k/train"
SIF="${PROJECT_ROOT}/tell_seg.sif"

# ============================================================
# CONFIGURATION: Select which mode to run
# ============================================================
# Set MODE to one of: standard, tune_hp, cross_validate, profile_best

MODE="standard"  # Change this to switch modes

# ============================================================
# END CONFIGURATION
# ============================================================

echo "============================================================"
echo "Tell Site Segmentation Training"
echo "Mode:       ${MODE}"
echo "Hostname:   $(hostname)"
echo "Job ID:     ${SLURM_JOB_ID}"
echo "Start time: $(date)"
echo "============================================================"

if [ ! -d "${PROJECT_ROOT}" ]; then
    echo "ERROR: Project root not found: ${PROJECT_ROOT}"
    exit 1
fi

if [ ! -f "${SIF}" ]; then
    echo "ERROR: Container not found: ${SIF}"
    exit 1
fi

module load tools/Apptainer/1.4.0

cd "${PROJECT_ROOT}"

export PYTHONUNBUFFERED=1
export WANDB_MODE=disabled

# ============================================================
# MODE 1: STANDARD TRAINING
# Single train/val/test split with early stopping
# ============================================================
if [ "${MODE}" == "standard" ]; then
    echo "Running STANDARD training..."
    apptainer exec --nv \
        --bind "${HOME}/Thesis:/workspace" \
        --bind "${HOME}/Thesis/bing_1k/train:/data" \
        "${HOME}/Thesis/tell_seg.sif" \
        python -u /workspace/archive/root-pipeline/train.py \
            --data_root /data \
            --batch_size 8 \
            --num_workers 4 \
            --seed 420 \
            --max_epochs 100 \
            --patience 15 \
            --lr 1e-3 \
            --weight_decay 1e-4 \
            --focal_alpha 0.75 \
            --focal_gamma 2.0 \
            --ckpt_dir /workspace/checkpoints \
            --wandb_project tell_segmentation \
            --wandb_offline \
            2>&1 | tee "${PROJECT_ROOT}/logs/training_$(date +%Y%m%d_%H%M%S).log"

# ============================================================
# MODE 2: HYPERPARAMETER TUNING
# Uses RayTune + Optuna to find best hyperparameters
# ============================================================
elif [ "${MODE}" == "tune_hp" ]; then
    echo "Running HYPERPARAMETER TUNING with RayTune..."
    apptainer exec --nv \
        --bind "${HOME}/Thesis:/workspace" \
        --bind "${HOME}/Thesis/bing_1k/train:/data" \
        "${HOME}/Thesis/tell_seg.sif" \
        python -u /workspace/archive/root-pipeline/train.py \
            --data_root /data \
            --batch_size 8 \
            --num_workers 4 \
            --seed 420 \
            --max_epochs 50 \
            --patience 10 \
            --ckpt_dir /workspace/checkpoints_tune \
            --wandb_project tell_segmentation \
            --wandb_offline \
            --tune_hp \
            --tune_samples 8 \
            --tune_scheduler asha \
            --profile \
            --profile_active 3 \
            2>&1 | tee "${PROJECT_ROOT}/logs/tuning_$(date +%Y%m%d_%H%M%S).log"

# ============================================================
# MODE 3: K-FOLD CROSS-VALIDATION
# Train and validate on 5 folds with best hyperparameters
# (Update lr, weight_decay, focal_alpha, focal_gamma with values from tuning)
# ============================================================
elif [ "${MODE}" == "cross_validate" ]; then
    echo "Running K-FOLD CROSS-VALIDATION..."
    apptainer exec --nv \
        --bind "${HOME}/Thesis:/workspace" \
        --bind "${HOME}/Thesis/bing_1k/train:/data" \
        "${HOME}/Thesis/tell_seg.sif" \
        python -u /workspace/archive/root-pipeline/train.py \
            --data_root /data \
            --batch_size 8 \
            --num_workers 4 \
            --seed 420 \
            --max_epochs 100 \
            --patience 15 \
            --lr 1e-3 \
            --weight_decay 1e-4 \
            --focal_alpha 0.75 \
            --focal_gamma 2.0 \
            --ckpt_dir /workspace/checkpoints_cv \
            --wandb_project tell_segmentation \
            --wandb_offline \
            --cross_validation \
            --n_splits 5 \
            --profile \
            --profile_active 5 \
            2>&1 | tee "${PROJECT_ROOT}/logs/cross_val_$(date +%Y%m%d_%H%M%S).log"

# ============================================================
# MODE 4: PROFILE BEST MODEL
# Train normally with profiling enabled to analyze bottlenecks
# ============================================================
elif [ "${MODE}" == "profile_best" ]; then
    echo "Running TRAINING with PROFILING..."
    apptainer exec --nv \
        --bind "${HOME}/Thesis:/workspace" \
        --bind "${HOME}/Thesis/bing_1k/train:/data" \
        "${HOME}/Thesis/tell_seg.sif" \
        python -u /workspace/archive/root-pipeline/train.py \
            --data_root /data \
            --batch_size 8 \
            --num_workers 4 \
            --seed 420 \
            --max_epochs 50 \
            --patience 15 \
            --lr 1e-3 \
            --weight_decay 1e-4 \
            --focal_alpha 0.75 \
            --focal_gamma 2.0 \
            --ckpt_dir /workspace/checkpoints_profile \
            --wandb_project tell_segmentation \
            --wandb_offline \
            --profile \
            --profile_wait 1 \
            --profile_warmup 2 \
            --profile_active 5 \
            2>&1 | tee "${PROJECT_ROOT}/logs/profile_$(date +%Y%m%d_%H%M%S).log"

else
    echo "ERROR: Unknown mode '${MODE}'"
    echo "Valid modes: standard, tune_hp, cross_validate, profile_best"
    exit 1
fi

echo "============================================================"
echo "Done."
echo "End time: $(date)"
echo "============================================================"
