#!/usr/bin/env bash
#SBATCH --job-name=manet_t15
#SBATCH --partition=interactive
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:15:00
#SBATCH --output=runs/manet_table15/logs/%x_%j.out
#SBATCH --error=runs/manet_table15/logs/%x_%j.err
#SBATCH --export=NONE
#SBATCH --qos=debug

# =============================================================================
# Fills MA-Net's missing rows in Table 15 (8-batch probe vs full-epoch).
#
# The original diagnostic (runs/profile_window_check/) was run for DeepLabV3+
# and SegFormer only. MA-Net was added later, but only to the *discard sweep*
# (runs/profile_sweep/, discard=5) -- which is the corrected measurement, not
# the uncorrected one Table 15 contrasts against.
#
# Two probes, both with NO discard, matching the original diagnostic exactly:
#   A  --profile_batches 8    the uncorrected default (expected ratio ~1.x)
#   B  --profile_batches 147  full epoch (expected ratio ~12)
#
# --profile_only exits after the policy decision, so neither trains.
# =============================================================================

set -euo pipefail

PROJECT_ROOT="${HOME}/Thesis"
SIF="${PROJECT_ROOT}/tell_seg.sif"

echo "============================================================"
echo "MA-Net Table 15 backfill -- uncorrected 8-batch and full-epoch probes"
echo "Hostname:  $(hostname)"
echo "Job ID:    ${SLURM_JOB_ID:-none}"
echo "Start:     $(date)"
echo "============================================================"

[ -f "${SIF}" ] || { echo "ERROR: no ${SIF}"; exit 1; }
module load tools/Apptainer/1.4.0
cd "${PROJECT_ROOT}"
export PYTHONUNBUFFERED=1
export PADS_DATA_ROOT="/workspace"

run_probe () {
    local NB="$1" TAG="$2"
    echo ""
    echo "------------------------------------------------------------"
    echo "PROBE ${TAG}: --profile_batches ${NB}, no discard"
    echo "------------------------------------------------------------"
    apptainer exec --nv \
        --bind "${PROJECT_ROOT}:/workspace" \
        --env PYTHONPATH="/workspace/.pylibs" \
        --env PADS_PROJECT_ROOT="/workspace/runs/manet_table15/${TAG}" \
        --env PADS_DATA_ROOT="/workspace" \
        "${SIF}" \
        python -u /workspace/3-pads/manet/train.py \
            --epochs 1 \
            --batch_size 32 \
            --accumulate 1 \
            --precision 32 \
            --val_crop random \
            --num_workers 0 \
            --profile_only \
            --profile_batches "${NB}"
}

run_probe 8   "b8_nodiscard"
run_probe 147 "b147_nodiscard"

echo "============================================================"
echo "Done. End: $(date)"
echo "============================================================"
