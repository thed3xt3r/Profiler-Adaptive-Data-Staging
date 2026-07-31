#!/usr/bin/env bash
#SBATCH --job-name=segformer_pads_hpc
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=runs/segformer/logs/%x_%j.out
#SBATCH --error=runs/segformer/logs/%x_%j.err
#SBATCH --export=NONE
#SBATCH --qos=besteffort

# =============================================================================
# MA-Net on Iris, using the reference (Casini et al.) configuration.
#
# Differs from PAD_segformer.slurm in that every parameter the reference fixes is
# passed EXPLICITLY rather than inherited from a default, so the run is
# self-documenting and cannot drift if defaults change.
#
#   batch 32, no accumulation   -- V100 has the memory; a 4 GB card does not,
#                                  and BatchNorm over 8 samples is not the same
#                                  statistic as over 32
#   precision 32                -- V100 is sm_70: bf16 needs sm_80 and would
#                                  fail or fall back to slow emulation
#   focal alpha unset (None)    -- the reference uses smp's default
#   val_crop random             -- the reference random-crops at evaluation
#   no --scale_input            -- the reference standardises raw 0-255 data
#
# MODE:
#   standard  training only
#   pads      profiler selects the data policy, then trains on it (+ torch profiler)
# =============================================================================

set -euo pipefail

PROJECT_ROOT="${HOME}/Thesis"
SIF="${PROJECT_ROOT}/tell_seg.sif"
MODE="${1:-standard}"
EPOCHS="${2:-20}"

echo "============================================================"
echo "MA-Net (segformer + EfficientNet-B3 + Focal) -- reference config"
echo "Mode:       ${MODE}"
echo "Epochs:     ${EPOCHS}"
echo "Hostname:   $(hostname)"
echo "Job ID:     ${SLURM_JOB_ID:-none}"
echo "Start:      $(date)"
echo "============================================================"

[ -d "${PROJECT_ROOT}" ]            || { echo "ERROR: no ${PROJECT_ROOT}"; exit 1; }
[ -f "${SIF}" ]                     || { echo "ERROR: no ${SIF}"; exit 1; }
[ -d "${PROJECT_ROOT}/bing_1k/train" ] || { echo "ERROR: no dataset"; exit 1; }

mkdir -p "${PROJECT_ROOT}/runs/segformer/logs" "${PROJECT_ROOT}/runs/checkpoints_segformer"

module load tools/Apptainer/1.4.0
cd "${PROJECT_ROOT}"

export PYTHONUNBUFFERED=1
# Outputs land under ${PROJECT_ROOT}/runs/ (train.py default). Set this to
# ${PROJECT_ROOT} to restore the original flat ~/Thesis layout.
export PADS_PROJECT_ROOT="/workspace/runs"
export PADS_DATA_ROOT="/workspace"

# NOTE: no PYTORCH_CUDA_ALLOC_CONF=expandable_segments here -- that option needs
# torch >= 2.1 and the container ships 2.0.1.

echo "Ensuring compatible NumPy (<2) in .pylibs..."
apptainer exec --bind "${PROJECT_ROOT}:/workspace" "${SIF}" \
    pip install --target /workspace/.pylibs "numpy<2" --upgrade --quiet

COMMON_ARGS=(
    --epochs "${EPOCHS}"
    --batch_size 32
    --accumulate 1
    --precision 32
    --num_workers 8
    --val_crop random
)

case "${MODE}" in
  standard) EXTRA_ARGS=() ;;
  pads)     EXTRA_ARGS=(--profile_policy --profile) ;;
  *)        echo "ERROR: mode must be standard|pads"; exit 1 ;;
esac

apptainer exec --nv \
    --bind "${PROJECT_ROOT}:/workspace" \
    --env PYTHONPATH="/workspace/.pylibs" \
    --env PADS_PROJECT_ROOT="${PADS_PROJECT_ROOT}" \
    --env PADS_DATA_ROOT="${PADS_DATA_ROOT}" \
    "${SIF}" \
    python -u /workspace/PADS/segformer/train.py "${COMMON_ARGS[@]}" "${EXTRA_ARGS[@]}"

echo "============================================================"
echo "Done. End: $(date)"
echo "============================================================"
