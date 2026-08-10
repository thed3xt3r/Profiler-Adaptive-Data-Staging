#!/usr/bin/env bash
#SBATCH --job-name=baseline_deeplab
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=runs/baseline/logs/%x_%j.out
#SBATCH --error=runs/baseline/logs/%x_%j.err
#SBATCH --export=NONE
#SBATCH --qos=normal

# =============================================================================
# Baseline data-path measurements -- DeepLabV3+ (EfficientNet-B3 + Focal)
#
# This stage establishes the reference points every later stage is measured
# against. It deliberately has no trainer of its own: it runs the SAME trainer
# as 3-pads/ with the adaptive layer switched off, so that model, dataset,
# hyperparameters and hardware are held identical and the ONLY thing varying
# across the thesis's five methods is the data path. A separate copy of
# train.py here would be free to drift, and any drift would silently
# invalidate the comparison.
#
#   METHOD 1  default loader  --data_policy loose  --num_workers 0
#   METHOD 2  tuned loader    --data_policy loose  --num_workers 4
#   METHOD 4  full pre-stage  --full_prestage      --num_workers 4  (the ceiling)
#
# Method 4 uses --full_prestage, NOT --data_policy stage: the former copies the
# whole dataset to node-local scratch up front and is the static ceiling PADS is
# measured against, while the latter is PADS's own adaptive rolling staging.
# Passing 'stage' here would compare PADS against itself.
#
# Methods 3 (WebDataset) and 5 (PADS) live in 2-webdataset/ and 3-pads/.
#
# Each method gets its own RUN_TAG. Without it, methods 1 and 2 -- identical
# but for num_workers -- resolve to the same output directory and the second
# job "resumes" the first's finished checkpoint, reporting done without
# training a step.
#
# Usage: sbatch run_baseline_deeplab.slurm [METHOD] [EPOCHS]
#   METHOD  1|2|4  (default: 1)
#   EPOCHS  integer (default: 20)
# =============================================================================

set -euo pipefail

PROJECT_ROOT="${HOME}/Thesis"
SIF="${PROJECT_ROOT}/tell_seg.sif"
METHOD="${1:-1}"
EPOCHS="${2:-20}"

case "${METHOD}" in
  1) POLICY_ARGS=(--data_policy loose); NUM_WORKERS=0
     TAG="baseline_m1_default";  DESC="default loader (loose, 0 workers)" ;;
  2) POLICY_ARGS=(--data_policy loose); NUM_WORKERS=4
     TAG="baseline_m2_tuned";    DESC="tuned loader (loose, 4 workers)" ;;
  4) POLICY_ARGS=(--full_prestage);     NUM_WORKERS=4
     TAG="baseline_m4_prestage"; DESC="full pre-stage ceiling (static, 4 workers)" ;;
  *) echo "ERROR: METHOD must be 1, 2 or 4 (methods 3 and 5 live elsewhere)"; exit 1 ;;
esac

echo "============================================================"
echo "Baseline -- DeepLabV3+ (EfficientNet-B3 + Focal)"
echo "Method:      ${METHOD}  -- ${DESC}"
echo "Epochs:      ${EPOCHS}"
echo "Run tag:     ${TAG}"
echo "Hostname:    $(hostname)"
echo "Job ID:      ${SLURM_JOB_ID:-none}"
echo "Start:       $(date)"
echo "============================================================"

[ -d "${PROJECT_ROOT}" ]               || { echo "ERROR: no ${PROJECT_ROOT}"; exit 1; }
[ -f "${SIF}" ]                        || { echo "ERROR: no ${SIF}"; exit 1; }
[ -d "${PROJECT_ROOT}/bing_1k/train" ] || { echo "ERROR: no dataset"; exit 1; }

mkdir -p "${PROJECT_ROOT}/runs/baseline/logs"

module load tools/Apptainer/1.4.0
cd "${PROJECT_ROOT}"

export PYTHONUNBUFFERED=1
export PADS_PROJECT_ROOT="/workspace/runs/baseline/tag_${TAG}"
export PADS_DATA_ROOT="/workspace"

# Read-only check. Fix .pylibs ONCE with ops/fix_numpy_abi.slurm, never per-job:
# a live 'pip install --target' can race a sibling job already importing numpy.
NUMPY_CHECK=$(apptainer exec --bind "${PROJECT_ROOT}:/workspace" "${SIF}" \
    python -c "import numpy; print(numpy.__version__)" 2>&1) || true
case "${NUMPY_CHECK}" in
    1.*) echo "  numpy ${NUMPY_CHECK} OK" ;;
    *)   echo "ERROR: numpy in .pylibs is not a healthy 1.x install (got: ${NUMPY_CHECK})"
         echo "       Fix it once: sbatch ops/fix_numpy_abi.slurm"
         exit 1 ;;
esac

# Reference configuration (Casini et al.), passed explicitly so the run is
# self-documenting and cannot drift if a default changes:
#   batch 32, no accumulation -- BatchNorm over 8 samples is not the same
#                                statistic as over 32
#   precision 32              -- V100 is sm_70; bf16 needs sm_80
#   val_crop random           -- the reference random-crops at evaluation
apptainer exec --nv \
    --bind "${PROJECT_ROOT}:/workspace" \
    --env PYTHONPATH="/workspace/.pylibs" \
    --env PADS_PROJECT_ROOT="${PADS_PROJECT_ROOT}" \
    --env PADS_DATA_ROOT="${PADS_DATA_ROOT}" \
    "${SIF}" \
    python -u /workspace/3-pads/deeplab/train.py \
        --epochs "${EPOCHS}" \
        --batch_size 32 \
        --accumulate 1 \
        --precision 32 \
        --val_crop random \
        "${POLICY_ARGS[@]}" \
        --num_workers "${NUM_WORKERS}"

echo "============================================================"
echo "Done. End: $(date)"
echo "============================================================"
