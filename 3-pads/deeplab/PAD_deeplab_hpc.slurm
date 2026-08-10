#!/usr/bin/env bash
#SBATCH --job-name=deeplab_pads_hpc
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=runs/deeplab/logs/%x_%j.out
#SBATCH --error=runs/deeplab/logs/%x_%j.err
#SBATCH --export=NONE
#SBATCH --qos=normal

# =============================================================================
# DeepLabV3Plus on Iris, using the reference (Casini et al.) configuration.
#
# Differs from PAD_deeplab.slurm in that every parameter the reference fixes is
# passed EXPLICITLY rather than inherited from a default, so the run is
# self-documenting and cannot drift if defaults change.
#
#   batch 32, no accumulation   -- V100 has the memory; a 4 GB card does not,
#                                  and BatchNorm over 8 samples is not the same
#                                  statistic as over 32
#   precision 32                -- matches the reference; V100 is sm_70 so bf16
#                                  is unavailable in any case
#   focal alpha unset (None)    -- the reference uses smp's default
#   val_crop random             -- the reference random-crops at evaluation
#   no --scale_input            -- the reference standardises raw 0-255 data
#
# BEFORE ANY PADS RUN:
#   Build the shard archives once on the cluster -- POLICY=shard cannot be
#   selected without them (TarArcheoDataset falls back to loose silently):
#     sbatch 3-pads/build_tars.slurm
#
# TREAT THE FIRST JOB AS A TEST, NOT A RESULT:
#   Batch-32 memory headroom, container package versions against the local
#   changes, and smp 0.3.3's FocalLoss(alpha=...) signature have not been
#   verified on Iris. Run the MA-Net smoke test first (see PAD_manet_hpc.slurm)
#   and confirm it tracks toward the previous ~0.68 reference run before
#   submitting the full loose/shard/stage comparison matrix for this model too.
#
# Usage: sbatch PAD_deeplab_hpc.slurm [MODE] [EPOCHS] [POLICY] [FOCAL_ALPHA]
#   MODE         standard|pads   (default: standard)
#   EPOCHS       integer          (default: 20)
#   POLICY       loose|shard|stage, or empty to let MODE decide (default: empty)
#                Forces --data_policy; needed to A/B the policies in the
#                comparison matrix. Overrides pads-mode profiling when both are set.
#   FOCAL_ALPHA  Focal loss alpha (positive-class weight), or empty for the
#                code's own default (empty -> None, unweighted). model.py's
#                comment claims the reference uses 0.75; train.py's --help
#                claims the reference uses None -- these contradict each
#                other, and every run so far used None (~7-12% of pixels are
#                the positive class, so an unweighted loss lets the model
#                coast on predicting background). Worth testing 0.75 against
#                whatever got the earlier ~0.68 result.
#
# Example comparison-matrix submissions (after the MA-Net smoke test passes):
#   sbatch --job-name=deeplab_loose PAD_deeplab_hpc.slurm standard 20 loose
#   sbatch --job-name=deeplab_shard PAD_deeplab_hpc.slurm standard 20 shard
#   sbatch --job-name=deeplab_stage PAD_deeplab_hpc.slurm standard 20 stage
#   sbatch --job-name=deeplab_loose_alpha75 PAD_deeplab_hpc.slurm standard 20 loose 0.75
# =============================================================================

set -euo pipefail

PROJECT_ROOT="${HOME}/Thesis"
SIF="${PROJECT_ROOT}/tell_seg.sif"
MODE="${1:-standard}"
EPOCHS="${2:-20}"
POLICY="${3:-}"
FOCAL_ALPHA="${4:-}"
NUM_WORKERS="${5:-8}"
RUN_TAG="${6:-}"

echo "============================================================"
echo "DeepLabV3Plus (ResNet-50 + Focal) -- reference config"
echo "Mode:        ${MODE}"
echo "Epochs:      ${EPOCHS}"
echo "Policy:      ${POLICY:-<auto/unset>}"
echo "Focal alpha: ${FOCAL_ALPHA:-<code default: None>}"
echo "Num workers: ${NUM_WORKERS}"
echo "Run tag:    ${RUN_TAG:-<none>}"
echo "Hostname:   $(hostname)"
echo "Job ID:     ${SLURM_JOB_ID:-none}"
echo "Start:      $(date)"
echo "============================================================"

[ -d "${PROJECT_ROOT}" ]            || { echo "ERROR: no ${PROJECT_ROOT}"; exit 1; }
[ -f "${SIF}" ]                     || { echo "ERROR: no ${SIF}"; exit 1; }
[ -d "${PROJECT_ROOT}/bing_1k/train" ] || { echo "ERROR: no dataset"; exit 1; }

if [ "${POLICY}" = "shard" ]; then
    if [ ! -f "${PROJECT_ROOT}/bing_1k_tars/originals.tar" ] || [ ! -f "${PROJECT_ROOT}/bing_1k_tars/negs.tar" ]; then
        echo "ERROR: POLICY=shard requested but ${PROJECT_ROOT}/bing_1k_tars/{originals,negs}.tar are missing."
        echo "       Build them once first: sbatch 3-pads/build_tars.slurm"
        exit 1
    fi
fi

mkdir -p "${PROJECT_ROOT}/runs/deeplab/logs" "${PROJECT_ROOT}/runs/checkpoints_deeplab"

module load tools/Apptainer/1.4.0
cd "${PROJECT_ROOT}"

export PYTHONUNBUFFERED=1
# Outputs land under ${PROJECT_ROOT}/runs/ (train.py default) -- or, when a
# POLICY and/or FOCAL_ALPHA is forced, under a matching
# ${PROJECT_ROOT}/runs/policy_<policy>[/alpha_<alpha>]/ subtree. Without this
# split, any two runs sharing the same isolation key share one
# checkpoints_<model>/last.ckpt: the second run "resumes" the first one's
# already-finished checkpoint and immediately reports done without training a
# single step of its own (hit in practice: a "shard" run completed in under 3
# minutes instead of ~9, having silently inherited an already-max_epochs'd
# loose checkpoint -- and a rerun of plain "loose" would just as silently
# collide with an "loose"+alpha=0.75 result sitting in the same place).
RUN_SUFFIX=""
if [ -n "${POLICY}" ]; then
    RUN_SUFFIX="${RUN_SUFFIX}/policy_${POLICY}"
fi
if [ -n "${FOCAL_ALPHA}" ]; then
    RUN_SUFFIX="${RUN_SUFFIX}/alpha_${FOCAL_ALPHA}"
fi
if [ -n "${RUN_TAG}" ]; then
    RUN_SUFFIX="${RUN_SUFFIX}/tag_${RUN_TAG}"
fi
# RUN_TAG exists because POLICY/FOCAL_ALPHA alone do not disambiguate runs
# that differ only in NUM_WORKERS or in which partition/hardware ran them --
# e.g. RQ1's "default" (num_workers=0) vs "tuned" (num_workers=4) jobs are
# both policy=loose, alpha=unset, and would otherwise share one
# checkpoints_<model>/last.ckpt and silently collide (see the comment above
# PADS_PROJECT_ROOT's original use for the same failure mode with POLICY).
export PADS_PROJECT_ROOT="/workspace/runs${RUN_SUFFIX}"
export PADS_DATA_ROOT="/workspace"

# NOTE: no PYTORCH_CUDA_ALLOC_CONF=expandable_segments here -- that option needs
# torch >= 2.1 and the container ships 2.0.1.

echo "Checking NumPy in .pylibs..."
# Read-only check, NOT a write. A live `pip install --target` here -- even
# flock-serialized against other pip installs -- can still race against a
# DIFFERENT concurrently-running job's python process importing numpy for
# the first time (the flock only protects pip invocations from each other,
# not from a sibling job's already-running training process). Hit in
# practice as a transient "ModuleNotFoundError: numpy._typing._add_docstring"
# in another job's log while two matrix jobs started together. Fix .pylibs
# ONCE with ops/fix_numpy_abi.slurm before submitting jobs, not per-job.
NUMPY_CHECK=$(apptainer exec --bind "${PROJECT_ROOT}:/workspace" "${SIF}" \
    python -c "import numpy; print(numpy.__version__)" 2>&1) || true
case "${NUMPY_CHECK}" in
    1.*) echo "  numpy ${NUMPY_CHECK} OK" ;;
    *)   echo "ERROR: numpy in .pylibs is not a healthy 1.x install (got: ${NUMPY_CHECK})"
         echo "       Fix it once: sbatch ops/fix_numpy_abi.slurm"
         exit 1 ;;
esac

COMMON_ARGS=(
    --epochs "${EPOCHS}"
    --batch_size 32
    --accumulate 1
    --precision 32
    --num_workers "${NUM_WORKERS}"
    --val_crop random
)

if [ -n "${POLICY}" ]; then
    COMMON_ARGS+=(--data_policy "${POLICY}")
fi

if [ -n "${FOCAL_ALPHA}" ]; then
    COMMON_ARGS+=(--focal_alpha "${FOCAL_ALPHA}")
fi

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
    python -u /workspace/3-pads/deeplab/train.py "${COMMON_ARGS[@]}" "${EXTRA_ARGS[@]}"

echo "============================================================"
echo "Done. End: $(date)"
echo "============================================================"
