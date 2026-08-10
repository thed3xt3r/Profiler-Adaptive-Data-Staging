#!/usr/bin/env bash
#SBATCH --job-name=segformer_pads_l40s
#SBATCH --partition=l40s
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=runs/segformer/logs/%x_%j.out
#SBATCH --error=runs/segformer/logs/%x_%j.err
#SBATCH --export=NONE
#SBATCH --qos=besteffort

# =============================================================================
# SegFormer on the L40S partition -- fast-iteration variant of
# PAD_segformer_hpc.slurm.
#
# Same reference (Casini et al.) defaults as the V100 script, run on
# hardware that was sitting mostly idle (6/8 L40S GPUs free) when this was
# written, vs. the gpu (V100) partition which was heavily contended. Use
# this for quick turnaround; use PAD_segformer_hpc.slurm for the config you
# actually want to report, so results stay comparable to earlier V100 runs.
#
#   batch 32, no accumulation   -- same as the V100 script; batch size changes
#                                  BatchNorm statistics, so it stays pinned
#                                  even though L40S (48 GB) has headroom to
#                                  raise it
#   precision 32 by default     -- matches the V100 script's reference config.
#                                  UNLIKE the V100 (sm_70, no native bf16),
#                                  L40S is Ada Lovelace (sm_89) and runs
#                                  bf16-mixed natively -- pass a 4th arg of
#                                  "bf16-mixed" to opt into it. That is a
#                                  DEVIATION from the pinned reference config,
#                                  traded for speed; don't use it for numbers
#                                  you intend to compare against the V100 runs.
#   focal alpha unset (None)    -- the reference uses smp's default
#   val_crop random             -- the reference random-crops at evaluation
#   no --scale_input            -- the reference standardises raw 0-255 data
#
# BEFORE ANY PADS RUN:
#   Build the shard archives once on the cluster -- POLICY=shard cannot be
#   selected without them (TarArcheoDataset falls back to loose silently):
#     sbatch PADS/build_tars.slurm
#
# TREAT THE FIRST JOB AS A TEST, NOT A RESULT:
#   Batch-32 memory headroom, container package versions against the local
#   changes, and smp 0.3.3's FocalLoss(alpha=...) signature have not been
#   verified on this partition either. Run the MA-Net smoke test first (see
#   PAD_manet_l40s.slurm) and confirm it tracks toward the previous ~0.68
#   reference run before submitting the full loose/shard/stage comparison
#   matrix for this model too.
#
# Usage: sbatch PAD_segformer_l40s.slurm [MODE] [EPOCHS] [POLICY] [PRECISION] [FOCAL_ALPHA]
#   MODE         standard|pads   (default: standard)
#   EPOCHS       integer          (default: 20)
#   POLICY       loose|shard|stage, or empty to let MODE decide (default: empty)
#   PRECISION    Lightning precision string (default: 32, matching the
#                reference). Don't mix precisions within one comparison matrix.
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
#   sbatch --job-name=segformer_loose_l40s PAD_segformer_l40s.slurm standard 20 loose
#   sbatch --job-name=segformer_shard_l40s PAD_segformer_l40s.slurm standard 20 shard
#   sbatch --job-name=segformer_stage_l40s PAD_segformer_l40s.slurm standard 20 stage
#   sbatch --job-name=segformer_loose_alpha75 PAD_segformer_l40s.slurm standard 20 loose 32 0.75
# =============================================================================

set -euo pipefail

PROJECT_ROOT="${HOME}/Thesis"
SIF="${PROJECT_ROOT}/tell_seg.sif"
MODE="${1:-standard}"
EPOCHS="${2:-20}"
POLICY="${3:-}"
PRECISION="${4:-32}"
FOCAL_ALPHA="${5:-}"
NUM_WORKERS="${6:-8}"
RUN_TAG="${7:-}"

echo "============================================================"
echo "SegFormer (MiT-B0 + Focal) -- L40S fast-iteration"
echo "Mode:        ${MODE}"
echo "Epochs:      ${EPOCHS}"
echo "Policy:      ${POLICY:-<auto/unset>}"
echo "Precision:   ${PRECISION}"
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
        echo "       Build them once first: sbatch PADS/build_tars.slurm"
        exit 1
    fi
fi

mkdir -p "${PROJECT_ROOT}/runs/segformer/logs" "${PROJECT_ROOT}/runs/checkpoints_segformer"

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
# torch >= 2.1 and the container ships 2.0.1, regardless of GPU generation.

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
    --precision "${PRECISION}"
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
    python -u /workspace/PADS/segformer/train.py "${COMMON_ARGS[@]}" "${EXTRA_ARGS[@]}"

echo "============================================================"
echo "Done. End: $(date)"
echo "============================================================"
