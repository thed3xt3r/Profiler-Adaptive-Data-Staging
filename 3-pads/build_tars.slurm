#!/usr/bin/env bash
#SBATCH --job-name=pads_build_tars
#SBATCH --partition=l40s
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=build_tars_%j.log
#SBATCH --export=NONE
#SBATCH --qos=besteffort


# =============================================================================
# One-time prerequisite for the PADS 'shard' policy.
#
# TarArcheoDataset reads two archives (bing_1k_tars/originals.tar,
# bing_1k_tars/negs.tar) that nothing else in the repo produces. Without them,
# --data_policy shard (and the profiler auto-selecting "shard") silently falls
# back to "loose" -- see the [PADS] warning in PAD_*_hpc.slurm output.
#
# Run this ONCE before any PADS job that might select shard (including the
# policy-comparison matrix). It is not meant to be re-run per training job:
# it rebuilds ~5-6 GB of archives unconditionally on every invocation, which
# would otherwise turn into wasted I/O on every single submission.
#
# Usage: sbatch 3-pads/build_tars.slurm
# =============================================================================

set -euo pipefail

PROJECT_ROOT="${HOME}/Thesis"
SIF="${PROJECT_ROOT}/tell_seg.sif"

echo "============================================================"
echo "PADS - building shard archives"
echo "Hostname:   $(hostname)"
echo "Job ID:     ${SLURM_JOB_ID:-none}"
echo "Start:      $(date)"
echo "============================================================"

[ -d "${PROJECT_ROOT}" ]               || { echo "ERROR: no ${PROJECT_ROOT}"; exit 1; }
[ -f "${SIF}" ]                        || { echo "ERROR: no ${SIF}"; exit 1; }
[ -d "${PROJECT_ROOT}/bing_1k/train" ] || { echo "ERROR: no dataset at ${PROJECT_ROOT}/bing_1k/train"; exit 1; }

module load tools/Apptainer/1.4.0
cd "${PROJECT_ROOT}"

echo "Checking NumPy in .pylibs..."
# Read-only check, NOT a write -- see ops/fix_numpy_abi.slurm for why this no
# longer reinstalls into .pylibs on every job (a concurrent write here can
# corrupt a DIFFERENT job's in-progress numpy import).
NUMPY_CHECK=$(apptainer exec --bind "${PROJECT_ROOT}:/workspace" "${SIF}" \
    python -c "import numpy; print(numpy.__version__)" 2>&1) || true
case "${NUMPY_CHECK}" in
    1.*) echo "  numpy ${NUMPY_CHECK} OK" ;;
    *)   echo "ERROR: numpy in .pylibs is not a healthy 1.x install (got: ${NUMPY_CHECK})"
         echo "       Fix it once: sbatch ops/fix_numpy_abi.slurm"
         exit 1 ;;
esac

apptainer exec --bind "${PROJECT_ROOT}:/workspace" "${SIF}" \
    python /workspace/3-pads/build_tars.py --data_root /workspace

echo "============================================================"
echo "Done. End: $(date)"
echo "Archives are at: ${PROJECT_ROOT}/bing_1k_tars/{originals,negs}.tar"
echo "Now safe to submit PAD_*_hpc.slurm jobs with POLICY=shard."
echo "============================================================"
