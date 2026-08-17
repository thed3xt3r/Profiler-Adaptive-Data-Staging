#!/usr/bin/env bash
#SBATCH --job-name=discovery
#SBATCH --partition=gpu
#SBATCH --qos=besteffort
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=runs/discovery/logs/%x_%j.out
#SBATCH --error=runs/discovery/logs/%x_%j.err
#SBATCH --export=NONE

# Full discovery run: three models over 55,008 tiles, then four QGIS layers.
#
# Measured cost (job 5674000, real tiles, real checkpoints):
#   deeplab 34.6 ms/tile, manet 43.5 ms/tile, segformer 32.3 ms/tile
#   => ~1.7 h of inference, ~2-2.5 h wall clock including the .npy writes.
# The 6 h limit leaves headroom for besteffort preemption and the stitch step.
#
# Output volume: infer_tiles.py writes one 1024x1024 float16 .npy per tile per
# model = 116 GB per model, 346 GB for all three. All three sets must coexist:
# stitch() reads every model's prediction for a tile at once to build the
# agreement overlay, so they cannot be produced and deleted one model at a time.
# The three sparse deflate GeoTIFFs land on top of that; verified round trip in
# ops/test_float16_roundtrip.slurm (job 5674018).
#
# That is why output goes to SCRATCH. HOME is 358.1G against a 500G soft quota,
# so 346G cannot land there; SCRATCH allows 10T. The 165,024 .npy files also sit
# well inside SCRATCH's 1M inode quota. SCRATCH is not backed up and is purged,
# so copy runs/qgis_layers back to HOME when the run completes.
#
# PYTHONPATH note: .geolibs supplies fiona/rasterio/shapely and deliberately has no
# numpy of its own, so those bind to the container's numpy 1.24.3 (the version
# torch 2.0.1 was built against). Do not reinstall numpy into it.

set -euo pipefail
PROJECT_ROOT="${HOME}/Thesis"
module load tools/Apptainer/1.4.0
cd "${PROJECT_ROOT}"
export PYTHONUNBUFFERED=1

SCRATCH_OUT="/mnt/scratch/users/${USER}/discovery"
mkdir -p "${SCRATCH_OUT}"

echo "=== quota before the run (needs ~346 GB free on SCRATCH) ==="
df-ulhpc 2>/dev/null || echo "(df-ulhpc unavailable on compute node)"
echo

apptainer exec --nv \
  --bind "${PROJECT_ROOT}:/workspace" \
  --bind "${SCRATCH_OUT}:/scratch_out" \
  --env PYTHONPATH="/workspace/.geolibs:/workspace/3-pads/discovery" \
  --env TRANSFORMERS_OFFLINE=1 --env HF_HUB_OFFLINE=1 \
  "${PROJECT_ROOT}/tell_seg.sif" \
  python -u /workspace/3-pads/discovery/run_discovery.py \
    /workspace/3-pads/discovery/config.hpc.json
