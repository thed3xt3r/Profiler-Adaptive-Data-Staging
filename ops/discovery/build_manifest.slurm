#!/usr/bin/env bash
#SBATCH --job-name=build_manifest
#SBATCH --partition=interactive
#SBATCH --cpus-per-task=4
#SBATCH --time=00:55:00
#SBATCH --output=runs/discovery/logs/%x_%j.out
#SBATCH --error=runs/discovery/logs/%x_%j.err
#SBATCH --export=NONE
#SBATCH --qos=debug

# Reconstructs the QGIS render manifest that the discovery pipeline needs.
#
# Two corrections against the command run on the login node:
#   * the login node has Python 3.6.8 and no pip, so fiona/shapely can never
#     import there; this runs inside the container with .geolibs on the path;
#   * --images-dir was /workspace/discovery_tiles/images, which does not exist.
#     The 55,008 tiles are at /workspace/images.
#
# .geolibs carries its own numpy 2.2.6, which would shadow the container's numpy.
# That is harmless here and for stitching, but it must NOT be put on PYTHONPATH
# for infer_tiles.py, which needs the container's numpy/torch pairing.

set -euo pipefail
PROJECT_ROOT="${HOME}/Thesis"
module load tools/Apptainer/1.4.0
cd "${PROJECT_ROOT}"
export PYTHONUNBUFFERED=1

apptainer exec \
  --bind "${PROJECT_ROOT}:/workspace" \
  --env PYTHONPATH=/workspace/.geolibs \
  "${PROJECT_ROOT}/tell_seg.sif" \
  python -u /workspace/3-pads/discovery/reconstruct_render_manifest.py \
    --project-area          /workspace/qgis/project_area.gpkg \
    --known-site-exclusion  /workspace/qgis/known_site_exclusion.gpkg \
    --images-dir            /workspace/images \
    --output                /workspace/discovery_tiles/tiles.json \
    --take-first            55008
