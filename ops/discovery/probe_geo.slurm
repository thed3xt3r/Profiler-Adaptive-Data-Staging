#!/usr/bin/env bash
#SBATCH --job-name=probe_geo
#SBATCH --partition=interactive
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=runs/discovery/logs/%x_%j.out
#SBATCH --error=runs/discovery/logs/%x_%j.err
#SBATCH --export=NONE
#SBATCH --qos=debug

# reconstruct_render_manifest.py needs fiona + shapely; stitch_tile_predictions.py
# needs rasterio. The login node has Python 3.6.8 without pip, so the container is
# the only viable interpreter. Find out what it already has, and if the geo stack
# is absent try to add it to a separate .geolibs prefix (kept out of .pylibs so a
# bad wheel cannot break the training/inference runs).

set -uo pipefail
PROJECT_ROOT="${HOME}/Thesis"
module load tools/Apptainer/1.4.0
cd "${PROJECT_ROOT}"
export PYTHONUNBUFFERED=1

echo "=== what the container already has ==="
apptainer exec --bind "${PROJECT_ROOT}:/workspace" "${PROJECT_ROOT}/tell_seg.sif" \
  python -c "
import sys; print('python', sys.version.split()[0])
for m in ('fiona','shapely','rasterio','pyproj','osgeo'):
    try:
        mod=__import__(m); print(f'  OK      {m:9}', getattr(mod,'__version__','?'))
    except Exception as e: print(f'  MISSING {m:9}', type(e).__name__)
" 2>&1 | grep -viE "warning|pkg_resources"

echo
echo "=== attempting install into .geolibs (needs outbound network) ==="
apptainer exec --bind "${PROJECT_ROOT}:/workspace" "${PROJECT_ROOT}/tell_seg.sif" \
  python -m pip install --no-cache-dir --target /workspace/.geolibs \
    fiona shapely rasterio 2>&1 | tail -15

echo
echo "=== re-check with .geolibs on the path ==="
apptainer exec --bind "${PROJECT_ROOT}:/workspace" \
  --env PYTHONPATH=/workspace/.geolibs \
  "${PROJECT_ROOT}/tell_seg.sif" \
  python -c "
for m in ('fiona','shapely','rasterio'):
    try:
        mod=__import__(m); print(f'  OK      {m:9}', getattr(mod,'__version__','?'))
    except Exception as e: print(f'  MISSING {m:9}', type(e).__name__, str(e)[:120])
" 2>&1 | grep -viE "warning|pkg_resources"
