#!/usr/bin/env bash
#SBATCH --job-name=overviews
#SBATCH --partition=batch
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=runs/discovery/logs/%x_%j.out
#SBATCH --error=runs/discovery/logs/%x_%j.err
#SBATCH --export=NONE

# Builds an overview pyramid on the agreement overlay so QGIS stops rendering
# 114.9 Gpixel on every pan.
#
# gdaladdo is not installed anywhere on this cluster (the container has no GDAL
# CLI), so this uses rasterio's `rio overview`, which calls the same GDAL
# BuildOverviews routine underneath. COMPRESS_OVERVIEW is passed as an
# environment variable because GDAL reads config options from the environment.
#
# nearest, not average: this is a categorical raster whose eight colours are the
# agreement classes. Averaging would blend white (all three models) with dark
# grey (no detection) into greys that correspond to no class at all.

set -euo pipefail
PROJECT_ROOT="${HOME}/Thesis"
module load tools/Apptainer/1.4.0
cd "${PROJECT_ROOT}"
export PYTHONUNBUFFERED=1

TIF="/workspace/runs/discovery/qgis_layers/model_agreement_provenance_rgba.tif"

echo "=== before ==="
ls -lh "${PROJECT_ROOT}/runs/discovery/qgis_layers/model_agreement_provenance_rgba.tif"

apptainer exec \
  --bind "${PROJECT_ROOT}:/workspace" \
  --env PYTHONPATH="/workspace/.geolibs" \
  --env PATH="/workspace/.geolibs/bin:/opt/conda/bin:/usr/bin:/bin" \
  --env COMPRESS_OVERVIEW=DEFLATE \
  --env GDAL_NUM_THREADS=ALL_CPUS \
  "${PROJECT_ROOT}/tell_seg.sif" \
  rio overview --build 2,4,8,16,32,64,128 --resampling nearest "${TIF}"

echo
echo "=== after ==="
ls -lh "${PROJECT_ROOT}/runs/discovery/qgis_layers/model_agreement_provenance_rgba.tif"

apptainer exec \
  --bind "${PROJECT_ROOT}:/workspace" \
  --env PYTHONPATH="/workspace/.geolibs" \
  "${PROJECT_ROOT}/tell_seg.sif" \
  python -u -c "
import rasterio
with rasterio.open('${TIF}') as s:
    print('overviews per band:', [s.overviews(i) for i in s.indexes])
" 2>&1 | grep -viE "warning|pkg_resources"
