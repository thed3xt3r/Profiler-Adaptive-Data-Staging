#!/usr/bin/env bash
#SBATCH --job-name=stitch
#SBATCH --partition=batch
#SBATCH --cpus-per-task=4
#SBATCH --time=1-00:00:00
#SBATCH --output=runs/discovery/logs/%x_%j.out
#SBATCH --error=runs/discovery/logs/%x_%j.err
#SBATCH --export=NONE

# Stitch only. Job 5674038 completed inference for all three models (165,024 .npy
# on SCRATCH) but was going to hit its 6 h wall clock about an hour into the
# remaining stitch, so it was cancelled and the stitch restarted here.
#
# Measured on the cancelled run: 300 ms/tile => ~4.6 h for 55,008 tiles. The 1 day
# limit gives ~5x headroom.
#
# No GPU: stitch() is numpy + rasterio only. Three np.load per tile, the
# _probability validation passes, the eight-mask RGBA build, and four
# deflate-compressed windowed writes. It was only ever on a GPU node because
# run_discovery.py bundles it with inference.
#
# Default QoS deliberately, not besteffort: this cannot be preempted without
# losing everything, since stitch() has no resume.

set -euo pipefail
PROJECT_ROOT="${HOME}/Thesis"
module load tools/Apptainer/1.4.0
cd "${PROJECT_ROOT}"
export PYTHONUNBUFFERED=1

SCRATCH_OUT="/mnt/scratch/users/${USER}/discovery"

apptainer exec \
  --bind "${PROJECT_ROOT}:/workspace" \
  --bind "${SCRATCH_OUT}:/scratch_out" \
  --env PYTHONPATH="/workspace/.geolibs:/workspace/3-pads/discovery" \
  "${PROJECT_ROOT}/tell_seg.sif" \
  python -u -c "
import json, sys
from pathlib import Path
sys.path.insert(0,'/workspace/3-pads/discovery')
from stitch_tile_predictions import stitch

cfg=json.loads(Path('/workspace/3-pads/discovery/config.hpc.json').read_text())
NAMES=('deeplab','manet','segformer')
out=Path(cfg['output_dir'])
pred=out/'tile_probabilities'
dirs={n:pred/n for n in NAMES}
for n,d in dirs.items():
    print('{:10} {} predictions'.format(n, len(list(d.glob('*.npy')))), flush=True)
outs=stitch(cfg['manifest'], dirs, out/'qgis_layers', float(cfg.get('threshold',0.5)))
print()
print('SUCCESS - add these four layers to QGIS:')
for o in outs:
    print('  {}  ({:.2f} GB)'.format(o, Path(o).stat().st_size/1e9))
"
