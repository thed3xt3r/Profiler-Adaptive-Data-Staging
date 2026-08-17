#!/usr/bin/env bash
#SBATCH --job-name=preflight_disc
#SBATCH --partition=interactive
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=runs/discovery/logs/%x_%j.out
#SBATCH --error=runs/discovery/logs/%x_%j.err
#SBATCH --export=NONE
#SBATCH --qos=debug

# run_discovery.py has no preflight-only mode: run() falls straight into ~2 h of
# inference. This replicates every one of its checks first, so a missing file or
# a numpy/torch ABI clash surfaces in seconds rather than after the queue wait.
#
# The ABI question is real: .geolibs installed numpy 2.2.6, and PYTHONPATH takes
# precedence over the container's site-packages, so it would shadow whatever numpy
# torch was compiled against. Test with and without it.

set -uo pipefail
PROJECT_ROOT="${HOME}/Thesis"
module load tools/Apptainer/1.4.0
cd "${PROJECT_ROOT}"
export PYTHONUNBUFFERED=1

echo "############ A: container numpy only (no .geolibs) ############"
apptainer exec --nv --bind "${PROJECT_ROOT}:/workspace" \
  "${PROJECT_ROOT}/tell_seg.sif" \
  python -c "import numpy,torch; print('  container numpy', numpy.__version__, '| torch', torch.__version__)" \
  2>&1 | grep -viE "warning|pkg_resources"

echo
echo "############ B: .geolibs on PYTHONPATH (numpy 2.2.6 shadows) ############"
apptainer exec --nv --bind "${PROJECT_ROOT}:/workspace" \
  --env PYTHONPATH=/workspace/.geolibs \
  "${PROJECT_ROOT}/tell_seg.sif" \
  python -c "
import numpy; print('  numpy', numpy.__version__)
for m in ('torch','rasterio','fiona','shapely'):
    try:
        mod=__import__(m); print(f'  OK      {m:9}', getattr(mod,'__version__','?'))
    except Exception as e: print(f'  BROKEN  {m:9}', type(e).__name__, str(e)[:150])
" 2>&1 | grep -viE "warning|pkg_resources"

echo
echo "############ C: .geolibs WITHOUT its numpy (rasterio on container numpy) ############"
# NumPy 2-built wheels are runtime-compatible with numpy>=1.19, so dropping the
# bundled numpy should let rasterio/fiona bind to the container's numpy instead.
rm -rf /tmp/geolibs_nonumpy && mkdir -p /tmp/geolibs_nonumpy
cp -r "${PROJECT_ROOT}"/.geolibs/* /tmp/geolibs_nonumpy/ 2>/dev/null
rm -rf /tmp/geolibs_nonumpy/numpy /tmp/geolibs_nonumpy/numpy.libs /tmp/geolibs_nonumpy/numpy-*.dist-info
apptainer exec --nv --bind "${PROJECT_ROOT}:/workspace" --bind /mnt/scratch/users/${USER}/discovery:/scratch_out \
  --env PYTHONPATH=/geolibs \
  "${PROJECT_ROOT}/tell_seg.sif" \
  python -c "
import numpy; print('  numpy', numpy.__version__)
for m in ('torch','rasterio','fiona','shapely','lightning','segmentation_models_pytorch','transformers','PIL'):
    try:
        mod=__import__(m); print(f'  OK      {m:28}', getattr(mod,'__version__','?'))
    except Exception as e: print(f'  BROKEN  {m:28}', type(e).__name__, str(e)[:150])
" 2>&1 | grep -viE "warning|pkg_resources|newly initialized|should probably TRAIN|shapes did not match|decode_head"

echo
echo "############ D: full run_discovery preflight (config C arrangement) ############"
apptainer exec --nv --bind "${PROJECT_ROOT}:/workspace" --bind /mnt/scratch/users/${USER}/discovery:/scratch_out \
  --env PYTHONPATH="/workspace/.geolibs:/workspace/3-pads/discovery" \
  --env TRANSFORMERS_OFFLINE=1 --env HF_HUB_OFFLINE=1 \
  "${PROJECT_ROOT}/tell_seg.sif" \
  python -u -c "
import json, sys
from pathlib import Path
sys.path.insert(0,'/workspace/3-pads/discovery')
cfg=json.loads(Path('/workspace/3-pads/discovery/config.hpc.json').read_text())
from stitch_tile_predictions import load_manifest
manifest=Path(cfg['manifest']); tiles_root=Path(cfg['tiles_root'])
project_root=Path(cfg['project_root']); out=Path(cfg['output_dir'])
errs=[]
for label,p,isdir in (('manifest',manifest,False),('tiles_root',tiles_root,True),
                      ('project_root',project_root,True),('output_dir',out,True)):
    ok = p.is_dir() if isdir else p.is_file()
    print(f'  {\"OK  \" if ok else \"FAIL\"} {label:13} {p}')
    if not ok: errs.append(label)
for n,p in cfg['checkpoints'].items():
    p=Path(p); ok=p.is_file(); print(f'  {\"OK  \" if ok else \"FAIL\"} ckpt {n:9} {ok}')
    if not ok: errs.append(n)
for n in ('deeplab','manet','segformer'):
    p=project_root/'3-pads'/n/'model.py'
    if not p.is_file(): errs.append('model.py '+n)
print('  OK   model.py x3' if not any('model.py' in e for e in errs) else '  FAIL model.py')
tiles,meta=load_manifest(manifest)
print(f'  OK   load_manifest      {len(tiles)} tiles')
missing=[t for t in tiles if not (tiles_root/t.filename).is_file()]
print(f'  {\"OK  \" if not missing else \"FAIL\"} all images present  missing={len(missing)}')
import torch; print(f'  {\"OK  \" if torch.cuda.is_available() else \"FAIL\"} CUDA available')
from transformers import AutoConfig
AutoConfig.from_pretrained('nvidia/segformer-b0-finetuned-ade-512-512',local_files_only=True)
print('  OK   SegFormer offline cache')
print()
print('PREFLIGHT CLEAN' if not errs and not missing else 'PREFLIGHT FAILED: '+str(errs))
" 2>&1 | grep -viE "warning|pkg_resources"
