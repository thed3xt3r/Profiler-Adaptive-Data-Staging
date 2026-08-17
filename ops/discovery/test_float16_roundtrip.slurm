#!/usr/bin/env bash
#SBATCH --job-name=fp16_test
#SBATCH --partition=interactive
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:25:00
#SBATCH --output=runs/discovery/logs/%x_%j.out
#SBATCH --error=runs/discovery/logs/%x_%j.err
#SBATCH --export=NONE
#SBATCH --qos=debug

# Proves the float16 change survives the full path on real data: three models ->
# .npy -> _probability() -> stitch() -> four GeoTIFFs. Uses a 4-tile manifest cut
# from the real one, so the grid/CRS/extent checks in stitch() are exercised for
# real rather than assumed.
#
# Also quantifies the precision actually lost, and confirms no pixel changes side
# of the 0.5 threshold -- the only thing half precision could plausibly break.

set -euo pipefail
PROJECT_ROOT="${HOME}/Thesis"
module load tools/Apptainer/1.4.0
cd "${PROJECT_ROOT}"
export PYTHONUNBUFFERED=1

apptainer exec --nv \
  --bind "${PROJECT_ROOT}:/workspace" \
  --env PYTHONPATH="/workspace/.geolibs:/workspace/3-pads/discovery" \
  --env TRANSFORMERS_OFFLINE=1 --env HF_HUB_OFFLINE=1 \
  "${PROJECT_ROOT}/tell_seg.sif" \
  python -u -c "
import json, shutil, sys
from pathlib import Path
import numpy as np
sys.path.insert(0,'/workspace/3-pads/discovery')
import infer_tiles as it
from stitch_tile_predictions import stitch, _probability

CK='/workspace/0-reproduction/runs/{}_full/lightning_logs/version_0/checkpoints/epoch=19-step=2940.ckpt'
NAMES=('deeplab','manet','segformer')
work=Path('/workspace/runs/discovery/_fp16_test')
if work.exists(): shutil.rmtree(work)
work.mkdir(parents=True)

# 4-tile manifest cut from the real one
full=json.loads(Path('/workspace/discovery_tiles/tiles.json').read_text())
sub=dict(full); sub['tiles']=full['tiles'][:4]; sub['tile_count']=4
mpath=work/'tiles.json'; mpath.write_text(json.dumps(sub))
print('subset tiles:', [t['tile_id'] for t in sub['tiles']])

dirs={}
for n in NAMES:
    d=work/'pred'/n
    it.run(n, mpath, '/workspace/images', CK.format(n), d, '/workspace',
           'cuda', 512, 512, 256, 4)
    dirs[n]=d
    f=sorted(d.glob('*.npy'))[0]
    a=np.load(f)
    print(f'  {n:10} dtype={a.dtype} shape={a.shape} bytes={f.stat().st_size/1e6:.2f} MB '
          f'range=[{a.min():.4f},{a.max():.4f}]')

print()
print('=== precision actually lost vs float32 ===')
for n in NAMES:
    m=it._load('/workspace',n,CK.format(n),'cuda')
    p=Path('/workspace/images')/sub['tiles'][0]['filename']
    ref=it._predict(m,n,str(p),'cuda',512,512,256,4).astype('float32')  # full precision
    got=_probability(sorted(dirs[n].glob('*.npy'))[0], 1024)            # via float16
    err=np.abs(ref-got)
    flip=np.sum((ref>=0.5)!=(got>=0.5))
    print(f'  {n:10} max_abs_err={err.max():.2e}  mean={err.mean():.2e}  '
          f'threshold flips={flip} / {ref.size}')
    del m

print()
print('=== stitch ===')
outs=stitch(mpath, dirs, work/'qgis_layers', 0.5)
import rasterio
for o in outs:
    with rasterio.open(o) as src:
        print(f'  {Path(o).name:42} {src.width}x{src.height} {src.count}band {src.dtypes[0]} '
              f'{Path(o).stat().st_size/1e6:7.2f} MB')
print()
print('ROUND TRIP OK')
" 2>&1 | grep -viE "warning|pkg_resources|newly initialized|should probably TRAIN|shapes did not match|decode_head"
