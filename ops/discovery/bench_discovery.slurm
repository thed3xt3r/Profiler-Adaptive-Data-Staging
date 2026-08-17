#!/usr/bin/env bash
#SBATCH --job-name=disc_bench
#SBATCH --partition=interactive
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=runs/discovery/logs/%x_%j.out
#SBATCH --error=runs/discovery/logs/%x_%j.err
#SBATCH --export=NONE
#SBATCH --qos=debug

# Measures the real per-tile cost of 3-pads/discovery/infer_tiles.py::_predict
# on actual images/ tiles, so the 55,008-tile runtime can be projected from
# measurement rather than from the training-time throughput benchmark (which
# timed GPU compute only, with data already resident).
#
# Splits the per-tile cost into read+decode, GPU, and stitch/save, because the
# loop is serial: no DataLoader, no workers, so those costs add rather than
# overlap.

set -euo pipefail
PROJECT_ROOT="${HOME}/Thesis"
module load tools/Apptainer/1.4.0
cd "${PROJECT_ROOT}"
export PYTHONUNBUFFERED=1

apptainer exec --nv \
  --bind "${PROJECT_ROOT}:/workspace" \
  --env PYTHONPATH="/workspace/.pylibs:/workspace/3-pads/discovery" \
  --env TRANSFORMERS_OFFLINE=1 --env HF_HUB_OFFLINE=1 \
  "${PROJECT_ROOT}/tell_seg.sif" \
  python -u -c "
import sys, time, glob, io, os
import numpy as np, torch
import torch.nn.functional as F
from PIL import Image
sys.path.insert(0,'/workspace/3-pads/discovery')
import infer_tiles as it

CK='/workspace/0-reproduction/runs/{}_full/lightning_logs/version_0/checkpoints/epoch=19-step=2940.ckpt'
N=40
all_tiles=sorted(glob.glob('/workspace/images/*.jpg'))
TOTAL=len(all_tiles)
# spread across the directory so we sample the filesystem, not one stripe
idx=np.linspace(0,TOTAL-1,N).astype(int)
tiles=[all_tiles[i] for i in idx]
print('total tiles on disk:', TOTAL)

# ---- read+decode cost, cold-ish (these were not touched this job) ----
t0=time.perf_counter(); nbytes=0
for p in tiles:
    with open(p,'rb') as f: raw=f.read()
    nbytes+=len(raw)
    a=np.asarray(Image.open(io.BytesIO(raw)).convert('RGB'))
io_s=time.perf_counter()-t0
print(f'read+decode : {io_s/N*1000:7.1f} ms/tile   ({nbytes/N/1024:.0f} KB/tile, {a.shape})')

# ---- per-model end-to-end _predict (warm cache now) ----
for name in ('deeplab','manet','segformer'):
    m=it._load('/workspace',name,CK.format(name),'cuda')
    # warm up cuda/cudnn so init cost does not land in the average
    for p in tiles[:3]:
        it._predict(m,name,p,'cuda',512,512,256,4)
    torch.cuda.synchronize()
    t0=time.perf_counter()
    for p in tiles:
        out=it._predict(m,name,p,'cuda',512,512,256,4)
    torch.cuda.synchronize()
    e2e=(time.perf_counter()-t0)/N

    # ---- isolate pure GPU: same 4 windows, tensor already on device ----
    img=np.asarray(Image.open(tiles[0]).convert('RGB'))
    patches=np.stack([img[y:y+512,x:x+512] for y in (0,512) for x in (0,512)])
    t=torch.from_numpy(patches).permute(0,3,1,2).float()
    if name=='segformer': t=t/255.0
    t=F.interpolate(t,size=(256,256),mode='bilinear',align_corners=False).cuda()
    with torch.inference_mode():
        for _ in range(5): m(t)
    torch.cuda.synchronize()
    t0=time.perf_counter()
    with torch.inference_mode():
        for _ in range(30): m(t)
    torch.cuda.synchronize()
    gpu=(time.perf_counter()-t0)/30

    print(f'{name:<10} e2e={e2e*1000:7.1f} ms/tile  gpu={gpu*1000:6.1f} ms/tile  '
          f'gpu_util={gpu/e2e*100:4.1f}%  ->  {e2e*TOTAL/3600:6.2f} h for {TOTAL} tiles')
    del m; torch.cuda.empty_cache()

print()
print('output volume: 1024x1024 float32 npy per tile =',1024*1024*4/1e6,'MB')
print('  per model :',TOTAL*1024*1024*4/1e9,'GB')
print('  all three :',3*TOTAL*1024*1024*4/1e9,'GB')
"
