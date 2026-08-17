#!/usr/bin/env bash
#SBATCH --job-name=disc_ckpt
#SBATCH --partition=interactive
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
#SBATCH --output=runs/discovery_check/logs/%x_%j.out
#SBATCH --error=runs/discovery_check/logs/%x_%j.err
#SBATCH --export=NONE
#SBATCH --qos=debug

# Does 3-pads/discovery/infer_tiles.py load the 0-reproduction checkpoints?
# It builds ArcheoModel from 3-pads/<arch>/model.py and calls
# load_state_dict(..., strict=True), so any key mismatch fails loudly here
# rather than after hours of inference.

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
import sys, torch
sys.path.insert(0,'/workspace/3-pads/discovery')
import infer_tiles as it

CK='/workspace/0-reproduction/runs/{}_full/lightning_logs/version_0/checkpoints/epoch=19-step=2940.ckpt'
for name in ('deeplab','manet','segformer'):
    ck = CK.format(name)
    try:
        m = it._load('/workspace', name, ck, 'cpu')
        n = sum(p.numel() for p in m.parameters())
        x = torch.zeros(1,3,256,256)
        with torch.no_grad():
            y = m(x)
        y = y.logits if hasattr(y,'logits') else y
        print(f'OK   {name:<10} params={n/1e6:.1f}M  forward{tuple(x.shape)} -> {tuple(y.shape)}')
    except Exception as e:
        print(f'FAIL {name:<10} {type(e).__name__}: {str(e)[:220]}')
"
