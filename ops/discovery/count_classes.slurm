#!/usr/bin/env bash
#SBATCH --job-name=count_classes
#SBATCH --partition=batch
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --output=runs/discovery/logs/%x_%j.out
#SBATCH --error=runs/discovery/logs/%x_%j.err
#SBATCH --export=NONE
# Exact per-class pixel counts over the full-resolution agreement raster, so the
# thesis can quote real areas rather than figures sampled off a downsampled
# overview (nearest resampling drops isolated pixels and understates rare classes).
set -euo pipefail
PROJECT_ROOT="${HOME}/Thesis"
module load tools/Apptainer/1.4.0
cd "${PROJECT_ROOT}"
export PYTHONUNBUFFERED=1
apptainer exec --bind "${PROJECT_ROOT}:/workspace" \
  --env PYTHONPATH="/workspace/.geolibs" "${PROJECT_ROOT}/tell_seg.sif" \
  python -u -c "
import rasterio, numpy as np
p='/workspace/runs/discovery/qgis_layers/model_agreement_provenance_rgba.tif'
LAB={0:'No detection',1:'DeepLab only',2:'MA-Net only',3:'DeepLab + MA-Net',
     4:'SegFormer only',5:'DeepLab + SegFormer',6:'MA-Net + SegFormer',7:'All three'}
counts=np.zeros(8,dtype=np.int64); written=0; total=0
with rasterio.open(p) as s:
    px_m2=abs(s.transform.a*s.transform.e)
    for n,(_,w) in enumerate(s.block_windows(1),1):
        a=s.read(window=w)
        r,g,b,al=a[0],a[1],a[2],a[3]
        m=al==255
        total+=r.size; written+=int(m.sum())
        if not m.any(): continue
        code=((r==255).astype(np.uint8)|((g==255).astype(np.uint8)<<1)|((b==255).astype(np.uint8)<<2))[m]
        counts+=np.bincount(code,minlength=8)
        if n%20000==0: print('  block',n,flush=True)
print()
print('pixel area  : %.4f m2' % px_m2)
print('total px    : %d' % total)
print('written px  : %d  (%.2f%% of bbox)' % (written,100*written/total))
print()
print('%-24s %14s %12s %10s' % ('class','pixels','km2','% surveyed'))
for k in range(8):
    print('%-24s %14d %12.3f %9.4f%%' % (LAB[k],counts[k],counts[k]*px_m2/1e6,100*counts[k]/written))
det=counts[1:].sum()
print()
print('any detection : %d px = %.2f km2 (%.4f%% of surveyed)' % (det,det*px_m2/1e6,100*det/written))
print('multi-model   : %d px = %.2f km2' % (counts[[3,5,6,7]].sum(),counts[[3,5,6,7]].sum()*px_m2/1e6))
print('all three     : %d px = %.3f km2' % (counts[7],counts[7]*px_m2/1e6))
" 2>&1 | grep -viE "warning|pkg_resources"
