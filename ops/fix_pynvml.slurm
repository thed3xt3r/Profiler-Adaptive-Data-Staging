#!/usr/bin/env bash
#SBATCH --job-name=fix_pynvml
#SBATCH --partition=batch
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --output=fix_pynvml_%j.log
#SBATCH --export=NONE
#SBATCH --qos=besteffort

set -euo pipefail

module load tools/Apptainer/1.4.0

# GPUUtilMonitor (PADS/*/gpu_monitor.py) needs pynvml to poll NVML for GPU
# utilisation -- confirmed missing from tell_seg.sif. Same mechanism as
# fix_numpy_abi.slurm: install into .pylibs (bind-mounted, prepended to
# PYTHONPATH ahead of the container's own site-packages) rather than
# rebuilding the .sif image. No --nv / GPU needed for a pip install, so this
# runs on the CPU batch partition rather than competing for a GPU slot.
#
# Installed as "nvidia-ml-py" (the current, maintained PyPI package),
# importable as `import pynvml` -- NOT the old deprecated "pynvml" package
# on PyPI, which is a different, unmaintained project despite the name.
apptainer exec --bind "${HOME}/Thesis:/workspace" "${HOME}/Thesis/tell_seg.sif" \
    pip install --target /workspace/.pylibs nvidia-ml-py

echo "Verifying import..."
apptainer exec --bind "${HOME}/Thesis:/workspace" \
    --env PYTHONPATH="/workspace/.pylibs" \
    "${HOME}/Thesis/tell_seg.sif" \
    python -c "import pynvml; print('pynvml import OK, from:', pynvml.__file__)"
