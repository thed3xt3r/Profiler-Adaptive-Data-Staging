#!/usr/bin/env bash
#SBATCH --job-name=score_tiles
#SBATCH --partition=l40s
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=runs/figs/logs/%x_%j.out
#SBATCH --error=runs/figs/logs/%x_%j.err
#SBATCH --export=NONE
#SBATCH --qos=besteffort

set -euo pipefail
PROJECT_ROOT="${HOME}/Thesis"
SIF="${PROJECT_ROOT}/tell_seg.sif"
module load tools/Apptainer/1.4.0
cd "${PROJECT_ROOT}"
export PYTHONUNBUFFERED=1
apptainer exec --nv \
    --bind "${PROJECT_ROOT}:/workspace" \
    --env PYTHONPATH="/workspace/.pylibs" \
    "${SIF}" \
    python -u /workspace/4-inference/score_test_tiles.py
