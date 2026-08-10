#!/usr/bin/env bash
#SBATCH --job-name=fix_tokenizers
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=00:10:00
#SBATCH --output=fix_tokenizers.log
#SBATCH --export=NONE

set -euo pipefail

module load tools/Apptainer/1.4.0

# Force pip to install the correct tokenizers version matching the older transformers
apptainer exec --bind "${HOME}/Thesis:/workspace" "${HOME}/Thesis/tell_seg.sif" pip install --target /workspace/.pylibs "tokenizers>=0.19,<0.20" --upgrade
