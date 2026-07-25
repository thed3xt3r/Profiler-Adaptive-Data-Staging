#!/usr/bin/env bash
#SBATCH --job-name=fix_numpy_abi
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=00:10:00
#SBATCH --output=fix_numpy_abi.log
#SBATCH --export=NONE

set -euo pipefail

module load tools/Apptainer/1.4.0

# ~/Thesis/.pylibs picked up numpy 2.2.6 as an unconstrained transitive
# dependency of a package installed there with `pip install --target`
# (optuna>=4.x is the likely culprit: it declares "numpy" with no upper
# bound). .pylibs is prepended to PYTHONPATH ahead of the container's own
# site-packages, so that numpy 2.x shadows the container's pinned
# "numpy<2" — breaking ABI compatibility for anything compiled against
# NumPy 1.x's C API (scikit-learn's murmurhash extension via albumentations,
# wandb's data_types module via lightning's loggers, etc.).
#
# Force .pylibs back down to numpy<2 to match the container. Every package
# already installed in .pylibs (optuna, transformers, safetensors,
# webdataset, huggingface_hub) accepts numpy<2, so this does not break
# anything else there.
apptainer exec --bind "${HOME}/Thesis:/workspace" "${HOME}/Thesis/tell_seg.sif" \
    pip install --target /workspace/.pylibs "numpy<2" --upgrade --force-reinstall
