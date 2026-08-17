#!/usr/bin/env bash
#SBATCH --job-name=quota_check
#SBATCH --partition=interactive
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --output=runs/discovery/logs/%x_%j.out
#SBATCH --error=runs/discovery/logs/%x_%j.err
#SBATCH --export=NONE
#SBATCH --qos=debug
set -uo pipefail
cd "${HOME}/Thesis"
echo "=== mmlsquota ==="; timeout 60 mmlsquota --block-size auto 2>&1 | head -12
echo "=== quota -s ===";  timeout 60 quota -s 2>&1 | head -12
echo "=== df ===";        df -h "${HOME}/Thesis" | tail -2
