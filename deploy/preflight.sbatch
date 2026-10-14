#!/bin/bash
#SBATCH --job-name=liveact-preflight
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:05:00
#SBATCH --output=runs/slurm/preflight-%j.out
#SBATCH --error=runs/slurm/preflight-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
if [ -f /etc/profile.d/modules.sh ]; then
    source /etc/profile.d/modules.sh
    module load gcc/12.5
fi
.hopper-env/bin/python -m tools.run preflight
