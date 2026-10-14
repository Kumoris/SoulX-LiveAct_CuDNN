#!/bin/bash
#SBATCH --job-name=liveact-setup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=runs/slurm/setup-%j.out
#SBATCH --error=runs/slurm/setup-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
if [ -f /etc/profile.d/modules.sh ]; then
    source /etc/profile.d/modules.sh
    module load gcc/12.5
fi
export CUDA_VISIBLE_DEVICES="" CMAKE_BUILD_PARALLEL_LEVEL=4
python -m tools.bootstrap --mode "${LIVEACT_SETUP_MODE:-all}"
