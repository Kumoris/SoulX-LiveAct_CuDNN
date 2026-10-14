#!/bin/bash
#SBATCH --job-name=liveact-resident
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=02:00:00
#SBATCH --output=runs/slurm/resident-%j.out
#SBATCH --error=runs/slurm/resident-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
if [ -f /etc/profile.d/modules.sh ]; then
    source /etc/profile.d/modules.sh
    module load gcc/12.5
fi
export PATH="$PWD/.hopper-env/bin:$PWD/.liveact-env/bin:$PATH"
.hopper-env/bin/python -m tools.run resident --input "${LIVEACT_INPUT:-examples/input.json}" \
    --output "runs/resident-${SLURM_JOB_ID:?}"
