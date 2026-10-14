#!/bin/bash
#SBATCH --job-name=liveact-video
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=00:20:00
#SBATCH --output=runs/slurm/video-%j.out
#SBATCH --error=runs/slurm/video-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
if [ -f /etc/profile.d/modules.sh ]; then
    source /etc/profile.d/modules.sh
    module load gcc/12.5
fi
export PATH="$PWD/.hopper-env/bin:$PWD/.liveact-env/bin:$PATH"
.hopper-env/bin/python -m tools.run video --input "${LIVEACT_INPUT:-examples/input.json}" \
    --output "runs/video-${SLURM_JOB_ID:?}"
