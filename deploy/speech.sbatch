#!/bin/bash
# CPU inference; one GPU is the HPC3 ACD partition's minimum allocation.
#SBATCH --job-name=liveact-speech
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=runs/slurm/speech-%j.out
#SBATCH --error=runs/slurm/speech-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
if [ -f /etc/profile.d/modules.sh ]; then
    source /etc/profile.d/modules.sh
    module load gcc/12.5
fi
export CUDA_VISIBLE_DEVICES="" PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8 LIVEACT_TTS_VOICE="${LIVEACT_TTS_VOICE:-zf_003}"
.dialogue-env/bin/python -m tools.speech --host 0.0.0.0
