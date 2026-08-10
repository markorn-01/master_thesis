#!/bin/bash
#SBATCH --job-name=wind256
#SBATCH --partition=gpu-single
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1,gpumem_per_gpu:70GB
#SBATCH --constraint=fp64
#SBATCH --mem=96gb
#SBATCH --time=24:00:00
#SBATCH --export=NONE
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

module purge
module load devel/python
module load devel/cuda

if [[ ! -x .venv-helix/bin/python ]]; then
    echo "Missing .venv-helix. Follow cluster/HELIX.md before submitting."
    exit 1
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export JAX_PLATFORMS=cuda
export JAX_ENABLE_X64=0
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"

srun .venv-helix/bin/python experiments/wind_bubble/check_gpu_environment.py
srun .venv-helix/bin/python \
    experiments/wind_bubble/run_wind_bubble_convergence.py \
    --resolutions 256 \
    --t-end 0.20 \
    --checkpoint-segments 4 \
    --resume \
    --require-gpu \
    --output-dir outputs/wind_bubble_convergence
