#!/bin/bash
#SBATCH -p gpu,gpu_requeue
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH -t 2-00:00:00
#SBATCH -c 8
#SBATCH -o /n/holystore01/LABS/xlin/Lab/ahmadazim/log_err/diffugene_vqvae512_chr_%j.out
#SBATCH -e /n/holystore01/LABS/xlin/Lab/ahmadazim/log_err/diffugene_vqvae512_chr_%j.err

set -euo pipefail

conda init
eval "$(conda shell.bash hook)"
conda activate boltz1
module load cuda/12.4.1-fasrc01
module load cudnn/9.5.1.17_cuda12-fasrc01
module load R gcc cmake
nvidia-smi || true

export OMP_NUM_THREADS=8
export MKL_INTERFACE_LAYER=GNU,LP64
export MKL_THREADING_LAYER=GNU
export MKL_DYNAMIC=TRUE
export MKL_NUM_THREADS=8

# This wrapper expects to receive python -m DiffuGene.VAEembed.train_vqvae ... arguments
python -m DiffuGene.VAEembed.train_vqvae "$@"


