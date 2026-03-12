#!/bin/bash
#SBATCH -p gpu,gpu_requeue
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH -t 3-00:00:00
#SBATCH -c 2
#SBATCH -o /n/holystore01/LABS/xlin/Lab/ahmadazim/log_err/search_vqvae_%j.out
#SBATCH -e /n/holystore01/LABS/xlin/Lab/ahmadazim/log_err/search_vqvae_%j.err

conda init
eval "$(conda shell.bash hook)"
conda activate boltz1
module load cuda/12.4.1-fasrc01
module load cudnn/9.5.1.17_cuda12-fasrc01
module load R gcc cmake
nvidia-smi

export OMP_NUM_THREADS=2
export MKL_INTERFACE_LAYER=GNU,LP64
export MKL_THREADING_LAYER=GNU
export MKL_DYNAMIC=TRUE
export MKL_NUM_THREADS=2

set -euo pipefail

DATA_PATH="/n/holystore01/LABS/xlin/Lab/ahmadazim/genGen/VAEredesign/geneticBinary/ukb_allchr_unrel_britishWhite_genome_encoder_train_chr22_5000.raw"
RESULTS_CSV="/n/holystore01/LABS/xlin/Lab/ahmadazim/genGen/VAEredesign/results_vqvae_grid_hyperparam.csv"

srun python -u /n/home03/ahmadazim/WORKING/genGen/DiffuGene/src/DiffuGene/VAEembed/grid_search_vqvae.py \
  --data_path "$DATA_PATH" \
  --device cuda \
  --epochs 25 \
  --batch_size 32 \
  --results_csv "$RESULTS_CSV" \
  --pin_memory \
  --num_workers 2