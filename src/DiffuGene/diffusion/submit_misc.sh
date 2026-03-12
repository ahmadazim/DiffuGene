#!/bin/bash
#SBATCH -p hsph_gpu,gpu_h200,gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=300G
#SBATCH -t 1-00:00:00
#SBATCH -o /n/holystore01/LABS/xlin/Lab/ahmadazim/log_err/generate_invInvNoise_UKB_SiT_%j.out
#SBATCH -e /n/holystore01/LABS/xlin/Lab/ahmadazim/log_err/generate_invInvNoise_UKB_SiT_%j.err

set -eo pipefail

eval "$(conda shell.bash hook)"
conda activate diffugene

set -u

export OMP_NUM_THREADS=1
export MKL_INTERFACE_LAYER=GNU,LP64
export MKL_THREADING_LAYER=GNU
export MKL_DYNAMIC=TRUE
export MKL_NUM_THREADS=1

module load cuda/12.4.1-fasrc01
module load cudnn/9.5.1.17_cuda12-fasrc01
nvidia-smi


base=/n/home03/ahmadazim/WORKING/genGen/
model_dir=${base}UKBVQVAE/models/USiT_NC/

# python -u ${base}DiffuGene/src/DiffuGene/diffusion/generate_invInverseNoise.py \
#     --checkpoint-path ${base}UKBVQVAE/models/USiT_NC/USiT_M_unrelWhite_allchr_AE1d_NC.pth \
#     --input-dir ${base}UKBVQVAE/genomic_data/invNoise_embeddings/ \
#     --output-dir ${base}UKBVQVAE/genomic_data/invInvNoise_embeddings_M/ \
#     --steps 25 \
#     --model-batch-size 128 \
#     --samples-per-save 12000 \
#     --batch-indices 1 \
#     --hidden-dim 768

python -u ${base}DiffuGene/src/DiffuGene/diffusion/generate_invInverseNoise.py \
    --checkpoint-path ${base}UKBVQVAE/models/USiT_NC/USiT_H_unrelWhite_allchr_AE1d_NC.pth \
    --input-dir ${base}UKBVQVAE/genomic_data/invNoise_embeddings/ \
    --output-dir ${base}UKBVQVAE/genomic_data/invInvNoise_embeddings_H/ \
    --steps 25 \
    --model-batch-size 128 \
    --samples-per-save 12000 \
    --batch-indices 1 \
    --hidden-dim 1152