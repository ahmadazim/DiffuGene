#!/bin/bash
#SBATCH -p hsph_gpu,gpu_h200,gpu
#SBATCH --gres=gpu:4
#SBATCH --mem=100G
#SBATCH -t 3-00:00:00
#SBATCH -o /n/holystore01/LABS/xlin/Lab/ahmadazim/log_err/train_diffusion_USiT_%j.out
#SBATCH -e /n/holystore01/LABS/xlin/Lab/ahmadazim/log_err/train_diffusion_USiT_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20

set -eo pipefail

eval "$(conda shell.bash hook)"
conda activate diffugene

set -u

export OMP_NUM_THREADS=1
export MKL_INTERFACE_LAYER=GNU,LP64
export MKL_THREADING_LAYER=GNU
export MKL_DYNAMIC=TRUE
export MKL_NUM_THREADS=1

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

module load cuda/12.4.1-fasrc01
module load cudnn/9.5.1.17_cuda12-fasrc01
nvidia-smi

home=/n/home03/ahmadazim/WORKING/genGen/
base=${home}/UKBVQVAE/
# model_dir=${base}/models/USiT/
model_dir=${base}/models/USiT_NC/
src_root=${home}/DiffuGene/src

mkdir -p "${model_dir}"

# main params
# num_layers=29
# num_heads=16
# hidden_dim=1152

## for invNoise training:
num_layers=29
num_heads=16
hidden_dim=1152

# Unconditional USiT training on per-chromosome AE latents (chr*/batch*_latents.pt).
cd "${src_root}"
torchrun --standalone --nproc_per_node=4 -m DiffuGene.diffusion.train \
    --model-output-path ${model_dir}USiT_unrelWhite_allchr_AE1d_NC.pth \
    --train-embed-dataset-path ${base}/genomic_data/invNoise_embeddings/ \
    --model-type usit \
    --sit-use-udit \
    --sit-dropout 0.0 \
    --sit-mlp-ratio 4 \
    --batch-size 16 \
    --num-epochs 10 \
    --num-time-steps 1000 \
    --lr 2e-5 \
    --flow-norm-eps 1e-6 \
    --ema-decay 0.999 \
    --sit-num-layers ${num_layers} \
    --sit-num-heads ${num_heads} \
    --sit-qkv-bias \
    --sit-hidden-dim ${hidden_dim} \
    --flow-disable-latent-normalization 
    
    
    # --checkpoint-path ${model_dir}USiT_unrelWhite_allchr_AE1d_NC_epoch10.pth

    # --lr 1e-4 \
    # --model-output-path ${model_dir}USiT_unrelWhite_allchr_AE1d.pth \
    # --train-embed-dataset-path ${base}/genomic_data/AE_embeddings/ \