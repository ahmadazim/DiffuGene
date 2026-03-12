#!/bin/bash
#SBATCH -p hsph_gpu,gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH -c 8
#SBATCH -t 1-00:00:00
#SBATCH -o /n/holystore01/LABS/xlin/Lab/ahmadazim/log_err/train_UNET_regLat_AR_chr22_%j.out
#SBATCH -e /n/holystore01/LABS/xlin/Lab/ahmadazim/log_err/train_UNET_regLat_AR_chr22_%j.err

conda init
eval "$(conda shell.bash hook)"
conda activate boltz1
module load cuda/12.4.1-fasrc01
module load cudnn/9.5.1.17_cuda12-fasrc01
module load R gcc cmake
nvidia-smi

export OMP_NUM_THREADS=1
export MKL_INTERFACE_LAYER=GNU,LP64
export MKL_THREADING_LAYER=GNU
export MKL_DYNAMIC=TRUE
export MKL_NUM_THREADS=1

# name=$1 #bc32_beta1e-06_ld0.001
chr=$1
python -u /n/home03/ahmadazim/WORKING/genGen/DiffuGene/notebooks/UNet_oneChrTest_vpred.py \
    --ae-model-path /n/home03/ahmadazim/WORKING/genGen/UKBVQVAE/models/VAEmasked_regLat/vae_chr${chr}_veryhighbetaKL.pt \
    --h5-batch-path /n/home03/ahmadazim/WORKING/genGen/UKBVQVAE/genomic_data/vqvae_h5_cache/chr${chr}/batch00001.h5 \
    --memmap-out /n/home03/ahmadazim/WORKING/genGen/UKBVQVAE/models/diffusion_test/UNET_regLat/latents_chr${chr}_memmap_regLat_veryhighbetaKL_vpred.npy \
    --model-out-path /n/home03/ahmadazim/WORKING/genGen/UKBVQVAE/models/diffusion_test/UNET_regLat/UNET_chr${chr}_regLat_veryhighbetaKL_vpred.pth \
    --encode-batch-size 256 \
    --train-batch-size 256 \
    --epochs 250 \
    --lr 1e-5 \
    --max-batches 8 \
    --use-channel-norm

# bc=$1
# python -u /n/home03/ahmadazim/WORKING/genGen/DiffuGene/scripts/VAE_hyperparamtune.py --ld-lambda 0.001 --beta 1e-6 --bottleneck-channels $bc

# python -u /n/home03/ahmadazim/WORKING/genGen/DiffuGene/src/DiffuGene/VAEembed/train.py \
#     --h5-dir /n/home03/ahmadazim/WORKING/genGen/UKBVQVAE/genomic_data/vqvae_h5_cache/ \
#     --chromosome 22 \
#     --spatial1d 512 \
#     --spatial2d 16 \
#     --latent-channels 64 \
#     --embed-dim 8 \
#     --ld-lambda 1e-3 \
#     --maf-lambda 1e-3 \
#     --ld-window 128 \
#     --epochs 50 \
#     --batch-size 256 \
#     --lr 5e-3 \
#     --weight-decay 0.1 \
#     --device cuda \
#     --grad-clip 5.0 \
#     --val-h5-dir /n/home03/ahmadazim/WORKING/genGen/UKBVQVAE/genomic_data/val_h5_cache/ \
#     --plateau-min-rel-improve 0.005 \
#     --plateau-patience 50 \
#     --plateau-mse-threshold 0.001 \
#     --save-path /n/home03/ahmadazim/WORKING/genGen/UKBVQVAE/models/VAEmasked/vae_chr22.pt