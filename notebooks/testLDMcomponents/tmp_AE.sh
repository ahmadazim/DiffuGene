#!/bin/bash
#SBATCH -p hsph_gpu,gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --mem=35G
#SBATCH -c 8
#SBATCH -t 1-00:00:00
#SBATCH -o /n/holystore01/LABS/xlin/Lab/ahmadazim/log_err/train_VAEmasked_regLatent_chr22_%j.out
#SBATCH -e /n/holystore01/LABS/xlin/Lab/ahmadazim/log_err/train_VAEmasked_regLatent_chr22_%j.err

conda init
eval "$(conda shell.bash hook)"
conda activate boltz1
module load cuda/12.4.1-fasrc01
module load cudnn/9.5.1.17_cuda12-fasrc01
module load R gcc cmake
nvidia-smi

# export OMP_NUM_THREADS=1
# export MKL_INTERFACE_LAYER=GNU,LP64
# export MKL_THREADING_LAYER=GNU
# export MKL_DYNAMIC=TRUE
# export MKL_NUM_THREADS=1

# name=$1 #bc32_beta1e-06_ld0.001
# chr=$1
# python -u /n/home03/ahmadazim/WORKING/genGen/DiffuGene/notebooks/FM_UNet_oneChrTest.py \
#     --ae-model-path /n/home03/ahmadazim/WORKING/genGen/UKBVQVAE/models/AEmasked/ae_chr${chr}.pt \
#     --h5-batch-path /n/home03/ahmadazim/WORKING/genGen/UKBVQVAE/genomic_data/vqvae_h5_cache/chr${chr}/batch00001.h5 \
#     --memmap-out /n/home03/ahmadazim/WORKING/genGen/UKBVQVAE/models/diffusion_test/FM_UNET/latents_chr${chr}_memmap.npy \
#     --model-out-path /n/home03/ahmadazim/WORKING/genGen/UKBVQVAE/models/diffusion_test/FM_UNET/FM_UNET_chr${chr}.pth \
#     --encode-batch-size 256 \
#     --train-batch-size 256 \
#     --epochs 50 \
#     --lr 1e-5 \
#     --max-batches 1

# bc=$1
# python -u /n/home03/ahmadazim/WORKING/genGen/DiffuGene/scripts/VAE_hyperparamtune.py --ld-lambda 0.001 --beta 1e-6 --bottleneck-channels $bc

# --spatial1d 512 \
# --spatial2d 16 \
chr=$1
spatial1d=$2
spatial2d=$3
python -u /n/home03/ahmadazim/WORKING/genGen/DiffuGene/src/DiffuGene/VAEembed/train.py \
    --h5-dir /n/home03/ahmadazim/WORKING/genGen/UKBVQVAE/genomic_data/vqvae_h5_cache/ \
    --chromosome ${chr} \
    --spatial1d ${spatial1d} \
    --spatial2d ${spatial2d} \
    --latent-channels 64 \
    --embed-dim 8 \
    --ld-lambda 1e-3 \
    --maf-lambda 1e-3 \
    --ld-window 128 \
    --beta-kl 0.1 \
    --tv-lambda 2e-2 \
    --robust-lambda 100 \
    --stable-lambda 3e-1 \
    --latent-noise-std 0.05 \
    --embed-noise-std 0.03 \
    --stage2-start-frac 0.7 \
    --latent-eval-max-batches 5 \
    --epochs 75 \
    --batch-size 256 \
    --lr 5e-3 \
    --weight-decay 0.1 \
    --device cuda \
    --grad-clip 5.0 \
    --val-h5-dir /n/home03/ahmadazim/WORKING/genGen/UKBVQVAE/genomic_data/val_h5_cache/ \
    --plateau-min-rel-improve 0.005 \
    --plateau-patience 50 \
    --plateau-mse-threshold 0.001 \
    --save-path /n/home03/ahmadazim/WORKING/genGen/UKBVQVAE/models/VAEmasked_regLat/vae_chr${chr}_veryhighbetaKL.pt

# for chr in {2..13}; do sbatch /n/home03/ahmadazim/WORKING/genGen/DiffuGene/notebooks/tmp_AE.sh ${chr} 1024 32; done
# sbatch /n/home03/ahmadazim/WORKING/genGen/DiffuGene/notebooks/tmp_AE.sh 14 512 16
# sbatch /n/home03/ahmadazim/WORKING/genGen/DiffuGene/notebooks/tmp_AE.sh 15 512 16
# sbatch /n/home03/ahmadazim/WORKING/genGen/DiffuGene/notebooks/tmp_AE.sh 16 1024 32
# for chr in {17..21}; do sbatch /n/home03/ahmadazim/WORKING/genGen/DiffuGene/notebooks/tmp_AE.sh ${chr} 512 16; done

# chr=$1
# python /n/home03/ahmadazim/WORKING/genGen/DiffuGene/notebooks/testLatent_AE.py \
#   --ae-model-path /n/home03/ahmadazim/WORKING/genGen/UKBVQVAE/models/AEmasked/ae_chr${chr}.pt \
#   --h5-batch-path /n/home03/ahmadazim/WORKING/genGen/UKBVQVAE/genomic_data/vqvae_h5_cache/chr${chr}/batch00001.h5 \
#   --max-batches 1 \
#   --encode-batch-size 256 \
#   --n-eval 1000 \
#   --latent-noise-std 0.05 \
#   --embed-noise-std 0.05 \
#   --output-metrics /n/home03/ahmadazim/WORKING/genGen/UKBVQVAE/models/diffusion_test/AEtest/chr${chr}_latent_penalties.npz

# python /n/home03/ahmadazim/WORKING/genGen/DiffuGene/src/DiffuGene/VAEembed/train_stage2.py \
#     --ae-checkpoints /n/home03/ahmadazim/WORKING/genGen/UKBVQVAE/models/AE/ \
#     --h5-dir /n/home03/ahmadazim/WORKING/genGen/UKBVQVAE/genomic_data/vqvae_h5_cache/ \
#     --chromosomes all \
#     --batch-size 256 \
#     --val-batch-size 256 \
#     --epochs 5 \
#     --lr 1e-4 \
#     --weight-decay 1e-2 \
#     --latent-loss-weight 0.05 \
#     --tv-lambda 2e-2 \
#     --robust-lambda 100.0 \
#     --stable-lambda 3e-1 \
#     --latent-noise-std 0.05 \
#     --embed-noise-std 0.03 \
#     --device cuda \
#     --val-h5-dir /n/home03/ahmadazim/WORKING/genGen/UKBVQVAE/genomic_data/val_h5_cache/ \
#     --num-workers 8