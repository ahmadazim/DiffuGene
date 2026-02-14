#!/bin/bash
#SBATCH -p hsph_gpu,gpu_h200,gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=450G
#SBATCH -t 1-00:00:00
#SBATCH -o /n/holystore01/LABS/xlin/Lab/ahmadazim/log_err/encode_batched_AE_%j.out
#SBATCH -e /n/holystore01/LABS/xlin/Lab/ahmadazim/log_err/encode_batched_AE_%j.err

conda init
eval "$(conda shell.bash hook)"
conda activate boltz1
module load cuda/12.4.1-fasrc01
module load cudnn/9.5.1.17_cuda12-fasrc01
module load R gcc cmake
nvidia-smi

srun bash -c '
export RANK=$SLURM_PROCID
export WORLD_SIZE=$SLURM_NTASKS
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
python -u /n/home03/ahmadazim/WORKING/genGen/DiffuGene/src/DiffuGene/VAEembed/train_stage2.py \
  --ae-checkpoints /n/home03/ahmadazim/WORKING/genGen/UKBVQVAE/models/AE/ \
  --h5-dir /n/home03/ahmadazim/WORKING/genGen/UKBVQVAE/genomic_data/vqvae_h5_cache/ \
  --val-h5-dir /n/home03/ahmadazim/WORKING/genGen/UKBVQVAE/genomic_data/val_h5_cache/ \
  --chromosomes all \
  --batch-size 256 \
  --val-batch-size 256 \
  --epochs 5 \
  --lr 1e-4 \
  --weight-decay 1e-2 \
  --latent-loss-weight 0.05 \
  --tv-lambda 2e-2 \
  --robust-lambda 100.0 \
  --stable-lambda 3e-1 \
  --latent-noise-std 0.05 \
  --embed-noise-std 0.03 \
  --device cuda \
  --num-workers 8
'

# DATA_ROOT=/n/home03/ahmadazim/WORKING/genGen/UKBVQVAE/genomic_data
# python /n/home03/ahmadazim/WORKING/genGen/DiffuGene/src/DiffuGene/VAEembed/encode_batched_AE.py \
#   --bfile ${DATA_ROOT}/geneticBinary/ukb_allchr_unrel_britishWhite \
#   --bim ${DATA_ROOT}/geneticBinary/ukb_allchr_unrel_britishWhite.bim \
#   --fam ${DATA_ROOT}/geneticBinary/ukb_allchr_unrel_britishWhite_conditional_diffusion_train.fam \
#   --chromosomes all \
#   --batch-size 12000 \
#   --h5-out-root ${DATA_ROOT}/ae_h5 \
#   --models-dir ${DATA_ROOT}/../models/AE \
#   --model-pattern ae_chr{chr}_homog.pt \
#   --latents-out-root ${DATA_ROOT}/AE_embeddings \
#   --layout-json ${DATA_ROOT}/../models/AE/ae_milp_layout.json \
#   --unified-out-root ${DATA_ROOT}/AE_embeddings/unified \
#   --device cuda 