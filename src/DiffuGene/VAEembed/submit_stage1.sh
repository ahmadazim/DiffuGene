#!/bin/bash
#SBATCH -p test
#SBATCH --mem=100G
#SBATCH -t 02:00:00
#SBATCH -o /n/holystore01/LABS/xlin/Lab/ahmadazim/log_err/diffugene_ae_tok_stage1submit_%j.out
#SBATCH -e /n/holystore01/LABS/xlin/Lab/ahmadazim/log_err/diffugene_ae_tok_stage1submit_%j.err

conda init
eval "$(conda shell.bash hook)"
conda activate diffugene

home=/n/home03/ahmadazim/WORKING/genGen/
database=${home}UKBVQVAE/genomic_data/
modelbase=${home}UKBVQVAE/models/

python -u ${home}DiffuGene/src/DiffuGene/VAEembed/train_orchestrator_tok.py \
    --h5-dir ${database}vqvae_h5_cache/ \
    --val-h5-dir ${database}val_h5_cache/ \
    --bim ${database}geneticBinary/ukb_allchr_unrel_britishWhite.bim \
    --output-dir ${modelbase}AE/ \
    --chromosomes all \
    --total-tokens 4096 \
    --latent-dim 256 \
    --min-tokens 64 \
    --max-tokens 1024 \
    --embed-dim 8 \
    --max-c 5 \
    --dropout 0.0 \
    --epochs 50 \
    --batch-size 128 \
    --lr 2e-4 \
    --weight-decay 0.0 \
    --device cuda \
    --grad-clip 1.0 \
    --use-slurm \
    --slurm-script ${home}DiffuGene/scripts/slurm_train_ae_single.sh 