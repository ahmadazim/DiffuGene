#!/bin/bash
#SBATCH -p gpu_requeue,gpu,hsph_gpu,gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH -t 02:00:00
#SBATCH -o /n/holystore01/LABS/xlin/Lab/ahmadazim/log_err/diffugene_ae_encode_%j.out
#SBATCH -e /n/holystore01/LABS/xlin/Lab/ahmadazim/log_err/diffugene_ae_encode_%j.err

conda init
eval "$(conda shell.bash hook)"
conda activate diffugene

CHROM=$1

home=/n/home03/ahmadazim/WORKING/genGen/
database=${home}UKBVQVAE/genomic_data/
modelbase=${home}UKBVQVAE/models/

python -u ${home}DiffuGene/src/DiffuGene/VAEembed/encode_batched_AE.py \
    --bfile ${database}geneticBinary/ukb_allchr_unrel_britishWhite_251K.bed \
    --bim ${database}geneticBinary/ukb_allchr_unrel_britishWhite_251K.bim \
    --fam ${database}geneticBinary/ukb_allchr_unrel_britishWhite_251K.fam \
    --chromosomes ${CHROM} \
    --batch-size 12000 \
    --models-dir ${modelbase}AE/ \
    --model-pattern ae_chr{chr}.pt \
    --latents-out-root ${database}AE_embeddings/ \
    --h5-out-root ${database}orig251K_h5_cache/ \
    --encode-batch-size 256
