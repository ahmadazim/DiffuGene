#!/bin/bash

# Set paths
BASENAME="all_hm3_45k"
PLINK_DIR="/n/home03/ahmadazim/WORKING/pilotMME/genomic_data/indiv45k/"
CHR=22

# Set paths to pre-trained models
diffdir="/n/home03/ahmadazim/WORKING/genGen/DiffuGene/"
BLOCK_FILE="${diffdir}data/haploblocks/all_hm3_15k_chr22_blocks.blocks.det"
PCA_LOADINGS_DIR="${diffdir}data/haploblocks_embeddings_4PC/loadings/"
VAE_MODEL="${diffdir}models/joint_embed/vae_model_4PCscale.pt"

# Output path
OUTPUT_DIR="${diffdir}data/VAE_embeddings/"
OUTPUT_FILE="${OUTPUT_DIR}${BASENAME}_chr${CHR}_VAE_latents_4PCscale.pt"
WORK_DIR="${diffdir}data/work_encode/"

# Model parameters
PCA_K=4
GRID_H=16
GRID_W=16
BLOCK_DIM=4
POS_DIM=16
LATENT_CHANNELS=32

echo "Encoding genetic data: ${BASENAME}"
echo "Chromosome: ${CHR}"
echo "Output: ${OUTPUT_FILE}"
echo "Starting encoding at $(date)"

python ${diffdir}scripts/encode_genetic_data.py \
    --basename ${BASENAME} \
    --genetic-binary-folder ${PLINK_DIR} \
    --chromosome ${CHR} \
    --block-file ${BLOCK_FILE} \
    --pca-loadings-dir ${PCA_LOADINGS_DIR} \
    --vae-model-path ${VAE_MODEL} \
    --output-path ${OUTPUT_FILE} \
    --work-dir ${WORK_DIR} \
    --pca-k ${PCA_K} \
    --grid-h ${GRID_H} \
    --grid-w ${GRID_W} \
    --block-dim ${BLOCK_DIM} \
    --pos-dim ${POS_DIM} \
    --latent-channels ${LATENT_CHANNELS} 

echo "Script completed at $(date)"

# Optionally decode latents to get back to original space
python ${diffdir}src/DiffuGene/joint_embed/decode_vae_latents.py \
    --latents-file ${OUTPUT_FILE} \
    --model-file ${VAE_MODEL} \
    --embeddings-dir ${diffdir}data/haploblocks_embeddings_4PC/ \
    --spans-file ${WORK_DIR}encoding_${BASENAME}_chr${CHR}/${BASENAME}_chr${CHR}_blocks_4PC_inference.csv \
    --output-file ${OUTPUT_DIR}${BASENAME}_chr${CHR}_VAE_decoded_4PCscale.pt \
    --batch-size 256 \
    --chromosome ${CHR}