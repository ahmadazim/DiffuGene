#!/bin/bash
#SBATCH -p gpu_test,gpu,gpu_requeue
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH -t 01:00:00
#SBATCH -o /n/holystore01/LABS/xlin/Lab/ahmadazim/log_err/diffugene_UKB_encode_%j.out
#SBATCH -e /n/holystore01/LABS/xlin/Lab/ahmadazim/log_err/diffugene_UKB_encode_%j.err

conda init
eval "$(conda shell.bash hook)"
conda activate boltz1
module load cuda/12.4.1-fasrc01
module load cudnn/9.5.1.17_cuda12-fasrc01
module load R gcc cmake
nvidia-smi

# # Set paths
# PLINK_DIR="/n/home03/ahmadazim/WORKING/genGen/UKB/genomic_data/geneticBinary/"
# BED_BASENAME="ukb_allchr_unrel_britishWhite"
# FAM_BASENAME="ukb_allchr_unrel_britishWhite_conditional_diffusion_train"
# CHROMOSOMES="all"

# # Set paths to pre-trained models
# diffdir="/n/home03/ahmadazim/WORKING/genGen/DiffuGene/"

# # PCA embeddings directory (contains loadings/, means/, metadata/ subdirs)
# PCA_EMBEDDINGS_DIR="${diffdir}../UKB/genomic_data/haploblocks_embeddings/"

# # Trained VAE model 
# VAE_MODEL="${diffdir}../UKB/models/joint_embed/VAE_unrelWhite_allchr_4PC_64z_checkpoint_400.pt"

# # Output path
# OUTPUT_DIR="${diffdir}../UKB/genomic_data/VAE_embeddings/"
# OUTPUT_FILE="${OUTPUT_DIR}${FAM_BASENAME}_VAE_latents_4PC_64z_checkpoint_200.pt"
# WORK_DIR="${diffdir}../UKB/genomic_data/work_encode/"

# # Model parameters (matching trained model)
# PCA_K=4
# GRID_H=64
# GRID_W=64
# POS_DIM=16
# LATENT_CHANNELS=128

# echo "Encoding genetic data: ${FAM_BASENAME}"
# echo "Chromosomes: ${CHROMOSOMES} (auto-discovery mode)"
# echo "Output: ${OUTPUT_FILE}"
# echo "Starting encoding at $(date)"

# # Create output directory
# mkdir -p ${OUTPUT_DIR}

# python ${diffdir}scripts/encode_genetic_data.py \
#     --global-bfile ${PLINK_DIR}/${BED_BASENAME} \
#     --keep-fam-file ${PLINK_DIR}/${FAM_BASENAME}.fam \
#     --chromosomes ${CHROMOSOMES} \
#     --pca-embeddings-dir ${PCA_EMBEDDINGS_DIR} \
#     --vae-model-path ${VAE_MODEL} \
#     --output-path ${OUTPUT_FILE} \
#     --work-dir ${WORK_DIR} \
#     --pca-k ${PCA_K} \
#     --grid-h ${GRID_H} \
#     --grid-w ${GRID_W} \
#     --pos-dim ${POS_DIM} \
#     --latent-channels ${LATENT_CHANNELS}

# echo "Script completed at $(date)"

# # Optional: Decode latents to get back to original space (if needed for validation)
# if [ "$1" = "--decode" ]; then
#     echo "Decoding latents back to SNP space..."
    
#     python ${diffdir}src/DiffuGene/joint_embed/decode_vae_latents.py \
#         --latents-file ${OUTPUT_FILE} \
#         --model-file ${VAE_MODEL} \
#         --embeddings-dir ${PCA_EMBEDDINGS_DIR} \
#         --spans-file ${WORK_DIR}encoding_${BASENAME}/${BASENAME}_blocks_4PC_inference.csv \
#         --output-file ${OUTPUT_DIR}${BASENAME}_all_chr_VAE_decoded_4PCscale.pt \
#         --batch-size 256
        
#     echo "Decoding completed at $(date)"
# fi

# echo "All operations completed successfully!"

wd=/n/home03/ahmadazim/WORKING/genGen/UKB/
cd ${wd}../DiffuGene/src/

python -m DiffuGene.joint_embed.infer \
    --model-path ${wd}models/joint_embed/VAE_unrelWhite_allchr_4PC_64z.pt \
    --spans-file ${wd}genomic_data/work_encode/encoding_ukb_allchr_unrel_britishWhite_conditional_diffusion_train/ukb_allchr_unrel_britishWhite_conditional_diffusion_train_blocks_4PC_inference.csv \
    --recoded-dir ${wd}genomic_data/work_encode/encoding_ukb_allchr_unrel_britishWhite_conditional_diffusion_train/recoded_blocks/ \
    --embeddings-dir ${wd}genomic_data/work_encode/encoding_ukb_allchr_unrel_britishWhite_conditional_diffusion_train/embeddings/ \
    --output-path ${wd}genomic_data/VAE_embeddings/ukb_allchr_unrel_britishWhite_conditional_diffusion_train_VAE_latents_4PC_64z.pt \
    --grid-h 64 \
    --grid-w 64 \
    --block-dim 4 \
    --pos-dim 16 \
    --latent-channels 128