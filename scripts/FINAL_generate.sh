#!/bin/bash
#SBATCH -p hsph_gpu,gpu_h200,gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH -t 03:00:00
#SBATCH -o /n/holystore01/LABS/xlin/Lab/ahmadazim/log_err/FINAL_generate_%j.out
#SBATCH -e /n/holystore01/LABS/xlin/Lab/ahmadazim/log_err/FINAL_generate_%j.err

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


MODEL_PATH=/n/home03/ahmadazim/WORKING/genGen/UKBVQVAE/models/diffusion/DDPM_unrelWhite_allchr_AE128z_epoch22.pth
PREDICTION_TYPE=v_prediction
NUM_STEPS=50
ETA=0.0
GUIDANCE_SCALE=7.0
COVARIATE_FILE=/n/home03/ahmadazim/WORKING/genGen/UKBVQVAE/covariates/all_covariates.csv
base=/n/home03/ahmadazim/WORKING/genGen/
home=${base}/UKBVQVAE/genomic_data/

gen_batch_idx=$1
type=$2 # "indep" or "synObs"
num_samples=8192
batch_size=64
start_index=$(((gen_batch_idx - 1)*num_samples + 1))

## To generate independent samples (starting from random noise)
if [ "${type}" == "indep" ]; then
    # generate latents
    python -u ${base}/DiffuGene/src/DiffuGene/generate/generate_HC.py \
        --model-path ${MODEL_PATH} \
        --prediction-type ${PREDICTION_TYPE} \
        --num-steps ${NUM_STEPS} \
        --num-samples ${num_samples} \
        --batch-size ${batch_size} \
        --eta ${ETA} \
        --guidance-scale ${GUIDANCE_SCALE} \
        --save-path /n/home03/ahmadazim/WORKING/genGen/UKBVQVAE/genomic_data/generated_samples/generated_latents_unrelWhite_allchr_AE128z_${num_samples}_genBatch${gen_batch_idx}.pt \
        --cond \
        --covariate-file ${COVARIATE_FILE} \
        --fam-file /n/home03/ahmadazim/WORKING/genGen/UKBVQVAE/genomic_data/geneticBinary/ukb_allchr_unrel_britishWhite_conditional_diffusion_train.fam \
        --binary-cols SEX_MALE CAD HYPERTENSION T2D T1D STROKE CKD HYPERLIPIDEMIA LUNG_CANCER BREAST_CANCER PROSTATE_CANCER COLORECTAL_CANCER PANCREATIC_CANCER BIPOLAR MAJOR_DEPRESSION RA IBD AD_DIMENTIA PARKINSONS ATRIAL_FIBRILLATION CHOL_LOW_MEDS ASSESSMENT_CENTER_10003 ASSESSMENT_CENTER_11001 ASSESSMENT_CENTER_11002 ASSESSMENT_CENTER_11003 ASSESSMENT_CENTER_11004 ASSESSMENT_CENTER_11005 ASSESSMENT_CENTER_11006 ASSESSMENT_CENTER_11007 ASSESSMENT_CENTER_11008 ASSESSMENT_CENTER_11009 ASSESSMENT_CENTER_11010 ASSESSMENT_CENTER_11011 ASSESSMENT_CENTER_11012 ASSESSMENT_CENTER_11013 ASSESSMENT_CENTER_11014 ASSESSMENT_CENTER_11016 ASSESSMENT_CENTER_11017 ASSESSMENT_CENTER_11018 ASSESSMENT_CENTER_11020 ASSESSMENT_CENTER_11021 ASSESSMENT_CENTER_11022 ASSESSMENT_CENTER_11023 \
        --categorical-cols SMOKING_STATUS ALCOHOL_INTAKE \
        --cov-start-index ${start_index}
    
    # decode latents to calls
    batch_size=128
    python -u ${base}/DiffuGene/src/DiffuGene/generate/decode.py \
        --latents-path ${home}/generated_samples/generated_latents_unrelWhite_allchr_AE128z_${num_samples}_genBatch${gen_batch_idx}.pt \
        --layout-json ${home}/../models/AE/ae_milp_layout.json \
        --tile-stats ${home}/AE_embeddings/unified/tile_norm_stats.pt \
        --models-dir ${home}/../models/AE \
        --model-pattern "ae_chr{chr}_homog.pt" \
        --batch-decode ${batch_size} \
        --return-logits \
        --output-dir ${home}/generated_samples/generated_decoded_unrelWhite_allchr_AE128z_${num_samples}_genBatch${gen_batch_idx}/
fi
# for gen_batch_idx in {1..10}; do sbatch /n/home03/ahmadazim/WORKING/genGen/DiffuGene/scripts/FINAL_generate.sh ${gen_batch_idx} "indep"; done
# for gen_batch_idx in {11..20}; do sbatch /n/home03/ahmadazim/WORKING/genGen/DiffuGene/scripts/FINAL_generate.sh ${gen_batch_idx} "indep"; done


## To generate synObs samples (starting from noised latents)
if [ "${type}" == "synObs" ]; then
    python -u /n/home03/ahmadazim/WORKING/genGen/DiffuGene/src/DiffuGene/generate/generate_synObs.py \
        --model-path ${MODEL_PATH} \
        --train-embed-dataset-path /n/home03/ahmadazim/WORKING/genGen/UKBVQVAE/genomic_data/AE_embeddings/unified/ukb_allchr_unrel_britishWhite_unrelWhite_allchr_AE128z_memmap.npy \
        --prediction-type ${PREDICTION_TYPE} \
        --num-steps ${NUM_STEPS} \
        --num-samples ${num_samples} \
        --batch-size ${batch_size} \
        --eta ${ETA} \
        --guidance-scale ${GUIDANCE_SCALE} \
        --save-path /n/home03/ahmadazim/WORKING/genGen/UKBVQVAE/genomic_data/generated_samples/synObs_latents_unrelWhite_allchr_AE128z_${num_samples}_genBatch${gen_batch_idx}.pt \
        --cond \
        --covariate-file ${COVARIATE_FILE} \
        --fam-file /n/home03/ahmadazim/WORKING/genGen/UKBVQVAE/genomic_data/geneticBinary/ukb_allchr_unrel_britishWhite_conditional_diffusion_train.fam \
        --binary-cols SEX_MALE CAD HYPERTENSION T2D T1D STROKE CKD HYPERLIPIDEMIA LUNG_CANCER BREAST_CANCER PROSTATE_CANCER COLORECTAL_CANCER PANCREATIC_CANCER BIPOLAR MAJOR_DEPRESSION RA IBD AD_DIMENTIA PARKINSONS ATRIAL_FIBRILLATION CHOL_LOW_MEDS ASSESSMENT_CENTER_10003 ASSESSMENT_CENTER_11001 ASSESSMENT_CENTER_11002 ASSESSMENT_CENTER_11003 ASSESSMENT_CENTER_11004 ASSESSMENT_CENTER_11005 ASSESSMENT_CENTER_11006 ASSESSMENT_CENTER_11007 ASSESSMENT_CENTER_11008 ASSESSMENT_CENTER_11009 ASSESSMENT_CENTER_11010 ASSESSMENT_CENTER_11011 ASSESSMENT_CENTER_11012 ASSESSMENT_CENTER_11013 ASSESSMENT_CENTER_11014 ASSESSMENT_CENTER_11016 ASSESSMENT_CENTER_11017 ASSESSMENT_CENTER_11018 ASSESSMENT_CENTER_11020 ASSESSMENT_CENTER_11021 ASSESSMENT_CENTER_11022 ASSESSMENT_CENTER_11023 \
        --categorical-cols SMOKING_STATUS ALCOHOL_INTAKE \
        --data-start-index ${start_index}
    
    # decode latents to calls
    batch_size=128
    python -u ${base}/DiffuGene/src/DiffuGene/generate/decode.py \
        --latents-path ${home}/generated_samples/synObs_latents_unrelWhite_allchr_AE128z_${num_samples}_genBatch${gen_batch_idx}.pt \
        --layout-json ${home}/../models/AE/ae_milp_layout.json \
        --tile-stats ${home}/AE_embeddings/unified/tile_norm_stats.pt \
        --models-dir ${home}/../models/AE \
        --model-pattern "ae_chr{chr}_homog.pt" \
        --batch-decode ${batch_size} \
        --return-logits \
        --output-dir ${home}/generated_samples/synObs_decoded_unrelWhite_allchr_AE128z_${num_samples}_genBatch${gen_batch_idx}/
fi
# for gen_batch_idx in {1..10}; do sbatch /n/home03/ahmadazim/WORKING/genGen/DiffuGene/scripts/FINAL_generate.sh ${gen_batch_idx} "synObs"; done
# for gen_batch_idx in {11..20}; do sbatch /n/home03/ahmadazim/WORKING/genGen/DiffuGene/scripts/FINAL_generate.sh ${gen_batch_idx} "synObs"; done

