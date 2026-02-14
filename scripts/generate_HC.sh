#!/bin/bash
#SBATCH -p hsph_gpu,gpu_h200,gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH -t 03:00:00
#SBATCH -o /n/holystore01/LABS/xlin/Lab/ahmadazim/log_err/generate_latents_HC_%j.out
#SBATCH -e /n/holystore01/LABS/xlin/Lab/ahmadazim/log_err/generate_latents_HC_%j.err

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

gen_batch_idx=$1
num_samples=8192
batch_size=64
start_index=$(((gen_batch_idx - 1)*num_samples + 1))

python -u /n/home03/ahmadazim/WORKING/genGen/DiffuGene/src/DiffuGene/generate/generate_HC.py \
  --model-path /n/home03/ahmadazim/WORKING/genGen/UKBVQVAE/models/diffusion/DDPM_unrelWhite_allchr_AE128z_epoch22.pth \
  --prediction-type v_prediction \
  --num-steps 50 \
  --num-samples ${num_samples} \
  --batch-size ${batch_size} \
  --eta 0.0 \
  --guidance-scale 7.0 \
  --save-path /n/home03/ahmadazim/WORKING/genGen/UKBVQVAE/genomic_data/generated_samples/generated_latents_unrelWhite_allchr_AE128z_${num_samples}_genBatch${gen_batch_idx}.pt \
  --cond \
  --covariate-file /n/home03/ahmadazim/WORKING/genGen/UKBVQVAE/covariates/all_covariates.csv \
  --fam-file /n/home03/ahmadazim/WORKING/genGen/UKBVQVAE/genomic_data/geneticBinary/ukb_allchr_unrel_britishWhite_conditional_diffusion_train.fam \
  --binary-cols SEX_MALE CAD HYPERTENSION T2D T1D STROKE CKD HYPERLIPIDEMIA LUNG_CANCER BREAST_CANCER PROSTATE_CANCER COLORECTAL_CANCER PANCREATIC_CANCER BIPOLAR MAJOR_DEPRESSION RA IBD AD_DIMENTIA PARKINSONS ATRIAL_FIBRILLATION CHOL_LOW_MEDS ASSESSMENT_CENTER_10003 ASSESSMENT_CENTER_11001 ASSESSMENT_CENTER_11002 ASSESSMENT_CENTER_11003 ASSESSMENT_CENTER_11004 ASSESSMENT_CENTER_11005 ASSESSMENT_CENTER_11006 ASSESSMENT_CENTER_11007 ASSESSMENT_CENTER_11008 ASSESSMENT_CENTER_11009 ASSESSMENT_CENTER_11010 ASSESSMENT_CENTER_11011 ASSESSMENT_CENTER_11012 ASSESSMENT_CENTER_11013 ASSESSMENT_CENTER_11014 ASSESSMENT_CENTER_11016 ASSESSMENT_CENTER_11017 ASSESSMENT_CENTER_11018 ASSESSMENT_CENTER_11020 ASSESSMENT_CENTER_11021 ASSESSMENT_CENTER_11022 ASSESSMENT_CENTER_11023 \
  --categorical-cols SMOKING_STATUS ALCOHOL_INTAKE \
  --cov-start-index ${start_index}

# for gen_batch_idx in {1..10}; do sbatch /n/home03/ahmadazim/WORKING/genGen/DiffuGene/scripts/generate_HC.sh ${gen_batch_idx}; done

# # DDPM:
# python -u /n/home03/ahmadazim/WORKING/genGen/DiffuGene/scripts/generate_HC_DDPM.py \
#   --model-path /n/home03/ahmadazim/WORKING/genGen/UKBVQVAE/models/diffusion/DDPM_unrelWhite_allchr_128z.pth \
#   --prediction-type epsilon \
#   --num-steps 1000 \
#   --num-samples ${num_samples} \
#   --batch-size ${batch_size} \
#   --eta 0.0 \
#   --guidance-scale 7.0 \
#   --save-path /n/home03/ahmadazim/WORKING/genGen/UKBVQVAE/genomic_data/generated_samples/DDPM_generated_latents_unrelWhite_allchr_128z_${num_samples}_genBatch${gen_batch_idx}.pt \
#   --cond \
#   --covariate-file /n/home03/ahmadazim/WORKING/genGen/UKBVQVAE/covariates/all_covariates.csv \
#   --fam-file /n/home03/ahmadazim/WORKING/genGen/UKBVQVAE/genomic_data/geneticBinary/ukb_allchr_unrel_britishWhite_conditional_diffusion_train.fam \
#   --binary-cols SEX_MALE CAD HYPERTENSION T2D T1D STROKE CKD HYPERLIPIDEMIA LUNG_CANCER BREAST_CANCER PROSTATE_CANCER COLORECTAL_CANCER PANCREATIC_CANCER BIPOLAR MAJOR_DEPRESSION RA IBD AD_DIMENTIA PARKINSONS ATRIAL_FIBRILLATION CHOL_LOW_MEDS ASSESSMENT_CENTER_10003 ASSESSMENT_CENTER_11001 ASSESSMENT_CENTER_11002 ASSESSMENT_CENTER_11003 ASSESSMENT_CENTER_11004 ASSESSMENT_CENTER_11005 ASSESSMENT_CENTER_11006 ASSESSMENT_CENTER_11007 ASSESSMENT_CENTER_11008 ASSESSMENT_CENTER_11009 ASSESSMENT_CENTER_11010 ASSESSMENT_CENTER_11011 ASSESSMENT_CENTER_11012 ASSESSMENT_CENTER_11013 ASSESSMENT_CENTER_11014 ASSESSMENT_CENTER_11016 ASSESSMENT_CENTER_11017 ASSESSMENT_CENTER_11018 ASSESSMENT_CENTER_11020 ASSESSMENT_CENTER_11021 ASSESSMENT_CENTER_11022 ASSESSMENT_CENTER_11023 \
#   --categorical-cols SMOKING_STATUS ALCOHOL_INTAKE \
#   --cov-start-index ${start_index}

# # for gen_batch_idx in {1..10}; do sbatch /n/home03/ahmadazim/WORKING/genGen/DiffuGene/scripts/generate_HC.sh ${gen_batch_idx}; done