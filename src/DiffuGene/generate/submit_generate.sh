#!/bin/bash
#SBATCH -p hsph_gpu,gpu_h200,gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=300G
#SBATCH -t 1-00:00:00
#SBATCH -o /n/holystore01/LABS/xlin/Lab/ahmadazim/log_err/generate_invInvNoise_UKB_SiT_%j.out
#SBATCH -e /n/holystore01/LABS/xlin/Lab/ahmadazim/log_err/generate_invInvNoise_UKB_SiT_%j.err

set -eo pipefail

eval "$(conda shell.bash hook)"
conda activate diffugene

set -u

export OMP_NUM_THREADS=1
export MKL_INTERFACE_LAYER=GNU,LP64
export MKL_THREADING_LAYER=GNU
export MKL_DYNAMIC=TRUE
export MKL_NUM_THREADS=1

module load cuda/12.4.1-fasrc01
module load cudnn/9.5.1.17_cuda12-fasrc01
nvidia-smi


base=/n/home03/ahmadazim/WORKING/genGen/
model_dir=${base}UKBVQVAE/models/
main_model_dir=${model_dir}/USiT/
inverse_model_dir=${model_dir}/USiT_NC/

batch=$1
num_samples=16384

python -u ${base}DiffuGene/src/DiffuGene/generate/generate.py \
    --main-checkpoint-path ${main_model_dir}/USiT_unrelWhite_allchr_AE1d_epoch12.pth \
    --inverse-checkpoint-path ${inverse_model_dir}/USiT_H_unrelWhite_allchr_AE1d_NC.pth \
    --memmap-dir ${main_model_dir} \
    --output-dir ${base}UKBVQVAE/genomic_data/generated_samples/generated_USiT_H_wNC_H/ \
    --output-prefix generated_decoded_USiT_H_wNC_H_unrelWhite_allchr_${batch} \
    --num-samples ${num_samples} \
    --batch-size 128 \
    --inverse-steps 25 \
    --main-steps 25 \
    --inverse-solver "dpm" \
    --main-solver "dpm" \
    --ae-models-dir ${model_dir}/AE/ \
    --inverse-hidden-dim 1152 \
    --main-hidden-dim 1152 \
    --seed ${batch}${num_samples}