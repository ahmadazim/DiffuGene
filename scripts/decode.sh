#!/bin/bash
#SBATCH -p hsph_gpu,gpu_h200,gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=250G
#SBATCH -t 01:00:00
#SBATCH -o /n/holystore01/LABS/xlin/Lab/ahmadazim/log_err/decode_AE128z_%j.out
#SBATCH -e /n/holystore01/LABS/xlin/Lab/ahmadazim/log_err/decode_AE128z_%j.err

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
batch_size=128

base=/n/home03/ahmadazim/WORKING/genGen/
home=${base}/UKBVQVAE/genomic_data/
python -u ${base}/DiffuGene/src/DiffuGene/generate/decode.py \
    --latents-path ${home}/generated_samples/generated_latents_unrelWhite_allchr_AE128z_${num_samples}_genBatch${gen_batch_idx}.pt \
    --layout-json ${home}/../models/AE/ae_milp_layout.json \
    --tile-stats ${home}/AE_embeddings/unified/tile_norm_stats.pt \
    --models-dir ${home}/../models/AE \
    --model-pattern "ae_chr{chr}_homog.pt" \
    --batch-decode ${batch_size} \
    --return-logits \
    --output-dir ${home}/generated_samples/generated_decoded_unrelWhite_allchr_AE128z_${num_samples}_genBatch${gen_batch_idx}/

# for gen_batch_idx in {1..10}; do sbatch /n/home03/ahmadazim/WORKING/genGen/DiffuGene/scripts/decode.sh ${gen_batch_idx}; done


base=/n/home03/ahmadazim/WORKING/genGen/
home=${base}/UKBVQVAE/genomic_data/
python -u ${base}/DiffuGene/src/DiffuGene/generate/decode.py \
    --latents-path ${home}/generated_samples/synObs_latents_unrelWhite_allchr_AE128z_512_genBatch1.pt \
    --layout-json ${home}/../models/AE/ae_milp_layout.json \
    --tile-stats ${home}/AE_embeddings/unified/tile_norm_stats.pt \
    --models-dir ${home}/../models/AE \
    --model-pattern "ae_chr{chr}_homog.pt" \
    --batch-decode 64 \
    --return-logits \
    --output-dir ${home}/generated_samples/synObs_decoded_unrelWhite_allchr_AE128z_512_genBatch1/
