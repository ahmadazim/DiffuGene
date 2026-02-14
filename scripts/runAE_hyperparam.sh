#!/bin/bash
#SBATCH -p hsph_gpu,gpu,gpu_requeue
#SBATCH --gres=gpu:1
#SBATCH --mem=5G
#SBATCH -t 02:00:00
#SBATCH -o /n/holystore01/LABS/xlin/Lab/ahmadazim/log_err/hyperparam_tune_AE_%j.out
#SBATCH -e /n/holystore01/LABS/xlin/Lab/ahmadazim/log_err/hyperparam_tune_AE_%j.err

conda init
eval "$(conda shell.bash hook)"
conda activate boltz1
module load cuda/12.4.1-fasrc01
module load cudnn/9.5.1.17_cuda12-fasrc01
module load R gcc cmake
nvidia-smi

# chrom 1:
# K1: 512, 1024
# K2: 32, 64
# C: 16, 32, 64
# chrom 22:
# K1: 256, 512
# K2: 16, 32
# C: 16, 32, 64

chrom=$1
K1=$2
K2=$3
C=$4
python -u /n/home03/ahmadazim/WORKING/genGen/DiffuGene/scripts/hyperparam_tune_AE.py --chrom ${chrom} --K1 ${K1} --K2 ${K2} --C ${C}

# chrom=1
# for K1 in {512,1024}; do
#     for K2 in {32,64}; do
#         for C in {16,32,64}; do
#             sbatch /n/home03/ahmadazim/WORKING/genGen/DiffuGene/scripts/runAE_hyperparam.sh ${chrom} ${K1} ${K2} ${C}
#         done
#     done
# done

# chrom=22
# for K1 in {256,512}; do
#     for K2 in {16,32}; do
#         for C in {16,32,64}; do
#             sbatch /n/home03/ahmadazim/WORKING/genGen/DiffuGene/scripts/runAE_hyperparam.sh ${chrom} ${K1} ${K2} ${C}
#         done
#     done
# done