#!/bin/bash
#SBATCH -p gpu_requeue
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH -t 1-00:00:00
#SBATCH -o /n/holystore01/LABS/xlin/Lab/ahmadazim/log_err/diffugene_pipeline_%j.out
#SBATCH -e /n/holystore01/LABS/xlin/Lab/ahmadazim/log_err/diffugene_pipeline_%j.err

conda init
eval "$(conda shell.bash hook)"
conda activate boltz1
module load cuda/12.4.1-fasrc01
module load cudnn/9.5.1.17_cuda12-fasrc01
module load R gcc cmake
nvidia-smi
unset CUDA_VISIBLE_DEVICES

yaml_filename=$1

difdir=/n/home03/ahmadazim/WORKING/genGen/DiffuGene
${difdir}/scripts/run_pipeline.sh --config ${difdir}/src/DiffuGene/config/${yaml_filename} 

# for pc in 3 4 5; do sbatch /n/home03/ahmadazim/WORKING/genGen/DiffuGene/scripts/slurm_wrapper.sh pipeline_${pc}PC.yaml; done