#!/bin/bash
#SBATCH -p gpu,gpu_requeue
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH -t 3-00:00:00
#SBATCH -c 16
#SBATCH -o /n/holystore01/LABS/xlin/Lab/ahmadazim/log_err/diffugene_pipeline_UKB_%j.out
#SBATCH -e /n/holystore01/LABS/xlin/Lab/ahmadazim/log_err/diffugene_pipeline_UKB_%j.err

conda init
eval "$(conda shell.bash hook)"
conda activate boltz1
module load cuda/12.4.1-fasrc01
module load cudnn/9.5.1.17_cuda12-fasrc01
module load R gcc cmake
nvidia-smi

export OMP_NUM_THREADS=16
export MKL_INTERFACE_LAYER=GNU,LP64
export MKL_THREADING_LAYER=GNU
export MKL_DYNAMIC=TRUE
export MKL_NUM_THREADS=16

yaml_file=$1

difdir=/n/home03/ahmadazim/WORKING/genGen/DiffuGene
${difdir}/scripts/run_pipeline.sh --config ${yaml_file} 

# sbatch /n/home03/ahmadazim/WORKING/genGen/DiffuGene/scripts/slurm_wrapper.sh /n/home03/ahmadazim/WORKING/genGen/UKB/pipeline_ukb_allchr_unrel_britishWhite.yaml
# sbatch /n/home03/ahmadazim/WORKING/genGen/DiffuGene/scripts/slurm_wrapper.sh /n/home03/ahmadazim/WORKING/genGen/UKB/pipeline_ukb_allchr_unrel_britishWhite_condDiff.yaml