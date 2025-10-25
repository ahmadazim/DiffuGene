#!/bin/bash
#SBATCH -p hsph_gpu,gpu
#SBATCH --gres=gpu:4
#SBATCH --mem=1000G
#SBATCH -t 3-00:00:00
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH -o /n/holystore01/LABS/xlin/Lab/ahmadazim/log_err/diffugene_pipeline_UKB_condDiff_regVQVAE_%j.out
#SBATCH -e /n/holystore01/LABS/xlin/Lab/ahmadazim/log_err/diffugene_pipeline_UKB_condDiff_regVQVAE_%j.err

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

yaml_file=$1

difdir=/n/home03/ahmadazim/WORKING/genGen/DiffuGene
${difdir}/scripts/run_pipeline.sh --config ${yaml_file} 

# sbatch /n/home03/ahmadazim/WORKING/genGen/DiffuGene/scripts/slurm_wrapper.sh /n/home03/ahmadazim/WORKING/genGen/DiffuGene/src/DiffuGene/config/pipeline.yaml