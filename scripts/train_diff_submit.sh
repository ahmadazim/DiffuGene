#!/bin/bash
#SBATCH -p hsph_gpu,gpu_h200,gpu,gpu_requeue
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH -t 1-00:00:00
#SBATCH -o /n/holystore01/LABS/xlin/Lab/ahmadazim/log_err/train_1dAE_%j.out
#SBATCH -e /n/holystore01/LABS/xlin/Lab/ahmadazim/log_err/train_1dAE_%j.err

conda init
eval "$(conda shell.bash hook)"
conda activate diffugene
module load cuda/12.4.1-fasrc01
module load cudnn/9.5.1.17_cuda12-fasrc01
nvidia-smi

export OMP_NUM_THREADS=1
export MKL_INTERFACE_LAYER=GNU,LP64
export MKL_THREADING_LAYER=GNU
export MKL_DYNAMIC=TRUE
export MKL_NUM_THREADS=1

# python -u /n/home03/ahmadazim/WORKING/genGen/DiffuGene/scripts/train_1dAE.py
if [ $1 == "unet" ]; then
    python -u /n/home03/ahmadazim/WORKING/genGen/DiffuGene/scripts/train_diff.py --model_type unet
elif [ $1 == "dit" ]; then
    python -u /n/home03/ahmadazim/WORKING/genGen/DiffuGene/scripts/train_diff.py --model_type dit
elif [ $1 == "udit" ]; then
    python -u /n/home03/ahmadazim/WORKING/genGen/DiffuGene/scripts/train_diff.py --model_type udit
elif [ $1 == "sit" ]; then
    python -u /n/home03/ahmadazim/WORKING/genGen/DiffuGene/scripts/train_diff.py --model_type sit
elif [ $1 == "usit" ]; then
    python -u /n/home03/ahmadazim/WORKING/genGen/DiffuGene/scripts/train_diff.py --model_type usit
elif [ $1 == "normalize_usit" ]; then
    python -u /n/home03/ahmadazim/WORKING/genGen/DiffuGene/scripts/train_diff.py --model_type usit --normalize_latents
fi

# for type in unet dit udit sit usit normalize_usit; do sbatch /n/home03/ahmadazim/WORKING/genGen/DiffuGene/scripts/train_diff_submit.sh $type; done