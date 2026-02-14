#!/bin/bash
#SBATCH -p serial_requeue,xlin,hsph,sapphire
#SBATCH --mem=200G
#SBATCH -t 02:00:00
#SBATCH -o /n/holystore01/LABS/xlin/Lab/ahmadazim/log_err/postGen_align_CV_%j.out
#SBATCH -e /n/holystore01/LABS/xlin/Lab/ahmadazim/log_err/postGen_align_CV_%j.err

conda init
eval "$(conda shell.bash hook)"
conda activate boltz1
module load R gcc cmake

seed=$1
chrNo=$2

python -u /n/home03/ahmadazim/WORKING/genGen/DiffuGene/scripts/postGen_align.py --seed ${seed} --chrNo ${chrNo}