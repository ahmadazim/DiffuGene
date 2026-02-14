#!/bin/bash
#SBATCH -p serial_requeue,xlin,hsph,sapphire
#SBATCH --mem=100G
#SBATCH -t 1-00:00:00
#SBATCH -o /n/holystore01/LABS/xlin/Lab/ahmadazim/log_err/PCalign_apply_AE128z_%j.out
#SBATCH -e /n/holystore01/LABS/xlin/Lab/ahmadazim/log_err/PCalign_apply_AE128z_%j.err

conda init
eval "$(conda shell.bash hook)"
conda activate boltz1

gen_batch_idx=$1
type=$2
num_samples=8192

base=/n/home03/ahmadazim/WORKING/genGen/
home=${base}/UKBVQVAE/genomic_data/
python -u ${base}/DiffuGene/src/DiffuGene/postGen/pcaProcrustes_align.py \
    --generated-dir ${home}/generated_samples/${type}_decoded_unrelWhite_allchr_AE128z_${num_samples}_genBatch${gen_batch_idx}/ \
    --alignment-dir ${home}/../models/PCalign/ \
    --output-dir ${home}/generated_samples/${type}_decoded_aligned_unrelWhite_allchr_AE128z_${num_samples}_genBatch${gen_batch_idx}/

# sbatch /n/home03/ahmadazim/WORKING/genGen/DiffuGene/scripts/PCalign_apply.sh 1 synObs
# for i in {1..10}; do sbatch /n/home03/ahmadazim/WORKING/genGen/DiffuGene/scripts/PCalign_apply.sh ${i} generated; done
# for i in {1..10}; do sbatch /n/home03/ahmadazim/WORKING/genGen/DiffuGene/scripts/PCalign_apply.sh ${i} synObs; done
# for i in {11..20}; do sbatch /n/home03/ahmadazim/WORKING/genGen/DiffuGene/scripts/PCalign_apply.sh ${i} generated; done
# for i in {11..20}; do sbatch /n/home03/ahmadazim/WORKING/genGen/DiffuGene/scripts/PCalign_apply.sh ${i} synObs; done