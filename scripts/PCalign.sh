#!/bin/bash
#SBATCH -p serial_requeue,xlin,hsph,sapphire
#SBATCH --mem=100G
#SBATCH -t 1-00:00:00
#SBATCH -o /n/holystore01/LABS/xlin/Lab/ahmadazim/log_err/PCalign_AE128z_%j.out
#SBATCH -e /n/holystore01/LABS/xlin/Lab/ahmadazim/log_err/PCalign_AE128z_%j.err

conda init
eval "$(conda shell.bash hook)"
conda activate boltz1

gen_batch_idx=$1
num_samples=8192

base=/n/home03/ahmadazim/WORKING/genGen/
home=${base}/UKBVQVAE/genomic_data/
python -u ${base}/DiffuGene/src/DiffuGene/postGen/pcaProcrustes_fit.py \
    --orig-h5-root ${home}/ae_h5 \
    --generated ${home}/generated_samples/synObs_decoded_unrelWhite_allchr_AE128z_${num_samples}_genBatch${gen_batch_idx}/ \
    --n-fit 8192 \
    --k-pcs 5000 \
    --models-dir ${home}/../models/PCalign/ \
    --out-dir ${home}/generated_samples/synObs_decoded_aligned_unrelWhite_allchr_AE128z_${num_samples}_genBatch${gen_batch_idx}/

# only need to run for 1 batch:
# sbatch /n/home03/ahmadazim/WORKING/genGen/DiffuGene/scripts/PCalign.sh 1