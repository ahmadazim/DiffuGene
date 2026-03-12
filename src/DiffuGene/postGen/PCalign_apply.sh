#!/bin/bash
#SBATCH -p serial_requeue,xlin,hsph,sapphire
#SBATCH --mem=100G
#SBATCH -t 1-00:00:00
#SBATCH -o /n/holystore01/LABS/xlin/Lab/ahmadazim/log_err/PCalign_apply_AE128z_%j.out
#SBATCH -e /n/holystore01/LABS/xlin/Lab/ahmadazim/log_err/PCalign_apply_AE128z_%j.err

set -eo pipefail

eval "$(conda shell.bash hook)"
conda activate diffugene

type=$1 # synObs or generated
batch_no=$2

generated_dir=/n/home03/ahmadazim/WORKING/genGen/UKBVQVAE/genomic_data/generated_samples/${type}_USiT_H_wNC_H/
alignment_dir=/n/home03/ahmadazim/WORKING/genGen/UKBVQVAE/models/PCalign_USiT/
output_dir=/n/home03/ahmadazim/WORKING/genGen/UKBVQVAE/genomic_data/generated_samples/${type}_aligned_USiT_H_wNC_H/
output_prefix=${type}_decoded_aligned_USiT_H_wNC_H_batch${batch_no}
chromosomes=all

base=/n/home03/ahmadazim/WORKING/genGen/

mkdir -p ${output_dir}

python -u ${base}/DiffuGene/src/DiffuGene/postGen/pcaProcrustes_align.py \
    --generated-dir ${generated_dir} \
    --generated-batch-number ${batch_no} \
    --alignment-dir ${alignment_dir} \
    --output-dir ${output_dir} \
    --output-prefix ${output_prefix} \
    --chromosomes ${chromosomes}

# for i in {1..5}; do sbatch /n/home03/ahmadazim/WORKING/genGen/DiffuGene/src/DiffuGene/postGen/PCalign_apply.sh synObs ${i}; done
# for i in {1..5}; do sbatch /n/home03/ahmadazim/WORKING/genGen/DiffuGene/src/DiffuGene/postGen/PCalign_apply.sh generated ${i}; done