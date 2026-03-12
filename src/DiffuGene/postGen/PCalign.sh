#!/bin/bash
#SBATCH -p serial_requeue,xlin,hsph,sapphire
#SBATCH --mem=200G
#SBATCH -t 1-00:00:00
#SBATCH -o /n/holystore01/LABS/xlin/Lab/ahmadazim/log_err/PCalign_USiT_%j.out
#SBATCH -e /n/holystore01/LABS/xlin/Lab/ahmadazim/log_err/PCalign_USiT_%j.err

eval "$(conda shell.bash hook)"
conda activate diffugene

gen_batch_idx=$1
num_samples_fit=$2
num_pcs=$3
chromosomes=$4

base=/n/home03/ahmadazim/WORKING/genGen/
home=${base}/UKBVQVAE/genomic_data/

# mkdir -p ${base}/UKBVQVAE/models/PCalign_USiT/test_${num_pcs}_chr${chromosomes}/
# mkdir -p ${home}/generated_samples/generated_USiT_H_wNC_H_aligned_${num_pcs}_chr${chromosomes}/

python -u ${base}/DiffuGene/src/DiffuGene/postGen/pcaProcrustes_fit.py \
    --orig-h5-root ${home}/orig251K_h5_cache/ \
    --generated ${home}/generated_samples/generated_USiT_H_wNC_H/ \
    --generated-batch-number ${gen_batch_idx} \
    --n-fit ${num_samples_fit} \
    --k-pcs ${num_pcs} \
    --chromosomes ${chromosomes} \
    --models-dir ${base}/UKBVQVAE/models/PCalign_USiT/ \
    --out-dir ${home}/generated_samples/generated_USiT_H_wNC_H_aligned/
    # --models-dir ${base}/UKBVQVAE/models/PCalign_USiT/test_${num_pcs}_chr${chromosomes}/ \
    # --out-dir ${home}/generated_samples/generated_USiT_H_wNC_H_aligned_${num_pcs}_chr${chromosomes}/

# testing:
# for k in 500 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000; do sbatch /n/home03/ahmadazim/WORKING/genGen/DiffuGene/src/DiffuGene/postGen/PCalign.sh 1 16384 ${k} 22; done
# running 5k for every chromosome:
# for chr in {1..21}; do sbatch /n/home03/ahmadazim/WORKING/genGen/DiffuGene/src/DiffuGene/postGen/PCalign.sh 1 16384 5000 ${chr}; done