#!/bin/bash
#SBATCH -p xlin,hsph,sapphire,serial_requeue
#SBATCH --mem=100G
#SBATCH -t 1-00:00:00
#SBATCH -c 1
#SBATCH -o /n/holystore01/LABS/xlin/Lab/ahmadazim/log_err/export_snp_major_float32_%j.out
#SBATCH -e /n/holystore01/LABS/xlin/Lab/ahmadazim/log_err/export_snp_major_float32_%j.err

set -eo pipefail

eval "$(conda shell.bash hook)"
conda activate diffugene

set -u

export OMP_NUM_THREADS=1
export MKL_INTERFACE_LAYER=GNU,LP64
export MKL_THREADING_LAYER=GNU
export MKL_DYNAMIC=TRUE
export MKL_NUM_THREADS=1

type=$1 # synObs or generated

name_suffix="_USiT_H_wNC_H"

base=/n/home03/ahmadazim/WORKING/genGen/
gen_dir=${base}UKBVQVAE/genomic_data/generated_samples/
bim_src=${gen_dir}../geneticBinary/ukb_allchr_1t22_unrel_britishWhite.bim

python -u ${base}DiffuGene/src/DiffuGene/postGen/export_snp_major_float32.py \
    --input-dir ${gen_dir}${type}_decoded_aligned${name_suffix}/ \
    --bim ${bim_src} \
    --output-dir ${gen_dir}${type}_decoded_aligned${name_suffix}_float32/ \
    --output-prefix ${type}_decoded_aligned${name_suffix} \
    --file-prefix-pattern "${type}_decoded_aligned${name_suffix}_*" \
    --chromosomes all \
    --batch-numbers all