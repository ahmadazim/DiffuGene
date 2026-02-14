#!/bin/bash
#SBATCH -p xlin,hsph,sapphire,serial_requeue
#SBATCH --mem=200G
#SBATCH -t 1-00:00:00
#SBATCH -c 1
#SBATCH -o /n/holystore01/LABS/xlin/Lab/ahmadazim/log_err/writePlink_%j.out
#SBATCH -e /n/holystore01/LABS/xlin/Lab/ahmadazim/log_err/writePlink_%j.err

conda init
eval "$(conda shell.bash hook)"
conda activate boltz1
module load R gcc cmake

type=$1  # e.g. "synObs_decoded_aligned" or "generated_decoded"
gen_batch_idx=$2

num_samples=8192
base_dir=/n/home03/ahmadazim/WORKING/genGen/UKBVQVAE/genomic_data/generated_samples/
pt_dir=${base_dir}${type}_unrelWhite_allchr_AE128z_${num_samples}_genBatch${gen_batch_idx}/
out_prefix=${pt_dir}${type}_unrelWhite_allchr_AE128z_${num_samples}_genBatch${gen_batch_idx}

bim_src=${base_dir}../geneticBinary/ukb_allchr_1t22_unrel_britishWhite.bim
covar_csv=${base_dir}../../covariates/all_covariates_conditionalTrain.csv

python -u /n/home03/ahmadazim/WORKING/genGen/DiffuGene/src/DiffuGene/postGen/writePlink.py \
  --pt-dir ${pt_dir} \
  --bim ${bim_src} \
  --out-prefix ${out_prefix} \
  --chromosomes "all" \
  --covar-csv ${covar_csv}

# for i in {1..10}; do sbatch /n/home03/ahmadazim/WORKING/genGen/DiffuGene/scripts/writePlink.sh generated_decoded $i; done
# sbatch /n/home03/ahmadazim/WORKING/genGen/DiffuGene/scripts/writePlink.sh synObs_decoded 2
# for i in {1..10}; do sbatch /n/home03/ahmadazim/WORKING/genGen/DiffuGene/scripts/writePlink.sh generated_decoded_aligned $i; done
# for i in {1..10}; do sbatch /n/home03/ahmadazim/WORKING/genGen/DiffuGene/scripts/writePlink.sh synObs_decoded_aligned $i; done
# for i in {11..20}; do sbatch /n/home03/ahmadazim/WORKING/genGen/DiffuGene/scripts/writePlink.sh generated_decoded_aligned $i; done
# for i in {11..20}; do sbatch /n/home03/ahmadazim/WORKING/genGen/DiffuGene/scripts/writePlink.sh synObs_decoded_aligned $i; done

# type=synObs_decoded_aligned
# ls ${type}_unrelWhite_allchr_AE128z_8192_genBatch*/${type}_unrelWhite_allchr_AE128z_8192_genBatch*.bed \
#   | sed 's/\.bed$//' \
#   | grep -v 'genBatch1$' \
#   > merge_list_${type}.txt
# plink \
#   --bfile ${type}_unrelWhite_allchr_AE128z_8192_genBatch11/${type}_unrelWhite_allchr_AE128z_8192_genBatch11 \
#   --merge-list merge_list_${type}.txt \
#   --allow-no-sex \
#   --make-bed \
#   --out ${type}_unrelWhite_allchr_AE128z_81920_2


# type=generated_decoded_aligned
# ls ${type}_unrelWhite_allchr_AE128z_8192_genBatch*/${type}_unrelWhite_allchr_AE128z_8192_genBatch*.bed \
#   | sed 's/\.bed$//' \
#   | grep -v 'genBatch1$' \
#   > merge_list_${type}.txt
# plink \
#   --bfile ${type}_unrelWhite_allchr_AE128z_8192_genBatch11/${type}_unrelWhite_allchr_AE128z_8192_genBatch11 \
#   --merge-list merge_list_${type}.txt \
#   --allow-no-sex \
#   --make-bed \
#   --out ${type}_unrelWhite_allchr_AE128z_81920_2