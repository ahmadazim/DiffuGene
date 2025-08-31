#!/bin/bash
#SBATCH -p xlin,hsph,sapphire,serial_requeue
#SBATCH --mem=300G
#SBATCH -t 3-00:00:00
#SBATCH -c 1
#SBATCH -o /n/holystore01/LABS/xlin/Lab/ahmadazim/log_err/store_decoded_binary_merge_%j.out
#SBATCH -e /n/holystore01/LABS/xlin/Lab/ahmadazim/log_err/store_decoded_binary_merge_%j.err

conda init
eval "$(conda shell.bash hook)"
conda activate boltz1
module load R gcc cmake

key=$1 # 294Korig, 500K

UKBdir=/n/home03/ahmadazim/WORKING/genGen/UKB/
decoded_dir=${UKBdir}genomic_data/generated_samples/decoded_snps_ukb_allchr_unrel_britishWhite_unrelWhite_allchr_4PC_64z_${key}/

python ~/WORKING/genGen/DiffuGene/scripts/store_decoded_binary_merge.py \
  --batch-dir ${decoded_dir} \
  --bim-file ${decoded_dir}ukb_allchr_unrel_britishWhite_${key}.bim \
  --fam-file ${decoded_dir}ukb_allchr_unrel_britishWhite_${key}.fam \
  --output-prefix ${decoded_dir}ukb_allchr_unrel_britishWhite_${key}_merged \
  --plink /n/home03/ahmadazim/TOOLS/plink

# sbatch /n/home03/ahmadazim/WORKING/genGen/DiffuGene/scripts/store_decoded_binary_merge.sh 294Korig
# sbatch /n/home03/ahmadazim/WORKING/genGen/DiffuGene/scripts/store_decoded_binary_merge.sh 500K