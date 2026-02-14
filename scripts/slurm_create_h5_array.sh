#!/bin/bash
#SBATCH --job-name=ae_h5_batches
#SBATCH --array=1-22
#SBATCH --cpus-per-task=1
#SBATCH --mem=30G
#SBATCH --time=05:00:00
#SBATCH --partition=xlin,serial_requeue,hsph,sapphire
#SBATCH --output=/n/home03/ahmadazim/WORKING/genGen/logs/ae_h5_%A_%a.out
#SBATCH --error=/n/home03/ahmadazim/WORKING/genGen/logs/ae_h5_%A_%a.err

set -euo pipefail

conda init
eval "$(conda shell.bash hook)"
conda activate boltz1
module load cuda/12.4.1-fasrc01
module load cudnn/9.5.1.17_cuda12-fasrc01
module load R gcc cmake


#   --bfile ${DATA_ROOT}/geneticBinary/ukb_allchr_unrel_britishWhite \
#   --bim ${DATA_ROOT}/geneticBinary/ukb_allchr_unrel_britishWhite.bim \
#   --fam ${DATA_ROOT}/geneticBinary/ukb_allchr_unrel_britishWhite.fam \
#   --chromosomes all \
#   --batch-size 12000 \
#   --h5-out-root ${DATA_ROOT}/ae_h5 \
#   --models-dir ${DATA_ROOT}/../models/AE \
#   --model-pattern ae_chr{chr}_homog.pt \
#   --latents-out-root ${DATA_ROOT}/AE_embeddings \
#   --layout-json ${DATA_ROOT}/../models/AE/ae_milp_layout.json \
#   --unified-out-root ${DATA_ROOT}/AE_embeddings/unified \
#   --device cuda \
#   --encode-batch-size 128

# --- EDIT THESE PATHS BEFORE SUBMITTING ---
DATA_ROOT=/n/home03/ahmadazim/WORKING/genGen/UKBVQVAE/genomic_data/
BFILE="${DATA_ROOT}/geneticBinary/ukb_allchr_unrel_britishWhite"
BIM="${DATA_ROOT}/geneticBinary/ukb_allchr_unrel_britishWhite.bim"
FAM="${DATA_ROOT}/geneticBinary/ukb_allchr_unrel_britishWhite_conditional_diffusion_train.fam"
OUT_DIR="${DATA_ROOT}/ae_h5"
BATCH_SIZE=12000

SCRIPT="/n/home03/ahmadazim/WORKING/genGen/DiffuGene/scripts/create_h5_for_chr.py"

CHR="${SLURM_ARRAY_TASK_ID}"

if [[ ! -f "${SCRIPT}" ]]; then
    echo "Cannot find script at ${SCRIPT}" >&2
    exit 1
fi

python "${SCRIPT}" \
    --bfile "${BFILE}" \
    --bim "${BIM}" \
    --fam "${FAM}" \
    --out-dir "${OUT_DIR}" \
    --chromosome "${CHR}" \
    --batch-size "${BATCH_SIZE}"

