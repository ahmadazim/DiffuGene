
# unaligned 
python -u /n/home03/ahmadazim/WORKING/genGen/DiffuGene/src/DiffuGene/generate/viz_decoded_quality.py \
    --orig-h5-root /n/home03/ahmadazim/WORKING/genGen/UKBVQVAE/genomic_data/vqvae_h5_cache \
    --generated /n/home03/ahmadazim/WORKING/genGen/UKBVQVAE/genomic_data/generated_samples/generated_decoded_unrelWhite_allchr_AE128z_2500_genBatch1/ \
    --out-dir /n/home03/ahmadazim/WORKING/genGen/UKBVQVAE/genomic_data/generated_samples/generated_decoded_unrelWhite_allchr_AE128z_2500_genBatch1/quality_viz \
    --bim-path /n/home03/ahmadazim/WORKING/genGen/UKBVQVAE/genomic_data/geneticBinary/ukb_allchr_unrel_britishWhite.bim \
    --haploblock-dir /n/home03/ahmadazim/WORKING/genGen/UKB6PC/genomic_data/haploblocks/ \
    --min-snps-per-block 50 \
    --max-blocks 5 \
    --ld-max-rows 2500 \
    --pca-samples-cap 2500 

# aligned (float) 
python -u /n/home03/ahmadazim/WORKING/genGen/DiffuGene/src/DiffuGene/generate/viz_decoded_quality.py \
    --orig-h5-root /n/home03/ahmadazim/WORKING/genGen/UKBVQVAE/genomic_data/vqvae_h5_cache \
    --generated /n/home03/ahmadazim/WORKING/genGen/UKBVQVAE/genomic_data/generated_samples/generated_decoded_aligned_unrelWhite_allchr_AE128z_8192_genBatch1/ \
    --out-dir /n/home03/ahmadazim/WORKING/genGen/UKBVQVAE/genomic_data/generated_samples/generated_decoded_aligned_unrelWhite_allchr_AE128z_8192_genBatch1/quality_viz_float \
    --bim-path /n/home03/ahmadazim/WORKING/genGen/UKBVQVAE/genomic_data/geneticBinary/ukb_allchr_unrel_britishWhite.bim \
    --haploblock-dir /n/home03/ahmadazim/WORKING/genGen/UKB6PC/genomic_data/haploblocks/ \
    --min-snps-per-block 50 \
    --max-blocks 5 \
    --ld-max-rows 2500 \
    --pca-samples-cap 2500 \
    --generated-mode aligned_float 

# aligned (rounded)
python -u /n/home03/ahmadazim/WORKING/genGen/DiffuGene/src/DiffuGene/generate/viz_decoded_quality.py \
    --orig-h5-root /n/home03/ahmadazim/WORKING/genGen/UKBVQVAE/genomic_data/vqvae_h5_cache \
    --generated /n/home03/ahmadazim/WORKING/genGen/UKBVQVAE/genomic_data/generated_samples/generated_decoded_aligned_unrelWhite_allchr_AE128z_8192_genBatch1/ \
    --out-dir /n/home03/ahmadazim/WORKING/genGen/UKBVQVAE/genomic_data/generated_samples/generated_decoded_aligned_unrelWhite_allchr_AE128z_8192_genBatch1/quality_viz_rounded \
    --bim-path /n/home03/ahmadazim/WORKING/genGen/UKBVQVAE/genomic_data/geneticBinary/ukb_allchr_unrel_britishWhite.bim \
    --haploblock-dir /n/home03/ahmadazim/WORKING/genGen/UKB6PC/genomic_data/haploblocks/ \
    --min-snps-per-block 50 \
    --max-blocks 5 \
    --ld-max-rows 2500 \
    --pca-samples-cap 2500 \
    --generated-mode aligned_rounded 