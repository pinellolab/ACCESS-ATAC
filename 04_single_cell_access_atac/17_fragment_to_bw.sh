#!/bin/bash
in_dir=../../results/04_single_cell_access_atac/16_split_fragment
out_dir=../../results/04_single_cell_access_atac/17_fragment_to_bw

# Create output directory if it does not exist
if [ ! -d ${out_dir} ]; then
  mkdir -p ${out_dir}
fi

# Sort downsampled files by chr, start, end and ignore header
for sample in Basal Ciliated Club Tuft;
do

# python scripts/fragment_to_bw_atac.py \
#   --input_fragments ${in_dir}/${sample}.fragments.tsv.gz \
#   --out_dir ${out_dir} \
#   --out_name ${sample}_ATAC \
#   --extend_size 50 \
#   --normalize \
#   --chrom_size_file ../../data/refdata-cellranger-arc-GRCm39-2024-A/fasta/genome.chrom.sizes


python scripts/fragment_to_bw_access.py \
  --input_fragments ${in_dir}/${sample}.fragments.tsv.gz \
  --out_dir ${out_dir} \
  --out_name ${sample}_ACCESS \
  --extend_size 50 \
  --normalize \
  --chrom_size_file ../../data/refdata-cellranger-arc-GRCm39-2024-A/fasta/genome.chrom.sizes

done

