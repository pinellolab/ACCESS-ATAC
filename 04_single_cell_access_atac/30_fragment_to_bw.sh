#!/bin/bash
in_dir=../../results/04_single_cell_access_atac/26_split_fragment
out_dir=../../results/04_single_cell_access_atac/30_fragment_to_bw

# Create output directory if it does not exist
if [ ! -d ${out_dir} ]; then
  mkdir -p ${out_dir}
fi

for cell_type in Basal Ciliated Club Tuft Hillock;
do

python scripts/fragment_to_bw_atac.py \
  --input_fragments ${in_dir}/${cell_type}.fragments.tsv.gz \
  --out_dir ${out_dir} \
  --out_name ${cell_type}_ATAC \
  --extend_size 0 \
  --normalize \
  --chrom_size_file ../../data/refdata-cellranger-arc-GRCm39-2024-A/fasta/genome.chrom.sizes


python scripts/fragment_to_bw_access.py \
  --input_fragments ${in_dir}/${cell_type}.fragments.tsv.gz \
  --out_dir ${out_dir} \
  --out_name ${cell_type}_ACCESS \
  --extend_size 0 \
  --normalize \
  --chrom_size_file ../../data/refdata-cellranger-arc-GRCm39-2024-A/fasta/genome.chrom.sizes

done

