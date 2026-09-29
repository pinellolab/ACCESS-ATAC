in_dir=../../results/04_single_cell_access_atac/06_add_barcodes
out_dir=../../results/04_single_cell_access_atac/07_bam_to_fragment

# Create output directory if it does not exist
if [ ! -d ${out_dir} ]; then
  mkdir -p ${out_dir}
fi

BC="CATTCCGTCGACTATG"

# samtools view -@ 8 -b -d CB:${BC} ${in_dir}/batch1_in1.bam > ${out_dir}/${BC}.bam
# samtools index ${out_dir}/${BC}.bam

# # sort by read name in preparation for fragment file generation
# samtools sort -@ 64 -n -o ${out_dir}/${BC}.sorted.bam ${out_dir}/${BC}.bam

python scripts/bam_to_fragments.py \
--bam ${out_dir}/${BC}.sorted.bam \
--out ${out_dir}/${BC}.fragments.tsv \
--cell-tag CB \
--min-mapq 30 \
--add-access \
--ref-fasta ../../data/refdata-cellranger-arc-GRCm39-2024-A/fasta/genome.fa

# sort and unique the fragment file
cat ${out_dir}/${BC}.fragments.tsv | \
sort -k1,1 -k2,2n -k3,3n -k4,4 -k5,5 | \
uniq -c | \
awk 'BEGIN{OFS="\t"}{print $2,$3,$4,$5,$1,$6}' > ${out_dir}/${BC}.fragments.sorted.tsv

mv ${out_dir}/${BC}.fragments.sorted.tsv ${out_dir}/${BC}.fragments.tsv

python scripts/aggregate_replicates.py \
--input ${out_dir}/${BC}.fragments.tsv \
--output ${out_dir}/${BC}.fragments.aggregated.tsv