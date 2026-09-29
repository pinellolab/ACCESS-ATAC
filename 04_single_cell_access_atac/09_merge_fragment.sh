#!/usr/bin/env bash

in_dir=../../results/04_single_cell_access_atac/08_bam_to_fragment
out_dir=../../results/04_single_cell_access_atac/09_merge_fragment

# Create output directory if it does not exist
if [ ! -d ${out_dir} ];  then
  mkdir -p ${out_dir}
fi

# combine all samples
cat ${in_dir}/*.fragments.tsv \
| sort --parallel 64 -k1,1 -k2,2n -k3,3n -k4,4 \
| uniq -c \
| awk 'BEGIN{OFS="\t"}{print $2,$3,$4,$5,$1,$6}' | bgzip -c > ${out_dir}/merged.fragments.tsv.gz

# aggregate fragments across replicates
python scripts/aggregate_replicates.py \
--input ${out_dir}/merged.fragments.tsv.gz \
--output ${out_dir}/merged.fragments.aggregated.tsv

bgzip -c ${out_dir}/merged.fragments.aggregated.tsv > ${out_dir}/merged.fragments.aggregated.tsv.gz
tabix -p bed ${out_dir}/merged.fragments.aggregated.tsv.gz

rm ${out_dir}/merged.fragments.aggregated.tsv

# generate ATAC fragments file 
zcat ${out_dir}/merged.fragments.aggregated.tsv.gz | cut -f1-5 | bgzip -c > ${out_dir}/merged.atac.tsv.gz
tabix -p bed ${out_dir}/merged.atac.tsv.gz
