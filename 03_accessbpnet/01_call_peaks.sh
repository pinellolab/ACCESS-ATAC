#!/bin/bash
#SBATCH --job-name=01_call_peaks
#SBATCH --output=logs/01_call_peaks.out
#SBATCH --error=logs/01_call_peaks.err
#SBATCH --mem=64G
#SBATCH --time=3:00:00

# set -euo pipefail

# --- environment ---
conda activate macs2

in_dir=../../results/01_process_access/02_filter
out_dir=../../results/03_accessbpnet/01_call_peaks

mkdir -p ${out_dir}

for cell_type in HepG2;
do

BAM=${in_dir}/${cell_type}.concurrent.accessatac.bam

# call peaks with MACS2
macs2 callpeak \
    -t ${BAM} \
    -f BAM -n ${cell_type} --outdir ${out_dir} \
    -g hs \
    --nomodel --nolambda --shift -100 --extsize 200 \
    --keep-dup all \
    -q 0.01 --bdg

# # only keep peaks from chromosomes 1-22, X, and Y
awk '$1 ~ /^chr[0-9XY]+$/' ${out_dir}/${cell_type}_peaks.narrowPeak > ${out_dir}/${cell_type}_peaks_filtered.narrowPeak

# remove blacklist
module load bedtools

bedtools intersect -v \
-a ${out_dir}/${cell_type}_peaks_filtered.narrowPeak \
-b ../../data/Blacklist/hg38-blacklist.v2.bed | \
sort -k1,1 -k2,2n > ${out_dir}/${cell_type}_peaks_filtered_rm_blacklist.narrowPeak

# select top 100000 peaks
sort -k 8gr,8gr ${out_dir}/${cell_type}_peaks_filtered_rm_blacklist.narrowPeak | \
head -n 100000 | sort -k1,1 -k2,2n > ${out_dir}/${cell_type}_peaks_filtered_rm_blacklist_top100k.narrowPeak

done