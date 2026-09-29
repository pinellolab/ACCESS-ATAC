#!/bin/bash
#SBATCH --job-name=04_bam_summary
#SBATCH --output=logs/04_bam_summary.out
#SBATCH --error=logs/04_bam_summary.err
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=12:00:00

set -euo pipefail

conda activate access

in_dir=../../results/01_process_access/02_filter
out_dir=../../results/01_process_access/04_bam_summary
blacklist=../../data/Blacklist/hg38-blacklist.v2.bed
mkdir -p ${out_dir} logs

for sample in HepG2 K562; do
    echo "[$(date)] ${sample}"

    access_bam=${in_dir}/${sample}.concurrent.accessatac.bam
    atac_bam=${in_dir}/${sample}.encode.atacseq.bam

    # bin genome-wide read coverage from the two BAMs
    multiBamSummary bins -p 32 \
        -b ${access_bam} ${atac_bam} \
        --labels ACCESS_ATAC ATAC \
        --binSize 10000 \
        --blackListFileName ${blacklist} \
        -o ${out_dir}/${sample}.npz \
        --outRawCounts ${out_dir}/${sample}.counts.tab

    # Spearman correlation as a scatterplot (clearer than heatmap for 2 samples)
    plotCorrelation -in ${out_dir}/${sample}.npz \
        -c spearman \
        -p scatterplot \
        --log1p \
        --removeOutliers \
        --plotTitle "${sample}: ACCESS-ATAC vs ATAC" \
        -o ${out_dir}/${sample}.png
done

echo "[$(date)] done"