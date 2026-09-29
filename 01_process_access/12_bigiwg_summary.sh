#!/bin/bash
#SBATCH --job-name=12_bigiwg_summary
#SBATCH --output=logs/12_bigiwg_summary.out
#SBATCH --error=logs/12_bigiwg_summary.err
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=12:00:00

set -euo pipefail

# conda activate access

THREADS=32
peak_dir=../../results/01_process_access/10_call_peaks
out_dir=../../results/01_process_access/12_peak_correlation
mkdir -p ${out_dir}

for sample in HepG2 K562; do
    echo "[$(date)] ${sample}"

    access_peak=${peak_dir}/${sample}_accessatac_peaks.filtered.narrowPeak
    atac_peak=${peak_dir}/${sample}_atacseq_peaks.filtered.narrowPeak
    access_bw=../../results/01_process_access/06_bam2bw_access_atac/${sample}_atac_ext50.bw
    atac_bw=../../results/01_process_access/07_bam2bw_atac/${sample}_atac_ext50.bw

    # 1. merge the two peak sets -> comparison regions
    cat ${access_peak} ${atac_peak} \
        | sort -k1,1 -k2,2n \
        | bedtools merge > ${out_dir}/${sample}.merged_peaks.bed
    echo "  merged peaks: $(wc -l < ${out_dir}/${sample}.merged_peaks.bed)"

    # 2. signal in each merged peak from the two Tn5 insertion bigWigs
    multiBigwigSummary BED-file \
        --BED ${atac_peak} \
        -b ${access_bw} ${atac_bw} \
        --labels ACCESS_Tn5 ATAC_Tn5 \
        -o ${out_dir}/${sample}_peak_summary.npz \
        --outRawCounts ${out_dir}/${sample}_peak_summary.tab \
        -p ${THREADS}

    # 3. Spearman correlation (matches Fig 1f; robust to skewed signal)
    plotCorrelation -in ${out_dir}/${sample}_peak_summary.npz \
        -c spearman \
        -p scatterplot \
        --log1p --removeOutliers \
        --plotTitle "${sample}: Tn5 insertion in peaks (ACCESS vs ATAC)" \
        -o ${out_dir}/${sample}_spearman.png

    # also output Pearson for reference
    plotCorrelation -in ${out_dir}/${sample}_peak_summary.npz \
        -c pearson \
        -p scatterplot \
        --log1p --removeOutliers \
        --plotTitle "${sample}: Tn5 insertion in peaks (ACCESS vs ATAC)" \
        -o ${out_dir}/${sample}_pearson.png
done

echo "[$(date)] done"