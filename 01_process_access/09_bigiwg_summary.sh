#!/bin/bash
#SBATCH --job-name=09_bigiwg_summary
#SBATCH --output=logs/09_bigiwg_summary.out
#SBATCH --error=logs/09_bigiwg_summary.err
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=12:00:00

set -euo pipefail

# --- environment ---
conda activate access          # adjust if your samtools env has another name

access_dir=../../results/01_process_access/06_bam2bw_access_atac
encode_dir=../../results/01_process_access/07_bam2bw_atac

out_dir=../../results/01_process_access/09_bigwig_summary

mkdir -p ${out_dir}

for sample in HepG2 K562; do
    for ext in noext ext50 ext100;
    do

    echo ${sample}

    multiBigwigSummary bins -p 32 -b \
    ${access_dir}/${sample}_atac_${ext}.bw \
    ${encode_dir}/${sample}_atac_${ext}.bw \
    --binSize 1000 \
    --blackListFileName ../../data/Blacklist/hg38-blacklist.v2.bed \
    -o ${out_dir}/${sample}.${ext}.npz \
    --labels ACCESS_ATAC ATAC \
    --outRawCounts ${out_dir}/${sample}.${ext}.tab

    # 3. Spearman correlation (matches Fig 1f; robust to skewed signal)
    plotCorrelation -in ${out_dir}/${sample}.${ext}.npz \
        -c spearman \
        -p scatterplot \
        --log1p --removeOutliers \
        --plotTitle "${sample}: Tn5 insertion in peaks (ACCESS vs ATAC)" \
        -o ${out_dir}/${sample}_spearman.${ext}.png

    # also output Pearson for reference
    plotCorrelation -in ${out_dir}/${sample}.${ext}.npz \
        -c pearson \
        -p scatterplot \
        --log1p --removeOutliers \
        --plotTitle "${sample}: Tn5 insertion in peaks (ACCESS vs ATAC)" \
        -o ${out_dir}/${sample}_pearson.${ext}.png
done
done
