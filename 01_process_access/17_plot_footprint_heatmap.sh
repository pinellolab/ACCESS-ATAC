#!/bin/bash
#SBATCH --job-name=17_plot_footprint_heatmap
#SBATCH --output=logs/17_plot_footprint_heatmap_%A_%a.out
#SBATCH --error=logs/17_plot_footprint_heatmap_%A_%a.err
#SBATCH --array=0-1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00

set -euo pipefail

conda activate access

in_dir=../../results/01_process_access/14_get_mpbs
out_dir=../../results/01_process_access/17_plot_footprint_heatmap
bw_dir=../../results/01_process_access/06_bam2bw_access_atac

# --- task table: 2 samples, both assays plotted together ---
# index:    0       1
samples=( K562    HepG2 )

i=${SLURM_ARRAY_TASK_ID}
sample=${samples[$i]}

# both bigWigs for this sample; ACCESS first so rows are sorted by ACCESS signal
bw_access=${bw_dir}/${sample}_access_noext.bw
bw_atac=${bw_dir}/${sample}_atac_noext.bw

mkdir -p ${out_dir}/${sample}

echo "[$(date)] plotting footprint heatmaps: ${sample} (ACCESS + ATAC)"

for tf_mpbs in ${in_dir}/${sample}/*.bed; do
    name=$(basename ${tf_mpbs} .bed)
    prefix=${out_dir}/${sample}/${name}

    # Compute signal matrix for both assays at once, centered at motif midpoint
    computeMatrix reference-point \
        --referencePoint center \
        -b 50 -a 50 \
        --binSize 1 \
        -S ${bw_access} ${bw_atac} \
        -R ${tf_mpbs} \
        --missingDataAsZero \
        --numberOfProcessors ${SLURM_CPUS_PER_TASK} \
        -o ${prefix}.matrix.gz

    # Two side-by-side heatmaps sharing the same row order (sorted by first sample = ACCESS)
    plotHeatmap \
        -m ${prefix}.matrix.gz \
        --sortRegions descend \
        --sortUsing mean \
        --sortUsingSamples 1 \
        --colorMap Purples Greens \
        --whatToShow "heatmap and colorbar" \
        --refPointLabel "Motif center" \
        --regionsLabel "${name}" \
        --samplesLabel "ACCESS" "ATAC" \
        --xAxisLabel "Distance to motif center (bp)" \
        --heatmapHeight 12 \
        --heatmapWidth 4 \
        --dpi 300 \
        -o ${prefix}.heatmap.png

done

echo "[$(date)] done: ${sample}"