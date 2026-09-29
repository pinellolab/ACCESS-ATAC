#!/bin/bash
#SBATCH --job-name=08_plot_heatmap_vs_atac
#SBATCH --output=logs/08_plot_heatmap_vs_atac.out
#SBATCH --error=logs/08_plot_heatmap_vs_atac.err
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=12:00:00

set -euo pipefail

# --- environment ---
conda activate access          # adjust if your samtools env has another name

access_dir=../../results/01_process_access/06_bam2bw_access_atac
encode_dir=../../results/01_process_access/07_bam2bw_atac

out_dir=../../results/01_process_access/08_plot_heatmap_vs_atac

# Create output directory if it does not exist
if [ ! -d ${out_dir} ]; then
  mkdir -p ${out_dir}
fi

tss_file=${out_dir}/tss.bed

module load bedtools/2.31.1

bedtools slop -i ../../data/bap/bap/anno/TSS/hg38.refGene.TSS.bed \
    -g ../../data/bap/bap/anno/bedtools/chrom_hg38.sizes -b 1000 > ${tss_file}

for sample in K562 HepG2;
do

computeMatrix reference-point \
-S ${access_dir}/${sample}_atac_ext50.bw ${encode_dir}/${sample}_atac_ext50.bw \
-R ${tss_file} \
--referencePoint center \
--missingDataAsZero \
-a 2000 -b 2000 -p 64 \
--binSize 1 --skipZeros \
--outFileName ${out_dir}/${sample}.ext50.tab.gz

plotHeatmap \
-m ${out_dir}/${sample}.ext50.tab.gz \
-out ${out_dir}/${sample}.ext50.tab.pdf \
--heatmapHeight 10  \
--heatmapWidth 6  \
--colorMap Purples Greens \
--samplesLabel ACCESS ATAC \
--whatToShow 'heatmap and colorbar' \
--refPointLabel enh.center \
--plotTitle '' \
--zMin 0 --zMax 100 --dpi 300

done