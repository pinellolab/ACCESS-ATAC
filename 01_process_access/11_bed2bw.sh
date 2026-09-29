#!/bin/bash
#SBATCH --job-name=11_bed2bw
#SBATCH --output=logs/11_bed2bw.out
#SBATCH --error=logs/11_bed2bw.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=6:00:00

set -euo pipefail

# conda activate access
# module load bedtools/2.31.1

THREADS=16
SORT_MEM=32G
IN_DIR=../../results/01_process_access/10_call_peaks   # sorted insertion BEDs
OUT_DIR=../../results/01_process_access/11_bed2bw
CHROM_SIZES=../../data/GRCh38/core.chrom.sizes
mkdir -p ${OUT_DIR}

for sample in HepG2 K562; do
    for assay in accessatac atacseq; do
        bed=${IN_DIR}/${sample}.${assay}.insertions.bed
        bg=${OUT_DIR}/${sample}.${assay}.bedgraph
        bw=${OUT_DIR}/${sample}.${assay}.bw

        # CPM scale factor (1e6 / total insertions) so tracks are comparable
        n=$(wc -l < "${bed}")
        scale=$(awk -v n=${n} 'BEGIN{printf "%.8f", 1000000/n}')
        echo "[$(date)] ${sample} ${assay}: ${n} insertions, scale=${scale}"

        # BED already coordinate-sorted; genomecov -> re-sort bedGraph for bigWig
        bedtools genomecov -i "${bed}" -g "${CHROM_SIZES}" -bg -scale ${scale} \
          | LC_ALL=C sort -k1,1 -k2,2n --parallel=${THREADS} -S ${SORT_MEM} \
          > "${bg}"

        bedGraphToBigWig "${bg}" "${CHROM_SIZES}" "${bw}"
        rm "${bg}"
        echo "[$(date)] wrote ${bw}"
    done
done

echo "[$(date)] done"