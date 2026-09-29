#!/bin/bash
#SBATCH --job-name=10_call_peaks
#SBATCH --output=logs/10_call_peaks.out
#SBATCH --error=logs/10_call_peaks.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=120:00:00        # long, to avoid timing out

set -euo pipefail

conda activate macs2
module load bedtools/2.31.1

THREADS=16                     # keep in sync with cpus-per-task
SORT_MEM=32G                   # lowered; 32G is enough and safe

IN_DIR=../../results/01_process_access/05_subsample   # already downsampled BAMs
OUT_DIR=../../results/01_process_access/10_call_peaks
BLACKLIST=../../data/Blacklist/hg38-blacklist.v2.bed
mkdir -p ${OUT_DIR}

# # ---------------------------------------------------------------------------
# # Step 1: BAM -> Tn5 insertion BED (one line per insertion = one 5' cut site)
# #   single-end ACCESS : each read -> 1 insertion
# #   paired-end  ATAC  : each mate -> 1 insertion (both ends of a fragment)
# #   Tn5 offset: +4 on + strand, -5 on - strand.
# # ---------------------------------------------------------------------------
# bam_to_insertion_bed () {
#     local bam="$1" out_bed="$2"
#     bedtools bamtobed -i "${bam}" \
#       | awk 'BEGIN{OFS="\t"}
#              {
#                if ($6 == "+") { start = $2 + 4 } else { start = $3 - 5 }
#                if (start < 0) start = 0;
#                print $1, start, start+1, ".", ".", $6
#              }' \
#       | LC_ALL=C sort -k1,1 -k2,2n --parallel=${THREADS} -S ${SORT_MEM} \
#       > "${out_bed}"
# }

# echo "[$(date)] Step 1: BAM -> insertion BED"
# for sample in HepG2 K562; do
#     for assay in accessatac atacseq; do
#         bam=${IN_DIR}/${sample}.${assay}.bam
#         bed=${OUT_DIR}/${sample}.${assay}.insertions.bed
#         bam_to_insertion_bed "${bam}" "${bed}"
#         echo "  [${sample} ${assay}] insertions = $(wc -l < ${bed})"
#     done
# done

# ---------------------------------------------------------------------------
# Step 2: call peaks on the insertion BED (-f BED: each line = one tag)
#   BED input treats single-end and paired-end identically.
# ---------------------------------------------------------------------------
echo "[$(date)] Step 2: MACS2 callpeak (-f BED)"
for sample in HepG2 K562; do
    for assay in atacseq; do
        bed=${OUT_DIR}/${sample}.${assay}.insertions.bed
        name=${sample}_${assay}

        macs2 callpeak \
            -t ${bed} \
            -f BED -n ${name} \
            --outdir ${OUT_DIR} \
            -g hs \
            --nomodel --shift -100 --extsize 200 \
            -q 0.0001

        bedtools intersect -v \
            -a ${OUT_DIR}/${name}_peaks.narrowPeak \
            -b ${BLACKLIST} \
            | LC_ALL=C sort -k1,1 -k2,2n \
            > ${OUT_DIR}/${name}_peaks.filtered.narrowPeak

        echo "  [${name}] $(wc -l < ${OUT_DIR}/${name}_peaks.filtered.narrowPeak) peaks"
    done
done

echo "[$(date)] done"