#!/bin/bash
#SBATCH --job-name=02_filter_bam
#SBATCH --output=logs/02_filter_bam.out
#SBATCH --error=logs/02_filter_bam.err
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=8:00:00

set -euo pipefail

conda activate access

THREADS=64
OUT_DIR=../../results/01_process_access/02_filter
mkdir -p ${OUT_DIR}

# Main chromosomes to keep: chr1-22 + chrX.
# chrY excluded (K562/HepG2 are female-derived); chrM excluded to avoid the
# very high mitochondrial signal that distorts ATAC analyses.
CHROMS=$(echo chr{1..22} chrX)

#############################
# ACCESS-ATAC (single-end, Ultima; already cz-dedup'd, so keep all -- no 0x400)
IN_DIR=../../data/Concurrent_ACCESS_ATAC
for sample in HepG2 K562; do
    in_bam=${IN_DIR}/062424_${sample}_concurrent_ACCESS-ATAC.dedup.hg38_core.bam
    out_bam=${OUT_DIR}/${sample}.concurrent.accessatac.bam

    # index input if needed (required for region-based extraction)
    [ -f "${in_bam}.bai" ] || samtools index -@ ${THREADS} "${in_bam}"

    # drop unmapped/secondary/supplementary/QC-fail (-F 0xB04), MAPQ>=30, main chroms
    samtools view -@ ${THREADS} -b -F 0xB04 -q 30 "${in_bam}" ${CHROMS} > "${out_bam}"
    samtools index -@ ${THREADS} "${out_bam}"
    samtools flagstat -@ ${THREADS} "${out_bam}" \
        > "${OUT_DIR}/${sample}.concurrent.accessatac.qc.txt"
done

#############################
# ENCODE ATAC-seq (paired-end; duplicates marked-but-not-removed, so drop them)
IN_DIR=../../data/encode_atac
for sample in HepG2 K562; do
    in_bam=${IN_DIR}/${sample}.bam
    out_bam=${OUT_DIR}/${sample}.encode.atacseq.bam

    [ -f "${in_bam}.bai" ] || samtools index -@ ${THREADS} "${in_bam}"

    # keep properly paired (-f 0x2); drop unmapped/secondary/supplementary/QC-fail/dup
    # (-F 0xF04); MAPQ>=30; main chroms
    samtools view -@ ${THREADS} -b -f 0x2 -F 0xF04 -q 30 "${in_bam}" ${CHROMS} > "${out_bam}"
    samtools index -@ ${THREADS} "${out_bam}"
    samtools flagstat -@ ${THREADS} "${out_bam}" \
        > "${OUT_DIR}/${sample}.encode.atacseq.qc.txt"
done