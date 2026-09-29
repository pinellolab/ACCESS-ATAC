#!/bin/bash
#SBATCH --job-name=05_subsample
#SBATCH --output=logs/05_subsample.out
#SBATCH --error=logs/05_subsample.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=8:00:00

set -euo pipefail

conda activate access

THREADS=32
IN_DIR=../../results/01_process_access/02_filter
OUT_DIR=../../results/01_process_access/05_subsample
mkdir -p ${OUT_DIR}

SEED=42
TARGET=200000000        # <-- set the number of reads to subsample to

for sample in HepG2 K562; do
    echo "[$(date)] ${sample}"

    access_bam=${IN_DIR}/${sample}.concurrent.accessatac.bam
    atac_bam=${IN_DIR}/${sample}.encode.atacseq.bam

    # --- ACCESS (single-end): count all primary reads ---
    n_access=$(samtools view -c -F 0x900 "${access_bam}")
    frac_access=$(awk -v t=${TARGET} -v n=${n_access} 'BEGIN{printf "%.8f", t/n}')
    samtools view -@ ${THREADS} -b -s ${SEED}${frac_access} \
        "${access_bam}" > "${OUT_DIR}/${sample}.accessatac.bam"
    samtools index -@ ${THREADS} "${OUT_DIR}/${sample}.accessatac.bam"

    # --- ATAC (paired-end): only properly paired reads (-f 0x2) ---
    n_atac=$(samtools view -c -f 0x2 -F 0x900 "${atac_bam}")
    frac_atac=$(awk -v t=${TARGET} -v n=${n_atac} 'BEGIN{printf "%.8f", t/n}')
    samtools view -@ ${THREADS} -b -f 0x2 -F 0x900 -s ${SEED}${frac_atac} \
        "${atac_bam}" > "${OUT_DIR}/${sample}.atacseq.bam"
    samtools index -@ ${THREADS} "${OUT_DIR}/${sample}.atacseq.bam"

    echo "  ACCESS: ${n_access} -> $(samtools view -c -F 0x900 ${OUT_DIR}/${sample}.accessatac.bam)"
    echo "  ATAC:   ${n_atac} -> $(samtools view -c -F 0x900 ${OUT_DIR}/${sample}.atacseq.bam)"
done

echo "[$(date)] done"