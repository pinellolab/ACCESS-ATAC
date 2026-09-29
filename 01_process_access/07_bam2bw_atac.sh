#!/bin/bash
#SBATCH --job-name=07_bam2bw_atac
#SBATCH --output=logs/07_bam2bw_atac_%A_%a.out
#SBATCH --error=logs/07_bam2bw_atac_%A_%a.err
#SBATCH --array=0-2
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00

set -euo pipefail

conda activate access

IN_DIR=../../results/01_process_access/05_subsample
OUT_DIR=../../results/01_process_access/07_bam2bw_atac

FASTA=../../data/GRCh38/GRCh38.primary_assembly.genome.fa

# --- task table: 8 combinations = 2 samples x 2 events x 2 extend settings ---
# index:      0        1        2        3        4        5        6        7
# samples=( HepG2    HepG2   K562     K562   )
# events=(   tn5      tn5     tn5      tn5    )
# extends=(   0        50     0        50     )

samples=( HepG2   K562   )
events=(   tn5    tn5    )
extends=(  100    100     )

i=${SLURM_ARRAY_TASK_ID}
sample=${samples[$i]}
event=${events[$i]}
extend=${extends[$i]}

bam=${IN_DIR}/${sample}.atacseq.bam

# event flag: access is the default (no --event), tn5 needs --event tn5
if [ "${event}" = "tn5" ]; then
    event_flag="--event tn5"
    out_tag="atac"
else
    event_flag=""
    out_tag="access"
fi

# extend flag: 0 means no extension, otherwise pass --extend_size
if [ "${extend}" = "0" ]; then
    extend_flag=""
    out_name="${sample}_${out_tag}_noext"
else
    extend_flag="--extend_size ${extend}"
    out_name="${sample}_${out_tag}_ext${extend}"
fi

echo "[$(date)] task ${i}: sample=${sample} event=${event} extend=${extend} -> ${out_name}"

deamtools bam2bw \
    --bam ${bam} \
    --fasta ${FASTA} \
    --out_dir ${OUT_DIR} \
    --out_name ${out_name} \
    ${event_flag} \
    ${extend_flag} \
    --mode count \
    --min_coverage 1 \
    --min_mapq 0 \
    --min_baseq 0 \
    --threads 8

echo "[$(date)] done task ${i}"