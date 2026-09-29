#!/bin/bash
#SBATCH --job-name=02_bam2bw
#SBATCH --output=logs/02_bam2bw_%A_%a.out
#SBATCH --error=logs/02_bam2bw_%A_%a.err
#SBATCH --array=0-3
#SBATCH --cpus-per-task=18
#SBATCH --mem=48G
#SBATCH --time=12:00:00

set -euo pipefail

conda activate access

IN_DIR=../../results/01_process_access/02_filter
OUT_DIR=../../results/03_accessbpnet/02_bam2bw
FASTA=../../data/GRCh38/GRCh38.primary_assembly.genome.fa
mkdir -p ${OUT_DIR} logs

# --- task table: 4 combinations = 2 samples x 2 events ---
# index:    0      1      2      3
samples=( HepG2  HepG2  K562   K562 )
events=(  edit   tn5    edit   tn5  )

i=${SLURM_ARRAY_TASK_ID}
sample=${samples[$i]}
event=${events[$i]}

bam=${IN_DIR}/${sample}.concurrent.accessatac.bam

# event flag: edit (ACCESS) is the default (no --event); tn5 needs --event tn5
if [ "${event}" = "tn5" ]; then
    event_flag="--event tn5"
    out_tag="atac"
else
    event_flag=""
    out_tag="access"
fi

out_name="${sample}_${out_tag}"

echo "[$(date)] task ${i}: sample=${sample} event=${event} -> ${out_name}"

deamtools bam2bw \
    --bam ${bam} \
    --fasta ${FASTA} \
    --out_dir ${OUT_DIR} \
    --out_name ${out_name} \
    ${event_flag} \
    --mode count \
    --min_coverage 1 \
    --min_mapq 0 \
    --min_baseq 0 \
    --threads 18

echo "[$(date)] done task ${i}"