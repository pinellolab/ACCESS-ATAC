#!/bin/bash
#SBATCH --job-name=03_prep_nonpeaks
#SBATCH --output=logs/03_prep_nonpeaks_%a.out
#SBATCH --error=logs/03_prep_nonpeaks_%a.err
#SBATCH --array=0-1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=8:00:00

set -euo pipefail

conda activate chrombpnet

peak_dir=../../results/03_accessbpnet/01_call_peaks
out_dir=../../results/03_accessbpnet/03_chrombpnet_bias
GENOME=../../data/GRCh38/GRCh38.primary_assembly.genome.fa
GENOME_SIZE=../../data/GRCh38/core.chrom.sizes
BLACKLIST=../../data/Blacklist/hg38-blacklist.v2.bed
mkdir -p logs

samples=(K562 HepG2)
sample=${samples[$SLURM_ARRAY_TASK_ID]}

mkdir -p ${out_dir}/${sample}
PEAK_FILE=${peak_dir}/${sample}_peaks_filtered_rm_blacklist_top100k.narrowPeak

# clean any previous auxiliary output
rm -rf ${out_dir}/${sample}/nonpeaks_auxiliary

# Prepare non-peaks (shared by both assays of this sample)
chrombpnet prep nonpeaks \
    -g ${GENOME} \
    -p ${PEAK_FILE} \
    -c ${GENOME_SIZE} \
    -fl ./fold_0.json \
    -br ${BLACKLIST} \
    -o ${out_dir}/${sample}/nonpeaks

echo "[$(date)] nonpeaks done for ${sample}"