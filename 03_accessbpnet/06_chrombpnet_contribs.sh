#!/bin/bash
#SBATCH --job-name=06_chrombpnet_contribs
#SBATCH --output=logs/06_chrombpnet_contribs_%a.out
#SBATCH --error=logs/06_chrombpnet_contribs_%a.err
#SBATCH --array=0-3
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --partition=a40-quad
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1

set -euo pipefail

conda activate chrombpnet

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

model_dir=../../results/03_accessbpnet/05_run_chrombpnet
out_dir=../../results/03_accessbpnet/06_chrombpnet_contribs
peak_dir=../../results/03_accessbpnet/01_call_peaks
GENOME=../../data/GRCh38/GRCh38.primary_assembly.genome.fa
CHROM_SIZES=../../data/GRCh38/GRCh38.primary_assembly.chrom.sizes

# --- task table: 4 = 2 samples x 2 assays ---
# index:   0      1       2      3
samples=( K562   K562   HepG2  HepG2 )
assays=(  ATAC   ACCESS ATAC   ACCESS )

i=${SLURM_ARRAY_TASK_ID}
sample=${samples[$i]}
assay=${assays[$i]}

PEAK_FILE=${peak_dir}/${sample}_peaks_filtered_rm_blacklist_top100k.narrowPeak

mkdir -p ${out_dir}/${sample}/${assay}

echo "[$(date)] contribs: ${sample} ${assay}"

chrombpnet contribs_bw \
    -m ${model_dir}/${sample}/${assay}/models/chrombpnet_nobias.h5 \
    -r ${PEAK_FILE} \
    -g ${GENOME} \
    -c ${CHROM_SIZES} \
    -op ${out_dir}/${sample}/${assay}/${sample}

echo "[$(date)] done: ${sample} ${assay}"