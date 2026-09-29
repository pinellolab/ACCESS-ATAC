#!/bin/bash
#SBATCH --job-name=04_chrombpnet_bias
#SBATCH --output=logs/04_chrombpnet_bias_%a.out
#SBATCH --error=logs/04_chrombpnet_bias_%a.err
#SBATCH --array=0-3
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=120:00:00
#SBATCH --partition=a40-quad
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1

set -euo pipefail

conda activate chrombpnet

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

peak_dir=../../results/03_accessbpnet/01_call_peaks
in_dir=../../results/03_accessbpnet/02_bam2bw
out_dir=../../results/03_accessbpnet/03_chrombpnet_bias
GENOME=../../data/GRCh38/GRCh38.primary_assembly.genome.fa
CHROM_SIZES=../../data/GRCh38/GRCh38.primary_assembly.chrom.sizes
mkdir -p logs

# --- task table: 4 = 2 samples x 2 assays ---
# index:   0      1       2       3
samples=( K562   K562   HepG2   HepG2 )
assays=(  ATAC   ACCESS ATAC    ACCESS )

i=${SLURM_ARRAY_TASK_ID}
sample=${samples[$i]}
assay=${assays[$i]}

PEAK_FILE=${peak_dir}/${sample}_peaks_filtered_rm_blacklist_top100k.narrowPeak

if [ "${assay}" = "ATAC" ]; then
    BW_FILE=${in_dir}/${sample}_atac.bw
else
    BW_FILE=${in_dir}/${sample}_access.bw
fi

echo "[$(date)] bias: sample=${sample} assay=${assay}"

chrombpnet bias pipeline \
    -ibw ${BW_FILE} \
    -d ${assay} \
    -g ${GENOME} \
    -c ${CHROM_SIZES} \
    -p ${PEAK_FILE} \
    -n ${out_dir}/${sample}/nonpeaks_negatives.bed \
    -fl ./fold_0.json \
    -b 0.5 \
    -o ${out_dir}/${sample}/${assay} \
    -fp ${sample}

echo "[$(date)] done: ${sample} ${assay}"