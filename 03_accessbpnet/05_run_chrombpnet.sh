#!/bin/bash
#SBATCH --job-name=05_run_chrombpnet
#SBATCH --output=logs/05_run_chrombpnet_%a.out
#SBATCH --error=logs/05_run_chrombpnet_%a.err
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
bias_dir=../../results/03_accessbpnet/03_chrombpnet_bias
out_dir=../../results/03_accessbpnet/05_run_chrombpnet
GENOME=../../data/GRCh38/GRCh38.primary_assembly.genome.fa
CHROM_SIZES=../../data/GRCh38/GRCh38.primary_assembly.chrom.sizes

mkdir -p ${out_dir}

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

# fresh output dir (chrombpnet requires the output dir not to pre-exist)
rm -rf ${out_dir}/${sample}/${assay}
mkdir -p ${out_dir}/${sample}

echo "[$(date)] chrombpnet: sample=${sample} assay=${assay}"

chrombpnet pipeline \
    -ibw ${BW_FILE} \
    -d ${assay} \
    -g ${GENOME} \
    -c ${CHROM_SIZES} \
    -p ${PEAK_FILE} \
    -n ${bias_dir}/${sample}/nonpeaks_negatives.bed \
    -fl ./fold_0.json \
    -b ${bias_dir}/${sample}/${assay}/models/${sample}_bias.h5 \
    -o ${out_dir}/${sample}/${assay}

echo "[$(date)] done: ${sample} ${assay}"