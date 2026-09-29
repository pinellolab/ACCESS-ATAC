#!/bin/bash
#SBATCH --job-name=16_variant_score_prediction
#SBATCH --partition=a100-40g
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/16_variant_score_prediction_%A_%a.out
#SBATCH --error=logs/16_variant_score_prediction_%A_%a.err

set -euo pipefail

conda activate chrombpnet

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

in_dir=../../results/03_accessbpnet/15_prepare_snp
out_dir=../../results/03_accessbpnet/16_variant_score_prediction
model_dir=../../results/03_accessbpnet/05_run_chrombpnet/HepG2
GENOME=../../data/GRCh38/GRCh38.primary_assembly.genome.fa
CHROM_SIZES=../../data/GRCh38/GRCh38.primary_assembly.chrom.sizes

mkdir -p ${out_dir}

# Map array task ID to assay
assays=(ATAC ACCESS)
assay=${assays[$SLURM_ARRAY_TASK_ID]}

echo "Task ${SLURM_ARRAY_TASK_ID}: scoring variants for ${assay}"

python ../../github/variant-scorer/src/variant_scoring.py \
    -l ${in_dir}/snp_variants.tsv \
    -g ${GENOME} \
    -m ${model_dir}/${assay}/models/chrombpnet_nobias.h5 \
    -o ${out_dir}/${assay} \
    -s ${CHROM_SIZES} \
    -sc chrombpnet \
    --forward_only