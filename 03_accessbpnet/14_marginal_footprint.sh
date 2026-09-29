#!/bin/bash
#SBATCH --job-name=14_marginal_footprint
#SBATCH --output=logs/14_marginal_footprint_%a.out
#SBATCH --error=logs/14_marginal_footprint_%a.err
#SBATCH --array=0-7
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=120:00:00
#SBATCH --partition=a40-quad
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1

set -euo pipefail

conda activate chrombpnet

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}

out_dir=../../results/03_accessbpnet/14_marginal_footprint
model_dir=../../results/03_accessbpnet/05_run_chrombpnet
peak_dir=../../results/03_accessbpnet/01_call_peaks
GENOME=../../data/GRCh38/GRCh38.primary_assembly.genome.fa
CHROM_SIZES=../../data/GRCh38/GRCh38.primary_assembly.chrom.sizes

# --- task table: 8 = 2 samples x 2 assays x 2 models ---
# index:    0      1      2      3      4      5      6      7
samples=(  K562   K562   K562   K562   HepG2  HepG2  HepG2  HepG2 )
assays=(   ATAC   ATAC   ACCESS ACCESS ATAC   ATAC   ACCESS ACCESS )
modes=(    corrected uncorrected corrected uncorrected corrected uncorrected corrected uncorrected )

i=${SLURM_ARRAY_TASK_ID}
sample=${samples[$i]}
assay=${assays[$i]}
mode=${modes[$i]}

# corrected = bias-factorized model; uncorrected = full model
if [[ ${mode} == "corrected" ]]; then
    model_h5=${model_dir}/${sample}/${assay}/models/chrombpnet_nobias.h5
else
    model_h5=${model_dir}/${sample}/${assay}/models/chrombpnet.h5
fi

mkdir -p ${out_dir}/${sample}/${assay}/${mode}

chrombpnet footprints \
    -m ${model_h5} \
    -r ../../results/03_accessbpnet/03_chrombpnet_bias/${sample}/nonpeaks_negatives.bed \
    -g ${GENOME} \
    -fl ./fold_0.json \
    -op ${out_dir}/${sample}/${assay}/${mode}/ \
    -pwm_f ./../../results/03_accessbpnet/13_prepare_pwm/motif_to_pwm.tsv

echo "[$(date)] done: ${sample} ${assay} ${mode}"