#!/bin/bash
#SBATCH --job-name=08_run_tfmodisco
#SBATCH --output=logs/08_run_tfmodisco_%a.out
#SBATCH --error=logs/08_run_tfmodisco_%a.err
#SBATCH --array=0-7
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00

set -euo pipefail

conda activate chrombpnet

in_dir=../../results/03_accessbpnet/06_chrombpnet_contribs
out_dir=../../results/03_accessbpnet/08_run_tfmodisco
MOTIFS=../../data/Motifs/JASPAR2026_CORE_vertebrates_non-redundant_pfms_meme.txt
mkdir -p logs

# --- task table: 8 = 2 samples x 2 assays x 2 score types ---
# index:      0        1        2        3        4        5        6        7
samples=(   K562     K562     K562     K562     HepG2    HepG2    HepG2    HepG2  )
assays=(    ATAC     ATAC     ACCESS   ACCESS   ATAC     ATAC     ACCESS   ACCESS )
scores=(    counts   profile  counts   profile  counts   profile  counts   profile )

i=${SLURM_ARRAY_TASK_ID}
sample=${samples[$i]}
assay=${assays[$i]}
score=${scores[$i]}

mkdir -p ${out_dir}/${sample}/${assay}

echo "[$(date)] TF-MoDISco: ${sample} ${assay} ${score}_scores"

# motif discovery
modisco motifs -w 500 \
    -i ${in_dir}/${sample}/${assay}/${sample}.${score}_scores.h5 \
    -n 1000000 \
    -o ${out_dir}/${sample}/${assay}/${score}_scores.h5

# report
modisco report \
    -i ${out_dir}/${sample}/${assay}/${score}_scores.h5 \
    -o ${out_dir}/${sample}/${assay}/${score}_scores/ \
    -s ${out_dir}/${sample}/${assay}/${score}_scores/ \
    -m ${MOTIFS}

echo "[$(date)] done: ${sample} ${assay} ${score}_scores"
