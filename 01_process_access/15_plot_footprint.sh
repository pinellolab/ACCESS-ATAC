#!/bin/bash
#SBATCH --job-name=15_plot_footprint
#SBATCH --output=logs/15_plot_footprint_%A_%a.out
#SBATCH --error=logs/15_plot_footprint_%A_%a.err
#SBATCH --array=0-3
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00

set -euo pipefail

conda activate access

in_dir=../../results/01_process_access/14_get_mpbs
out_dir=../../results/01_process_access/15_plot_footprint
bw_dir=../../results/01_process_access/06_bam2bw_access_atac
script=../../github/ACCESS-ATAC/plotting/plot_all_footprint.py
mkdir -p logs

# --- task table: 4 = 2 samples x 2 assays ---
# index:    0       1       2       3
samples=( K562    K562    HepG2   HepG2 )
assays=(  ACCESS  ATAC    ACCESS  ATAC  )

i=${SLURM_ARRAY_TASK_ID}
sample=${samples[$i]}
assay=${assays[$i]}

# pick the right bigWig for this assay
if [ "${assay}" = "ACCESS" ]; then
    bw=${bw_dir}/${sample}_access_noext.bw
else
    bw=${bw_dir}/${sample}_atac_noext.bw
fi

mkdir -p ${out_dir}/${sample}/${assay}

echo "[$(date)] plotting footprints: ${sample} ${assay}"

for tf_mpbs in ${in_dir}/${sample}/*.bed; do
    name=$(basename ${tf_mpbs} .bed)

    # skip if already done
    if ls ${out_dir}/${sample}/${assay}/${name}* >/dev/null 2>&1; then
        continue
    fi

    python ${script} \
        --bw_files ${bw} \
        --labels ${sample} \
        --bed_file ${tf_mpbs} \
        --extend 500 \
        --out_dir ${out_dir}/${sample}/${assay} \
        --out_name ${name}
done

echo "[$(date)] done: ${sample} ${assay}"