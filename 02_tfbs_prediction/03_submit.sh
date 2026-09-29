#!/bin/bash
#SBATCH --job-name=03_prepare_data
#SBATCH --output=logs/03_prepare_data_%A_%a.out
#SBATCH --error=logs/03_prepare_data_%A_%a.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --array=0-773%50          # 774 tasks, 50 concurrent

set -euo pipefail

conda activate access

META=../../results/27_compare_tfbs_prediction/01_download_peaks/tfbs_peaks_metadata.csv

# --- build (cell_type, tf) task list from the metadata csv ---
# csv has a 'sample' (cell_type) column and a 'tf' column; skip header
TASK_LIST=$(awk -F',' 'NR>1 {print $1"\t"$2}' "${META}" | sort -u)
# NOTE: adjust column indices ($1, $2) to match your csv layout.
# If 'sample' and 'tf' are not columns 1 and 2, change accordingly.

N_TASKS=$(printf "%s\n" "${TASK_LIST}" | grep -c . || true)
if [ "${SLURM_ARRAY_TASK_ID}" -ge "${N_TASKS}" ]; then
    echo "index ${SLURM_ARRAY_TASK_ID} >= N_TASKS ${N_TASKS}, nothing to do"
    exit 0
fi

line=$(printf "%s\n" "${TASK_LIST}" | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")
cell_type=$(echo "${line}" | cut -f1)
tf=$(echo "${line}" | cut -f2)

echo "[$(date)] task ${SLURM_ARRAY_TASK_ID}: ${cell_type} ${tf}"
python 03_prepare_data.py --cell_type "${cell_type}" --tf "${tf}"
echo "[$(date)] done"
