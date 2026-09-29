#!/bin/bash
#SBATCH --job-name=02_create_labels
#SBATCH --output=logs/02_create_labels_%A_%a.out
#SBATCH --error=logs/02_create_labels_%A_%a.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --array=0-773%50          # 774 tasks, 50 concurrent

set -euo pipefail

conda activate access

IN_DIR=../../results/27_compare_tfbs_prediction/01_download_peaks

# --- build the full (cell_type, tf) task list from the input peak files ---
# one line per task: "<cell_type>\t<tf>"
TASK_LIST=$(
  for cell_type in K562 HepG2; do
    for f in ${IN_DIR}/${cell_type}/*.bed; do
      [ -e "$f" ] || continue
      tf=$(basename "$f" .bed)
      printf "%s\t%s\n" "${cell_type}" "${tf}"
    done
  done | sort -u
)

# total number of tasks
N_TASKS=$(printf "%s\n" "${TASK_LIST}" | grep -c . || true)

# guard: if this array index is beyond the task list, exit quietly
if [ "${SLURM_ARRAY_TASK_ID}" -ge "${N_TASKS}" ]; then
    echo "array index ${SLURM_ARRAY_TASK_ID} >= N_TASKS ${N_TASKS}, nothing to do"
    exit 0
fi

# pick this task's (cell_type, tf) by line number
line=$(printf "%s\n" "${TASK_LIST}" | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")
cell_type=$(echo "${line}" | cut -f1)
tf=$(echo "${line}" | cut -f2)

echo "[$(date)] task ${SLURM_ARRAY_TASK_ID}: ${cell_type} ${tf}"

python ./02_create_labels.py --cell_type "${cell_type}" --tf "${tf}"

echo "[$(date)] done"
