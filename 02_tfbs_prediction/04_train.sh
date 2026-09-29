#!/bin/bash
#SBATCH --job-name=04_train
#SBATCH --output=logs/04_train.out
#SBATCH --error=logs/04_train.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=120:00:00
#SBATCH --partition=a40-quad
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1

set -euo pipefail

conda activate cell2net          # env with the DL framework for train.py

in_dir=../../results/02_tfbs_prediction/03_prepare_data
out_dir=../../results/02_tfbs_prediction/04_train
mkdir -p ${out_dir}

for cell_type in HepG2 K562; do

  # create output subdirectories for all assays once
  for assay in seq accessatac_tn5 accessatac_dddss accessatac_both; do
    mkdir -p ${out_dir}/${cell_type}/${assay}/{model,logs,prediction,metric}
  done

  # outer loop: TF
  for in_file in ${in_dir}/${cell_type}/train/*.npz; do
    tf_name=$(basename ${in_file} .npz)

    # inner loop: assay
    for assay in seq accessatac_tn5 accessatac_dddss accessatac_both; do

      if [ -f ${out_dir}/${cell_type}/${assay}/prediction/${tf_name}.csv ]; then
        echo "${tf_name} (${assay}) already processed, skip."
        continue
      fi

      echo "[$(date)] Training ${cell_type} ${tf_name} using ${assay}..."

      python model/train.py \
        --train_data ${in_dir}/${cell_type}/train/${tf_name}.npz \
        --valid_data ${in_dir}/${cell_type}/valid/${tf_name}.npz \
        --test_data  ${in_dir}/${cell_type}/test/${tf_name}.npz \
        --assay ${assay} \
        --batch_size 64 \
        --epochs 30 \
        --cuda 0 \
        --model_dir  ${out_dir}/${cell_type}/${assay}/model \
        --log_dir    ${out_dir}/${cell_type}/${assay}/logs \
        --pred_dir   ${out_dir}/${cell_type}/${assay}/prediction \
        --metric_dir ${out_dir}/${cell_type}/${assay}/metric \
        --out_name ${tf_name}

    done
  done
done

echo "[$(date)] all done"
