#!/usr/bin/env python3
"""
Prepare TFBS-prediction training data (one-hot sequence + accessibility signals)
for one cell type and one TF. Splits into train/valid/test by chromosome.

Usage:
    python prepare_data.py --cell_type K562 --tf CTCF
"""
import os
import json
import argparse

import numpy as np
import pandas as pd
import pysam
import pyBigWig


# ---- fixed config ----
IN_DIR    = "../../results/02_tfbs_prediction/02_create_labels"
OUT_DIR   = "../../results/02_tfbs_prediction/03_prepare_data"
FASTA     = "../../data/GRCh38/GRCh38.primary_assembly.genome.fa"
FOLD_JSON = "../46_accessbpnet/fold_0.json"
BW_DIR    = "../../results/01_process_access"
WIDTH     = 256


def one_hot_encode(seq):
    nuc_d = {
        "A": [1.0, 0.0, 0.0, 0.0],
        "C": [0.0, 1.0, 0.0, 0.0],
        "G": [0.0, 0.0, 1.0, 0.0],
        "T": [0.0, 0.0, 0.0, 1.0],
        "N": [0.0, 0.0, 0.0, 0.0],
    }
    return np.array([nuc_d[x] for x in seq], dtype=np.float32)


def get_data(df, fasta, bw_access_tn5, bw_access_dddss):
    seq_list, s_acc_tn5, s_acc_dddss, lab = [], [], [], []
    for chrom, start, end, label in zip(df.chrom, df.start, df.end, df.label):
        seq = str(fasta.fetch(chrom, start, end)).upper()
        if len(seq) != WIDTH:
            continue
        # skip sequences with characters outside ACGTN
        if not set(seq).issubset(set("ACGTN")):
            continue

        b  = np.nan_to_num(np.array(bw_access_tn5.values(chrom, start, end)))
        d  = np.nan_to_num(np.array(bw_access_dddss.values(chrom, start, end)))

        # log1p to stabilize variance
        b, d = np.log1p(b), np.log1p(d)

        seq_list.append(one_hot_encode(seq))
        s_acc_tn5.append(b)
        s_acc_dddss.append(d)
        lab.append(label)

    return (np.array(seq_list, dtype=np.float32),
            np.array(s_acc_tn5, dtype=np.float32),
            np.array(s_acc_dddss, dtype=np.float32),
            np.array(lab, dtype=np.int8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell_type", required=True)
    ap.add_argument("--tf", required=True)
    args = ap.parse_args()
    cell_type, tf = args.cell_type, args.tf

    labels_file = f"{IN_DIR}/{cell_type}/{tf}.labels.tsv"
    if not os.path.exists(labels_file):
        print(f"[{cell_type} {tf}] labels file not found, skipping")
        return

    # all three splits already done?
    if all(os.path.exists(f"{OUT_DIR}/{cell_type}/{d}/{tf}.npz")
           for d in ("train", "valid", "test")):
        print(f"[{cell_type} {tf}] all splits exist, skipping")
        return

    for d in ("train", "valid", "test"):
        os.makedirs(f"{OUT_DIR}/{cell_type}/{d}", exist_ok=True)

    with open(FOLD_JSON) as f:
        data_split = json.load(f)

    df_peaks = pd.read_csv(labels_file, sep="\t")
    df_peaks.columns = ["chrom", "start", "end", "label"]

    fasta = pysam.FastaFile(FASTA)
    bw_access_tn5   = pyBigWig.open(f"{BW_DIR}/06_bam2bw_access_atac/{cell_type}_atac_noext.bw")
    bw_access_dddss = pyBigWig.open(f"{BW_DIR}/06_bam2bw_access_atac/{cell_type}_access_noext.bw")

    for d in ("train", "valid", "test"):
        out_npz = f"{OUT_DIR}/{cell_type}/{d}/{tf}.npz"
        if os.path.exists(out_npz):
            continue
        sub = df_peaks[df_peaks["chrom"].isin(data_split[d])].reset_index(drop=True)
        seq, s_acc_tn5, s_acc_dddss, label = get_data(
            sub, fasta, bw_access_tn5, bw_access_dddss)
        np.savez_compressed(out_npz,
                            seq=seq,
                            signal_accessatac_tn5=s_acc_tn5,
                            signal_accessatac_dddss=s_acc_dddss,
                            label=label)
        print(f"[{cell_type} {tf} {d}] n={len(label)} pos={int(label.sum())}")

    for bw in (bw_access_tn5, bw_access_dddss):
        bw.close()
    print(f"[{cell_type} {tf}] done")


if __name__ == "__main__":
    main()
