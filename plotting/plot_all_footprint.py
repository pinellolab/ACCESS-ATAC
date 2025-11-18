import warnings

warnings.filterwarnings("ignore")

import argparse
import pyranges as pr
import logging
import pandas as pd
import numpy as np
import pyBigWig
import matplotlib.pyplot as plt
import seaborn as sns

# plt.rcParams['pdf.fonttype'] = 42

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

def get_signal(grs, bw_file, label) -> pd.DataFrame:
    bw = pyBigWig.open(bw_file)
    
    window_size = grs.End.values[0] - grs.Start.values[0]
    signal = np.zeros(shape=(len(grs), window_size))

    for i, (chrom, start, end) in enumerate(zip(grs.Chromosome, grs.Start, grs.End)):
        signal[i] = bw.values(chrom, start, end)

    signal[np.isnan(signal)] = 0
    signal = np.mean(signal, axis=0)
    
    return signal

def get_all_signal(grs, bw_files: list[str], labels: list[str], extend: int) -> pd.DataFrame:
    # extend regions
    mid = (grs.End + grs.Start) // 2
    grs.Start = mid - extend
    grs.End = mid + extend

    df_list = []
    for bw_file, label in zip(bw_files, labels):
        signal = get_signal(grs=grs, bw_file=bw_file, label=label)
        df = pd.DataFrame(data={"position":range(-extend, extend), "signal": signal, "data": label})
        df_list.append(df)
    
    df = pd.concat(df_list)
    
    return df

def parse_args():
    parser = argparse.ArgumentParser(
        description="This script generates BigWig file from a BAM file",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # Required parameters
    parser.add_argument("--bw_files", type=str, default=None)
    parser.add_argument("--labels", type=str, default=None)
    parser.add_argument("--bed_file", type=str, default=None)
    parser.add_argument("--extend", type=int, default=100)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--out_name", type=str, default="counts")

    return parser.parse_args()


def main():
    args = parse_args()

    logging.info(f"Reading regions from {args.bed_file}")
    grs = pr.read_bed(args.bed_file)
    
    # get bw files
    bw_files = args.bw_files.strip().split(",")
    labels = args.labels.strip().split(",")

    df = get_all_signal(grs, bw_files, labels, args.extend)
    
    colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#ffff33", "#a65628", "#f781bf",
              "#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854"]

    fig, ax = plt.subplots(1, 1, figsize=[6, 4])
    sns.lineplot(data=df, x="position", y="signal", hue="data", ax=ax, palette=colors)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.title(args.out_name)

    fig.tight_layout()
    plt.savefig(f'{args.out_dir}/{args.out_name}.png')
    df.to_csv(f"{args.out_dir}/{args.out_name}.csv", index=False)

    logging.info("Done")


if __name__ == "__main__":
    main()
