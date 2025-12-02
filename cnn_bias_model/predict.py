import argparse
import os
import sys
import numpy as np
import subprocess as sp
import warnings
import torch
import logging
import pyranges as pr
import pysam
import pandas as pd
from model import BiasNet
from utils import one_hot_encode

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

if not sys.warnoptions:
    warnings.simplefilter("ignore")


def parse_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--regions", type=str, default=None, help="BED file of input genomic regions")
    parser.add_argument("--ref_fasta", type=str, default=None, help="FASTQ file for reference genome")
    parser.add_argument("--k", type=int, default=128, help="Input sequence size")
    parser.add_argument("--model_path", type=str, default=None, help="Model path")
    parser.add_argument("--chrom_size_file", type=str, default=None, help="chrom_size_file")
    parser.add_argument("--out_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--out_name", type=str, default=None, help="Output name")
    return parser.parse_args()


def main():
    args = parse_args()

    fasta = pysam.FastaFile(args.ref_fasta)

    if args.regions:
        logging.info("Loading input regions")
        grs = pr.read_bed(args.regions)
    else:
        logging.info("Using whole genome")
        df = pd.read_csv(args.chrom_size_file, header=None, sep="\t")
        df.columns = ["Chromosome", "End"]
        df["Start"] = 0
        df = df[["Chromosome", "Start", "End"]]
        df["End"] = df["End"] - 1
        grs = pr.from_dict(df)

    # Setup model
    logging.info("Creating model")
    model = BiasNet(seq_len=args.k)
    state_dict = torch.load(args.model_path)
    model.load_state_dict(state_dict["state_dict"])
    device = torch.device("cuda")
    model.to(device)
    model.eval()

    logging.info("Predicting values for input regions")
    wig_filename = os.path.join(args.out_dir, "{}.wig".format(args.out_name))
    bw_filename = os.path.join(args.out_dir, "{}.bw".format(args.out_name))

    with open(wig_filename, "w") as f:
        for chrom, start, end in zip(grs.Chromosome, grs.Start, grs.End):
            n = (end - start) // args.k
            preds = []

            for i in range(n + 1):
                if start + (i + 1) * args.k > end:
                    seq = str(fasta.fetch(chrom, start + i * args.k, end)).upper()
                    seq = seq + ("N" * (args.k - len(seq)))
                else:
                    seq = str(
                        fasta.fetch(chrom, start + i * args.k, start + (i + 1) * args.k)
                    ).upper()

                # forward prediction
                x = torch.tensor(one_hot_encode(seq=seq))
                x = x.unsqueeze(dim=0)
                pred = model(x.to(device)).detach().cpu().view(-1).numpy()
                
                # for A, T or N, set the prediction as zero
                seq = np.array(list(seq))
                pred[np.where(seq == 'A')] = 0
                pred[np.where(seq == 'T')] = 0
                pred[np.where(seq == 'N')] = 0
                
                preds.append(pred)
                
            pred = np.concatenate(preds).tolist()
            pred = pred[: (end - start)]
            f.write(f"fixedStep chrom={chrom} start={start+1} step=1\n")
            f.write("\n".join(str(e) for e in pred))
            f.write("\n")

    logging.info(f"Predicting finished")
    sp.run(["wigToBigWig", wig_filename, args.chrom_size_file, bw_filename])
    os.remove(wig_filename)


if __name__ == "__main__":
    main()
