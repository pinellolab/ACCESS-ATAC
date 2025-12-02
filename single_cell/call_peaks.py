import pysam
import pandas as pd
import argparse
import logging
import warnings

warnings.filterwarnings("ignore")

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="This script adds barcode to bam file",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # Required parameters
    parser.add_argument("--bam_file", type=str, default=None)
    parser.add_argument("--barcode_file", type=str, default=None)
    parser.add_argument("--bc_tag", type=str, default="CB")
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--out_name", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    infile = pysam.AlignmentFile(args.bam_file, "rb")
    outfile = pysam.AlignmentFile(
        f"{args.out_dir}/{args.out_name}.bam", "wb", template=infile
    )

    logging.info("Reading barcode file")
    df = pd.read_csv(args.barcode_file, compression="gzip", sep=",")
    dict_barcode = dict(zip(df['Identifier'], df['Barcode']))

    del df

    logging.info("Adding barcode to bam file")
    iter = infile.fetch(until_eof=True)
    for read in iter:
        read.set_tag(args.bc_tag, dict_barcode[read.qname], replace=False)
        outfile.write(read)

    infile.close()
    outfile.close()
    logging.info(f"Done!")


if __name__ == "__main__":
    main()
