import os
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
        description="This script filters bam file by a list of barcodes",
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
        f"{args.out_dir}/{args.out_name}.bam", "wb", 
        template=infile
    )

    logging.info("Reading barcode file")
    df = pd.read_csv(args.barcode_file)
    sel_barcodes = df['barcode'].tolist()
    
    logging.info(f"Number of valid barcodes: {len(sel_barcodes)}")
    
    iter = infile.fetch(until_eof=True)
    for read in iter:
        barcode = read.get_tag(args.bc_tag)
        if barcode in sel_barcodes:
            outfile.write(read)

    infile.close()
    outfile.close()
    logging.info(f"Done!")


if __name__ == "__main__":
    main()
    
