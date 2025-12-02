# Add cell barcodes to a bam file from a fastq file.

import argparse
import logging
import warnings
import gzip
from Bio import SeqIO

warnings.filterwarnings("ignore")

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

def parse_args():
    parser = argparse.ArgumentParser(
        description="This script adds barcode to fastq file",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # Required parameters
    parser.add_argument("--reads_fastq", type=str, default=None)
    parser.add_argument("--barcodes_fastq", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--out_name", type=str, default=None)
    return parser.parse_args()

def main():
    args = parse_args()

    output_fastq = f"{args.out_dir}/{args.out_name}.fastq.gz"

    # read barcodes from the fastq file
    with gzip.open(args.reads_fastq, 'rt') as r_fq, gzip.open(args.barcodes_fastq, 'rt') as bc_fq, gzip.open(output_fastq, 'wt') as out_fq:
        reads_iter = SeqIO.parse(r_fq, "fastq")
        bc_iter    = SeqIO.parse(bc_fq,    "fastq")
        
        for r, b in zip(reads_iter, bc_iter):
            # Check if read name matches barcode name
            if r.id != b.id:
                logging.warning(f"Read name {r.id} does not match barcode name {b.id}. Skipping this pair.")
                break
            
            # Extract barcode sequence
            barcode = str(b.seq)
            orig_id = r.id  # e.g. "HWUSI-EAS100R:6:73:941:1973#0/1"
            if r.description:
                # description includes id + comment
                new_id = orig_id + "_" + barcode
                # preserve rest of description
                rest = r.description[len(orig_id):]  # leading space included
                r.id = new_id
                r.description = new_id + rest
            else:
                r.id = orig_id + "_" + barcode
                r.description = r.id  # or leave blank if preferred

            # Write modified read to output fastq
            SeqIO.write(r, out_fq, "fastq")
        
    logging.info("Done!")

if __name__ == "__main__":
    main()