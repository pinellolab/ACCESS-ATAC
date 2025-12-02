# Add cell barcodes to a bam file from a fastq file.

import argparse
import logging
import warnings
import gzip

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

    logging.info(f"Adding barcodes to fastq file {args.reads_fastq} using barcodes from {args.barcodes_fastq}")
    # read barcodes from the fastq file
    with gzip.open(args.reads_fastq, 'rt') as r_fq, gzip.open(args.barcodes_fastq, 'rt') as bc_fq, gzip.open(output_fastq, 'w') as out_fq:
        while True:
            header = r_fq.readline()
            if not header:
                break  # EOF

            seq = r_fq.readline()
            plus = r_fq.readline()
            qual = r_fq.readline()

            bc_header = bc_fq.readline()
            bc_seq = bc_fq.readline()
            bc_plus = bc_fq.readline()
            bc_qual = bc_fq.readline()

            # Extract read name from both files
            read_name = header.split()[0][1:]  # remove '@' and get the read name
            barcode_name = bc_header.split()[0][1:]  # remove '@' and get the barcode name

            # Check if read name matches barcode name
            if read_name != barcode_name:
                logging.error("Read names do not match between reads and barcodes fastq files!")
                logging.error("Terminating the process.")
                logging.error(f"Please check the input fastq files: {args.reads_fastq} and {args.barcodes_fastq}")
                logging.error(f"Read name: {read_name}, Barcode name: {barcode_name}")
                break

            # Extract barcode sequence
            barcode = bc_seq.strip()

            # Modify the read header (line 0)
            header = header.strip().split(" ")[0]
            header_with_barcode = f"{header}_{barcode}\n"
            header_with_barcode = header_with_barcode.replace(" ", "_")
    
            # Write modified record
            out_fq.write(header_with_barcode.encode("utf-8")) # header
            out_fq.write(seq.encode("utf-8"))  # sequence
            out_fq.write(plus.encode("utf-8"))  # '+'
            out_fq.write(qual.encode("utf-8"))  # quality

    logging.info("Done!")

if __name__ == "__main__":
    main()