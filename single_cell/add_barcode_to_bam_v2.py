import pysam
import argparse
import logging
import warnings
import pandas as pd

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
    parser.add_argument("--corrected_barcode", type=str, default=None)
    parser.add_argument("--bc_tag", type=str, default="CB")
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--out_name", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    # read corrected barcodes
    barcode_dict = None
    if args.corrected_barcode:
        df = pd.read_csv( args.corrected_barcode, sep="\t", header=None)
        barcode_dict = dict(zip(df[0], df[1]))
        logging.info(f"Loaded {len(barcode_dict)} corrected barcodes")

    # read sequence fastq to get original barcodes
    logging.info("Reading fastq file to get original barcodes")
    fastq_barcodes = {}
    with pysam.FastxFile(args.barcode_file) as fq:
        for entry in fq:
            read_name = entry.name.split(" ")[0]
            barcode = entry.sequence
            fastq_barcodes[read_name] = barcode
    logging.info(f"Loaded {len(fastq_barcodes)} reads from fastq")

    # add barcode to bam file
    logging.info("Adding barcode to bam file")
    infile = pysam.AlignmentFile(args.bam_file, "rb")
    outfile = pysam.AlignmentFile(
        f"{args.out_dir}/{args.out_name}.bam", "wb", template=infile
    )
    
    iter = infile.fetch(until_eof=True)
    for read in iter:
        read_name = read.query_name
        # Get original barcode from fastq
        original_barcode = fastq_barcodes.get(read_name, None)
        if original_barcode is not None:
            barcode = original_barcode
            # Correct barcode if correction file is provided
            if barcode_dict and barcode in barcode_dict:
                barcode = barcode_dict[barcode]

            # Set the barcode tag
            read.set_tag(args.bc_tag, barcode, replace=False)
            outfile.write(read)

    infile.close()
    outfile.close()
    logging.info("Done!")


if __name__ == "__main__":
    main()
