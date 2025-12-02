import pysam
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
    logging.info("Adding barcode to bam file")
    iter = infile.fetch(until_eof=True)
    for read in iter:
        barcode = read.qname.split("_")[-1]  # Assuming barcode is the last part of qname
        if not barcode:
            logging.warning(f"Read {read.qname} does not have a valid barcode.")
            continue

        # Set the barcode tag
        read.set_tag(args.bc_tag, barcode, replace=False)
        outfile.write(read)

    infile.close()
    outfile.close()
    logging.info("Done!")


if __name__ == "__main__":
    main()
