import os
import argparse
import pysam
import sys
import logging
import polars as pl
logging.basicConfig(level=logging.INFO)

from utils import read_chrom_sizes

def parse_args():
    parser = argparse.ArgumentParser(
        description="This script generates BigWig file from a fragment.tsv.gz file",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # Required parameters
    parser.add_argument("--input_fragment", help = "Path to the coordinate-sorted fragment file.")
    parser.add_argument("--chrom_size", help = "Path to the chromosome sizes file.")
    parser.add_argument("--skip_rows", help = "Number of rows to skip in the fragment file.", type=int, default=0)
    parser.add_argument("--out_dir", help = "Path to the output directory.")
    parser.add_argument("--out_name", help = "Name for the output fragment file.")

    return parser.parse_args()


def build_header(chrom_sizes):
    """Return a pysam header dict with SQ lines built from sizes."""
    hd = {"HD": {"VN": "1.6"}, "SQ": []}
    for chrom, length in chrom_sizes:
        hd["SQ"].append({"SN": chrom, "LN": int(length)})
    return hd

def main():
    args = parse_args()

    out_path = f"{args.out_dir}/{args.out_name}.bam"

    logging.info(f"Loading chromosome sizes from {args.chrom_size}")
    chrom_sizes = read_chrom_sizes(args.chrom_size)

    logging.info(f"Building BAM header")
    header = build_header(chrom_sizes)

    logging.info(f"Reading fragments from {args.input_fragments}")
    df_fragments = pl.read_csv(
        args.input_fragments,
        skip_rows=args.skip_rows,
        has_header=True,
        separator="\t",
        use_pyarrow=False,
    )

    logging.info(f"Creating BAM file at {out_path}")
    bam_file = pysam.AlignmentFile(out_path, "wb", header=header)

    logging.info(f"Reading fragments from {args.input_fragment}")

    for row in df_fragments.iter_rows():
        chrom = row[0]
        start = row[1]
        end = row[2]

        a = pysam.AlignedSegment(bam_file.header)
        a.query_name = name
        a.reference_id = bam_file.get_tid(chrom)
        a.reference_start = start
        a.reference_end = end
        a.mapping_quality = score
        a.is_reverse = False  # Default to forward if strand is unknown

        # CIGAR string: assuming full-length match
        cigar_length = end - start
        a.cigar = [(0, cigar_length)]  # 0: BAM_CMATCH

        bam_file.write(a)


    bam_file.close()
    logging.info(f"BAM file creation completed.")


if __name__ == '__main__':
    main()


