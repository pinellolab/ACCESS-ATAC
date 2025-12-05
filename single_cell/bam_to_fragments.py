import argparse
import pysam
import gzip
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser(description="Generate fragments from name-sorted BAM")
    parser.add_argument("--bam", "-i", required=True, help="Input BAM file (name-sorted)")
    parser.add_argument("--out", "-o", required=True, help="Output fragments file (tsv or tsv.gz)")
    parser.add_argument("--cell-tag", default="CB", help="Tag name storing cell barcode (default: CB)")
    parser.add_argument("--min-mapq", type=int, default=30,help="Minimum MAPQ for both mates to be kept (default: 30)")
    parser.add_argument("--tn5-shift", type=int, default=4, help="Tn5 shift in bp applied to 5' ends (default: 4, similar to 10x)")
    return parser.parse_args()

def five_prime_position(read):
    """
    Return the 5' genomic coordinate (0-based, inclusive) of a read.

    For ATAC-seq, the 5' end is:
      - reference_start for a forward read
      - reference_end - 1 for a reverse read
    """
    if read.is_reverse:
        return read.reference_end - 1
    else:
        return read.reference_start

def is_good_read_pair(r1, r2, min_mapq=30, allow_secondary=False):
    """
    Check basic filters for a proper useful ATAC-seq read pair.
    """
    # paired and proper pair
    if not (r1.is_paired and r2.is_paired):
        return False
    if not (r1.is_proper_pair and r2.is_proper_pair):
        return False

    # both mapped
    if r1.is_unmapped or r2.is_unmapped:
        return False
    if r1.mate_is_unmapped or r2.mate_is_unmapped:
        return False

    # same reference
    if r1.reference_id != r2.reference_id:
        return False

    # filter secondary / supplementary if not allowed
    if not allow_secondary:
        if r1.is_secondary or r2.is_secondary:
            return False
        if r1.is_supplementary or r2.is_supplementary:
            return False

    # MAPQ filter
    if r1.mapping_quality < min_mapq or r2.mapping_quality < min_mapq:
        return False

    return True

def open_output(path):
    """
    Open output as text handle; use gzip if filename ends with .gz.
    """
    if path.endswith(".gz"):
        return gzip.open(path, "wt")
    else:
        return open(path, "w")
    
def main():
    args = parse_args()

    bam = pysam.AlignmentFile(args.bam, "rb")


    with open_output(args.out) as out:
        prev_read = None
        for read in bam:
            # We assume name-sorted BAM, so mates appear consecutively.
            if prev_read is None:
                prev_read = read
                continue

            if read.query_name != prev_read.query_name:
                # Orphaned read; start new pair candidate with current read
                prev_read = read
                continue

            # Here: read and prev_read share the same query_name -> form a pair
            r1, r2 = prev_read, read

            # Normalize r1 as read1 if possible
            if r2.is_read1 and not r1.is_read1:
                r1, r2 = r2, r1

            # Decide if this pair passes filters
            if not is_good_read_pair(r1, r2,
                                    min_mapq=args.min_mapq):
                prev_read = None
                continue

            # Get cell barcode from either mate
            cb = None
            try:
                cb = r1.get_tag(args.cell_tag)
            except KeyError:
                try:
                    cb = r2.get_tag(args.cell_tag)
                except KeyError:
                    cb = None

            if cb is None:
                # No cell barcode -> skip
                prev_read = None
                continue

            # Compute Tn5-shifted fragment boundaries
            chrom = bam.get_reference_name(r1.reference_id)
            p1 = five_prime_position(r1)
            p2 = five_prime_position(r2)

            # Apply Tn5 shift
            # +strand (5' left): +shift
            # -strand (5' right): -shift
            # A simple approximation: shift each 5' end independently
            # if not r1.is_reverse:
            #     p1_shifted = p1 + args.tn5_shift
            # else:
            #     p1_shifted = p1 - args.tn5_shift

            # if not r2.is_reverse:
            #     p2_shifted = p2 + args.tn5_shift
            # else:
            #     p2_shifted = p2 - args.tn5_shift

            frag_start = min(p1, p2)
            frag_end = max(p1, p2) + 1  # end is exclusive

            out.write(f"{chrom}\t{frag_start}\t{frag_end}\t{cb}\n")
            # Reset prev_read to None, because we consumed a pair
            prev_read = None

        bam.close()

if __name__ == '__main__':
    main()