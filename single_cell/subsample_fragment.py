import argparse
import gzip
from typing import Tuple, Iterator, TextIO, Dict
import math

def parse_args():
    parser = argparse.ArgumentParser(description="Generate fragments from name-sorted BAM")
    parser.add_argument("--input", "-i", required=True, help="Input fragments file (tsv or tsv.gz)")
    parser.add_argument("--output", "-o", required=True, help="Output fragments file (tsv or tsv.gz)")
    parser.add_argument("--frac", type=float, default=0.2, help="Fraction of reads to keep (default: 0.2)")
    return parser.parse_args()

def open_maybe_gzip(path: str, mode: str) -> TextIO:
    # mode: "rt" or "wt"
    if path.endswith(".gz"):
        return gzip.open(path, mode)
    return open(path, mode)

def open_output(path):
    """
    Open output as text handle; use gzip if filename ends with .gz.
    """
    if path.endswith(".gz"):
        return gzip.open(path, "wt")
    else:
        return open(path, "w")


def parse_line_5col(line: str) -> Tuple[str, int, int, str, int]:
    # chrom start end barcode count [optional extra columns...]
    parts = line.rstrip("\n").split("\t")
    if len(parts) < 5:
        raise ValueError("Line has <5 columns")
    chrom = parts[0]
    start = int(parts[1])
    end = int(parts[2])
    barcode = parts[3]
    count = int(parts[4])
    return chrom, start, end, barcode, count

def iter_frag_rows(in_path: str) -> Iterator[Tuple[int, str, int, str, int, int]]:
    """
    Yield (lineno_1based, chrom, start, end, barcode, count)
    Skips blank/malformed lines.
    """
    with open_maybe_gzip(in_path, "rt") as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                chrom, start, end, barcode, count = parse_line_5col(line)
            except Exception:
                continue
            if count <= 0:
                continue
            yield lineno, chrom, start, end, barcode, count

def compute_k(total: int, fraction: float, rounding: str) -> int:
    x = total * fraction
    if rounding == "floor":
        return int(math.floor(x))
    if rounding == "ceil":
        return int(math.ceil(x))
    # "round" with +0.5 then floor (avoids banker's rounding)
    return int(math.floor(x + 0.5))


def main():
    args = parse_args()

    # -------- Pass 1: total fragment count per barcode (sum of column 5) --------
    totals: Dict[str, int] = {}
    for _lineno, _chrom, _start, _end, bc, cnt in iter_frag_rows(args.input):
        totals[bc] = totals.get(bc, 0) + 1

    # -------- Pass 2: subsample fragments and write output --------


if __name__ == '__main__':
    main()