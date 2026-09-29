import argparse
from dataclasses import dataclass, field
import gzip
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple
import csv
import logging

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

def parse_args():
    parser = argparse.ArgumentParser(description="Generate fragments from name-sorted BAM")
    parser.add_argument("--input", "-i", required=True, help="Input BAM file (name-sorted)")
    parser.add_argument("--output", "-o", required=True, help="Output fragments file (tsv or tsv.gz)")
    return parser.parse_args()

def _open_maybe_gzip(path: str):
    p = Path(path)
    if p.suffix == ".gz":
        return gzip.open(p, "rt", newline="")  # text mode  [oai_citation:3‡Python documentation](https://docs.python.org/3/library/gzip.html?utm_source=chatgpt.com)
    return open(p, "rt", newline="")

def _parse_edit_positions(field: str) -> Set[int]:
    """
    field example: "3050099|3050114|3050130" or "" or "."
    """
    field = field.strip()
    if not field or field == ".":
        return set()
    return {int(x) for x in field.split("|") if x}

def open_output(path):
    """
    Open output as text handle; use gzip if filename ends with .gz.
    """
    if path.endswith(".gz"):
        return gzip.open(path, "wt")
    else:
        return open(path, "w")

@dataclass
class AggRecord:
    chrom: str
    start: int
    end: int
    barcode: str
    total_reps: int = 0
    edit_counts: Dict[int, int] = field(default_factory=dict)

def main():
    args = parse_args()

    agg: Dict[Tuple[str, int, int, str], AggRecord] = {}
    with _open_maybe_gzip(args.input) as f:
        reader = csv.reader(f, delimiter="\t")
        for row_i, row in enumerate(reader, start=1):
            if not row or len(row) < 6:
                # skip blank/malformed lines
                logging.warning(f"Skipping malformed line {row_i}: {row}")
                continue

            chrom = row[0]
            start = int(row[1])
            end = int(row[2])
            cell_barcode = row[3]
            count = int(row[4])
            edit_field = row[5]
            edit_positions = _parse_edit_positions(edit_field)

            key = (chrom, start, end, cell_barcode)
            rec = agg.get(key)

            if rec is None:
                rec = AggRecord(chrom=chrom, start=start, end=end, 
                                barcode=cell_barcode)
                agg[key] = rec

            rec.total_reps += count
            # update per-edit weighted counts
            for pos in edit_positions:
                rec.edit_counts[pos] = rec.edit_counts.get(pos, 0) + count

    # materialize output
    with open_output(args.output) as out:
        for rec in agg.values():
            threshold = rec.total_reps / 2.0
            kept = [pos for pos, c in rec.edit_counts.items() if c >= threshold]
            kept.sort()
            edits_str = "|".join(map(str, kept)) if kept else "\t"
            out.write(f"{rec.chrom}\t{rec.start}\t{rec.end}\t{rec.barcode}\t{rec.total_reps}\t{edits_str}\n")

if __name__ == '__main__':
    main()