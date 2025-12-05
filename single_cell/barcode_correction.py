#!/usr/bin/env python3
"""
Barcode correction for 10x-style barcodes (scATAC/i5 index),
producing a *deduplicated* mapping:

  original_barcode <TAB> corrected_barcode

Inputs:
  - One or more FASTQ files: each read's sequence is the observed barcode (with quality).
  - One whitelist file: text (or .gz) with one valid barcode per line.

Logic:

  Pass 1 (all FASTQs):
    - Count exact matches of observed barcodes to whitelist
      → prior abundance for each valid barcode (global).

  Pass 2 (all FASTQs):
    - For each *distinct* observed barcode sequence (original):
        * On its first appearance, use that read's quality string to decide correction:
            - if original in whitelist → corrected = original
            - else:
                - find all whitelist barcodes at Hamming distance 1
                - for each candidate:
                    score ~ (prior_count + 1) * error_prob_at_mismatch_position
                - convert scores to probabilities
                - if best candidate prob > prob_threshold (default 0.9):
                      corrected = best_candidate
                  else:
                      corrected = original (no correction)

  Output:
    - For each unique original barcode observed across all FASTQs,
      write exactly one line:
        original_barcode <TAB> corrected_barcode

Notes:
  - Invalid barcodes that cannot be confidently corrected will have
    corrected_barcode == original_barcode.
"""

import argparse
import gzip
from collections import Counter
from typing import Set, List, Tuple, TextIO, Dict
import sys

import logging
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

def open_maybe_gz(path: str):
    """Open a plain or gzipped text file in text mode."""
    if path.endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "rt")


def load_whitelist(path: str) -> Set[str]:
    wl = set()
    with open_maybe_gz(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            wl.add(line)
    if not wl:
        raise ValueError("Whitelist is empty")
    return wl

def neighbors_hamming1(seq: str) -> List[Tuple[str, int]]:
    """
    Generate all sequences at Hamming distance 1 from `seq`.

    Returns:
      list of (neighbor_sequence, mismatch_pos)
    """
    bases = ("A", "C", "G", "T")
    out: List[Tuple[str, int]] = []
    s = list(seq)
    for i, orig in enumerate(s):
        for b in bases:
            if b == orig:
                continue
            s[i] = b
            out.append(("".join(s), i))
        s[i] = orig
    return out


def phred_error_prob(q_char: str) -> float:
    """Convert a FASTQ quality character to error probability."""
    q = ord(q_char) - 33  # Sanger encoding
    return 10.0 ** (-q / 10.0)


def pass1_count_exact(fastq_paths: List[str], whitelist: Set[str]) -> Counter:
    """
    First pass across ALL fastqs:
      count exact matches of observed barcodes to whitelist.
    """
    counts = Counter()
    for path in fastq_paths:
        logging.info(f"Pass1: scanning {path}...")
        with open_maybe_gz(path) as f:
            while True:
                header = f.readline()
                if not header:
                    break
                seq = f.readline().strip()
                plus = f.readline()
                qual = f.readline().strip()
                if not qual:
                    break
                if seq in whitelist:
                    counts[seq] += 1
    return counts


def decide_correction_for_one_barcode(
    original: str,
    qual: str,
    whitelist: Set[str],
    prior_counts: Counter,
    prob_threshold: float,
) -> str:
    """
    Given a single observed barcode sequence and its quality string,
    decide corrected barcode according to 10x-style model.
    """
    # Case 1: already a valid barcode
    if original in whitelist:
        return original

    # Case 2: try to correct invalid barcode
    candidates: List[Tuple[str, int]] = []
    for nb, pos in neighbors_hamming1(original):
        if nb in whitelist:
            candidates.append((nb, pos))

    if not candidates:
        # no valid neighbor at distance 1: cannot correct
        return original

    # Score candidates: score ~ (prior_count + 1) * error_prob_at_mismatch
    scores = []
    for bc, pos in candidates:
        prior = prior_counts.get(bc, 0) + 1  # add-1 smoothing
        err_p = phred_error_prob(qual[pos])
        score = prior * err_p
        scores.append((bc, score))

    total_score = sum(s for _, s in scores)
    if total_score <= 0:
        return original

    best_bc, best_score = max(scores, key=lambda x: x[1])
    prob_best = best_score / total_score

    if prob_best >= prob_threshold:
        return best_bc
    else:
        return original


def pass2_build_unique_mapping(
    fastq_paths: List[str],
    whitelist: Set[str],
    prior_counts: Counter,
    prob_threshold: float,
) -> Dict[str, str]:
    """
    Second pass across all FASTQs:
      build a mapping dict:
        original_barcode -> corrected_barcode
      with exactly one entry per unique original barcode.
    """
    mapping: Dict[str, str] = {}

    for path in fastq_paths:
        sys.stderr.write(f"[INFO] Pass2: processing {path}...\n")
        with open_maybe_gz(path) as f:
            while True:
                header = f.readline()
                if not header:
                    break
                seq = f.readline().strip()
                plus = f.readline()
                qual = f.readline().strip()
                if not qual:
                    break

                original = seq
                if original in mapping:
                    # already decided for this barcode, skip
                    continue

                corrected = decide_correction_for_one_barcode(
                    original,
                    qual,
                    whitelist,
                    prior_counts,
                    prob_threshold,
                )
                mapping[original] = corrected

    return mapping


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Deduplicated 10x-style barcode correction for one or more FASTQs. "
            "Output: unique lines of 'original_barcode<TAB>corrected_barcode'."
        )
    )
    ap.add_argument(
        "fastqs",
        nargs="+",
        help="One or more FASTQ files where the read sequence is the barcode "
             "(supports .gz).",
    )
    ap.add_argument(
        "whitelist",
        help="Text file (.txt or .gz) with one valid barcode per line.",
    )
    ap.add_argument(
        "-o",
        "--output",
        default="-",
        help="Output mapping file: 'original\\tcorrected'. "
             "Use '-' for stdout (default).",
    )
    ap.add_argument(
        "--prob-threshold",
        type=float,
        default=0.9,
        help="Minimum posterior probability required to accept a correction "
             "(default: 0.9).",
    )
    args = ap.parse_args()

    whitelist = load_whitelist(args.whitelist)
    lengths = {len(bc) for bc in whitelist}
    if len(lengths) != 1:
        raise ValueError("Whitelist barcodes must all have the same length.")
    bc_len = lengths.pop()
    logging.info(
        f"Loaded {len(whitelist)} whitelist barcodes (length={bc_len})."
    )

    # Pass 1: global prior across ALL fastqs
    logging.info("Pass1: counting exact matches across all FASTQs...")
    prior_counts = pass1_count_exact(args.fastqs, whitelist)
    logging.info(
        f"Non-zero exact-match barcodes: {len(prior_counts)}."
    )

    # Pass 2: build unique mapping
    logging.info("Pass2: building unique original->corrected mapping...")
    mapping = pass2_build_unique_mapping(
        args.fastqs,
        whitelist,
        prior_counts,
        prob_threshold=args.prob_threshold,
    )
    logging.info(
        f"Unique observed barcodes: {len(mapping)} (one line per barcode)."
    )

    total_unique = len(mapping)
    corrected_count = 0
    for original, corrected in mapping.items():
        if original != corrected and (corrected in whitelist) and (original not in whitelist):
            corrected_count += 1

    frac = corrected_count / total_unique if total_unique > 0 else 0.0
    logging.info(
        f"Correctable barcodes (invalid -> valid, changed): "
        f"{corrected_count} ({frac:.3%} of unique observed)."
    )

    # Output
    with open(args.output, "w") as out_f:
        for original, corrected in mapping.items():
            out_f.write(f"{original}\t{corrected}\n")

    logging.info("Done.")


if __name__ == "__main__":
    main()