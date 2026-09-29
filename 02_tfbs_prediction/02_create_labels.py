#!/usr/bin/env python3
"""
Create a TFBS labels file (positives from ChIP-seq summits + GC/chrom-matched
negatives) for one cell type and one TF.

Usage:
    python create_labels.py --cell_type K562 --tf CTCF
"""
import os
import gzip
import argparse
from collections import Counter

import numpy as np
import pyfaidx
import pybedtools


# ---- fixed config (paths & params) ----
IN_DIR      = "../../results/27_compare_tfbs_prediction/01_download_peaks"
OUT_DIR     = "../../results/02_tfbs_prediction/02_create_labels"
FASTA       = "../../data/GRCh38/GRCh38.primary_assembly.genome.fa"
BLACKLIST   = "../../data/Blacklist/hg38-blacklist.v2.bed"
FIXED_WIDTH = 256
GC_BINS     = 20
POOL_FACTOR = 50
EXCLUDE_MARGIN = 0
SEED        = 42


# ---- helpers ----
def open_maybe_gz(path):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path)


def load_summit_centered_peaks(narrowpeak_path, chrom_sizes, fixed_width):
    """narrowPeak -> fixed-width regions centered on the summit (col 10);
    falls back to the interval midpoint if the summit is missing/-1."""
    half = fixed_width // 2
    regions = []
    with open_maybe_gz(narrowpeak_path) as f:
        for line in f:
            if line.startswith(("#", "track", "browser")):
                continue
            fields = line.rstrip("\n").split("\t")
            chrom, start, end = fields[0], int(fields[1]), int(fields[2])
            if len(fields) >= 10 and fields[9] not in (".", "-1"):
                summit = start + int(fields[9])
            else:
                summit = (start + end) // 2
            s = summit - half
            e = summit - half + fixed_width
            if chrom not in chrom_sizes or s < 0 or e > chrom_sizes[chrom]:
                continue
            regions.append((chrom, s, e))
    return regions


def gc_content(genome, chrom, s, e):
    seq = str(genome[chrom][s:e]).upper()
    if not seq:
        return np.nan
    valid = len(seq) - seq.count("N")
    if valid == 0:
        return np.nan
    return (seq.count("G") + seq.count("C")) / valid


def generate_negatives(pos, genome, chrom_sizes, gc_edges, rng):
    """GC/chrom-matched negatives of the same count as positives."""
    pos_gc = np.array([gc_content(genome, c, s, e) for c, s, e in pos])
    ok = ~np.isnan(pos_gc)
    pos = [p for p, m in zip(pos, ok) if m]
    pos_gc = pos_gc[ok]
    pos_bin = np.clip(np.digitize(pos_gc, gc_edges) - 1, 0, GC_BINS - 1)

    pos_bed_str = "\n".join(
        f"{c}\t{max(0, s-EXCLUDE_MARGIN)}\t{e+EXCLUDE_MARGIN}" for c, s, e in pos
    )
    exclude = pybedtools.BedTool(pos_bed_str, from_string=True)
    if BLACKLIST:
        exclude = exclude.cat(pybedtools.BedTool(BLACKLIST), postmerge=True)
    exclude = exclude.sort().merge()
    excl_by_chrom = {}
    for iv in exclude:
        excl_by_chrom.setdefault(iv.chrom, []).append((iv.start, iv.end))

    def overlaps_excluded(chrom, s, e):
        for xs, xe in excl_by_chrom.get(chrom, []):
            if s < xe and e > xs:
                return True
        return False

    chrom_counts = Counter(c for c, _, _ in pos)
    negatives = []
    for chrom, n_needed in chrom_counts.items():
        clen = chrom_sizes[chrom]
        chrom_widths = [e - s for c, s, e in pos if c == chrom]
        pos_bins_chrom = [b for (c, _, _), b in zip(pos, pos_bin) if c == chrom]
        target_hist = Counter(pos_bins_chrom)

        pool_size = max(n_needed * POOL_FACTOR, 1000)
        cand, attempts, max_attempts = [], 0, pool_size * 20
        while len(cand) < pool_size and attempts < max_attempts:
            attempts += 1
            w = chrom_widths[rng.integers(len(chrom_widths))]
            s = int(rng.integers(0, max(1, clen - w)))
            e = s + w
            if overlaps_excluded(chrom, s, e):
                continue
            g = gc_content(genome, chrom, s, e)
            if np.isnan(g):
                continue
            if "N" in str(genome[chrom][s:e]).upper():
                continue
            cand.append((chrom, s, e, g))
        if not cand:
            print(f"  WARNING: no candidates on {chrom}")
            continue

        cand_bin = np.clip(np.digitize([c[3] for c in cand], gc_edges) - 1, 0, GC_BINS - 1)
        for b, cnt in target_hist.items():
            idx = np.where(cand_bin == b)[0]
            if len(idx) == 0:
                idx = np.argsort(np.abs(cand_bin - b))[:cnt]
            take = rng.choice(idx, size=min(cnt, len(idx)), replace=False)
            negatives.extend(cand[i][:3] for i in take)

    negatives.sort(key=lambda x: (x[0], x[1]))
    return pos, negatives


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell_type", required=True)
    ap.add_argument("--tf", required=True)
    ap.add_argument("--fixed_width", type=int, default=FIXED_WIDTH)
    args = ap.parse_args()

    cell_type, tf = args.cell_type, args.tf
    os.makedirs(os.path.join(OUT_DIR, cell_type), exist_ok=True)

    out_file = os.path.join(OUT_DIR, cell_type, f"{tf}.labels.tsv")
    if os.path.exists(out_file):
        print(f"[{cell_type} {tf}] already exists, skipping")
        return

    # input peak file: try .bed then .bed.gz
    in_file = f"{IN_DIR}/{cell_type}/{tf}.bed"
    if not os.path.exists(in_file):
        in_file = f"{IN_DIR}/{cell_type}/{tf}.bed.gz"
    if not os.path.exists(in_file):
        print(f"[{cell_type} {tf}] peak file not found, skipping")
        return

    genome = pyfaidx.Fasta(FASTA)
    chrom_sizes = {name: len(genome[name]) for name in genome.keys()}
    rng = np.random.default_rng(SEED)
    gc_edges = np.linspace(0, 1, GC_BINS + 1)

    pos = load_summit_centered_peaks(in_file, chrom_sizes, args.fixed_width)
    if not pos:
        print(f"[{cell_type} {tf}] no usable peaks, skipping")
        return

    pos, negatives = generate_negatives(pos, genome, chrom_sizes, gc_edges, rng)

    with open(out_file, "w") as f_out:
        f_out.write("chrom\tstart\tend\tlabel\n")
        for c, s, e in pos:
            f_out.write(f"{c}\t{s}\t{e}\t1\n")
        for c, s, e in negatives:
            f_out.write(f"{c}\t{s}\t{e}\t0\n")

    pos_gc = np.nanmean([gc_content(genome, c, s, e) for c, s, e in pos])
    neg_gc = np.nanmean([gc_content(genome, c, s, e) for c, s, e in negatives])
    print(f"[{cell_type} {tf}] pos={len(pos)} neg={len(negatives)} "
          f"GC pos={pos_gc:.3f} neg={neg_gc:.3f} -> {out_file}")


if __name__ == "__main__":
    main()
