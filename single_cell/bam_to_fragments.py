import argparse
import pysam
import gzip

def parse_args():
    parser = argparse.ArgumentParser(description="Generate fragments from name-sorted BAM")
    parser.add_argument("--bam", "-i", required=True, help="Input BAM file (name-sorted)")
    parser.add_argument("--out", "-o", required=True, help="Output fragments file (tsv or tsv.gz)")
    parser.add_argument("--cell-tag", default="CB", help="Tag name storing cell barcode (default: CB)")
    parser.add_argument("--min-mapq", type=int, default=30,help="Minimum MAPQ for both mates to be kept (default: 30)")
    parser.add_argument("--add-access", action="store_true", 
                        help="Whether to add ACCESS tag (default: False)")
    parser.add_argument("--ref-fasta", help="Reference FASTA file for ACCESS tag computation")
    return parser.parse_args()

def is_good_read_pair(
    r1,
    r2,
    *,
    min_mapq: int = 30,
    barcode_tag: str = "CB",
    require_barcode: bool = True,
    require_proper_pair: bool = True,
    allow_secondary: bool = False,
) -> bool:
    """
    10x-style filters for fragments generation (core idea):
      - MAPQ >= 30 on both reads
      - not mitochondrial
      - not chimerically mapped
      - maps to a primary (gene-containing) contig
    Plus basic sanity checks: paired, mapped, primary alignment.

    Notes:
      - 10x duplicate collapsing is separate; this function is meant to be applied to
        the representative/unique read-pair.
      - 'primary_contigs' should ideally come from your reference; if None, we use heuristics.
    """

    # Must be paired
    if not (r1.is_paired and r2.is_paired):
        return False

    # Both mapped
    if r1.is_unmapped or r2.is_unmapped:
        return False
    if r1.mate_is_unmapped or r2.mate_is_unmapped:
        return False

    # Primary alignment only (10x filters out supplementary/secondary for "high-quality")
    if not allow_secondary:
        if r1.is_secondary or r2.is_secondary:
            return False
        if r1.is_supplementary or r2.is_supplementary:
            return False

    # Must be on same reference/contig (otherwise chimeric)
    if r1.reference_id != r2.reference_id:
        return False

    # MAPQ filter
    if r1.mapping_quality < min_mapq or r2.mapping_quality < min_mapq:
        return False

    # Chimeric filter (practical approximation):
    # - split alignments often carry SA tag even if the record itself is primary
    if r1.has_tag("SA") or r2.has_tag("SA"):
        return False

    # Optional: require proper pair (some pipelines do; 10x's wording focuses on MAPQ/chimeric/contig)
    if require_proper_pair:
        if not (r1.is_proper_pair and r2.is_proper_pair):
            return False

    # Optional: require a corrected barcode tag exists (CB)
    if require_barcode:
        try:
            bc1 = r1.get_tag(barcode_tag)
            bc2 = r2.get_tag(barcode_tag)
        except KeyError:
            return False
        if not bc1 or not bc2:
            return False
        # usually should match for a pair
        if bc1 != bc2:
            return False

    return True

def get_access_signal(r1, r2, fasta) -> list[int]:
    """
    Compute ACCESS signal for a read pair based on DddSs edit sites.
    Return a list of integers representing the ACCESS edit positions within the fragment.
    """
    access_sites = []
    for read in [r1, r2]:
        # refer_seq = read.get_reference_sequence().upper()
        query_seq = read.query_sequence
        pairs = read.get_aligned_pairs(with_seq=True)

        # print(query_seq)
        for query_pos, ref_pos, ref_base in pairs:
            if ref_pos is None or query_pos is None:
                continue

            refer_base = fasta.fetch(read.reference_name, ref_pos, ref_pos + 1).upper()

            # refer_base = refer_base.upper()
            query_base = query_seq[query_pos]

            edit_site = ref_pos  # convert to reference coordinate
            # C -> T at forward strand
            if refer_base == "C" and query_base == "T":
                access_sites.append(edit_site)

            # G -> A at reverse strand
            elif refer_base == "G" and query_base == "A":
                access_sites.append(edit_site)

    # remove duplicates
    access_sites = list(dict.fromkeys(access_sites))
    access_sites.sort()

    return access_sites

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

    if args.add_access:
        fasta = pysam.FastaFile(args.ref_fasta)

    with open_output(args.out) as out:
        prev_read = None
        for read in bam:
            # We assume name-sorted BAM, so mates appear consecutively.
            if prev_read is None:
                prev_read = read
                continue

            # Debugging: process only one read name
            # if read.query_name != "AV242502:multiSeq_Arbab-Sherwood-labs-9:2511508312:1:20103:5052:0307":
            #     prev_read = None
            #     continue

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
            if not is_good_read_pair(r1, r2, min_mapq=args.min_mapq):
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

            s1, e1 = r1.reference_start, r1.reference_end
            s2, e2 = r2.reference_start, r2.reference_end

            frag_start = min(s1, s2) + 4
            frag_end = max(e1, e2) - 5

            if frag_end <= frag_start:
                # Invalid fragment (should not happen with proper pairs)
                prev_read = None
                continue

            # Write fragment line
            if not args.add_access:
                out.write(f"{chrom}\t{frag_start}\t{frag_end}\t{cb}\n")
            else:
                access_signal = get_access_signal(r1, r2, fasta)

                # remove edit sites outside fragment boundaries
                access_signal = [site for site in access_signal if frag_start <= site < frag_end]

                access_signal = "|".join(map(str, access_signal))
                out.write(f"{chrom}\t{frag_start}\t{frag_end}\t{cb}\t{access_signal}\n")
            prev_read = None

        bam.close()

if __name__ == '__main__':
    main()