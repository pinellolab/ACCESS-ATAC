import warnings
warnings.filterwarnings("ignore")

import logging
import pyranges as pr
import argparse
import numpy as np
import polars as pl
import pyBigWig
import numba

from utils import read_chrom_sizes

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

def parse_args():
    parser = argparse.ArgumentParser(
        description="This script generates BigWig file from a fragment.tsv.gz file",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # Required parameters
    parser.add_argument("--input_fragments", type=str, default=None)
    parser.add_argument("--bed_file", type=str, default=None)
    parser.add_argument("--extend_size", type=int, default=0)
    parser.add_argument("--skip_rows", type=int, default=0)
    parser.add_argument("--normalize", type=bool, default=False)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--out_name", type=str, default=None)
    parser.add_argument("--chrom_size_file", type=str, default=None)

    return parser.parse_args()


@numba.njit
def calculate_depth(chrom_size: int, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """
    Calculate genome depth for a given chromosome.

    This function computes the depth (coverage) at each base pair of a chromosome based on
    start and end positions of genomic fragments.

    Parameters
    ----------
    chrom_size :
        The size of the chromosome (total number of base pairs).
    starts :
        An array of start positions for the genomic fragments.
        Each value specifies the zero-based position where a fragment begins.
    ends :
        An array of end positions for the genomic fragments.
        Each value specifies the zero-based position where a fragment ends (exclusive).

    Returns
    -------
        A one-dimensional array of length `chrom_size`, where each position contains the
        depth (coverage) at that base pair.

    Notes
    -----
        - The `starts` and `ends` arrays must have the same length, as each pair defines a single fragment.
        - The depth is calculated as the count of overlapping fragments for each base pair.
        - This function uses Numba's Just-In-Time (JIT) compilation to optimize performance, making it suitable for processing large datasets.

    Examples
    --------
    >>> import numpy as np
    >>> import cell2net as cn
    >>> chrom_size = 10
    >>> starts = np.array([0, 2, 4])
    >>> ends = np.array([3, 6, 8])
    >>> depth = cn.pp.calculate_depth(chrom_size, starts, ends)
    >>> print(depth)
    array([1, 1, 2, 1, 2, 2, 1, 1, 0, 0], dtype=uint32)
    """
    # Initialize array for current chromosome to store the depth per basepair.
    chrom_depth = np.zeros(chrom_size, dtype=np.uint32)

    # Require same number of start and end positions.
    assert starts.shape[0] == ends.shape[0]

    for i in range(starts.shape[0]):
        # Add 1 depth for each basepair in the current fragment.
        chrom_depth[starts[i] : ends[i]] += numba.uint32(1)  # type: ignore

    return chrom_depth


@numba.njit
def collapse_consecutive_values(
    X: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Collapse consecutive identical values in an array.

    This function identifies segments of consecutive identical values in an input
    array and returns their start indices, unique values, and lengths of each segment.

    Parameters
    ----------
    X :
        A 1D array of values (integers or floats) to process.

    Returns
    -------
        A tuple containing:

            - idx (numpy.ndarray): Start indices of each segment of consecutive identical values.
            - values (numpy.ndarray): The unique values corresponding to each segment.
            - lengths (numpy.ndarray): The lengths (number of repetitions) of each segment.

    Notes
    -----
        - This function is optimized for performance using `numba.prange` for parallel processing of the input array.
        - The output `idx` array contains the indices where each segment starts in `X`.
        - The `values` array contains the unique values from the input array, and the `lengths` array contains the counts of consecutive occurrences of each value.
        - To reconstruct the original input, use `values.repeat(lengths)`.

    Examples
    --------
    >>> import numpy as np
    >>> import cell2net as cn
    >>> X = np.array([1, 1, 2, 2, 2, 3, 1, 1])
    >>> idx, values, lengths = cn.pp.collapse_consecutive_values(X)
    >>> print(idx)
    ... [0 2 5 6]
    >>> print(values)
    ... [1. 2. 3. 1.]
    >>> print(lengths)
    ... [2 3 1 2]
    >>> np.array_equal(X, values.repeat(lengths))
    ... True
    """
    # Length.
    n = X.shape[0]

    # Create idx array with enough space to store all possible indices
    # in case there are no consecutive values in the input that are
    # the same.
    idx = np.empty(n + 1, dtype=np.uint32)

    # First index position will always be zero.
    idx[0] = 0
    # Next index postion to fill in in idx.
    j = 1

    # Loop over the whole input array and store indices for only those
    # positions for which the previous value was different.
    for i in numba.prange(1, n):
        if X[i - 1] == X[i]:
            continue

        # Current value is different from previous value, so store the index
        # position of the current index i in idx[j].
        idx[j] = i

        # Increase next index position to fill in in idx.
        j += 1

    # Store length of input as last value. Needed to calculate the number of times
    # the last consecutive value gets repeated.
    idx[j] = n

    # Get all consecutive different values from the input.
    values = X[idx[:j]].astype(np.float32)

    # Calculate the number of times each value in values gets consecutively
    # repeated in the input.
    lengths = idx[1 : j + 1] - idx[:j]

    # Restrict indices array to same number of element than the values and lentgths
    # arrays.
    idx = idx[:j].copy()

    # To reconstruct the original input: X == values.repeat(lenghts)
    return idx, values, lengths


def fragments_to_coverage(
    df_fragments: pl.DataFrame,
    chrom_sizes: dict[str, int],
    normalize: bool = False,
    scaling_factor: float = 1.0,
    extend_cut_sites: int = 0,
):
    """
    Convert fragment data to genome coverage signal.

    This function processes fragment data and generates genome coverage or cut-site
    signal, which can be used for creating BigWig files or similar outputs.

    Parameters
    ----------
    df_fragments :
        A Polars DataFrame containing fragment data. Must include the columns:
        'Chromosome', 'Start', and 'End'.
    chrom_sizes:
        Dictionary mapping chromosome names to their respective sizes.
    normalize:
        If True, normalize the coverage values to Reads Per Million (RPM). Default is True.
    scaling_factor :
        A scaling factor to apply to the signal values. Only used if `normalize` is True.
        Default is 1.0.
    cut_sites:
        Use 1 bp Tn5 cut sites (start and end of each fragment) instead of whole
        fragment length for coverage calculation.
    extend_cut_sites:
        If set cut_sites, expand cut sites for both upstream and downstream, by default: 0

    Yields
    ------
    A tuple containing:

            - chroms (numpy.ndarray): Chromosome names for each coverage interval.
            - starts (numpy.ndarray): Start positions of coverage intervals.
            - ends (numpy.ndarray): End positions of coverage intervals.
            - values (numpy.ndarray): Signal values for each coverage interval.

    Notes
    -----
        - The `df_fragments` DataFrame is partitioned by chromosome for efficient processing.
        - The `chrom_sizes` dictionary defines the size of each chromosome and is used to initialize arrays.
        - If `cut_sites` is True, the coverage is computed at the fragment boundaries rather than the entire fragment range.
        - Normalization scales the signal to RPM, and an additional scaling factor can further adjust the signal values.
    """
    chrom_arrays = {}

    for chrom, chrom_size in chrom_sizes.items():
        chrom_arrays[chrom] = np.zeros(chrom_size, dtype=np.uint32)

    n_fragments = 0

    logging.info("Split fragments by chromosome")
    per_chrom_fragments_dfs = {
        str(chrom): fragments_chrom_df_pl
        for (chrom,), fragments_chrom_df_pl in df_fragments.partition_by(
            ["chrom"],
            as_dict=True,
        ).items()
    }

    logging.info("Calculate depth per chromosome:")
    for chrom in per_chrom_fragments_dfs:
        if chrom not in chrom_sizes:
            logging.warning(f"Skipping {chrom} as it is not in chrom sizes file.")
            continue

        edit_positions = per_chrom_fragments_dfs[chrom]["edit_positions"].to_list()
        # edit_positions = edit_positions[:10]  # for testing
        starts, ends = [], []
        for edit_position in edit_positions:
            for pos in edit_position.split("|"):
                pos = int(pos)
                starts.append(pos - extend_cut_sites)
                ends.append(pos + extend_cut_sites + 1)

        starts = np.array(starts)
        ends = np.array(ends)

        chrom_arrays[chrom] = calculate_depth(chrom_sizes[chrom], starts, ends)
        n_fragments += per_chrom_fragments_dfs[chrom].height

    # Calculate RPM scaling factor.
    rpm_scaling_factor = n_fragments / 1_000_000.0
    logging.info("Compact depth array per chromosome (make ranges for consecutive the same values and remove zeros):")
    for chrom in chrom_sizes:
        idx, values, lengths = collapse_consecutive_values(chrom_arrays[chrom])
        non_zero_idx = np.flatnonzero(values)

        if non_zero_idx.shape[0] == 0:
            # Skip chromosomes with no depth > 0.
            continue

        # Select only consecutive different values and calculate start and end
        # coordinates (in BED format) for each of those ranges.
        chroms = np.repeat(chrom, len(non_zero_idx))
        starts = idx[non_zero_idx]
        ends = idx[non_zero_idx] + lengths[non_zero_idx]
        values = values[non_zero_idx]

        if normalize:
            values = values / rpm_scaling_factor * scaling_factor
        elif scaling_factor != 1.0:
            values *= scaling_factor

        yield chroms, starts, ends, values


def main():
    args = parse_args()

    if args.bed_file:
        logging.info(f"Loading genomic regions from {args.bed_file}")
        grs = pr.read_bed(args.bed_file)
        grs = grs.merge()
    else:
        logging.info(f"Using whole genome")
        logging.info(f"Loading chromosome sizes from {args.chrom_size_file}")
        chrom_sizes = read_chrom_sizes(args.chrom_size_file)
        chromosome = list(chrom_sizes.keys())
        start = [0] * len(chromosome)
        end = list(chrom_sizes.values())
        grs = pr.from_dict({"Chromosome": chromosome, "Start": start, "End": end})

    logging.info(f"Total of {len(grs)} regions")

    logging.info(f"Reading fragments from {args.input_fragments}")
    df_fragments = pl.read_csv(
        args.input_fragments,
        skip_rows=args.skip_rows,
        has_header=False,
        separator="\t",
        use_pyarrow=False,
    )

    df_fragments = df_fragments.rename(
        {
            df_fragments.columns[0]: "chrom",
            df_fragments.columns[1]: "start",
            df_fragments.columns[2]: "end",
            df_fragments.columns[3]: "edit_positions",
        }
    )

    # remove rows then edit_positions is empty
    df_fragments = df_fragments.filter(pl.col("edit_positions").is_not_null())

    # subset df_fragments to have the same Chromosome as in grs
    gr_chroms = set(grs.Chromosome.tolist())
    df_fragments = df_fragments.filter(pl.col("chrom").is_in(gr_chroms))

    logging.info(f"Number of fragments: {df_fragments.height}")
    logging.info(f"Generating bigwig file in {args.out_dir}")

    if args.normalize:
        logging.info("Normalizing to RPM")
    else:
        logging.info("No normalization. Using raw counts.")

    bw_filename = f"{args.out_dir}/{args.out_name}.bw"

    with pyBigWig.open(bw_filename, "wb") as bw:
        bw.addHeader(list(chrom_sizes.items()))

        fragments_to_coverage_chrom_iter = fragments_to_coverage(
            df_fragments=df_fragments,
            chrom_sizes=chrom_sizes,
            normalize=args.normalize,
            scaling_factor=1.0,
            extend_cut_sites=args.extend_size,
        )

        for chroms, starts, ends, values in fragments_to_coverage_chrom_iter:
            bw.addEntries(chroms=chroms, starts=starts, ends=ends, values=values)

    logging.info("Done!")

if __name__ == "__main__":
    main()
