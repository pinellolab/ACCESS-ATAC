import os
import random
import numpy as np
import torch
import pyranges as pr
import pyfaidx


def set_seed(seed=42):
    """
    Set random seed
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def one_hot_encode(seq):
    """
    Given a DNA sequence, return its one-hot encoding
    """
    # Make sure seq has only allowed bases
    allowed = set("ACTGN")
    if not set(seq).issubset(allowed):
        invalid = set(seq) - allowed
        raise ValueError(
            f"Sequence contains chars not in allowed DNA alphabet (ACGTN): {invalid}"
        )

    # Dictionary returning one-hot encoding for each nucleotide
    nuc_d = {
        "A": [1.0, 0.0, 0.0, 0.0],
        "C": [0.0, 1.0, 0.0, 0.0],
        "G": [0.0, 0.0, 1.0, 0.0],
        "T": [0.0, 0.0, 0.0, 1.0],
        "N": [0.0, 0.0, 0.0, 1.0],
    }

    # Create array from nucleotide sequence
    vec = np.array([nuc_d[x] for x in seq], dtype=np.float32)

    return vec


def pad_and_split(grs: pr.PyRanges = None, fasta_file: str = None, k: int = 1024):
    """
    This function pad the input regions

    Parameters
    ----------
    grs : _type_
        _description_
    k : int, optional
        _description_, by default 1024
    """
    chroms, starts, ends = [], [], []

    for chrom, start, end in zip(grs.Chromosome, grs.Start, grs.End):
        # pad the regions
        n = (end - start) // k
        mid = (start + end) // 2
        start_new = mid - (n + 1) / 2 * k

        # split the regions
        for i in range(n + 1):
            chroms.append(chrom)
            starts.append(start_new + i * k)
            ends.append(start_new + (i + 1) * k)

    grs = pr.from_dict({"Chromosome": chroms, "Start": starts, "End": ends})

    # remove regions that are out-of-bounds of reference genome
    grs = grs[grs.Start>=0]
    pyf = pyfaidx.Fasta(fasta_file)
    grs = pr.genomicfeatures.genome_bounds(grs, chromsizes=pyf, clip=False)
    
    return grs


# Generates a random sequence containing A/C/G/T of length n
def random_seq(n):
    bases = ["A", "C", "G", "T"]
    rand_seq = "".join([np.random.choice(bases) for i in range(n)])
    return rand_seq
