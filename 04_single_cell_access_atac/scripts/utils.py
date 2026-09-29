import os
import numpy as np
import pandas as pd
import pysam
import pyranges as pr

def get_chrom_size_from_bam(bam: pysam.Samfile) -> pr.PyRanges:
    """
    Extract chromsome size from the input bam file

    Parameters
    ----------
    bam : pysam.Samfile
        Input bam file

    Returns
    -------
    pr.PyRanges
        A PyRanges object containing chromosome size. Note this is 0-based
    """

    chromosome = list(bam.references)
    start = [0] * len(chromosome)
    end = list(bam.lengths)
    end = [x - 1 for x in end]

    grs = pr.from_dict({"Chromosome": chromosome, "Start": start, "End": end})
    
    return grs

def read_chrom_sizes(chrom_size_file) -> dict[str, int]:
    chrom_sizes = {}
    with open(chrom_size_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            chrom = parts[0]
            size = int(parts[1])
            chrom_sizes[chrom] = size
    return chrom_sizes