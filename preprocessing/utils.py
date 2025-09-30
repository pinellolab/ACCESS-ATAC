
def read_chrom_sizes(chrom_size_file) -> dict[str, int]:
    chrom_sizes = {}
    with open(chrom_size_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            chrom = parts[0]
            size = int(parts[1])
            chrom_sizes[chrom] = size
    return chrom_sizes