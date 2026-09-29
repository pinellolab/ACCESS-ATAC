# ACCESS-ATAC

Analysis code for **ACCESS-ATAC**, an assay that measures chromatin accessibility two ways
from a single library:

- **ATAC signal** — Tn5 insertion (cut) sites, derived from fragment ends with a +4 / −5
  offset.
- **ACCESS signal** — per-base **DddA-family cytosine deaminase edits**, read off alignments
  as `C→T` (forward strand) and `G→A` (reverse strand) mismatches against the reference.

This repository contains the code used to generate both signal tracks and to compare them
against each other and against conventional ATAC-seq.

## Contents

| Directory | Purpose |
|---|---|
| [`01_process_access/`](01_process_access/) | Complete analysis pipeline for concurrent ACCESS-ATAC in HepG2 and K562, from raw BAM to TF footprint quantification. Start here to reproduce the analysis. |
| [`02_tfbs_prediction/`](02_tfbs_prediction/) | Benchmark of TF binding site prediction from ACCESS-ATAC signal versus sequence alone. Builds the training data, trains the classifier (bundled in `02_tfbs_prediction/model/`), and evaluates it. |
| [`preprocessing/`](#preprocessing) | Convert BAM and fragment files into bigWig signal tracks. |
| [`single_cell/`](#single_cell) | Cell-barcode correction and single-cell fragment generation. |
| [`plotting/`](#plotting) | Aggregate signal profiles over region sets. |

## Reproducing the analysis

The numbered directories contain the analyses as SLURM scripts and notebooks, each with
its own README documenting every step's inputs, outputs and parameters:
[`01_process_access/`](01_process_access/README.md) generates the signal tracks and
compares the two assays, and [`02_tfbs_prediction/`](02_tfbs_prediction/README.md) uses
those tracks to benchmark TF binding site prediction. The Python modules below are the
components those pipelines call, and can also be used independently.

All Python scripts are `argparse` command-line tools. They **import sibling modules by bare
module name** (`from utils import ...`), which resolves against the directory the script
itself lives in — so the module directories are run from inside:

```bash
cd preprocessing
python fragment_to_bw_atac.py \
    --input_fragments fragments.tsv.gz \
    --chrom_size_file hg38.chrom.sizes \
    --out_dir out --out_name sample_atac
```

Shared conventions: paired `--out_dir` / `--out_name` arguments produce
`{out_dir}/{out_name}.{ext}`; coordinates are 0-based half-open (BED convention); output
directories are **not** created automatically.

### Dependencies

Python: `numpy`, `scipy`, `pandas`, `polars`, `pyranges`, `pysam`, `pyBigWig`, `pyfaidx`,
`pybedtools`, `numba`, `torch`, `scikit-learn`, `matplotlib`, `seaborn`, `biopython`,
`tqdm`.

External tools: `samtools`, `bedtools`, `deeptools`, `MACS2`, UCSC `wigToBigWig` and
`bedGraphToBigWig`, and [RGT](https://reg-gen.readthedocs.io/) for motif matching. The
pipeline scripts also invoke `deamtools`, a separate command-line tool that is not part of
this repository.

Training the TFBS classifier (`02_tfbs_prediction/03_train.sh`) requires a CUDA device.

---

## File reference

### preprocessing

BAM and fragment files to bigWig signal tracks.

| File | Description |
|---|---|
| `fragment_to_bw_atac.py` | Fragment file → **Tn5 cut-site** bigWig. Reads columns 1–3, shifts the fragment start by `--forward_shift` (default 4) and the end by `−--reverse_shift` (default 5), then piles up both fragment ends as cut sites. `--extend_size` widens each cut site into a window; `--normalize` scales to RPM; `--bed_file` restricts to given regions, otherwise `--chrom_size_file` defines the whole genome. |
| `fragment_to_bw_access.py` | Fragment file → **deaminase edit** bigWig. Reads a 6-column fragment file and piles up the `\|`-separated edit positions in column 6 (rows with an empty or `none` field are skipped). Same `--extend_size`, `--normalize`, `--bed_file` / `--chrom_size_file` options as above. |
| `bam_to_bw_access.py` | BAM → edit signal bigWig, without an intermediate fragment file. `--out_type count` writes raw edit counts per base; `--out_type fraction` (default) writes edits ÷ per-base coverage, with positions below `--min_coverage` set to zero. Writes a WIG and converts it with `wigToBigWig`. |
| `bam_to_fragments_atac.py` | Coordinate-sorted BAM → fragment file. Keeps the forward read of each pair and applies the Tn5 offsets `--shift_plus` / `--shift_minus`. Emits `chrom start end`, or `chrom start end barcode 1` when `--bc_tag` is given. Adapted from [kundajelab/ENCODE_scatac](https://github.com/kundajelab/ENCODE_scatac/blob/master/workflow/scripts/bam_to_fragments.py). |
| `fragment_to_bam.py` | Fragment file → BAM. **Unfinished** — reads `args.input_fragments` while the flag is `--input_fragment`, and references undefined variables. |
| `utils.py` | Helpers imported by the scripts above: `read_chrom_sizes()` parses a chrom.sizes TSV into a dict; `get_chrom_size_from_bam()` builds a PyRanges of chromosome extents from a BAM header. |

Both `fragment_to_bw_*.py` scripts share the same implementation: a numba-compiled
per-base depth array (`calculate_depth`) followed by run-length encoding
(`collapse_consecutive_values`) before writing bigWig intervals.

### single_cell

Cell-barcode correction, barcode attachment, and fragment generation.

| File | Description |
|---|---|
| `barcode_correction.py` | 10x-style barcode correction. Takes one or more barcode FASTQs (positional) plus a whitelist (positional), and writes a deduplicated `original<TAB>corrected` mapping to `-o`. Pass 1 counts exact whitelist matches as a prior; pass 2 scores each distinct observed barcode's Hamming-distance-1 whitelist neighbours by `(prior + 1) × Phred error probability at the mismatch position`, accepting the best candidate when its posterior reaches `--prob-threshold` (default 0.9), otherwise leaving the barcode unchanged. |
| `add_barcode_to_fastq.py` | Appends `_<barcode>` to each read name in `--reads_fastq`, taking barcodes from the matching records of `--barcodes_fastq` and optionally remapping them through `--corrected_barcodes`. Aborts if read names disagree between the two files. Writes gzipped FASTQ. |
| `add_barcode_to_fastq_v2.py` | Same operation implemented with Biopython `SeqIO`; does not accept a correction map. |
| `add_barcode_to_bam.py` | Recovers the barcode from the read name (the last `_`-separated field, as written by `add_barcode_to_fastq.py`) and stores it in the `--bc_tag` tag (default `CB`). |
| `add_barcode_to_bam_v2.py` | Reads barcodes directly from `--barcode_file` (a FASTQ keyed by read name), optionally applies the `--corrected_barcode` mapping, and writes the `--bc_tag` tag. Avoids the read-name round trip. |
| `add_barcode.py` | Sets `--bc_tag` from `--barcode_file`, a gzipped CSV with `Identifier` and `Barcode` columns keyed by read name. |
| `bam_to_fragments.py` | **Name-sorted** BAM → fragment file. Pairs mates by consecutive `query_name` and keeps pairs passing 10x-style filters (MAPQ ≥ `--min-mapq` on both mates, properly paired, primary only, no `SA` tag, same contig, matching `--cell-tag`). Fragment boundaries use the Tn5 offsets +4 / −5. With `--add-access`, walks each pair's aligned positions against `--ref-fasta` and records C→T and G→A edit positions falling inside the fragment. |
| `aggregate_replicates.py` | Collapses PCR replicates sharing `(chrom, start, end, barcode)`, summing the replicate-count column, and keeps only edit positions supported by at least half the total replicate count. Handles a single edit column. |
| `aggregate_replicates_v2.py` | Same, for fragment files carrying **separate C→T and G→A columns** (the `bam_to_fragments.py --add-access` layout). |
| `filter_bam_by_barcode.py` | Keeps only reads whose `--bc_tag` value appears in the `barcode` column of `--barcode_file` (CSV). |
| `get_edit_ratio_per_barcode.py` | Per barcode, counts reads with and without any reference mismatch; writes a CSV of edited, non-edited and total read counts. |
| `check_fastq_quality.py` | Counts FASTQ records whose sequence and quality strings differ in length; writes a two-column summary. |
| `call_peaks.py` | **Misnamed** — byte-identical to `add_barcode.py`, and does not call peaks. Peak calling is done with MACS2 in `01_process_access/10_call_peaks.sh`. |
| `subsample_fragment.py` | Intended to subsample fragments per barcode. **Unfinished** — pass 1 counts fragments per barcode, pass 2 is not implemented. |

#### Fragment file formats

Column layouts vary by producer; check the column count before parsing. Edit positions are
`|`-joined integers in **reference coordinates** (not fragment offsets), empty or `.` when
absent.

| Columns | Layout | Written by |
|---|---|---|
| 4 | `chrom start end barcode` | `bam_to_fragments.py` |
| 6 | `chrom start end barcode c2t_edits g2a_edits` | `bam_to_fragments.py --add-access` |
| 6 | `chrom start end barcode n_replicates edits` | `aggregate_replicates.py` |
| 7 | `chrom start end barcode n_replicates c2t_edits g2a_edits` | `aggregate_replicates_v2.py` |

### plotting

| File | Description |
|---|---|
| `plot_all_footprint.py` | Averages bigWig signal over a BED region set, recentring each region on its midpoint ± `--extend`. Accepts comma-separated `--bw_files` and `--labels`, and writes a tidy CSV with `position`, `signal` and `data` columns. Used by `01_process_access/15_plot_footprint.sh`. The plotting block is commented out; only the CSV is written. |
