# 01_process_access

Compares **ACCESS-ATAC** against conventional ATAC-seq in HepG2 and K562.

ACCESS-ATAC gives two readouts from one library:

- **ATAC** — Tn5 cut sites. Extracted with [`deamtools`](https://github.com/lzj1769/deamTools) `bam2bw --event tn5`. Files tagged `*_atac_*`.
- **ACCESS** — DddSs deaminase edits (`C→T`, `G→A`). Extracted with no `--event` flag. Files tagged `*_access_*`.

The comparison data is ENCODE ATAC-seq. Both assays go through the same filtering and
depth matching, so differences come from the signal, not the processing.

## Running

Run the scripts in filename order. Notebooks are run interactively.

```bash
mkdir -p logs
sbatch 02_filter_bam.sh
sbatch 06_bam2bw_access_atac.sh    # scripts with #SBATCH --array are array jobs
```

Each script writes to `../../results/01_process_access/<NN_name>/`. `logs/` must exist
before you submit.

Conda environments: `access` for most scripts, `macs2` for `10_call_peaks.sh`. The scripts
run `conda activate` but do not initialise conda, so your shell must do that first.

## Paths to adapt

Paths assume this directory sits at `<project_root>/scripts/01_process_access/`.

- `../../data/Concurrent_ACCESS_ATAC/` — input ACCESS-ATAC BAMs
- `../../data/encode_atac/` — ENCODE ATAC-seq BAMs
- `../../data/GRCh38/` — genome FASTA and chrom sizes
- `../../data/Blacklist/hg38-blacklist.v2.bed` — ENCODE hg38 blacklist v2
- `../../data/bap/bap/anno/` — TSS annotation for the heatmap
- `~/rgtdata/motifs/jaspar_vertebrates/` — JASPAR PWMs, used by step 16

## Scripts

### QC

**`01_check_duplicate.sh`** — `samtools markdup` on the HepG2 BAM. Checks how many
duplicates remain after upstream deduplication. A diagnostic only; nothing reads its
output. Most of the script is commented out.

### Filtering and depth matching

**`02_filter_bam.sh`** — filters both assays to chr1–22 and chrX. chrY is dropped (both
lines are female-derived) and chrM too (its signal is very high). ACCESS-ATAC is
single-end and already deduplicated, so it gets `-F 0xB04 -q 30`. ENCODE ATAC is
paired-end with duplicates marked, so it gets `-f 0x2 -F 0xF04 -q 30`. Everything
downstream starts here.

**`03_bam2insertion.sh`** — matches depth by insertion count. Converts each BAM to a
single-base Tn5 insertion BED (+4 / −5 shift), downsamples both assays to the same
insertion count with a fixed seed, then writes CPM bigWigs. Note single-end ACCESS gives
one insertion per read, paired-end ATAC gives two per fragment.

**`04_bam_summary.sh`** — `multiBamSummary` at 10 kb, then a Spearman scatterplot. A quick
check on the filtered BAMs.

**`05_subsample.sh`** — matches depth by read count instead. Subsamples both assays to
2×10⁸ reads, seed 42. This runs in parallel with step 03, not after it. Steps 06 onwards
use this one.

### Signal tracks

**`06_bam2bw_access_atac.sh`** and **`07_bam2bw_atac.sh`** — the same script on different
inputs. `06` takes the ACCESS-ATAC BAMs, `07` the ENCODE ATAC BAMs. Both call
`deamtools bam2bw --mode count`. Each is an array job driven by a task table. The active
table is `samples=(HepG2 K562)`, `events=(tn5 tn5)`, `extends=(100 100)`. The original
8-combination table is kept above it as comments.

**`08_plot_heatmap_vs_atac.sh`** — TSS heatmap. `computeMatrix reference-point` over
±2 kb at bin size 1, then `plotHeatmap` with ACCESS and ATAC side by side.

**`09_bigiwg_summary.sh`** — genome-wide correlation. `multiBigwigSummary` at 1 kb for
each of `noext`, `ext50` and `ext100`, then Spearman and Pearson scatterplots. Writes to
`09_bigwig_summary/`.

### Peaks

**`10_call_peaks.sh`** — MACS2 in BED mode (`--nomodel --shift -100 --extsize 200
-q 0.0001`), then removes blacklist regions. BED input treats single-end and paired-end
the same way. The BAM → BED step is commented out, so the BED files must already exist.
The inner loop currently runs `atacseq` only.

**`11_bed2bw.sh`** — insertion BEDs → CPM bigWigs.

**`12_bigiwg_summary.sh`** — correlation inside peaks rather than genome-wide. Merges the
two peak sets, quantifies both bigWigs over them, writes Spearman and Pearson
scatterplots. Writes to `12_peak_correlation/`.

### TF footprints

**`13_motif_matching.sh`** — `rgt-motifanalysis matching` on the ACCESS-ATAC peaks. No
SLURM header; run it directly.

**`14_get_mpbs.ipynb`** — splits the motif hits into one BED per TF. TF names are
sanitised for use as filenames. Step 16 repeats the same sanitisation.

**`15_plot_footprint.sh`** — array job over 2 samples × {ACCESS, ATAC}. Runs
`plotting/plot_all_footprint.py` with `--extend 500` for each TF. Skips TFs already done.

**`16_estimate_footprint.ipynb`** — turns each profile into two numbers. For every
candidate width it computes `mean(flanks) − mean(centre)`. The maximum is the footprint
**height**; the width at that maximum is the footprint **width**. Motif widths come from
the JASPAR PWMs. Writes `footprint.csv`, `fp_height.csv`, `fp_width.csv` and plots.

**`17_plot_footprint_heatmap.sh`** — per TF, `computeMatrix` over ±50 bp for both bigWigs,
then `plotHeatmap`. Rows are sorted by ACCESS signal so both panels share a row order.

### Bundled script

**`plotting/plot_all_footprint.py`** — averages bigWig signal over a BED region set,
recentred on region midpoints ± `--extend`. Writes a CSV with `position`, `signal` and
`data`. Its plotting block is commented out.

## Known gaps

- Steps 15 and 17 read `*_noext.bw`, and step 09 reads three extension variants. The
  current task tables in `06` and `07` only produce `*_atac_ext100.bw`. Restore the
  commented-out tables first.
- `06` and `07` declare `--array=0-2` but their arrays hold two elements. The third task
  fails.
- The notebooks are committed without outputs.
