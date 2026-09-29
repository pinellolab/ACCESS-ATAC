# 01_process_access

Compares **concurrent ACCESS-ATAC** against conventional ATAC-seq in HepG2 and K562.

ACCESS-ATAC yields two readouts from one library. The **ATAC** readout is Tn5 cut sites,
taken from read ends with the usual +4 / −5 offset. The **ACCESS** readout is DddSs
deaminase edits, read off alignments as `C→T` and `G→A` mismatches against the reference.
Both are extracted from the same BAM by `deamtools bam2bw`: passing `--event tn5` gives
the Tn5 track and produces files tagged `*_atac_*`, while omitting `--event` gives the
edit track, tagged `*_access_*`.

The comparison dataset is ENCODE ATAC-seq for the same two cell lines. The central design
choice of this pipeline is that both assays go through **identical** filtering and depth
matching, so any difference in the results comes from the signal rather than from the
processing.

## Running

Scripts are SLURM job scripts, run in the numeric order of their filenames; notebooks are
run interactively at the point where their number falls.

```bash
mkdir -p logs
sbatch 02_filter_bam.sh
sbatch 06_bam2bw_access_atac.sh    # scripts with #SBATCH --array are array jobs
```

Each script writes to `../../results/01_process_access/<NN_name>/`, numbered to match the
script that produces it, and reads the output of earlier steps. The `#SBATCH --output`
directives require `logs/` to exist before submitting.

Conda environments: `access` for most scripts, `macs2` for `10_call_peaks.sh`. The scripts
call `conda activate` but do not initialise conda, so it must already be initialised in
the submitting shell — either through your shell profile, or by adding a line such as
`source "$(conda info --base)/etc/profile.d/conda.sh"` to each script.

### Paths that must be adapted

All paths are relative to a project root two levels above this directory, i.e. the layout
assumed is `<project_root>/scripts/01_process_access/`. The following are hard-coded:

- `../../data/Concurrent_ACCESS_ATAC/` — the input ACCESS-ATAC BAMs, named
  `062424_{HepG2,K562}_concurrent_ACCESS-ATAC.dedup.hg38_core.bam`
- `../../data/encode_atac/` — the comparison ENCODE ATAC-seq BAMs, named `{sample}.bam`
- `../../data/GRCh38/GRCh38.primary_assembly.genome.fa` — reference genome
- `../../data/GRCh38/core.chrom.sizes` and `GRCh38.primary_assembly.chrom.sizes` —
  chromosome sizes; `core` covers chr1–22 and chrX only
- `../../data/Blacklist/hg38-blacklist.v2.bed` — ENCODE hg38 blacklist v2
- `../../data/bap/bap/anno/TSS/hg38.refGene.TSS.bed` and
  `../../data/bap/bap/anno/bedtools/chrom_hg38.sizes` — TSS annotation and sizes for the
  heatmap step
- `~/rgtdata/motifs/jaspar_vertebrates/` — JASPAR PWMs, read by step 16

## The scripts

### Input QC

**`01_check_duplicate.sh`** runs `samtools markdup` on the HepG2 BAM to measure how many
coordinate-level duplicates remain after the upstream sequence-based deduplication. It is
a diagnostic, not part of the chain — nothing downstream reads its output. Only the
markdup call is active; the duplicate-rate summary, the per-chromosome breakdown and the
duplicate-vs-non-duplicate bigWigs are present but commented out.

### Filtering and depth matching

**`02_filter_bam.sh`** filters both assays to chr1–22 and chrX. chrY is dropped because
both cell lines are female-derived, and chrM because the very high mitochondrial signal
distorts ATAC analyses. The two assays need different flags: ACCESS-ATAC is single-end and
was already deduplicated upstream, so it takes `-F 0xB04 -q 30` and keeps everything else;
ENCODE ATAC is paired-end with duplicates marked but not removed, so it takes
`-f 0x2 -F 0xF04 -q 30`, which additionally requires proper pairing and drops the marked
duplicates. Writes `flagstat` output alongside each BAM. Everything downstream starts here.

**`03_bam2insertion.sh`** is the insertion-count route to depth matching, in three steps.
First it converts each BAM to a single-base Tn5 insertion BED with `bedtools bamtobed`,
shifting +4 on the plus strand and −5 on the minus strand; note that single-end ACCESS
contributes one insertion per read while paired-end ATAC contributes one per mate, so a
fragment yields two. Second, per cell line, it downsamples both assays to the *same*
insertion count — whichever is smaller — using `shuf` with a fixed seed so the result is
reproducible. Third, it converts each downsampled BED to a CPM-normalised bigWig via
`bedtools genomecov -bg -scale` and `bedGraphToBigWig`.

**`04_bam_summary.sh`** is a first sanity check on the filtered BAMs: `multiBamSummary
bins` at 10 kb with the blacklist excluded, then a Spearman scatterplot via
`plotCorrelation --log1p`. A scatterplot rather than a heatmap because there are only two
samples.

**`05_subsample.sh`** is the read-count route to depth matching, parallel to step 03 rather
than following it. It subsamples both assays to a fixed 2×10⁸ reads with `samtools view -s`
under seed 42. ATAC counts and samples only properly paired reads. Steps 03 and 05 are two
different notions of "equal depth" — matched insertions versus matched reads — and the
bigWig and peak-calling steps below build on `05`.

### Signal tracks

**`06_bam2bw_access_atac.sh`** and **`07_bam2bw_atac.sh`** are the same script pointed at
different inputs: `06` at the subsampled ACCESS-ATAC BAMs, `07` at the subsampled ENCODE
ATAC BAMs. Both call `deamtools bam2bw --mode count --min_coverage 1 --min_mapq 0
--min_baseq 0`. Each is a SLURM array job driven by a small task table of
`samples` / `events` / `extends`; the active table is `samples=(HepG2 K562)`,
`events=(tn5 tn5)`, `extends=(100 100)`, and the original eight-combination table
— two samples × {`edit`, `tn5`} × {no extension, 50 bp} — is retained just above it as
comments.

**`08_plot_heatmap_vs_atac.sh`** builds the TSS heatmap. It widens the TSS annotation by
1 kb with `bedtools slop`, then runs deeptools `computeMatrix reference-point` over ±2 kb
at bin size 1 and `plotHeatmap` with ACCESS and ATAC side by side, colour-capped at
`--zMin 0 --zMax 100` so the two panels are on a common scale.

**`09_bigiwg_summary.sh`** quantifies genome-wide agreement between the two tracks:
`multiBigwigSummary bins` at 1 kb with the blacklist excluded, repeated for each of the
`noext`, `ext50` and `ext100` bigWig variants, then both Spearman and Pearson
`plotCorrelation` scatterplots. Despite the filename it writes to `09_bigwig_summary/`.

### Peaks

**`10_call_peaks.sh`** calls peaks with MACS2 in BED mode:
`macs2 callpeak -f BED --nomodel --shift -100 --extsize 200 -q 0.0001 -g hs`, followed by
`bedtools intersect -v` against the blacklist. Feeding insertion BEDs rather than BAMs
means single-end and paired-end data are treated identically. Two caveats: the BAM →
insertion BED step is commented out, so the BED files must already exist in the output
directory, and the inner loop currently iterates over `atacseq` only.

**`11_bed2bw.sh`** converts those insertion BEDs to CPM-normalised bigWigs, the same
`genomecov` → `bedGraphToBigWig` route as step 03.

**`12_bigiwg_summary.sh`** measures agreement *within peaks* rather than genome-wide. It
merges the two peak sets with `bedtools merge` to define comparison regions, quantifies
both bigWigs over the ATAC peak set with `multiBigwigSummary BED-file`, and writes Spearman
and Pearson scatterplots. It writes to `12_peak_correlation/`, not to a directory matching
its own name.

### TF footprints

**`13_motif_matching.sh`** runs `rgt-motifanalysis matching --organism hg38` on the
ACCESS-ATAC peaks to produce motif-predicted binding sites (MPBS). It has no SLURM header
and is run directly.

**`14_get_mpbs.ipynb`** splits the combined MPBS BED into one BED per transcription factor,
under a per-cell-line subdirectory. TF names are sanitised for use as filenames — spaces
and `/` become `_`, parentheses are removed — and the same sanitisation is repeated in
step 16, so the two must stay in step.

**`15_plot_footprint.sh`** computes the footprint profiles. It is an array job over
2 samples × {ACCESS, ATAC}, and for each TF BED it runs the bundled
`plotting/plot_all_footprint.py` with `--extend 500`, writing the mean signal around motif
centres as a CSV. It skips any TF whose output already exists, so it can be resubmitted
after a timeout.

**`16_estimate_footprint.ipynb`** turns those profiles into two numbers per TF. Motif
widths come from the JASPAR PWM files in `~/rgtdata`. For every candidate footprint width
from the motif width up to 100 bp it computes a depth as `mean(flanking) − mean(centre)`;
the maximum over widths is the **footprint height** and the width at which that maximum
occurs is the **footprint width**. Writes `footprint.csv`, `fp_height.csv`, `fp_width.csv`,
per-motif diagnostic plots, and ACCESS-versus-ATAC boxplots and scatterplots.

**`17_plot_footprint_heatmap.sh`** is the visual counterpart to step 16. Per TF it runs
`computeMatrix reference-point` over ±50 bp at bin size 1 for the ACCESS and ATAC bigWigs
together, then `plotHeatmap` with rows sorted by mean ACCESS signal
(`--sortUsingSamples 1`) so both panels share a row order and can be read against each
other.

## Notes

- The notebooks are committed **without outputs**.
- `plotting/plot_all_footprint.py` is bundled here and called by step 15. It averages
  bigWig signal over a BED region set, recentring each region on its midpoint ±
  `--extend`, and writes a tidy CSV with `position`, `signal` and `data` columns. Its
  plotting block is commented out, so only the CSV is produced.
- Steps `15` and `17` read `*_access_noext.bw` and `*_atac_noext.bw`, and step `09` reads
  `noext`, `ext50` and `ext100`, but the active task tables in `06` and `07` generate only
  `*_atac_ext100.bw`. The other combinations must be produced by restoring the
  commented-out task tables before those steps can run.
- `06` and `07` declare `#SBATCH --array=0-2` while their task arrays hold two elements,
  so the third array task fails under `set -u`.
- Two scripts write to a directory whose name differs from their own:
  `09_bigiwg_summary.sh` → `09_bigwig_summary/`, and `12_bigiwg_summary.sh` →
  `12_peak_correlation/`.
