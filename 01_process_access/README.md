# 01_process_access

Analysis pipeline comparing **concurrent ACCESS-ATAC** against conventional ATAC-seq in
HepG2 and K562.

Both signals are extracted from the same ACCESS-ATAC BAM:

| Signal | Definition | `deamtools bam2bw` flag | Output name tag |
|---|---|---|---|
| **ATAC** | Tn5 cut sites (+4 / −5 offset) | `--event tn5` | `*_atac_*` |
| **ACCESS** | DddA deaminase edits (`C→T`, `G→A`) | omitted (default) | `*_access_*` |

The comparison dataset is ENCODE ATAC-seq for the same two cell lines. Both assays are put
through identical filtering and depth matching so that the two are directly comparable.

## Running

Scripts are SLURM job scripts, run in the numeric order of their filenames. Notebooks are
run interactively at the point where their number falls.

```bash
mkdir -p logs
sbatch 02_filter_bam.sh
sbatch 06_bam2bw_access_atac.sh    # scripts with #SBATCH --array are array jobs
```

Each script writes to `../../results/01_process_access/<NN_name>/`, numbered to match the
script that produces it, and reads the output of earlier steps. The `#SBATCH --output`
directives require `logs/` to exist before submitting.

### Paths that must be adapted

All paths are relative to a project root two levels above this directory, i.e. the layout
assumed is `<project_root>/scripts/01_process_access/`. The following are hard-coded and
need to be changed for another environment:

| Path | Used for |
|---|---|
| `../../data/Concurrent_ACCESS_ATAC/` | input ACCESS-ATAC BAMs |
| `../../data/encode_atac/` | input ENCODE ATAC-seq BAMs |
| `../../data/GRCh38/GRCh38.primary_assembly.genome.fa` | reference genome |
| `../../data/GRCh38/core.chrom.sizes` | chromosome sizes (chr1–22, chrX) |
| `../../data/Blacklist/hg38-blacklist.v2.bed` | ENCODE hg38 blacklist v2 |
| `../../data/bap/bap/anno/TSS/hg38.refGene.TSS.bed` | TSS annotation for heatmaps |
| `../../data/bap/bap/anno/bedtools/chrom_hg38.sizes` | chromosome sizes for `bedtools slop` |
| `../../github/ACCESS-ATAC/plotting/plot_all_footprint.py` | footprint profile script (step 15) |
| `~/rgtdata/motifs/jaspar_vertebrates/` | JASPAR PWMs, read by step 16 |

Conda environments: `access` for most scripts, `macs2` for `10_call_peaks.sh`. The scripts
call `conda activate` but do not initialise conda themselves, so conda must already be
initialised in the submitting shell — either through your shell profile, or by adding a
line such as `source "$(conda info --base)/etc/profile.d/conda.sh"` to each script.

Input BAM naming: `062424_{HepG2,K562}_concurrent_ACCESS-ATAC.dedup.hg38_core.bam`.

## Scripts

### Input QC (01)

| Script | Input | Output | What it does |
|---|---|---|---|
| `01_check_duplicate.sh` | `data/Concurrent_ACCESS_ATAC/` HepG2 BAM | `01_check_duplicate/` | `samtools markdup` to measure coordinate-level duplicates remaining after upstream deduplication. Only the markdup step is active; the duplicate-rate, per-chromosome and bigWig diagnostics are commented out. |

### Filtering, depth matching and signal tracks (02–12)

| Script | Input | Output | What it does |
|---|---|---|---|
| `02_filter_bam.sh` | `data/Concurrent_ACCESS_ATAC/`, `data/encode_atac/` | `02_filter/` | Filters both assays to chr1–22 and chrX (chrY excluded as both lines are female-derived; chrM excluded). ACCESS-ATAC (single-end, already deduplicated upstream): `-F 0xB04 -q 30`. ENCODE ATAC (paired-end): `-f 0x2 -F 0xF04 -q 30`, which also drops marked duplicates. Writes `flagstat` for each. |
| `03_bam2insertion.sh` | `02_filter/` | `03_bam2insertion/` | Three steps: (1) `bedtools bamtobed` → single-base Tn5 insertion BED, +4 on the plus strand and −5 on the minus strand; (2) per cell line, downsamples both assays to the **same insertion count** (the smaller of the two) with `shuf` under a fixed seed; (3) `bedtools genomecov -bg -scale` to CPM, then `bedGraphToBigWig`. |
| `04_bam_summary.sh` | `02_filter/` | `04_bam_summary/` | `multiBamSummary bins --binSize 10000` with the blacklist excluded, then `plotCorrelation -c spearman -p scatterplot --log1p`. |
| `05_subsample.sh` | `02_filter/` | `05_subsample/` | Subsamples both assays to a fixed **read** count (2×10⁸) with `samtools view -s`, seed 42. ATAC counts and samples only properly paired reads. This is the read-count counterpart to the insertion-count matching in step 03. |
| `06_bam2bw_access_atac.sh` | `05_subsample/` | `06_bam2bw_access_atac/` | `deamtools bam2bw --mode count --min_coverage 1 --min_mapq 0 --min_baseq 0` on the ACCESS-ATAC BAMs. The active task table is `samples=(HepG2 K562)`, `events=(tn5 tn5)`, `extends=(100 100)`; the original 8-combination table (2 samples × {`edit`, `tn5`} × {no extension, 50 bp}) is retained as comments. |
| `07_bam2bw_atac.sh` | `05_subsample/` | `07_bam2bw_atac/` | Same for the ENCODE ATAC BAMs. |
| `08_plot_heatmap_vs_atac.sh` | `06_bam2bw_access_atac/`, `07_bam2bw_atac/` | `08_plot_heatmap_vs_atac/` | deeptools `computeMatrix reference-point` ± 2 kb around TSS (`bedtools slop -b 1000`), bin size 1, then `plotHeatmap` with ACCESS and ATAC side by side (`--zMin 0 --zMax 100`). |
| `09_bigiwg_summary.sh` | `06_bam2bw_access_atac/`, `07_bam2bw_atac/` | `09_bigwig_summary/` | `multiBigwigSummary bins --binSize 1000` for each of `noext`, `ext50` and `ext100`, blacklist excluded, then both Spearman and Pearson `plotCorrelation` scatterplots. |
| `10_call_peaks.sh` | `10_call_peaks/` insertion BEDs | `10_call_peaks/` | `macs2 callpeak -f BED --nomodel --shift -100 --extsize 200 -q 0.0001 -g hs`, then `bedtools intersect -v` against the blacklist. Step 1 (BAM → insertion BED) is commented out, so the BED files must already exist. The inner loop currently runs `atacseq` only. |
| `11_bed2bw.sh` | `10_call_peaks/` insertion BEDs | `11_bed2bw/` | `bedtools genomecov -bg -scale` to CPM, then `bedGraphToBigWig`. |
| `12_bigiwg_summary.sh` | `10_call_peaks/`, `06_bam2bw_access_atac/`, `07_bam2bw_atac/` | `12_peak_correlation/` | Merges the two peak sets with `bedtools merge`, quantifies both bigWigs over the ATAC peak set with `multiBigwigSummary BED-file`, and writes Spearman and Pearson scatterplots. |

### TF footprint analysis (13–17)

| Script | Input | Output | What it does |
|---|---|---|---|
| `13_motif_matching.sh` | `10_call_peaks/` | `13_motif_matching/` | `rgt-motifanalysis matching --organism hg38` on the ACCESS-ATAC peaks, producing motif-predicted binding sites (MPBS). No SLURM header; run directly. |
| `14_get_mpbs.ipynb` | `13_motif_matching/` | `14_get_mpbs/` | Splits the combined MPBS BED into one BED per TF, under a per-cell-line subdirectory. TF names are sanitised for use as filenames (spaces and `/` → `_`, parentheses removed). |
| `15_plot_footprint.sh` | `14_get_mpbs/`, `06_bam2bw_access_atac/` | `15_plot_footprint/` | Array job over 2 samples × {ACCESS, ATAC}. For each TF BED, runs `plotting/plot_all_footprint.py` with `--extend 500` to write the mean signal profile around motif centres as a CSV. Skips TFs whose output already exists. |
| `16_estimate_footprint.ipynb` | `15_plot_footprint/` | `16_estimate_footprint/` | Quantifies each profile. Motif widths are read from the JASPAR PWM files in `~/rgtdata`. For each candidate footprint width from the motif width to 100 bp, computes footprint depth as `mean(flanks) − mean(centre)`; the maximum over widths gives the footprint height, and the width at that maximum gives the footprint width. Writes `footprint.csv`, `fp_height.csv`, `fp_width.csv` and per-motif diagnostic plots, plus ACCESS-vs-ATAC boxplots and scatterplots. |
| `17_plot_footprint_heatmap.sh` | `14_get_mpbs/`, `06_bam2bw_access_atac/` | `17_plot_footprint_heatmap/` | Array job over the 2 samples. Per TF, `computeMatrix reference-point` ± 50 bp at bin size 1 for the ACCESS and ATAC bigWigs together, then `plotHeatmap` with rows sorted by mean ACCESS signal so both panels share a row order. |

## Notes

- The notebooks are committed **without outputs**.
- Two scripts write to an output directory whose name differs from the script's own:
  `09_bigiwg_summary.sh` → `09_bigwig_summary/`, and `12_bigiwg_summary.sh` →
  `12_peak_correlation/`.
- Steps `15` and `17` read `*_access_noext.bw` and `*_atac_noext.bw`, and step `09` reads
  `noext`, `ext50` and `ext100`, but the active task tables in `06` and `07` generate only
  `*_atac_ext100.bw`. The other combinations must be generated by restoring the
  commented-out task tables before those steps can run.
- `06` and `07` declare `#SBATCH --array=0-2` while their task arrays hold two elements, so
  the third array task fails under `set -u`.
