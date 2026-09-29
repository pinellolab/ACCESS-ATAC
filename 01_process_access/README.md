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
| `~/rgtdata/motifs/jaspar_vertebrates/` | JASPAR PWMs, read by step 16 |

Conda environments: `access` for most scripts, `macs2` for `10_call_peaks.sh`. The scripts
call `conda activate` but do not initialise conda themselves, so conda must already be
initialised in the submitting shell — either through your shell profile, or by adding a
line such as `source "$(conda info --base)/etc/profile.d/conda.sh"` to each script.

Input BAM naming: `062424_{HepG2,K562}_concurrent_ACCESS-ATAC.dedup.hg38_core.bam`.

`plotting/plot_all_footprint.py` is bundled here and called by `15_plot_footprint.sh`. It averages bigWig signal over a BED region set, recentring each region on its midpoint ± `--extend`, and writes a tidy CSV with `position`, `signal` and `data` columns.