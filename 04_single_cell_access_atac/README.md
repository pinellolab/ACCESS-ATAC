# 04_single_cell_access_atac

Single-cell ACCESS-ATAC in mouse airway cells: read alignment, cell-barcode correction,
fragment generation, clustering and cell-type annotation (ArchR and Seurat), marker
identification, trajectory analysis, and TF footprinting on the annotated populations.

## Running

Scripts run in the numeric order of their filenames, 01–35; notebooks are run
interactively at the point where their number falls. Step 01 runs an
[nf-core](https://nf-co.re/)-style Nextflow pipeline (`nf-core-accessatacseq`) which lives
outside this repository; everything after it is in here.

Most steps write to `../../results/04_single_cell_access_atac/<NN_name>/`.

### Bundled processing scripts

`scripts/` holds the Python this pipeline calls, copied here so the directory is
self-contained:

| Script | Used by |
|---|---|
| `barcode_correction.py` | `03_correct_barcode.sh` |
| `add_barcode_to_bam_v2.py` | `06_add_barcodes.sh` |
| `bam_to_fragments.py` | `07`, `08_bam_to_fragment.sh` |
| `aggregate_replicates.py` | `07_bam_to_fragment.sh`, `09_merge_fragment.sh` |
| `fragment_to_bw_atac.py`, `fragment_to_bw_access.py` | `17`, `21`, `27`, `30_fragment_to_bw.sh` |
| `utils.py` | imported by the two `fragment_to_bw_*` scripts |
| `plot_all_footprint.py` | `32_plot_footprint.sh` |

The two `fragment_to_bw_*` scripts import `utils` by bare module name, which resolves
inside `scripts/` because Python puts the script's own directory first on the import path.
Keep `utils.py` alongside them.

### Paths that must be adapted

Unlike pipelines 01–03, this one uses **absolute** cluster paths rather than
`../../data/`. They are recorded here as they were run; change them for another
environment:

| Path | Provides |
|---|---|
| `/data/pinello/PROJECTS/2025_06_ZL_ACCESS/` | project root — data and results live under it |
| `/data/refdata-cellranger-arc-GRCm39-2024-A/` | mouse reference: `fasta/genome.fa`, `fasta/genome.chrom.sizes`, `genes/genes.gtf.gz`, `regions/tss.bed` |
| `/data/pinello/PROJECTS/2025_06_ZL_ACCESS/data/refdata-cellranger-arc-GRCm39-2024-A/` | the same reference, including `bwa_meth_index`, used by step 01 |
| `/data/10xGenomics/737K-cratac-v1.txt` | 10x scATAC barcode whitelist |
| `/data/Sitara_scACCESS_mouseAirwayCell_112125/`, `.../_121725/` | raw FASTQ, two sequencing batches |
| `/data/pinello/SHARED_SOFTWARE/envs/zl002_envs/zl_macs2/bin/macs2` | MACS2 binary passed to ArchR |
| `../../github/nf-core-accessatacseq/` | the Nextflow pipeline run by step 01, a separate repository |

`samples.csv` lists the input FASTQ pairs by absolute path and needs the same treatment.