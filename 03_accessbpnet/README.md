# 03_accessbpnet

Applies **ChromBPNet** to ACCESS-ATAC data: trains a bias model and a ChromBPNet model,
derives contribution scores, runs TF-MoDISco motif discovery, computes marginal
footprints, and scores genetic variants.

ChromBPNet was written for ATAC-seq and DNase-seq, which it summarises as Tn5 or DNase
cut-site counts. ACCESS-ATAC's second readout is per-base deaminase editing, which does not
fit that model. [`chrombpnet/`](chrombpnet/) is therefore a **modified copy of ChromBPNet**
that adds an `ACCESS` assay type and lets a precomputed bigWig be supplied directly — see
[Modifications to ChromBPNet](#modifications-to-chrombpnet).

## Running

Scripts are SLURM job scripts, run in the numeric order of their filenames. Notebooks are
run interactively at the point where their number falls.

```bash
mkdir -p logs
sbatch 01_call_peaks.sh
sbatch 02_bam2bw.sh
...
```

Each script writes to `../../results/03_accessbpnet/<NN_name>/`, numbered to match the
script that produces it. The one exception is `03_chrombpnet_bias/`, which is a shared
working directory for the bias-model stage: `03_prep_nonpeaks.sh` creates it and
`04_chrombpnet_bias.sh` writes into it.

The `#SBATCH --output` directives require `logs/` to exist before submitting.

### Paths that must be adapted

All paths are relative to a project root two levels above this directory, i.e. the layout
assumed is `<project_root>/scripts/03_accessbpnet/`.

| Path | Provides |
|---|---|
| `../../results/01_process_access/02_filter/` | filtered ACCESS-ATAC BAMs, from pipeline 01 |
| `../../data/GRCh38/GRCh38.primary_assembly.genome.fa` | reference genome |
| `../../data/GRCh38/GRCh38.primary_assembly.chrom.sizes`, `core.chrom.sizes` | chromosome sizes |
| `../../data/Blacklist/hg38-blacklist.v2.bed` | ENCODE hg38 blacklist v2 |
| `../../data/Motifs/JASPAR2026_CORE_vertebrates_*` | JASPAR motifs (PFM and MEME formats) |
| `../../data/Liver_caQTL/` | caQTL variants for the variant-scoring steps |
| `../../data/encode_chipseq/K562/Bigwig/` | ENCODE ChIP-seq bigWig, used for track visualisation |
| `~/rgtdata/hg38/gencode.v21.annotation.gtf` | gene annotation, read by `11_viz_tracks.ipynb` |
| `../../results/32_chrombpnet/` | a parallel ChromBPNet run not in this repository, used for comparison in the variant-effect notebooks |

`fold_0.json` in this directory defines the train / valid / test chromosome split.

Conda environments: `chrombpnet` for the model steps, `macs2` for peak calling, `access`
for the rest. The scripts call `conda activate` but do not initialise conda, so it must
already be initialised in the submitting shell.

## Modifications to ChromBPNet

`chrombpnet/` is a copy of **[lzj1769/chrombpnet](https://github.com/lzj1769/chrombpnet)**,
branch `access`, commit `a298e6b`. That fork's `master` is identical to upstream
**[kundajelab/chrombpnet](https://github.com/kundajelab/chrombpnet)** at `09938fd`, so the
changes below are exactly the delta against stock ChromBPNet. ChromBPNet is MIT licensed;
`chrombpnet/LICENSE` is included unchanged.

The changes fall into three groups.

**1. Accept a precomputed bigWig as input.** Stock ChromBPNet always derives its count
track from reads (`-ibam`, `-ifrag` or `-itag`) via `reads_to_bigwig`. A fourth input
option `-ibw` / `--input-bigwig-file` was added; when it is set the pipeline skips
`reads_to_bigwig` entirely and uses the supplied bigWig as the observed signal. This is
what allows an ACCESS edit track — generated outside ChromBPNet — to be modelled directly.

**2. An `ACCESS` assay type.** `-d` / `--data-type` now accepts `ACCESS` alongside `ATAC`
and `DNASE`, threaded through the `pipeline`, `bias`, `prep`, `pred_bw`, `contribs_bw` and
`footprints` subcommands. The assay determines which enzyme-bias motifs the bias model is
checked against:

| Assay | Bias motifs |
|---|---|
| ATAC | Tn5 motifs (upstream) |
| DNASE | `TTTACAAGTCCA` (upstream) |
| **ACCESS** | `ddd1_1` `ATTCA`, `ddd1_2` `ATTCC`, `ddd1_3` `ATTCG`, `ddd1_4` `ATTCT` |

The DddA motifs are added as `chrombpnet/data/motif_to_pwm.ACCESS.tsv`, registered in
`chrombpnet/data/__init__.py`, and branched on in `pipelines.py`. `reads_to_bigwig.py`
also gains an ACCESS path (`get_raw_signal_access`) that builds the track from C→T and
G→A editing sites rather than from read ends.

**3. Bug fixes and small behavioural changes**, independent of ACCESS:

| File | Change |
|---|---|
| `evaluation/make_bigwigs/bigwig_helper.py` | `if regions_used:` → `if regions_used is not None:` — the truth value of a non-empty NumPy array is ambiguous, so the original raised or silently took the wrong branch |
| `helpers/hyperparameters/find_bias_hyperparams.py` | outlier filter changed from strict `<`/`>` to `<=`/`>=`, so non-peaks exactly on the quantile boundary are kept |
| `helpers/make_gc_matched_negatives/get_gc_matched_negatives.py` | the test split's negative-to-positive ratio was hard-coded to `1`; it now follows `--neg_to_pos_ratio_train` like the other splits |
| `requirements.txt` | TensorFlow pinned to `2.8.1` instead of `2.8.0` |
| `.gitignore` | ignore `build/` and `chrombpnet.egg-info/` |

`parsers.py` and `pipelines.py` additionally carry a large amount of reformatting
(re-wrapped `add_argument` calls); the functional changes in them are the two features
above.

### What was not copied

`chrombpnet/evaluation/figure_notebooks/` (2.4 MB of upstream's own paper-figure notebooks,
with outputs) was omitted — it is unrelated to this pipeline. Everything else from the
`access` branch working tree is present. Fetch the full branch from the fork if you need it.

## Notes

- The notebooks are committed **without outputs**.
- `reads_to_bigwig.py` contains three calls to `parser.add_argumen(` (missing the final
  `t`, lines 47, 54 and 61). These are in that module's standalone argument parser, which
  the main `chrombpnet` entry point does not use, so the pipeline runs — but invoking
  `reads_to_bigwig.py` directly raises `AttributeError`. Carried over from the fork as-is.
- `18_eval_variant_score.ipynb` operates entirely outside this pipeline's own result
  tree, working against `32_chrombpnet/` throughout.
- `17_eval_variant_score.ipynb` and `18_eval_variant_score.ipynb` are the same evaluation
  applied to two different runs — `17` to this pipeline (and `19` reads its output), `18`
  to the external `32_chrombpnet` results.
