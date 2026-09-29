# 03_accessbpnet

Applies **ChromBPNet** to ACCESS-ATAC data: trains a bias model and a ChromBPNet model,
derives contribution scores, runs TF-MoDISco motif discovery, computes marginal
footprints, and scores genetic variants.

ChromBPNet was written for ATAC-seq and DNase-seq, which it summarises as Tn5 or DNase
cut-site counts. ACCESS-ATAC's second readout is per-base deaminase editing, which does not
fit that model. [`chrombpnet/`](chrombpnet/) is therefore a **modified copy of ChromBPNet**
that adds an `ACCESS` assay type and lets a precomputed bigWig be supplied directly. What
was changed there is documented in [`chrombpnet/README.md`](chrombpnet/README.md); how
this pipeline drives it is [below](#using-chrombpnet).

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

`fold_0.json` in this directory defines the train / valid / test chromosome split.


## Using ChromBPNet

`chrombpnet/` is a modified copy of
[kundajelab/chrombpnet](https://github.com/kundajelab/chrombpnet); what was changed and
why is documented in [`chrombpnet/README.md`](chrombpnet/README.md). This section covers
how the pipeline uses it.

Install it from the bundled copy:

```bash
conda activate chrombpnet
pip install -e chrombpnet/
```

### Training on ACCESS and on ATAC

Because ACCESS-ATAC yields two tracks from the same library, both are trained with the
**same commands** — the only differences are `-d` and which bigWig is passed to `-ibw`:

| | ATAC readout | ACCESS readout |
|---|---|---|
| bigWig | Tn5 insertion counts | deaminase edit counts |
| `-d` | `ATAC` | `ACCESS` |

Everything after the bigWig — peaks, background regions, fold split, and every downstream
subcommand — is identical. That is deliberate: it makes the two readouts directly
comparable, since any difference in the resulting models comes from the signal rather than
from the processing.

#### Step 0 — build the two bigWigs

Done outside ChromBPNet, from the same BAM. In this repository that is
[`../02_bam2bw.sh`](02_bam2bw.sh), which calls `deamtools bam2bw` twice per sample:
`--event tn5` for the ATAC track and the default (edit) mode for the ACCESS track,
producing `{sample}_atac.bw` and `{sample}_access.bw`.

#### Step 1 — background regions (once per sample, shared by both assays)

```bash
chrombpnet prep nonpeaks \
    -g   genome.fa \
    -p   peaks.narrowPeak \
    -c   chrom.sizes \
    -fl  fold_0.json \
    -br  blacklist.bed \
    -o   ${OUT}/nonpeaks
```

GC-matched negatives do not depend on the assay, so this runs once and both models reuse
`nonpeaks_negatives.bed`.

#### Step 2 — bias model (once per assay)

```bash
ASSAY=ACCESS                       # or ATAC
BW=${sample}_access.bw             # or ${sample}_atac.bw

chrombpnet bias pipeline \
    -ibw ${BW} \
    -d   ${ASSAY} \
    -g   genome.fa \
    -c   chrom.sizes \
    -p   peaks.narrowPeak \
    -n   ${OUT}/nonpeaks_negatives.bed \
    -fl  fold_0.json \
    -b   0.5 \
    -o   ${OUT}/${ASSAY} \
    -fp  ${sample}
```

`-b 0.5` is the bias threshold factor. Each assay needs **its own** bias model: the
enzymatic bias of Tn5 and of DddSs are different, which is exactly what `-d` selects the
motifs for. See [`../04_chrombpnet_bias.sh`](04_chrombpnet_bias.sh).

#### Step 3 — ChromBPNet model (once per assay)

```bash
chrombpnet pipeline \
    -ibw ${BW} \
    -d   ${ASSAY} \
    -g   genome.fa \
    -c   chrom.sizes \
    -p   peaks.narrowPeak \
    -n   ${OUT}/nonpeaks_negatives.bed \
    -fl  fold_0.json \
    -b   ${BIAS_OUT}/${ASSAY}/models/${sample}_bias.h5 \
    -o   ${OUT}/${ASSAY}
```

`-b` now takes the bias model trained in step 2 for the *same* assay. The output directory
must not already exist. See [`../05_run_chrombpnet.sh`](05_run_chrombpnet.sh).

#### Step 4 onwards — identical for both assays

These take a trained model and no longer need `-d` or a bigWig:

```bash
# contribution scores
chrombpnet contribs_bw -m models/chrombpnet_nobias.h5 \
    -r peaks.narrowPeak -g genome.fa -c chrom.sizes -op ${PREFIX}

# predicted signal tracks
chrombpnet pred_bw -bm models/bias_model_scaled.h5 \
    -cm models/chrombpnet.h5 -cmb models/chrombpnet_nobias.h5 \
    -r peaks.narrowPeak -g genome.fa -c chrom.sizes -op ${PREFIX}

# marginal footprints
chrombpnet footprints -m models/chrombpnet_nobias.h5 \
    -r nonpeaks_negatives.bed -g genome.fa -fl fold_0.json \
    -op ${PREFIX} -pwm_f motif_to_pwm.tsv
```

Use `chrombpnet_nobias.h5` for the bias-corrected model and `chrombpnet.h5` for the
uncorrected one; comparing the two marginal footprints shows how much enzymatic bias was
removed. See [`../06_chrombpnet_contribs.sh`](06_chrombpnet_contribs.sh),
[`../09_chrombpnet_pred.sh`](09_chrombpnet_pred.sh) and
[`../14_marginal_footprint.sh`](14_marginal_footprint.sh).

