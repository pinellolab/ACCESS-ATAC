# 03_accessbpnet

Trains **ChromBPNet** on ACCESS-ATAC, then uses the models for motif discovery, footprints
and variant scoring.

ChromBPNet was written for ATAC-seq and DNase-seq, which it counts as cut sites. The
ACCESS readout is base-level editing, which does not fit that. [`chrombpnet/`](chrombpnet/)
is therefore a modified copy. It adds an `ACCESS` assay type and accepts a bigWig directly.
The changes are described in [`chrombpnet/README.md`](chrombpnet/README.md).

Both readouts are modelled the same way. Only two things differ: `-d` and which bigWig is
passed. See [Using ChromBPNet](#using-chrombpnet).

## Running

Run the scripts in filename order. Notebooks are run interactively.

```bash
mkdir -p logs
sbatch 01_call_peaks.sh
sbatch 02_bam2bw.sh
...
```

Each script writes to `../../results/03_accessbpnet/<NN_name>/`. The exception is
`03_chrombpnet_bias/`, shared by steps 03 and 04. `logs/` must exist before you submit.

Conda environments: `chrombpnet` for the model steps, `macs2` for peak calling, `access`
for the rest. The scripts run `conda activate` but do not initialise conda, so your shell
must do that first.

Install the bundled ChromBPNet with `pip install -e chrombpnet/`.

The notebooks are committed without outputs. Step 11 needs an R kernel; the rest are
Python.

## Paths to adapt

Paths assume this directory sits at `<project_root>/scripts/03_accessbpnet/`.

- `../../results/01_process_access/02_filter/` — filtered ACCESS-ATAC BAMs, from pipeline 01
- `../../data/GRCh38/` — genome FASTA and chrom sizes
- `../../data/Blacklist/hg38-blacklist.v2.bed` — ENCODE hg38 blacklist v2
- `../../data/Motifs/JASPAR2026_CORE_vertebrates_*` — JASPAR motifs, PFM and MEME formats
- `../../data/Liver_caQTL/` — caQTL and background variants for the variant steps
- `../../data/encode_chipseq/K562/Bigwig/` — ENCODE ChIP-seq bigWig, for track plots
- `~/rgtdata/hg38/gencode.v21.annotation.gtf` — gene annotation, used by step 11
- `../../github/variant-scorer/` — a separate repository, called by step 16

`fold_0.json` in this directory holds the train/valid/test chromosome split.

## Scripts

### Inputs

**`01_call_peaks.sh`** — MACS2 on the ACCESS-ATAC BAM
(`--nomodel --nolambda --shift -100 --extsize 200 --keep-dup all -q 0.01`). Keeps chr1–22,
X and Y, then removes blacklist regions. Currently loops over HepG2 only.

**`02_bam2bw.sh`** — builds the two bigWigs with [`deamtools`](https://github.com/lzj1769/deamTools) `bam2bw`. Array job over
2 samples × 2 events. `--event tn5` gives `{sample}_atac.bw`; the default gives
`{sample}_access.bw`.

### Model training

**`03_prep_nonpeaks.sh`** — `chrombpnet prep nonpeaks`. Makes GC-matched background
regions. These do not depend on the assay, so both models reuse them.

**`04_chrombpnet_bias.sh`** — `chrombpnet bias pipeline`, once per assay. Array job over
2 samples × 2 assays. Each assay needs its own bias model, because Tn5 and DddSs have
different sequence bias.

**`05_run_chrombpnet.sh`** — `chrombpnet pipeline`, once per assay. Takes the bias model
from step 04. The output directory must not already exist.

**`06_chrombpnet_contribs.sh`** — `chrombpnet contribs_bw`. Contribution scores over the
peaks, from the bias-corrected model.

**`07_eval_chrombpnet.ipynb`** — collects `chrombpnet_metrics.json` from each run into one
table, so the four models can be compared.

**`09_chrombpnet_pred.sh`** — `chrombpnet pred_bw`. Predicted signal tracks from the bias,
full and bias-corrected models.

### Motif discovery

**`08_run_tfmodisco.sh`** — `modisco motifs` then `modisco report`. Array job over
2 samples × 2 assays × {counts, profile} scores, 8 tasks. Matches discovered motifs
against JASPAR.

**`10_extract_tfmodisco.ipynb`** — reads the TF-MoDISco HDF5 files and pulls out each
pattern's seqlet count, giving one table across all runs.

**`11_viz_tracks.ipynb`** — genome browser tracks with `Gviz`. An **R** notebook.

**`12_viz_contribution.ipynb`** — sequence logos of contribution scores, drawn with
`logomaker`.

### Marginal footprints

**`13_prepare_pwm.ipynb`** — writes `motif_to_pwm.tsv` for a short list of representative
TFs (CTCF, MAX, USF1, SP1 and others) from the JASPAR PFMs.

**`14_marginal_footprint.sh`** — `chrombpnet footprints`. Array job over 2 samples ×
2 assays × {corrected, uncorrected}, 8 tasks. `corrected` uses `chrombpnet_nobias.h5` and
`uncorrected` uses `chrombpnet.h5`. Comparing the two shows how much bias was removed.

**`20_viz_marginal_footprint.ipynb`** — plots the footprints from the HDF5 output.

### Variant effects

**`15_prepare_snp.ipynb`** — builds the variant list from the liver caQTL data. Splits
`variant_id` into alleles and combines caQTL variants with background variants from
non-caPeaks.

**`16_variant_score_prediction.sh`** — scores each variant with `variant_scoring.py` from
the external `variant-scorer` repository, using the bias-corrected HepG2 model. Array job
over the two assays.

**`17_eval_variant_score.ipynb`** — evaluates the scores. Takes the true label from the
`variant_id` suffix and computes average precision from `abs_logfc`.

**`18_eval_variant_score.ipynb`** — the same evaluation applied to an external ChromBPNet
run, for comparison. Not part of this pipeline's chain.

**`19_variant_effects_to_bw.ipynb`** — writes the per-variant effect sizes out as bigWig
tracks.

## Using ChromBPNet

Both readouts use the same commands. Only `-d` and the bigWig change:

- ATAC readout — `-d ATAC`, with `{sample}_atac.bw` (Tn5 counts)
- ACCESS readout — `-d ACCESS`, with `{sample}_access.bw` (edit counts)

Peaks, background regions and the fold split are shared. This keeps the two comparable.

Background regions, once per sample:

```bash
chrombpnet prep nonpeaks \
    -g genome.fa -p peaks.narrowPeak -c chrom.sizes \
    -fl fold_0.json -br blacklist.bed -o ${OUT}/nonpeaks
```

Bias model, once per assay:

```bash
chrombpnet bias pipeline \
    -ibw ${BW} -d ${ASSAY} \
    -g genome.fa -c chrom.sizes -p peaks.narrowPeak \
    -n ${OUT}/nonpeaks_negatives.bed -fl fold_0.json \
    -b 0.5 -o ${OUT}/${ASSAY} -fp ${sample}
```

ChromBPNet model, once per assay. `-b` now takes the bias model from the step above:

```bash
chrombpnet pipeline \
    -ibw ${BW} -d ${ASSAY} \
    -g genome.fa -c chrom.sizes -p peaks.narrowPeak \
    -n ${OUT}/nonpeaks_negatives.bed -fl fold_0.json \
    -b ${BIAS_OUT}/${ASSAY}/models/${sample}_bias.h5 \
    -o ${OUT}/${ASSAY}
```

Everything after this takes a trained model and needs no `-d`:

```bash
chrombpnet contribs_bw -m models/chrombpnet_nobias.h5 -r peaks.narrowPeak ...
chrombpnet pred_bw -bm models/bias_model_scaled.h5 -cm models/chrombpnet.h5 ...
chrombpnet footprints -m models/chrombpnet_nobias.h5 -r nonpeaks_negatives.bed ...
```
