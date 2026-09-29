# 02_tfbs_prediction

Tests how well **ACCESS-ATAC** signal predicts TF binding sites, compared to sequence alone.

For each (cell type, TF) pair, a classifier separates ChIP-seq peaks from matched
background. The same model is then retrained with different inputs, set by `--assay`:

- `seq` — one-hot sequence only
- `accessatac_tn5` — sequence + Tn5 insertions
- `accessatac_dddss` — sequence + deaminase edits
- `accessatac_both` — sequence + both tracks

Everything else stays fixed, so any difference comes from the signal.

The classifier is bundled in [`model/`](model/), so this directory is self-contained.

## Running

```bash
mkdir -p logs
# run 01_download_peaks.ipynb
sbatch 02_submit.sh      # array job, one task per (cell type, TF)
sbatch 03_submit.sh      # array job
sbatch 04_train.sh       # needs a GPU
# then run 05_eval.ipynb and 06_plot_curve.ipynb
```

Conda environments: `access` for steps 02–03, `cell2net` for step 04. The scripts run
`conda activate` but do not initialise conda, so your shell must do that first.

Steps 02–06 write to `../../results/02_tfbs_prediction/<NN_name>/`. Step 01 is the
exception: it writes to `../../results/27_compare_tfbs_prediction/01_download_peaks/`,
and steps 02 and 03 read from there.

## Paths to adapt

Paths assume this directory sits at `<project_root>/scripts/02_tfbs_prediction/`.

- `../../data/encode_chipseq/{K562,HepG2}.txt` — ENCODE download URLs, one per line
- `../../results/01_process_access/06_bam2bw_access_atac/` — the two ACCESS-ATAC bigWigs
- `../46_accessbpnet/fold_0.json` — train/valid/test chromosome split, shared with pipeline 03
- `../../data/GRCh38/GRCh38.primary_assembly.genome.fa` — reference genome
- `../../data/Blacklist/hg38-blacklist.v2.bed` — ENCODE hg38 blacklist v2

## Scripts

**`01_download_peaks.ipynb`** — downloads ENCODE ChIP-seq peaks with `wget`, then gunzips
and sorts them. The URL's file stem becomes the TF name used everywhere downstream. Also
writes `tfbs_peaks_metadata.csv`, which step 03 uses as its task list. Note: the
skip-if-exists check looks for `{name}.bed.gz`, but gunzip leaves `{name}.bed`, so
re-running downloads everything again.

**`02_create_labels.py`** — builds labels for one TF. Positives are 256 bp windows on
narrowPeak summits. Negatives are random regions matched to the positives on count,
chromosome, width and GC content. GC matching matters: without it a model can score well
on base composition alone. Output is a TSV of `chrom start end label`. Seed 42.

**`02_submit.sh`** — SLURM array wrapper for step 02. Builds its task list by globbing
`*.bed` in the peak directory. Exits quietly if the array index is past the end, so
`--array=0-773%50` just needs to be large enough.

**`03_prepare_data.py`** — turns labels into model input. One-hot encodes the 256 bp
sequence and extracts the two ACCESS-ATAC tracks, each `log1p`-transformed. Splits into
train/valid/test **by chromosome** using `fold_0.json`, so no test region shares a
chromosome with training data. Writes one npz per split.

**`03_submit.sh`** — array wrapper for step 03. Same as `02_submit.sh`, but takes its task
list from `tfbs_peaks_metadata.csv`.

Both Python steps skip existing output, so a failed array can be resubmitted as-is.

**`04_train.sh`** — trains every TF × assay combination by calling `model/train.py`. Skips
combinations that already have a prediction CSV. Four models per TF run sequentially in
one job, so this is slow; the script asks for 120 hours. Needs a GPU.

**`05_eval.ipynb`** — reads the test predictions. Computes precision–recall curves, AUPR,
and **precision at fixed recall** (0.01–0.4). Precision at low recall is the metric that
matters here, because binding sites are rare. Writes per-TF CSVs and plots.

**`06_plot_curve.ipynb`** — redraws the curves for `recall ≤ 0.2`, one figure per TF, with
a fixed colour per assay.

## The model

`model/` holds the classifier.

**`model/seq_encoder.py`** — `SeqEncoder`, the sequence branch. A stem convolution, then
four conv/BatchNorm/ELU/MaxPool blocks. Each block halves the sequence length and has a
residual 1×1 convolution. A 256 bp input becomes a `(B, 16, 1, 16)` embedding.

**`model/model.py`** — `ACCESSNet`. Each signal track is projected by a small MLP and
**added** to the sequence embedding. An MLP head then emits one logit. Addition keeps the
head's input size fixed no matter how many tracks are given, which is why all four assays
share one architecture. `forward()` takes at most two tracks. Also defines an unused
`AttentionBlock`.

**`model/train.py`** — the entry point. `BCEWithLogitsLoss`, Adam (lr 3e-4, weight decay
1e-4), `ReduceLROnPlateau`, early stopping after 10 epochs without improvement. Writes the
best checkpoint, a loss curve, test predictions and a PR curve. `peak_len` is hard-coded
to 256.

**`model/dataset.py`** — `ChromatinAccessibilityDataSet` and `get_dataloader()`.

**`model/utils.py`** — `set_seed()`, `one_hot_encode()`, `pad_and_split()`, `random_seq()`.

`train.py` imports its siblings by bare module name, which resolves inside `model/`.
`04_train.sh` runs it as `python model/train.py`, so keep the five files together.

## Architecture diagram

`network/accessnet.pdf` draws the network. `network/accessnet.tex` is the TikZ source;
rebuild with `pdflatex accessnet.tex`. The diagram was checked against the model and
matches.

It labels the two inputs "ATAC signal" and "ACCESS signal". Here *ATAC* means the Tn5
readout and *ACCESS* the deaminase readout of the same library. Both are ACCESS-ATAC
tracks. Neither is conventional ATAC-seq.
