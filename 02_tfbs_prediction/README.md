# 02_tfbs_prediction

Benchmarks how well **ACCESS-ATAC** signal predicts transcription factor binding sites,
compared against sequence alone.

For each (cell type, TF) pair, a binary classifier is trained to separate ChIP-seq peaks
from GC- and chromosome-matched background, and the same architecture is retrained with
different input signals. The comparison between those input sets is the result.

| `--assay` value | Model inputs |
|---|---|
| `seq` | one-hot sequence only |
| `accessatac_tn5` | sequence + ACCESS-ATAC Tn5 insertions |
| `accessatac_dddss` | sequence + ACCESS-ATAC deaminase edits |
| `accessatac_both` | sequence + both ACCESS-ATAC signals |

The classifier source is bundled in [`model/`](model/), so this directory is
self-contained: it downloads the ChIP-seq peaks, builds the training data, trains the
model, and evaluates the output.

## Running

```bash
mkdir -p logs
# run 01_download_peaks.ipynb
sbatch 02_submit.sh      # array job, one task per (cell type, TF)
sbatch 03_submit.sh      # array job
sbatch 04_train.sh       # requires a GPU
# then run 05_eval.ipynb and 06_plot_curve.ipynb
```

Steps 02 and 03 are SLURM array jobs sized `--array=0-773%50` for 774 (cell type, TF)
tasks. Each builds its own task list at submission time — step 02 from the peak `.bed`
files on disk, step 03 from `tfbs_peaks_metadata.csv` — and exits quietly when the array
index exceeds the number of tasks found, so the array bound only needs to be large enough.
Adjust it to match your own TF count.

Both Python scripts take `--cell_type` and `--tf`, and skip work whose output already
exists, so a partially failed array can be resubmitted as-is.

### Inputs from outside this directory

| Path | Provides |
|---|---|
| `../../data/encode_chipseq/{K562,HepG2}.txt` | ENCODE download URLs, one per line, read by step 01 |
| `../../results/01_process_access/06_bam2bw_access_atac/` | ACCESS-ATAC bigWigs — `{cell}_atac_noext.bw` (Tn5) and `{cell}_access_noext.bw` (edits) |
| `../46_accessbpnet/fold_0.json` | train / valid / test chromosome split |
| `../../data/GRCh38/GRCh38.primary_assembly.genome.fa` | reference genome |
| `../../data/Blacklist/hg38-blacklist.v2.bed` | ENCODE hg38 blacklist v2 |

Conda environments: `access` for steps 02–03, `cell2net` for step 04. The scripts call
`conda activate` but do not initialise conda, so it must already be initialised in the
submitting shell.

Paths are relative to a project root two levels above this directory, i.e. the layout
assumed is `<project_root>/scripts/02_tfbs_prediction/`.

**Note on the step 01 output path.** Steps 02–06 write to
`../../results/02_tfbs_prediction/<NN_name>/`, matching the script that produces each
directory. Step 01 instead writes to
`../../results/27_compare_tfbs_prediction/01_download_peaks/`, and steps 02 and 03 read
the peaks back from there. That path predates this pipeline; it is the one place where the
numbering convention does not hold.

## The model (`model/`, diagram in `network/`)

A binary classifier over a 256 bp window. `SeqEncoder` embeds the one-hot sequence with a
2D convolution tower (a stem convolution, then four conv/BatchNorm/ELU/MaxPool blocks each
halving the sequence length, with a residual 1×1 convolution per block). Each supplied
signal track is projected to the same dimension by an MLP (`Linear→ReLU→Dropout→Linear`)
and **added** to the sequence embedding; an MLP head then emits a single logit. `--assay`
controls which signal tracks are added, which is what the four settings vary.

| File | Description |
|---|---|
| `model/train.py` | Training entry point. `BCEWithLogitsLoss`, Adam (lr 3e-4, weight decay 1e-4), `ReduceLROnPlateau` (patience 2, factor 0.5), early stopping after 10 epochs without a new best validation loss. Writes the best-validation checkpoint, a per-epoch loss CSV and curve, test-set predictions, and a precision–recall curve with AUPR to `--model_dir`, `--log_dir`, `--pred_dir` and `--metric_dir`. |
| `model/model.py` | `ACCESSNet`, the architecture described above. Also defines an `AttentionBlock` that the current forward pass does not use. |
| `model/seq_encoder.py` | `SeqEncoder`, the convolution tower. |
| `model/dataset.py` | `ChromatinAccessibilityDataSet` and `get_dataloader()`; each item is a dict with `seq`, `signal_accessatac_tn5`, `signal_accessatac_dddss` and `label`. |
| `model/utils.py` | `set_seed()`, `one_hot_encode()`, `pad_and_split()`, `random_seq()`. |
| `network/accessnet.tex` | TikZ source for the architecture diagram below. Compile with `pdflatex accessnet.tex`. |
| `network/accessnet.pdf` | The rendered diagram. |

### Diagram

[`network/accessnet.pdf`](network/accessnet.pdf) draws the architecture. Its contents were
checked against the instantiated model at `peak_len=256`:

| Diagram | Code |
|---|---|
| Stem conv, 4 → 64 filters, width 5, ELU | `Conv2d(4, 64, kernel_size=(1,5), padding="same")` + `ELU` |
| Conv block × 4: width 3, BatchNorm, ELU, MaxPool÷2, 1×1 conv with residual | 4 blocks of `Conv2d(k=(1,3))` + `BatchNorm2d` + `ELU` + `MaxPool2d((1,2), stride=(1,2))`, each followed by `Conv2d(k=(1,1))` + `ELU` added back |
| filters 64 → 32 → 32 → 16 → 16 | block channels 64→32, 32→32, 32→16, 16→16 |
| `z_seq` ∈ ℝ^(L'·d), L' = L/16, d = 16 | encoder output `(B, 16, 1, 16)` for L = 256; `embd_len` = 16 × 16 = 256 |
| Signal branch: Linear L→32, ReLU, Dropout, Linear 32→L'·d | `Linear(256, 32)`, `ReLU`, `Dropout(0.25)`, `Linear(32, 256)` |
| Fusion: element-wise sum | `seq_embd + atac_signal + access_signal` |
| Head: Linear L'·d→32, ReLU, BatchNorm, Dropout; Linear 32→1 | `Flatten`, `Linear(256, 32)`, `ReLU`, `BatchNorm1d(32)`, `Dropout(0.25)`, `Linear(32, 1)` |
| Output ŷ with BCE loss | `BCEWithLogitsLoss` on a single logit |

The unused `AttentionBlock` in `model.py` is correctly absent from the diagram.

The diagram labels the two signal inputs "ATAC signal" and "ACCESS signal", following this
repository's convention where *ATAC* is the Tn5 readout and *ACCESS* the deaminase readout
of the same library. Both branches are fed ACCESS-ATAC tracks — `accessatac_tn5` and
`accessatac_dddss` respectively — not conventional ATAC-seq.

`train.py` imports its siblings by bare module name (`from model import ACCESSNet`), which
resolves against `model/` because Python puts the script's own directory first on the
import path. `04_train.sh` therefore calls `python model/train.py` from this directory; do
not move `train.py` away from the other four files.

`train.py` reads the npz keys `seq`, `signal_accessatac_tn5`, `signal_accessatac_dddss`
and `label` — exactly what `03_prepare_data.py` writes. `peak_len` is hard-coded to 256 in
`train.py` to match the 256 bp windows from step 02.

`ACCESSNet.forward()` takes at most two signal tracks, so `--assay` dispatches as follows.
The two signal-embedding modules are named `atac_signal_embd` and `access_signal_embd`;
whichever track is passed first goes through the first module.

| `--assay` | Call | Tracks added to the sequence embedding |
|---|---|---|
| `seq` | `model(seq)` | none |
| `accessatac_tn5` | `model(seq, accessatac_tn5)` | ACCESS-ATAC Tn5 |
| `accessatac_dddss` | `model(seq, accessatac_dddss)` | ACCESS-ATAC edits |
| `accessatac_both` | `model(seq, accessatac_tn5, accessatac_dddss)` | both ACCESS-ATAC tracks |

## Notes

- The notebooks are committed **without outputs**.
- `04_train.sh` trains four models per TF sequentially inside a single SLURM job. With
  hundreds of TFs this is long-running (the script requests 120 h); the existing-output
  check makes it safe to resubmit after a timeout.
- `03_prepare_data.py` reads the `noext` bigWigs. See the note in
  [`../01_process_access/README.md`](../01_process_access/README.md) about which bigWig
  variants its active task tables actually generate.
- `01_download_peaks.ipynb`'s skip-if-exists check never fires: it tests for
  `{name}.bed.gz`, but `gunzip` removes the `.gz` and the file kept on disk is
  `{name}.bed`. Re-running the notebook therefore re-downloads every peak file.
