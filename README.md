# FastGatedConformer

This repository contains the code and reproducibility materials for:

**FastGatedConformer: A Lightweight Gated Temporal Mixer for Cross-Subject EEG Motor Imagery Decoding**

FastGatedConformer is a lightweight modification of DBConformer for cross-subject EEG motor imagery decoding. It keeps the temporal and spatial EEG embedding design, replaces the temporal Transformer with a gated temporal mixer, and simplifies the spatial branch using channel-attentive pooling.

## Main Results

The experiments use leave-one-subject-out evaluation on three public motor imagery EEG datasets.

| Dataset | DBConformer | FastGatedConformer | Parameters Reduction vs. DBConformer |
|---|---:|---:|---:|
| BNCI2014001 | 77.099 | 76.343 | 73.22% |
| BNCI2014002 | 76.857 | 77.343 | 81.90% |
| BNCI2014004 | 69.278 | 70.333 | 74.39% |
| Macro average | 74.411 | 74.673 | - |

Compared with DBConformer, FastGatedConformer preserves comparable accuracy while reducing parameters and training runtime. Compared with EEGNet and ShallowConvNet, it obtains higher macro-average accuracy under the same LOSO protocol.

## Repository Structure

```text
DBConformer_LOSO.py              Main LOSO training/evaluation entry
run_batch_experiments.py         FastGatedConformer ablation runner
run_baseline_experiments.py      EEGNet and ShallowConvNet baseline runner
models/                          Model definitions
utils/                           Data loading, training utilities, logging
tools/                           Log summarization, plotting, statistics
docs/                            Draft manuscript and exported result tables
```

## Data

The datasets used in the paper are publicly available motor imagery EEG datasets:

- BNCI2014001
- BNCI2014002
- BNCI2014004

The data can be obtained through the public BNCI Horizon 2020 repository and the MOABB ecosystem, subject to the original dataset licenses and access conditions.

This release does not include raw or processed EEG data. Put preprocessed files under:

```text
data/BNCI2014001/X.npy
data/BNCI2014001/labels.npy
data/BNCI2014002/X.npy
data/BNCI2014002/labels.npy
data/BNCI2014004/X.npy
data/BNCI2014004/labels.npy
```

## Environment

The experiments were developed with Python and PyTorch. A typical environment should include:

```text
torch
numpy
pandas
scikit-learn
mne
moabb
matplotlib
```

Install dependencies according to your local CUDA/PyTorch version. For plotting and statistical tables only, `matplotlib` and `pandas` are sufficient.

## Run Experiments

Run FastGatedConformer on the three datasets:

```bash
python DBConformer_LOSO.py --backbone FastGatedConformer --data-names BNCI2014001 BNCI2014002 BNCI2014004 --seeds 1 2 3 4 5 --max-epoch 100 --device-id 0
```

Run DBConformer baseline:

```bash
python DBConformer_LOSO.py --backbone DBConformer --data-names BNCI2014001 BNCI2014002 BNCI2014004 --seeds 1 2 3 4 5 --max-epoch 100 --device-id 0
```

Run classic baselines:

```bash
python run_baseline_experiments.py --baselines EEGNet shallow --datasets BNCI2014001 BNCI2014002 BNCI2014004 --device-id 0
```

Run FastGatedConformer ablations:

```bash
python run_batch_experiments.py --device-id 0
```

## Summarize Logs

```bash
python tools/summarize_logs.py --dataset BNCI2014001
python tools/summarize_logs.py --dataset BNCI2014002
python tools/summarize_logs.py --dataset BNCI2014004
```

## Plot Figures and Statistics

The paper result CSV files are included under `docs/bspc_submission/figure_data/`.

Generate figures:

```bash
python tools/plot_bspc_figures.py
```

Generate statistical tests:

```bash
python tools/statistical_test.py
```

## Notes

- `data/`, `logs/`, `runs/`, checkpoints, and local IDE files are intentionally excluded.
- The SVG files in the manuscript preparation folder are reference sketches only and should not be submitted as final journal figures.
- Please cite DBConformer, EEGNet, ShallowConvNet/DeepConvNet, MOABB, and the original dataset papers when using this code.
