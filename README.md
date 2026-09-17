# Context-Enriched Heads for Wake-Word Detection

Code accompanying the ICASSP 2027 submission:

> A. Sloin, *"Context-Enriched Heads for Wake-Word Detection"*,
> submitted to ICASSP 2027.

A frozen max-pooling wake-word detector is used as a localizer; a small
linear head (65-386 parameters) is then trained on pooled statistics of
the backbone hidden states in a +/-150 ms window around each candidate
peak, replacing the peak-posterior threshold decision. Head training and
inference run entirely on CPU.

The label-noise audit of the Mobvoi corpus described in Sec. 4.3 of the
paper is released separately:
**<https://github.com/sloina/mobvoi-audit>**.

## Contents

```
extract_features.py    stage 1: dump backbone hidden-state windows around
                       candidate peaks to per-utterance .npz files
                       (derived from wekws/bin/score.py, Apache-2.0)
train_margin_head.py   stage 2: train the linear head (V1/V2/V2.5/V3,
                       hinge or logistic loss) on the .npz features
score_margin.py        score a test set with a trained head, writing
                       wekws-format score files
frr_at_fa_exact.py     FRR at a target FA/h by exact event counting
                       (event-based alarms with a refractory window)
run_snips_heads.ps1    reproduces Table 1 (Hey Snips, DS-TCN backbone)
run_mobvoi_heads.ps1   reproduces the head rows of Table 2 (Mobvoi, MDTC)
checkpoints/           the exact frozen baseline detectors used in the
                       paper (avg_30.pt for each corpus), so results can
                       be reproduced without retraining stage 1
```

## Results

FRR (%) at the paper's operating points, linear V3 head vs. the frozen
baseline (details, all variants, and cleaned-set numbers in the paper):

| Corpus (backbone) | Operating point | Baseline | +V3 head (386 params) |
|---|---|---|---|
| Hey Snips (DS-TCN, 22k) | 1.0 FA/h | 2.81 | **0.79 +/- 0.03** (logistic) |
| Mobvoi Hi Xiaowen (MDTC, 160k) | 0.5 FA/h | 0.80 | **0.56 +/- 0.05** (hinge) |
| Mobvoi Nihao Wenwen (MDTC, 160k) | 0.5 FA/h | 0.55 | **0.47 +/- 0.01** (hinge) |

Head training and inference run on CPU; the added decision latency is
150 ms (15 future frames).

## Requirements

- Python 3.9+, PyTorch (CPU is sufficient for stages 2-3), NumPy, PyYAML
- [wekws](https://github.com/wenet-e2e/wekws) (Apache-2.0), installed and
  able to train/evaluate its Hey Snips DS-TCN and Mobvoi MDTC recipes.
  `extract_features.py` runs from within a wekws checkout (it imports
  `wekws.dataset`, `wekws.model`, `wekws.utils`).
- **Note:** `extract_features.py` calls `model.forward_with_features(...)`,
  a small addition to the wekws KWS model that returns the backbone hidden
  states alongside the per-frame posteriors. The required modification is
  described in `PATCH_wekws_feature_extraction.md`.
- **Data:** Mobvoi Hotwords is freely available from
  [OpenSLR SLR87](https://www.openslr.org/87/). The Hey Snips corpus is
  distributed by Sonos upon request (see the wekws Hey Snips recipe for
  the current procedure).
- The run scripts are PowerShell (the experiments were run on Windows);
  they are thin loops over the three Python entry points and translate
  directly to bash.

## Pipeline

1. **Baseline detector.** Use the released checkpoints in `checkpoints/`
   (the exact frozen models from the paper), or retrain with the
   unmodified wekws recipe (DS-TCN on Hey Snips; MDTC on Mobvoi) and
   average checkpoints as usual. Note that single-run baseline training
   has visible seed variance, so retraining will shift the absolute
   numbers; the released checkpoints reproduce the paper exactly.
2. **Extract features** for each split (train/dev/test):

   ```
   python extract_features.py --config <recipe.yaml> \
       --test_data <split>_data.list \
       --checkpoint checkpoints/snips_dstcn_avg30.pt \
       --output_dir feats/<split> --context 15 \
       --min_duration 50            # positives-locating mask, train split
   ```

   For Mobvoi, run per keyword with `--context 31` (windows are stored
   wide and center-cropped at training time via `--stored_k`), setting
   `--keyword_index`/`--keyword_target` to the keyword's logit column and
   label id: 0/0 for Hi Xiaowen, 1/1 for Nihao Wenwen.
3. **Train, score, and evaluate the heads** with the two run scripts
   (edit the path block at the top of each). Each script trains every
   (variant, loss, seed) configuration to full convergence, scores the
   test set, and appends one exact FRR@FA line per configuration to a
   summary file. Completed configurations are skipped, so an interrupted
   sweep can simply be re-run.

## Citation

If you use this code, please cite the paper above. A full reference will
be added after the review process.

Author: Alba Sloin, Independent Researcher --
[ORCID 0009-0009-5953-1459](https://orcid.org/0009-0009-5953-1459).

## License

Apache License 2.0 (see `LICENSE`). `extract_features.py` is derived from
wekws (`wekws/bin/score.py`), Copyright (c) 2021 Binbin Zhang, 2022
Shaoqing Yu, and retains its original license header.
