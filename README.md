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
paper, including the cleaned test lists used below, is released
separately: **<https://github.com/sloina/mobvoi-audit>**.

## Contents

```
extract_features.py        stage 1: build the head's TRAINING pool - dump
                           backbone hidden-state windows around candidate
                           peaks of the train/dev splits to per-utterance
                           .npz files (derived from wekws/bin/score.py)
train_margin_head.py       stage 2: train the linear head (anchor, V1, V2,
                           V2.5, V3; hinge or logistic loss)
score_heads_protocols.py   stage 3: score every head on the TEST set in one
                           backbone pass, under the label-blind gated
                           protocol; writes wekws-format score files
frr_at_fa_exact.py         FRR at a target FA/h by exact event counting
                           (alarm events with a 500 ms refractory window)
run_snips_heads.ps1        reproduces Table 1 (Hey Snips, DS-TCN backbone)
run_mobvoi_heads.ps1       reproduces Table 2 (Mobvoi, MDTC backbone, four
                           cleaning levels of the test set)
checkpoints/               the exact frozen baseline detectors used in the
                           paper (avg_30.pt for each corpus), so results can
                           be reproduced without retraining the backbone
PATCH_wekws_feature_extraction.md   the one-method wekws addition needed
```

## Results

FRR (%) at the paper's operating points, linear V3 head vs. the frozen
baseline, full test sets (all variants, seeds and cleaned-set numbers in
the paper):

| Corpus (backbone) | Operating point | Baseline | +V3 head (386 params) |
|---|---|---|---|
| Hey Snips (DS-TCN, 22k) | 1.0 FA/h | 2.81 | **0.95 +/- 0.00** (logistic, 4 seeds) |
| Mobvoi Hi Xiaowen (MDTC, 160k) | 0.5 FA/h | 0.80 | **0.45 +/- 0.00** (hinge, 3 seeds) |
| Mobvoi Nihao Wenwen (MDTC, 160k) | 0.5 FA/h | 0.55 | **0.45 +/- 0.02** (hinge, 3 seeds) |

The contextual window requires 15 future frames (150 ms at a 10 ms
frame shift), which also subsumes the one-frame lookahead needed to
confirm an interior local maximum, plus any right context of the
backbone. For the segmented benchmarks, boundary frames are evaluated
one-sidedly. Away from recording boundaries, candidates are emitted
online with fixed lookahead.

## Training pool vs. test protocol

The two stages select candidate frames differently, and the difference
is deliberate.

**Training pool (stage 1, `extract_features.py`, train/dev splits).**
The utterance label is used. Every keyword utterance contributes one
window, at the global argmax of the baseline posterior after masking the
first 50 frames (matching the backbone's `min_duration`). Every negative
utterance (a non-keyword utterance; on Mobvoi this includes utterances of
the other keyword) contributes windows centered on up to 20 of its
highest-scoring interior local maxima with posterior >= 0.1; a negative
utterance with no such peak contributes its global argmax so that every
utterance is represented. The cap and the
fallback are sampling devices for building a finite training set only.

**Test protocol (stage 3, `score_heads_protocols.py`).** The label is
never read. For every utterance, every local maximum of the baseline
posterior that passes 0.1 is sent to the head - including a boundary
frame when its sole neighbour has no higher score. There is no cap and
no fallback: an utterance without such a peak produces no score. A keyword
is detected if any candidate crosses the head's threshold; false alarms
are counted as events with a 500 ms refractory window
(`frr_at_fa_exact.py`, identical counting to wekws `compute_det.py`).

The 0.1 gate is the first stage of the cascade: it restricts the head to
frames the baseline already considers plausible, i.e. to the region the
head was trained on. Local maxima are taken to avoid scoring runs of
adjacent frames; this is not a max-pooling requirement at inference.
Candidates are fixed by the frozen baseline; the head never moves them.
The baseline rows of the tables are the dense baseline posterior written
from the same backbone pass, evaluated with the same counting.

## Requirements

- Python 3.9+, PyTorch (CPU is sufficient for stages 2-3), NumPy, PyYAML
- [wekws](https://github.com/wenet-e2e/wekws) (Apache-2.0), installed and
  able to run its Hey Snips DS-TCN and Mobvoi MDTC recipes. The
  experiments used a copy of upstream `main` taken in August 2026; `main`
  is a moving target, so a later checkout may need small adjustments.
  `extract_features.py` and `score_heads_protocols.py` run from within a
  wekws checkout (they import `wekws.dataset`, `wekws.model`,
  `wekws.utils`) and call `model.forward_with_features(...)`, a small
  addition that returns the backbone hidden states alongside the
  per-frame posteriors: see `PATCH_wekws_feature_extraction.md`.
- **Data:** Mobvoi Hotwords is freely available from
  [OpenSLR SLR87](https://www.openslr.org/87/). The Hey Snips corpus is
  distributed by Sonos upon request (see the wekws Hey Snips recipe for
  the current procedure). The cleaned Mobvoi test lists come from the
  audit repository above.
- The run scripts are PowerShell (the experiments were run on Windows);
  they are thin loops over the three Python entry points and translate
  directly to bash.

## Pipeline

1. **Baseline detector.** Use the released checkpoints in `checkpoints/`
   (the exact frozen models from the paper), or retrain with the
   unmodified wekws recipe (DS-TCN on Hey Snips; MDTC on Mobvoi) and
   average checkpoints as usual. Single-run backbone training has
   visible seed variance, so retraining will shift the absolute numbers;
   the released checkpoints reproduce the paper exactly.
2. **Extract the training pool** for the train and dev splits (not the
   test split):

   ```
   python extract_features.py --config <recipe.yaml> \
       --test_data train_data.list \
       --checkpoint checkpoints/snips_dstcn_avg30.pt \
       --output_dir feats/train --context 15 \
       --min_duration 50            # positive-locating mask, train split
   ```

   `--output_dir` must be empty (or absent) before extraction: files are
   named by utterance key, and stale `.npz` files from an earlier run
   would silently join the pool.

   For Mobvoi, run once per keyword with `--context 31` (windows are
   stored wide and center-cropped at training time via `--stored_k 31`),
   setting `--keyword_index`/`--keyword_target` to the keyword's logit
   column and label id: 0/0 for Hi Xiaowen, 1/1 for Nihao Wenwen.
3. **Train, score and evaluate** with the two run scripts (edit the path
   block at the top of each). Each script trains every (variant, loss,
   seed) configuration to full convergence, runs one backbone pass over
   the test set that scores all heads under the gated protocol, and
   appends one exact FRR@FA line per configuration (and per cleaning
   level, for Mobvoi) to a summary file. Completed steps are skipped, so
   an interrupted sweep can simply be re-run.

## Citation

If you use this code, please cite:

```
A. Sloin, "Context-Enriched Heads for Wake-Word Detection,"
submitted to ICASSP 2027.
```

Author: Alba Sloin, Independent Researcher --
[ORCID 0009-0009-5953-1459](https://orcid.org/0009-0009-5953-1459).
A full citation will be added after the review process.

## License

Apache License 2.0 (see `LICENSE`). `extract_features.py` and
`score_heads_protocols.py` are derived from wekws (`wekws/bin/score.py`),
Copyright (c) 2021 Binbin Zhang, 2022 Shaoqing Yu, and retain its
license.
