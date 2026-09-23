# Copyright (c) 2021 Binbin Zhang(binbzha@qq.com)
#               2022 Shaoqing Yu(954793264@qq.com)
#               2026 Margin-head experiment: test-time head scoring
#
# Based on wekws/bin/score.py (data loading and backbone pass).
#
# Licensed under the Apache License, Version 2.0 (the "License");
#
# Stage 3: test-time scoring of the linear heads, in ONE backbone pass.
#
# Test protocol (label-blind, identical for every head and every
# utterance):
#   candidates = every local maximum of the baseline posterior p_t with
#                p_t >= --pre_thresh (0.1). Boundary frames are admitted
#                one-sidedly (t = 0 if p_0 >= p_1; t = T-1 if
#                p_{T-1} >= p_{T-2}), i.e. the sequence is padded with
#                -inf on both sides. No cap on the number of candidates,
#                no argmax fallback: an utterance without a candidate
#                yields no score at all.
#   each head scores only these candidates, through the SAME
#   cut_window / build_feature code used at training time (zero padding
#   at utterance edges included), so the feature semantics match the
#   training pool exactly.
#
# One score file per (keyword, head), in wekws format: a frame sequence
# of zeros with the head's sigmoid score at each candidate frame.
# frr_at_fa_exact.py then applies unchanged: the smallest threshold whose
# event-based alarm count (crossings merged by the 50-frame refractory
# walk) fits the FA budget; a positive is rejected iff no frame crosses
# it.
#
# Several keywords sharing one backbone (Mobvoi: one model, two
# classifier rows) are scored in the same pass. Jobs are given as a JSON
# list:
# [
#   {"name": "kw0", "keyword": "<HI_XIAOWEN>",  "keyword_index": 0,
#    "heads": ["runs/mobvoi_kw0_*.pt"], "gated_dir": "scores/kw0",
#    "baseline_out": "scores/baseline_kw0.score"},
#   {"name": "kw1", "keyword": "<NIHAO_WENWEN>", "keyword_index": 1,
#    "heads": ["runs/mobvoi_kw1_*.pt"], "gated_dir": "scores/kw1"}
# ]
# baseline_out (optional) writes the dense baseline posterior p_t of that
# keyword for every utterance, from the same pass, for the baseline rows
# of the tables.
#
# Latency note: the head window needs K future frames (K=15 => 150 ms at
# a 10 ms step), which subsumes the one-frame lookahead of the interior
# local-maximum test, plus any backbone right-context. On the segmented
# benchmarks the final frame is tested one-sidedly against the utterance
# end.

from __future__ import print_function

import argparse
import copy
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from wekws.dataset.init_dataset import init_dataset
from wekws.model.kws_model import init_model
from wekws.utils.checkpoint import load_checkpoint
from wenet.text.char_tokenizer import CharTokenizer


# ---------------------------------------------------------------------------
# feature path: identical to train_margin_head / extract_features
# ---------------------------------------------------------------------------

def build_feature(windows, peak_scores, variant, K):
    """Must match train_margin_head.build_feature exactly."""
    N, W, H = windows.shape
    assert W == 2 * K + 1
    t_star = K

    if variant == 'anchor':
        return windows[:, t_star, :]

    mean_t = windows.mean(axis=1)

    if variant == 'v1':
        return mean_t

    center = windows[:, t_star, :]
    ps = peak_scores.reshape(-1, 1)

    if variant == 'v2':
        return np.concatenate([mean_t, center, ps], axis=1)

    third = W // 3
    m_pre = windows[:, :third, :].mean(axis=1)
    m_mid = windows[:, third:W - third, :].mean(axis=1)
    m_post = windows[:, W - third:, :].mean(axis=1)

    if variant == 'v2.5':
        return np.concatenate([m_pre, m_mid, m_post], axis=1)

    if variant == 'v3':
        max_t = windows.max(axis=1)
        std_t = windows.std(axis=1)
        return np.concatenate(
            [m_pre, m_mid, m_post, center, max_t, std_t, ps], axis=1)

    raise ValueError(f'unknown variant {variant}')


def cut_window(h, t, K):
    """Zero-padded window h[t-K : t+K+1] - identical to extract_features."""
    T, hdim = h.shape
    lo, hi = max(0, t - K), min(T, t + K + 1)
    w = np.zeros((2 * K + 1, hdim), dtype=np.float32)
    offset = lo - (t - K)
    w[offset:offset + (hi - lo)] = h[lo:hi]
    return w


def find_peaks(p, pre_thresh):
    """Local maxima above the pre-threshold, with boundary frames evaluated
    one-sidedly for the segmented benchmarks (t=0 if p[0] >= p[1];
    t=T-1 if p[T-1] >= p[T-2]). No cap or argmax fallback."""
    T = len(p)
    if T == 0:
        return []
    if T == 1:
        return [0] if p[0] >= pre_thresh else []
    cand = []
    if p[0] >= pre_thresh and p[0] >= p[1]:
        cand.append(0)
    cand.extend(t for t in range(1, T - 1)
                if p[t] >= pre_thresh and p[t] >= p[t - 1]
                and p[t] >= p[t + 1])
    if p[T - 1] >= pre_thresh and p[T - 1] >= p[T - 2]:
        cand.append(T - 1)
    return cand


# ---------------------------------------------------------------------------
# head loading (train-set standardisation folded into the weights)
# ---------------------------------------------------------------------------

def load_heads(patterns):
    paths = []
    for pat in patterns:
        hits = sorted(glob.glob(pat))
        if not hits and os.path.exists(pat):
            hits = [pat]
        if not hits:
            sys.exit(f'no head checkpoint matches {pat}')
        paths.extend(hits)
    paths = sorted(set(paths))

    groups = {}
    for path in paths:
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        variant = ckpt['variant']
        K = int(ckpt['context'])
        mu = np.asarray(ckpt['mu'], dtype=np.float64)
        sd = np.asarray(ckpt['sd'], dtype=np.float64)
        lin = nn.Linear(mu.shape[0], 1)
        lin.load_state_dict(ckpt['state_dict'])
        w = lin.weight.detach().numpy().astype(np.float64).reshape(-1)
        b = float(lin.bias.detach().numpy().reshape(-1)[0])
        # (x - mu) / sd . w + b  ==  x . (w / sd) + (b - (mu / sd) . w)
        w_eff = (w / sd).astype(np.float32)
        b_eff = np.float32(b - float(np.dot(w, mu / sd)))

        g = groups.setdefault((variant, K), {'tags': [], 'W': [], 'b': []})
        tag = os.path.splitext(os.path.basename(path))[0]
        if tag in g['tags']:
            sys.exit(f'duplicate head tag {tag} ({path})')
        g['tags'].append(tag)
        g['W'].append(w_eff)
        g['b'].append(b_eff)
        print(f'loaded {os.path.basename(path)}: variant {variant}, K={K}, '
              f'dim={mu.shape[0]}, dev FRR was '
              f"{ckpt.get('dev_frr', float('nan')):.4f}")
    for g in groups.values():
        g['W'] = np.stack(g['W']).astype(np.float32)          # (n, D)
        g['b'] = np.asarray(g['b'], dtype=np.float32)          # (n,)
    return groups


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser(
        description='stage 3: score every head on the label-blind gated '
                    'candidates of the test set, in one backbone pass')
    parser.add_argument('--config', required=True, help='backbone config')
    parser.add_argument('--test_data', required=True, help='data list file')
    parser.add_argument('--dict', default='./dict', help='dict dir')
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--checkpoint', required=True,
                        help='backbone checkpoint')
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--pin_memory', action='store_true', default=False)
    parser.add_argument('--prefetch', default=100, type=int)
    parser.add_argument('--pre_thresh', type=float, default=0.1,
                        help='candidate gate on the baseline posterior')
    parser.add_argument('--jobs', required=True,
                        help='JSON file: list of {name, keyword, '
                             'keyword_index, heads, gated_dir, '
                             '[baseline_out]}')
    parser.add_argument('--expected_utts', type=int, default=None,
                        help='fail unless exactly this many utterances '
                             'were processed (catches silently-skipped '
                             'audio, e.g. a missing data disk)')
    return parser.parse_args()


def fmt(vec):
    # '0' instead of '0.000000': information-identical (float('0') == 0.0
    # and 6-decimal formatting already floors tiny values), but shrinks
    # the mostly-zero gated files ~4x.
    out = []
    for v in vec:
        s = '{:.6f}'.format(v)
        out.append('0' if s == '0.000000' else s)
    return ' '.join(out)


def main():
    args = get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    # utf-8-sig: Windows PowerShell writes UTF8 with a BOM
    with open(args.jobs, 'r', encoding='utf-8-sig') as fin:
        jobs = json.load(fin)
    if not isinstance(jobs, list) or not jobs:
        sys.exit(f'{args.jobs} must hold a non-empty JSON list')

    for job in jobs:
        for field in ('name', 'keyword', 'keyword_index', 'heads',
                      'gated_dir'):
            if field not in job:
                sys.exit(f'job missing field {field}: {job}')
        if job['heads']:
            job['groups'] = load_heads(job['heads'])
            job['Kmax'] = max(K for (_, K) in job['groups'])
        elif job.get('baseline_out'):
            job['groups'] = {}          # baseline-only job
            job['Kmax'] = 0
        else:
            sys.exit(f"job {job['name']}: no heads and no baseline_out")
        n = sum(len(g['tags']) for g in job['groups'].values())
        print(f"job {job['name']}: {n} heads, keyword_index "
              f"{job['keyword_index']}, pre_thresh {args.pre_thresh}, "
              f"baseline_out={job.get('baseline_out')}")

    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)

    test_conf = copy.deepcopy(configs['dataset_conf'])
    test_conf['filter_conf']['max_length'] = 102400
    test_conf['filter_conf']['min_length'] = 0
    test_conf['filter_conf']['min_output_input_ratio'] = 0
    test_conf['speed_perturb'] = False
    test_conf['spec_aug'] = False
    test_conf['shuffle'] = False
    feats_type = test_conf.get('feats_type', 'fbank')
    test_conf[f'{feats_type}_conf']['dither'] = 0.0
    test_conf['batch_conf']['batch_size'] = args.batch_size

    tokenizer = CharTokenizer(f'{args.dict}/dict.txt',
                              f'{args.dict}/words.txt',
                              unk='<filler>')
    test_dataset = init_dataset(data_list_file=args.test_data,
                                conf=test_conf, tokenizer=tokenizer,
                                split='test')
    loader_kwargs = {}
    if args.num_workers > 0:      # prefetch_factor is invalid with 0 workers
        loader_kwargs['prefetch_factor'] = args.prefetch
    test_data_loader = DataLoader(test_dataset,
                                  batch_size=None,
                                  pin_memory=args.pin_memory,
                                  num_workers=args.num_workers,
                                  **loader_kwargs)

    model = init_model(configs['model'])
    load_checkpoint(model, args.checkpoint)
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    model = model.to(device)
    model.eval()

    # open output files
    for job in jobs:
        os.makedirs(job['gated_dir'], exist_ok=True)
        job['fouts'] = {}
        for g in job['groups'].values():
            for tag in g['tags']:
                job['fouts'][tag] = open(
                    os.path.join(job['gated_dir'], tag + '.score'),
                    'w', encoding='utf8')
        job['fout_b'] = None
        if job.get('baseline_out'):
            os.makedirs(os.path.dirname(job['baseline_out']) or '.',
                        exist_ok=True)
            job['fout_b'] = open(job['baseline_out'], 'w', encoding='utf8')
        job['num_peaks'] = 0
        job['num_empty'] = 0

    num_utts = 0
    with torch.no_grad():
        for batch_idx, batch_dict in enumerate(test_data_loader):
            keys = batch_dict['keys']
            feats = batch_dict['feats'].to(device)
            feats_lengths = batch_dict['feats_lengths']

            probs, hidden, _ = model.forward_with_features(feats)
            probs = probs.cpu().numpy()      # (B, T, odim)
            hidden = hidden.cpu().numpy()    # (B, T, hdim)

            for i in range(len(keys)):
                key = keys[i]
                T = int(feats_lengths[i].item())
                h = hidden[i, :T, :]
                num_utts += 1

                for job in jobs:
                    p = probs[i, :T, job['keyword_index']]
                    keyword = job['keyword']

                    cand = find_peaks(p, args.pre_thresh)
                    job['num_peaks'] += len(cand)
                    if not cand:
                        job['num_empty'] += 1

                    if job['fout_b'] is not None:
                        job['fout_b'].write(f'{key} {keyword} {fmt(p)}\n')

                    # windows are cut once at Kmax and center-cropped per
                    # (variant, K) group
                    scores = {}
                    if cand and job['groups']:
                        Kmax = job['Kmax']
                        wins_max = np.stack(
                            [cut_window(h, t, Kmax) for t in cand]
                        ).astype(np.float32)
                        s = np.array([p[t] for t in cand], dtype=np.float32)
                        for (variant, K), g in job['groups'].items():
                            if K == Kmax:
                                wins = wins_max
                            else:
                                wins = wins_max[:, Kmax - K:Kmax + K + 1, :]
                            x = build_feature(wins, s, variant, K)
                            m = x @ g['W'].T + g['b']        # (N, n)
                            pr = 1.0 / (1.0 + np.exp(-m))
                            for j, tag in enumerate(g['tags']):
                                fs = np.zeros(T, dtype=np.float32)
                                for t, v in zip(cand, pr[:, j]):
                                    fs[t] = max(fs[t], float(v))
                                scores[tag] = fs
                    for tag, f in job['fouts'].items():
                        fs = scores.get(tag)
                        if fs is None:      # no candidate: all zeros
                            fs = np.zeros(T, dtype=np.float32)
                        f.write(f'{key} {keyword} {fmt(fs)}\n')

            if batch_idx % 10 == 0:
                stat = ', '.join(
                    "{}: {} peaks, {} peakless".format(
                        job['name'], job['num_peaks'], job['num_empty'])
                    for job in jobs)
                print(f'Progress batch {batch_idx} ({num_utts} utts; {stat})')
                sys.stdout.flush()

    for job in jobs:
        for f in job['fouts'].values():
            f.close()
        if job['fout_b'] is not None:
            job['fout_b'].close()
            print("job {}: baseline dense -> {}".format(
                job['name'], job['baseline_out']))
        print("job {}: {} candidates ({:.2f}/utt), {} utterances without "
              "a candidate; head scores -> {}".format(
                  job['name'], job['num_peaks'],
                  job['num_peaks'] / max(num_utts, 1),
                  job['num_empty'], job['gated_dir']))
    if num_utts == 0:
        sys.exit('FAILED: 0 utterances processed - the dataset yielded '
                 'nothing (audio paths unreachable?). Score files written '
                 'in this run are EMPTY; delete them.')
    if args.expected_utts is not None and num_utts != args.expected_utts:
        sys.exit(f'FAILED: processed {num_utts} utterances but expected '
                 f'{args.expected_utts} - some audio was silently '
                 f'skipped; score files are INCOMPLETE, delete them.')
    print(f'Done: {num_utts} utterances in one backbone pass.')


if __name__ == '__main__':
    main()
