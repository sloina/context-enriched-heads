# Copyright (c) 2021 Binbin Zhang(binbzha@qq.com)
#               2022 Shaoqing Yu(954793264@qq.com)
#               2026 Margin-head experiment: feature extraction
#
# Based on wekws/bin/score.py. Instead of writing frame-level scores,
# this script saves backbone representations (hidden states) in a window
# around candidate peaks, for training the linear decision head.
#
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import print_function

import argparse
import copy
import logging
import os
import sys

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from wekws.dataset.init_dataset import init_dataset
from wekws.model.kws_model import init_model
from wekws.utils.checkpoint import load_checkpoint
from wenet.text.char_tokenizer import CharTokenizer


def get_args():
    parser = argparse.ArgumentParser(
        description='extract backbone features around score peaks')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--test_data', required=True,
                        help='data list file (train/dev/test)')
    parser.add_argument('--dict', default='./dict', help='dict dir')
    parser.add_argument('--gpu', type=int, default=-1,
                        help='gpu id for this rank, -1 for cpu')
    parser.add_argument('--checkpoint', required=True, help='checkpoint model')
    parser.add_argument('--batch_size', default=16, type=int,
                        help='batch size for inference')
    parser.add_argument('--num_workers', default=0, type=int,
                        help='num of subprocess workers for reading')
    parser.add_argument('--pin_memory', action='store_true', default=False,
                        help='Use pinned memory buffers used for reading')
    parser.add_argument('--prefetch', default=100, type=int,
                        help='prefetch number')
    parser.add_argument('--output_dir', required=True,
                        help='directory for output npz files')
    # --- margin-head specific knobs ---
    parser.add_argument('--context', type=int, default=15,
                        help='K: window is [t*-K, t*+K] around each peak')
    parser.add_argument('--pre_thresh', type=float, default=0.1,
                        help='pre-threshold for candidate peaks in fillers')
    parser.add_argument('--peaks_cap', type=int, default=20,
                        help='max candidate peaks kept per filler utterance')
    parser.add_argument('--keyword_index', type=int, default=0,
                        help='output column of the keyword in model logits')
    parser.add_argument('--keyword_target', type=int, default=0,
                        help='target id that marks a keyword utterance '
                             '(VERIFY against your data!)')
    parser.add_argument('--min_duration', type=int, default=0,
                        help='mask first frames when locating the positive '
                             'peak, to match training min_duration')
    args = parser.parse_args()
    return args


def find_candidates(p, label, keyword_target, pre_thresh, peaks_cap,
                    min_duration):
    """Return list of candidate frame indices for feature windows."""
    T = len(p)
    if label == keyword_target:
        # Positive: single peak, optionally masking the first frames
        # (consistent with min_duration masking in max_pooling_loss).
        start = min(min_duration, T - 1)
        t_star = int(np.argmax(p[start:])) + start
        return [t_star]
    # Negative (filler): all local maxima above pre-threshold,
    # hardest first, capped for safety.
    cand = [t for t in range(1, T - 1)
            if p[t] >= pre_thresh and p[t] >= p[t - 1] and p[t] >= p[t + 1]]
    cand = sorted(cand, key=lambda t: -p[t])[:peaks_cap]
    if not cand:
        # Keep one peak window even for "easy" fillers, so every
        # utterance is represented in the training set.
        cand = [int(np.argmax(p))]
    return cand


def cut_window(h, t, K):
    """Zero-padded window h[t-K : t+K+1], shape (2K+1, hdim)."""
    T, hdim = h.shape
    lo, hi = max(0, t - K), min(T, t + K + 1)
    w = np.zeros((2 * K + 1, hdim), dtype=np.float32)
    offset = lo - (t - K)
    w[offset:offset + (hi - lo)] = h[lo:hi]
    return w


def main():
    args = get_args()
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

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
    test_data_loader = DataLoader(test_dataset,
                                  batch_size=None,
                                  pin_memory=args.pin_memory,
                                  num_workers=args.num_workers,
                                  prefetch_factor=args.prefetch)

    model = init_model(configs['model'])
    load_checkpoint(model, args.checkpoint)
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    model = model.to(device)
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)
    K = args.context
    num_utts = 0
    num_windows = 0

    with torch.no_grad():
        for batch_idx, batch_dict in enumerate(test_data_loader):
            keys = batch_dict['keys']
            feats = batch_dict['feats']
            targets = batch_dict['target'][:, 0]
            feats_lengths = batch_dict['feats_lengths']
            feats = feats.to(device)

            probs, hidden, _ = model.forward_with_features(feats)
            probs = probs.cpu().numpy()      # (B, T, odim)
            hidden = hidden.cpu().numpy()    # (B, T, hdim)

            for i in range(len(keys)):
                key = keys[i]
                T = int(feats_lengths[i].item())
                p = probs[i, :T, args.keyword_index]
                h = hidden[i, :T, :]
                label = int(targets[i].item())

                cand = find_candidates(p, label, args.keyword_target,
                                       args.pre_thresh, args.peaks_cap,
                                       args.min_duration)

                wins = [cut_window(h, t, K) for t in cand]
                times = np.array(cand, dtype=np.int64)
                scores = np.array([p[t] for t in cand], dtype=np.float32)

                # utterance keys may contain path separators
                safe_key = key.replace('/', '_').replace('\\', '_')
                np.savez_compressed(
                    os.path.join(args.output_dir, safe_key + '.npz'),
                    key=key,
                    windows=np.stack(wins).astype(np.float32),
                    peak_times=times,
                    peak_scores=scores,
                    label=label,
                    T=T)
                num_utts += 1
                num_windows += len(cand)

            if batch_idx % 10 == 0:
                print('Progress batch {} ({} utts, {} windows)'.format(
                    batch_idx, num_utts, num_windows))
                sys.stdout.flush()

    print('Done: {} utterances, {} windows -> {}'.format(
        num_utts, num_windows, args.output_dir))


if __name__ == '__main__':
    main()