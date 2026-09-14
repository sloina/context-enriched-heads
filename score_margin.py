# Stage-2 scoring for the linear decision head.
# (see train_margin_head.py for the variant definitions; build_feature
# here MUST stay identical to the training-side version)
#
# Writes each utterance as a frame sequence of zeros with the head's
# sigmoid score placed at each candidate peak's original time, so the
# unmodified wekws DET computation applies directly.

import argparse
import glob
import os
import sys

import numpy as np
import torch
import torch.nn as nn


def build_feature(windows, peak_scores, variant, K):
    """Must match train_margin_head.build_feature exactly."""
    N, W, H = windows.shape
    assert W == 2 * K + 1
    t_star = K

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

def crop_windows(windows, stored_k, k):
    if stored_k is None or stored_k == k:
        return windows
    assert stored_k > k, f'stored_k={stored_k} < requested context {k}'
    c = stored_k
    return windows[:, c - k : c + k + 1, :]


def main():
    ap = argparse.ArgumentParser(
        description='score the linear decision head (stage 2)')
    ap.add_argument('--feats', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--stored_k', type=int, default=None)
    ap.add_argument('--keyword', default='<HEY_SNIPS>')
    ap.add_argument('--score_file', required=True)
    args = ap.parse_args()

    ckpt = torch.load(args.model, map_location='cpu', weights_only=False)
    variant = ckpt['variant']
    K = ckpt['context']
    mu = np.asarray(ckpt['mu'], dtype=np.float32)
    sd = np.asarray(ckpt['sd'], dtype=np.float32)

    in_dim = mu.shape[0]
    model = nn.Linear(in_dim, 1)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    print(f"loaded {args.model}: variant {variant}, K={K}, dim={in_dim}, "
          f"dev FRR was {ckpt.get('dev_frr', float('nan')):.4f}")

    files = sorted(glob.glob(os.path.join(args.feats, '*.npz')))
    if not files:
        sys.exit(f'no npz files found in {args.feats}')

    n_done = 0
    with open(args.score_file, 'w', encoding='utf8') as fout, \
            torch.no_grad():
        for f in files:
            d = np.load(f, allow_pickle=True)
            key = str(d['key'])
            T = int(d['T'])
            times = d['peak_times'].astype(int)
            w = crop_windows(d['windows'].astype(np.float32), args.stored_k, K)
            s = d['peak_scores'].astype(np.float32)

            x = build_feature(w, s, variant, K)
            x = ((x - mu) / sd).astype(np.float32)
            margins = model(torch.from_numpy(x)).squeeze(1).numpy()
            probs = 1.0 / (1.0 + np.exp(-margins))

            frame_scores = np.zeros(T, dtype=np.float32)
            for t, p in zip(times, probs):
                t = min(max(int(t), 0), T - 1)
                frame_scores[t] = max(frame_scores[t], float(p))

            fout.write('{} {} {}\n'.format(
                key, args.keyword,
                ' '.join('{:.6f}'.format(v) for v in frame_scores)))

            n_done += 1
            if n_done % 2000 == 0:
                print(f'{n_done}/{len(files)} utterances scored')

    print(f'done: {n_done} utterances -> {args.score_file}')


if __name__ == '__main__':
    main()
