# Stage-2 linear decision head on frozen max-pooling KWS representations.
#
# Trains a small linear classifier over context windows of backbone hidden
# states centered at the peak-scoring frame, as described in:
#
#   A. Sloin, "Context-Enriched Heads for Wake-Word Detection",
#   submitted to ICASSP 2027.
#
# Feature variants (H = backbone hidden size, window = 2K+1 frames):
#   v1     window mean                                             (H)
#   v2     mean + peak frame + peak score                          (2H+1)
#   v2.5   means of the three window thirds                        (3H)
#   v3     thirds + peak frame + max + std + peak score            (6H+1)
#
# Losses:  hinge = clamp(1 - margin, 0);  ce = softplus(-margin) (logistic).
# Model selection: --select best (best dev-FRR epoch) or last (final epoch,
# full-convergence protocol; watch the loss column for a plateau).
#
# Runs entirely on CPU.

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Feature building
# ---------------------------------------------------------------------------

def build_feature(windows, peak_scores, variant, K):
    """windows: (N, 2K+1, H); peak_scores: (N,)."""
    N, W, H = windows.shape
    assert W == 2 * K + 1
    t_star = K

    mean_t = windows.mean(axis=1)

    if variant == 'v1':
        return mean_t

    center = windows[:, t_star, :]
    ps = peak_scores.reshape(-1, 1)

    if variant == 'v2':
        return np.concatenate([mean_t, center, ps], axis=1)          # 2H+1

    third = W // 3
    m_pre = windows[:, :third, :].mean(axis=1)
    m_mid = windows[:, third:W - third, :].mean(axis=1)
    m_post = windows[:, W - third:, :].mean(axis=1)

    if variant == 'v2.5':
        return np.concatenate([m_pre, m_mid, m_post], axis=1)        # 3H

    if variant == 'v3':
        max_t = windows.max(axis=1)
        std_t = windows.std(axis=1)
        return np.concatenate(
            [m_pre, m_mid, m_post, center, max_t, std_t, ps], axis=1)  # 6H+1

    raise ValueError(f'unknown variant {variant}')

# ---------------------------------------------------------------------------
# crop_windows
# ---------------------------------------------------------------------------

def crop_windows(windows, stored_k, k):
    """Center-crop stored (N, 2*stored_k+1, H) windows to (N, 2*k+1, H)."""
    if stored_k is None or stored_k == k:
        return windows
    assert stored_k > k, f'stored_k={stored_k} < requested context {k}'
    c = stored_k
    return windows[:, c - k : c + k + 1, :]

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_split(feat_dir, variant, K, stored_k=None, max_neg_per_utt=None,
               pos_label=0):
    """Load all npz in feat_dir, return X, y, and per-utterance index.
    Two-pass with a preallocated output: peak memory ~= final X size."""
    files = sorted(glob.glob(os.path.join(feat_dir, '*.npz')))
    if not files:
        sys.exit(f'no npz files found in {feat_dir}')

    # pass 1: count windows per file
    counts = np.empty(len(files), dtype=np.int64)
    for i, f in enumerate(files):
        d = np.load(f, allow_pickle=True)
        n = d['windows'].shape[0]
        label = int(d['label'])
        is_pos = (label == pos_label)
        if max_neg_per_utt is not None and not is_pos:
            n = min(n, max_neg_per_utt)
        counts[i] = n
    total = int(counts.sum())

    # feature dim from the first file
    d0 = np.load(files[0], allow_pickle=True)
    w0_win = crop_windows(d0['windows'].astype(np.float32), stored_k, K)[:1]
    dim = build_feature(w0_win, d0['peak_scores'][:1].astype(np.float32),
                        variant, K).shape[1]

    X = np.empty((total, dim), dtype=np.float32)
    y = np.empty(total, dtype=np.float32)
    utt_keys, utt_slices = [], []

    # pass 2: fill in place
    pos = 0
    for f in files:
        d = np.load(f, allow_pickle=True)
        w = crop_windows(d['windows'].astype(np.float32), stored_k, K)
        s = d['peak_scores'].astype(np.float32)
        label = int(d['label'])
        is_pos = (label == pos_label)
        if max_neg_per_utt is not None and not is_pos:
            w, s = w[:max_neg_per_utt], s[:max_neg_per_utt]
        n = w.shape[0]
        X[pos:pos + n] = build_feature(w, s, variant, K)
        y[pos:pos + n] = 1.0 if is_pos else -1.0
        utt_keys.append(str(d['key']))
        utt_slices.append((pos, pos + n, 0 if is_pos else 1))
        pos += n

    print(f'{feat_dir}: {len(files)} utts, {X.shape[0]} windows, '
          f'dim {X.shape[1]}, positives {(y > 0).sum()}')
    return X, y, utt_keys, utt_slices


def load_filler_hours(data_list, keyword='<HEY_SNIPS>'):
    """Total duration (hours) of non-keyword utterances."""
    total = 0.0
    with open(data_list, 'r', encoding='utf8') as f:
        for line in f:
            obj = json.loads(line)
            if obj['txt'].upper() != keyword.upper():
                total += float(obj['duration'])
    return total / 3600.0


# ---------------------------------------------------------------------------
# Dev metric: FRR at the threshold giving target FA/h
# ---------------------------------------------------------------------------

def frr_at_fa(scores, utt_slices, filler_hours, target_fa_per_hour=1.0):
    pos_scores, neg_scores = [], []
    for lo, hi, label in utt_slices:
        if label == 0:
            pos_scores.append(scores[lo:hi].max())
        else:
            neg_scores.extend(scores[lo:hi].tolist())
    pos_scores = np.array(pos_scores)
    neg_scores = np.array(neg_scores)

    k = max(1, int(round(target_fa_per_hour * filler_hours)))
    if len(neg_scores) <= k:
        thr = neg_scores.min() - 1.0
    else:
        thr = np.sort(neg_scores)[-k]
    frr = float((pos_scores < thr).mean())
    return frr, float(thr)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description='train the linear decision head (stage 2)')
    ap.add_argument('--pos_label', type=int, default=0,
                    help='label id of the positive keyword in the npz files')
    ap.add_argument('--keyword_txt', default='<HEY_SNIPS>',
                    help='keyword string in data.list, for filler hours')
    ap.add_argument('--train_feats', required=True)
    ap.add_argument('--dev_feats', required=True)
    ap.add_argument('--dev_list', required=True)
    ap.add_argument('--variant', default='v1',
                    choices=['v1', 'v2', 'v2.5', 'v3'])
    ap.add_argument('--context', type=int, default=15)
    ap.add_argument('--stored_k', type=int, default=None)
    ap.add_argument('--epochs', type=int, default=300)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight_decay', type=float, default=1e-4)
    ap.add_argument('--batch_size', type=int, default=1024)
    ap.add_argument('--target_fa', type=float, default=1.0)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--out_model', required=True)
    ap.add_argument('--loss', default='hinge', choices=['hinge', 'ce'],
                    help='hinge = clamp(1-m,0); ce = softplus(-m) (logistic)')
    ap.add_argument('--select', default='last', choices=['best', 'last'],
                    help='best = save best dev-FRR epoch; '
                         'last = save every epoch, last wins (convergence)')
    ap.add_argument('--no_standardize', action='store_true', default=False,
                    help='skip feature standardization (mu=0, sd=1); '
                         'raw pass-through features')
    args = ap.parse_args()
    print('args:', vars(args))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    Xtr, ytr, _, _ = load_split(args.train_feats, args.variant, args.context,
                                args.stored_k, pos_label=args.pos_label)
    Xdv, ydv, dv_keys, dv_slices = load_split(
        args.dev_feats, args.variant, args.context, args.stored_k,
        pos_label=args.pos_label)
    filler_hours = load_filler_hours(args.dev_list, args.keyword_txt)
    print(f'dev filler hours: {filler_hours:.2f}')

    # standardize using train statistics only (chunked, in-place)
    if args.no_standardize:
        mu = np.zeros(Xtr.shape[1], dtype=np.float32)
        sd = np.ones(Xtr.shape[1], dtype=np.float32)
        print('standardization DISABLED (--no_standardize): raw features')
    else:
        mu = Xtr.mean(axis=0)
        sd = np.zeros_like(mu)
        CH = 20000
        for i in range(0, Xtr.shape[0], CH):
            sd += ((Xtr[i:i+CH] - mu) ** 2).sum(axis=0)
        sd = np.sqrt(sd / Xtr.shape[0]) + 1e-8
        mu = mu.astype(np.float32); sd = sd.astype(np.float32)
        for i in range(0, Xtr.shape[0], CH):
            Xtr[i:i+CH] = (Xtr[i:i+CH] - mu) / sd
        for i in range(0, Xdv.shape[0], CH):
            Xdv[i:i+CH] = (Xdv[i:i+CH] - mu) / sd

    Xtr_t = torch.from_numpy(Xtr)
    ytr_t = torch.from_numpy(ytr)
    Xdv_t = torch.from_numpy(Xdv)

    n_pos = float((ytr > 0).sum())
    n_neg = float((ytr < 0).sum())
    w_pos = n_neg / (n_pos + n_neg)
    w_neg = n_pos / (n_pos + n_neg)
    print(f'class weights: pos {w_pos:.3f}, neg {w_neg:.3f}')

    model = nn.Linear(Xtr.shape[1], 1)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)

    best = {'frr': 1.1, 'epoch': -1}
    N = Xtr_t.shape[0]
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(N)
        total_loss = 0.0
        for i in range(0, N, args.batch_size):
            idx = perm[i:i + args.batch_size]
            xb, yb = Xtr_t[idx], ytr_t[idx]
            margin = model(xb).squeeze(1) * yb
            if args.loss == 'ce':
                per_sample = torch.nn.functional.softplus(-margin)  # logistic
            else:
                per_sample = torch.clamp(1.0 - margin, min=0.0)     # hinge
            w = torch.where(yb > 0, w_pos, w_neg)
            loss = (per_sample * w).sum() / w.sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss) * len(idx)

        model.eval()
        with torch.no_grad():
            dv_scores = model(Xdv_t).squeeze(1).numpy()
        frr, thr = frr_at_fa(dv_scores, dv_slices, filler_hours,
                             args.target_fa)
        is_best = frr < best['frr']
        if is_best:
            best = {'frr': frr, 'epoch': epoch, 'thr': thr}
        save_now = is_best if args.select == 'best' else True
        marker = ''
        if save_now:
            torch.save({'state_dict': model.state_dict(),
                        'mu': mu, 'sd': sd,
                        'variant': args.variant,
                        'context': args.context,
                        'loss_type': args.loss,
                        'select': args.select,
                        'dev_frr': frr, 'dev_thr': thr,
                        'args': vars(args)}, args.out_model)
            marker = ('  <-- best, saved' if args.select == 'best'
                      else '  saved (last wins)')
        print(f'epoch {epoch:3d}  loss {total_loss / N:.5f}  '
              f'dev FRR@{args.target_fa}FA/h = {frr:.4f} '
              f'(thr {thr:.3f}){marker}')

    print(f"\nbest: epoch {best['epoch']}, "
          f"dev FRR@{args.target_fa}FA/h = {best['frr']:.4f}")
    if args.select == 'last':
        print('NOTE: select=last -> the SAVED model is the final epoch, '
              'not the best one. Check the loss column for a plateau.')
    print(f'model saved to {args.out_model}')


if __name__ == '__main__':
    main()
