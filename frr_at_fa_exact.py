# Exact FRR @ target FA/h, replicating wekws compute_det.py counting:
#   FA:  walk frames; each score >= thr is an alarm, then skip window_shift
#   FRR: positive utterance rejected iff max(frame scores) < thr
# Instead of a threshold grid, binary-search the minimal threshold whose
# event-based FA/h is <= target, over the sorted unique filler peak scores.

import argparse
import json
import bisect


def load(keyword, label_file, score_file):
    score_table = {}
    with open(score_file, 'r', encoding='utf8') as fin:
        for line in fin:
            arr = line.strip().split()
            if arr[1] == keyword and arr[0] not in score_table:
                score_table[arr[0]] = list(map(float, arr[2:]))
    kw, fl, dur = {}, {}, 0.0
    with open(label_file, 'r', encoding='utf8') as fin:
        for line in fin:
            obj = json.loads(line.strip())
            key, txt = obj['key'], obj['txt'].upper()
            assert key in score_table, f'key {key} not in score file'
            if txt == keyword:
                kw[key] = score_table[key]
            else:
                fl[key] = score_table[key]
                dur += obj['duration']
    return kw, fl, dur


def fa_count(fl, thr, shift):
    n = 0
    for scores in fl.values():
        i, L = 0, len(scores)
        while i < L:
            if scores[i] >= thr:
                n += 1
                i += shift
            else:
                i += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--score_file', required=True)
    ap.add_argument('--test_data', required=True)
    ap.add_argument('--keyword', required=True)
    ap.add_argument('--window_shift', type=int, default=50)
    ap.add_argument('--fa_target', type=float, default=1.0)
    args = ap.parse_args()

    kw, fl, dur = load(args.keyword, args.test_data, args.score_file)
    hours = dur / 3600.0
    print(f'positives {len(kw)}, fillers {len(fl)}, hours {hours:.4f}')
    max_alarms = args.fa_target * hours  # allowed alarm events

    # candidate thresholds: unique nonzero filler frame scores
    cand = sorted({s for v in fl.values() for s in v if s > 0.0})
    if not cand:
        print('no nonzero filler scores?!')
        return

    # binary search: smallest candidate thr with fa_count(thr) <= max_alarms
    lo, hi = 0, len(cand) - 1
    best = None
    while lo <= hi:
        mid = (lo + hi) // 2
        thr = cand[mid]
        n = fa_count(fl, thr, args.window_shift)
        if n <= max_alarms:
            best = (thr, n)
            hi = mid - 1
        else:
            lo = mid + 1
    if best is None:  # even the highest score alarms too much (impossible-ish)
        thr, n = cand[-1] + 1e-9, 0
    else:
        thr, n = best
        # sanity: verify monotonicity assumption around the found point
        if best[0] != cand[0]:
            below_idx = bisect.bisect_left(cand, thr) - 1
            n_below = fa_count(fl, cand[below_idx], args.window_shift)
            if n_below <= max_alarms:
                print(f'note: non-monotone spot below ({n_below} alarms), '
                      f'using lower threshold {cand[below_idx]:.6f}')
                thr, n = cand[below_idx], n_below

    frr_n = sum(1 for v in kw.values() if max(v) < thr)
    print(f'threshold {thr:.6f}  FA {n}/{hours:.2f}h = {n / hours:.6f}/h  '
          f'FRR {frr_n}/{len(kw)} = {100.0 * frr_n / len(kw):.4f}%')


if __name__ == '__main__':
    main()
