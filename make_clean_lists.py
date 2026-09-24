# Build the three cleaned Mobvoi test lists of Table 2 from the full test
# list and the removed-key snapshots in audit_lists/<snapshot>/.
#
# Levels are cumulative:
#   -bad   = full   minus removed_bad.txt
#   -ultra = -bad   minus removed_ultra.txt
#   -vh    = -ultra minus removed_vh.txt
# A removed key is dropped for both keywords (it leaves the negatives of
# the other keyword too), matching the paper's protocol.
#
# Usage:
#   python make_clean_lists.py --test_list mobvoi_test_data.list \
#       --keys_dir audit_lists --out_dir lists
# writes lists/mobvoi_test_nobad.list, lists/mobvoi_test_nobad_noultra.list,
# lists/mobvoi_test_nobad_noultra_novh.list

import argparse
import json
import os
import sys


def read_keys(path):
    with open(path, 'r', encoding='utf8') as f:
        keys = {line.strip() for line in f if line.strip()}
    if not keys:
        sys.exit(f'no keys in {path}')
    return keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--test_list', required=True)
    ap.add_argument('--keys_dir', required=True)
    ap.add_argument('--out_dir', required=True)
    args = ap.parse_args()

    levels = [('nobad', 'removed_bad.txt'),
              ('nobad_noultra', 'removed_ultra.txt'),
              ('nobad_noultra_novh', 'removed_vh.txt')]

    with open(args.test_list, 'r', encoding='utf8') as f:
        lines = [line for line in f if line.strip()]
    keys_in_list = [json.loads(line)['key'] for line in lines]
    print(f'{args.test_list}: {len(lines)} utterances')

    os.makedirs(args.out_dir, exist_ok=True)
    removed = set()
    for name, keyfile in levels:
        new = read_keys(os.path.join(args.keys_dir, keyfile))
        missing = new - set(keys_in_list)
        if missing:
            sys.exit(f'{keyfile}: {len(missing)} keys not in the test list, '
                     f'e.g. {sorted(missing)[:3]}')
        removed |= new
        out = os.path.join(args.out_dir, f'mobvoi_test_{name}.list')
        n = 0
        with open(out, 'w', encoding='utf8') as f:
            for line, key in zip(lines, keys_in_list):
                if key not in removed:
                    f.write(line)
                    n += 1
        print(f'{keyfile}: -{len(new)} -> {out}: {n} utterances')


if __name__ == '__main__':
    main()
