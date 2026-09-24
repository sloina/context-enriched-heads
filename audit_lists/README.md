# Cleaning-level key lists (ICASSP 2027 submission snapshot)

These are the utterance keys removed from the Mobvoi test list to form the
three cleaning levels of Table 2 in the paper, exactly as used for the
submitted results (September 2026):

| file | utterances | level it defines |
|---|---|---|
| `removed_bad.txt`   | 167 | `-bad`   = full minus bad |
| `removed_ultra.txt` |  13 | `-ultra` = `-bad` minus ultra-hard |
| `removed_vh.txt`    | 372 | `-vh`    = `-ultra` minus very-hard |

The levels are cumulative. One key (the Mobvoi utterance id) per line.
These lists are the ground truth for the paper's tables. The census
files in mobvoi-audit reflect later verification rounds, so a few of
these utterances carry a different label there; that does not change
the lists used here.
Removing a key drops the utterance for both keywords, so a flagged
Hi Xiaowen positive also disappears from the Nihao Wenwen negatives
(and vice versa), as stated in the paper.

**These lists are a frozen snapshot, not the final audit.** The full
listening census of the positive test and development sets has since
been completed and is released at <https://github.com/sloina/mobvoi-audit>;
its annotations may continue to be revised. Use that repository for the
labels themselves; use these files only to reproduce the paper's tables.

`make_clean_lists.py` (repository root) turns them into the three
wekws-format data lists that `run_mobvoi_heads.ps1` evaluates on.
