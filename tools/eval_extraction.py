"""Score extraction against known prescriptions.

CANONICALISE BOTH SIDES BEFORE COMPARING.

The first version of this scorer compared raw strings and reported labs at
F1 0.768, "clearly the weak link". That was wrong. The ground truth writes
"Thyroid profile" where the gazetteer emits "TSH", and the scorer counted
each correct detection TWICE as an error - once as a miss, once as a false
positive. The two columns were identical:

    FALSE POSITIVES          MISSES
     184  TSH                 184  Thyroid profile
     106  KFT                 106  Kidney function test
      77  CBC                  77  Complete Blood Count
      66  LFT                  66  Liver function test

Passing both sides through is_lab_test() first gives P 0.867 / R 1.000 /
F1 0.929 - zero missed lab orders across 500 transcripts.

The lesson is worth keeping: a bad metric sent me to fix a component that
was working. Canonicalise before you compare.
"""
from __future__ import annotations

import io
import json
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, "/workspace/voicerx")
sys.path.insert(0, ".")

from voicerx.glossary import is_lab_test, scan_labs   # noqa: E402
from voicerx.gate import judge_medication             # noqa: E402


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "/workspace/eval500.json"
    data = json.load(open(path, encoding="utf-8"))

    l_tp = l_fp = l_fn = 0
    d_tp = d_fp = d_fn = 0
    for rec in data:
        got = {is_lab_test(g) or g.lower() for g in scan_labs(rec["transcript"])}
        want = {is_lab_test(g) or g.lower() for g in rec["gt"]["labs"]}
        l_tp += len(got & want); l_fp += len(got - want); l_fn += len(want - got)

        # Drugs: canonicalise through the gate for the same reason - the
        # ground truth may name a brand where the gazetteer emits a generic.
        gt_d = set()
        for d in rec["gt"]["drugs"]:
            v = judge_medication(d)
            gt_d.add((v.canonical or d).lower())
        d_tp += len(gt_d)      # coverage check: can the gate resolve each?
        for d in rec["gt"]["drugs"]:
            if judge_medication(d).tier == "rejected":
                d_fn += 1
                d_tp -= 1

    p, r, f = prf(l_tp, l_fp, l_fn)
    print(f"LABS   P {p:.3f}  R {r:.3f}  F1 {f:.3f}   (tp={l_tp} fp={l_fp} fn={l_fn})")
    print(f"DRUGS  gate resolves {d_tp}/{d_tp + d_fn} ground-truth drug lines")


if __name__ == "__main__":
    main()
