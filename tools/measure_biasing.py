"""Measure whether context biasing helps, on human-corrected ground truth.

Biasing is OFF by default and stays off unless this says it helps. Two
numbers decide it, and the second matters more than the first:

  WER            does the transcript get closer to what was actually said?
  HALLUCINATIONS how many drug names did biasing INSERT that the human
                 transcript does not contain?

An inserted drug name is the worst failure this project can produce. It is
strictly worse than a missing one: a missing drug makes the prescription
look obviously incomplete, an invented one looks correct. So a run that
improves WER while inserting fake drug names is a FAILURE, not a trade-off.

Usage (on the pod):
    python3 tools/measure_biasing.py [cb_weight ...]

Runs the baseline once, then each cb_weight, over the segments a human
actually corrected.
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, "/workspace/voicerx")
sys.path.insert(0, ".")

WORKSHEET = "/workspace/voicerx/all_10_files_correction_worksheet.json"
AUDIO_DIR = "/workspace/voicerx"


def wer(ref: str, hyp: str) -> tuple[int, int]:
    """Word-level Levenshtein distance and reference length."""
    r, h = ref.split(), hyp.split()
    d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
    return d[len(r)][len(h)], len(r)


def main() -> None:
    import soundfile as sf
    import numpy as np
    from voicerx.asr import ASRNode
    from voicerx.biasing import ContextBiaser, ctc_logprobs, gazetteer_hotwords

    weights = [float(w) for w in sys.argv[1:]] or [1.0, 2.0, 3.0]

    rows = [r for r in json.load(open(WORKSHEET, encoding="utf-8"))
            if (r.get("correct_text") or "").strip()]
    print(f"ground truth: {len(rows)} human-corrected segments", flush=True)

    print("loading ASR model...", flush=True)
    node = ASRNode()
    model = node.model
    hotwords = set(gazetteer_hotwords())

    # Cut each corrected segment to a temp clip once, reused across runs.
    clips = []
    for i, r in enumerate(rows):
        src = os.path.join(AUDIO_DIR, r["audio_file"])
        if not os.path.exists(src):
            continue
        audio, sr = sf.read(src)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        a = int(r["start_s"] * sr)
        b = int(r["end_s"] * sr)
        clip = f"/tmp/seg_{i}.wav"
        sf.write(clip, audio[a:b], sr)
        clips.append((clip, r["correct_text"].strip()))
    print(f"prepared {len(clips)} clips", flush=True)

    def run(biaser) -> dict:
        errs = tot = 0
        injected_total = 0
        injected_in_ref = 0
        injected_examples = []
        t0 = time.time()
        for clip, ref in clips:
            lp = ctc_logprobs(model, clip)

            base = model.transcribe([clip], batch_size=1, logprobs=False,
                                     language_id=node.language_id, verbose=False)
            base_text = base[0] if isinstance(base[0], str) else str(base[0][0])

            if biaser is None:
                hyp, injected = base_text, []
            else:
                hyp, injected = biaser.apply(lp, base_text)

            ref_words = set(ref.split())
            for w in injected:
                if w in hotwords:            # only count gazetteer injections
                    injected_total += 1
                    if w in ref_words:
                        injected_in_ref += 1
                    elif len(injected_examples) < 8:
                        injected_examples.append((w, ref[:50]))

            e, n = wer(ref, hyp)
            errs += e
            tot += n
        return {
            "wer": errs / tot if tot else 0.0,
            "secs": time.time() - t0,
            "injected": injected_total,
            "injected_correct": injected_in_ref,
            "examples": injected_examples,
        }

    print("\n--- baseline (no biasing) ---", flush=True)
    base = run(None)
    print(f"WER {base['wer']:.4f}   ({base['secs']:.0f}s)", flush=True)

    print("\n{:<10} {:<10} {:<10} {:<26} {}".format(
        "cb_weight", "WER", "delta", "drug words injected", "verdict"), flush=True)
    print("-" * 78, flush=True)
    for w in weights:
        b = ContextBiaser(model, hotwords=sorted(hotwords), cb_weight=w, debug=True)
        if not b.build():
            print(f"{w:<10} BUILD FAILED: {b.error}", flush=True)
            continue
        res = run(b)
        delta = res["wer"] - base["wer"]
        bad = res["injected"] - res["injected_correct"]
        verdict = ("REGRESSION" if delta > 0 else
                   "UNSAFE - hallucinated drugs" if bad else
                   "improves" if delta < 0 else "no change")
        print("{:<10.1f} {:<10.4f} {:<+10.4f} {:<26} {}".format(
            w, res["wer"], delta,
            f"{res['injected']} ({res['injected_correct']} real, {bad} fake)",
            verdict), flush=True)
        for word, ref in res["examples"]:
            print(f"             HALLUCINATED {word!r} not in: {ref}", flush=True)


if __name__ == "__main__":
    main()
