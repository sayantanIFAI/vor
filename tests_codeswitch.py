# -*- coding: utf-8 -*-
"""The test that was missing when this shipped and fabricated instructions.

Every distinct segment from the sixteen real OPD recordings, asserted to
produce NOTHING. The first version of codeswitch.py was tested only on
phrases that ought to match, so it looked correct and invented a duration on
ten of those sixteen consultations - "9-1 hours" assembled out of a sentence
about taking medicine twice daily.

A feature that reads speech has to be tested on speech it must ignore.
"""
import json, sys

from voicerx.codeswitch import read_duration, read_instructions

SEGMENTS = json.load(open("tests_codeswitch_corpus.json", encoding="utf-8"))

# "টেস্ট" IS the English word test, so these are correct readings, not noise.
ALLOWED = {"test"}

fails = []
for text in SEGMENTS:
    dur = read_duration(text)
    if dur:
        fails.append(("fabricated duration %r" % dur, text))
    for adv in read_instructions(text):
        if adv not in ALLOWED:
            fails.append(("fabricated advice %r" % adv, text))

# Real code-switched speech must still be read.
POSITIVE = [
    ("রেস্ট ফর থ্রি টু ফাইভ দইস", "3-5 days", "rest"),
    ("সেভেন দইস", "7 days", None),
    ("থ্রি উইক পরে", "3 weeks", None),
]
for text, want_dur, want_adv in POSITIVE:
    got = read_duration(text)
    if got != want_dur:
        fails.append(("duration %r, wanted %r" % (got, want_dur), text))
    if want_adv and want_adv not in read_instructions(text):
        fails.append(("advice %r missing" % want_adv, text))

print("=" * 66)
print("  %d real segments checked, %d positive phrases" % (len(SEGMENTS), len(POSITIVE)))
if fails:
    for why, text in fails[:12]:
        print("  FAIL %-34s <- %s" % (why, text[:40]))
    print("  %d failed" % len(fails))
    sys.exit(1)
print("  all passed - nothing invented, real phrases still read")
print("=" * 66)
