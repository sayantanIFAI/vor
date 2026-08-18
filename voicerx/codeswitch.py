# -*- coding: utf-8 -*-
"""English spoken in Bengali script - the half of the consultation nothing read.

Indian clinicians code-switch constantly, and the ASR writes the English in
Bengali script exactly as it hears it:

    রেস্ট ফর থ্রি টু ফাইভ দইস     rest for three to five days
    ফিভার প্রোফাইল                fever profile
    ওয়ার্ম ওয়াটার                warm water

Every scanner in this package matches Bengali vocabulary, so none of that
existed. A duration, a rest instruction and a follow-up interval were spoken
aloud and the record came back with follow_up=None.

THE MECHANISM IS ALREADY HERE
-----------------------------
gate.romanize() has turned Bengali script into Latin since PHASE 4, used
only for drug names. Run token by token it cracks these open:

    রেস্ট -> rest    থ্রি -> thri    ফাইভ -> faibh    দইস -> dis

which is close enough to fuzzy-match an English word. So this needs no new
phonetics and no language detection - romanize, then compare against a small
closed lexicon of the English words a doctor actually code-switches into:
numbers, time units, and instructions.

WHY A LEXICON AND NOT A DICTIONARY
----------------------------------
Matching against all of English would let any garbled token find something.
The vocabulary a doctor switches into is small and predictable, and keeping
it small is what makes a hit meaningful. Anything outside it stays untouched
and reaches the reviewer as it always did.
"""
from __future__ import annotations

from difflib import SequenceMatcher

# Deliberately small. Numbers, the units they attach to, and the handful of
# instruction words that carry clinical weight.
# Keyed on what romanize() ACTUALLY produces, not on correct English.
# "থ্রি" romanises to "thri", which scores 0.67 against "three" - under any
# floor loose enough to be safe. The same lesson the brand transliteration
# taught: match the form the ASR emits, and let the dictionary spelling be
# the label rather than the key.
NUMBERS: dict[str, int] = {
    "one": 1, "oyan": 1, "wan": 1,
    "two": 2, "tu": 2,
    "three": 3, "thri": 3, "thre": 3, "tri": 3,
    # "four" only. "ফর" is the preposition FOR and romanises to the same
    # "phor" - listing it as a number read "for three to five days" as
    # 4 then 3. A word that is two things is worth neither.
    "four": 4,
    "five": 5, "faibh": 5, "phaibh": 5, "faib": 5,
    "six": 6, "siks": 6, "sike": 6,
    "seven": 7, "sebhen": 7, "saten": 7,
    "eight": 8, "et": 8, "at": 8,
    "nine": 9, "nain": 9,
    "ten": 10, "fifteen": 15, "phiphtin": 15,
    "twenty": 20, "thirty": 30,
}
UNITS: dict[str, str] = {
    "day": "days", "days": "days", "dis": "days", "deis": "days",
    "dej": "days", "dais": "days",
    "week": "weeks", "weeks": "weeks", "uik": "weeks", "wik": "weeks",
    "month": "months", "months": "months", "manth": "months",
    "hour": "hours", "hours": "hours", "aoyar": "hours",
    "night": "nights", "nights": "nights", "nait": "nights",
}
INSTRUCTIONS: dict[str, str] = {
    "rest": "rest", "gargle": "gargle", "warm": "warm",
    "water": "water", "oyatar": "water", "oyarm": "warm", "fluid": "fluids", "fluids": "fluids",
    "follow": "follow up", "followup": "follow up", "review": "review",
    "repeat": "repeat", "continue": "continue", "stop": "stop",
    "avoid": "avoid", "profile": "profile", "test": "test",
}
_JOINERS = {"to", "tu", "for", "fr", "and", "after", "aftar"}

# A romanised token is a noisy rendering of an English word, so the bar is
# lower than for a drug name - but high enough that ordinary Bengali does not
# drift into the lexicon.
_FLOOR = 0.72


def _best(token: str, table: dict) -> tuple[str | None, float]:
    best, score = None, 0.0
    for word in table:
        s = SequenceMatcher(a=token, b=word).ratio()
        if s > score:
            best, score = word, s
    return (best, score) if score >= _FLOOR else (None, 0.0)


def read_tokens(text: str) -> list[tuple[str, str, str]]:
    """(original, romanised, english) for every token that resolves."""
    from .gate import romanize

    out = []
    for tok in (text or "").split():
        rom = romanize(tok)
        if len(rom) < 2:
            continue
        for table in (NUMBERS, UNITS, INSTRUCTIONS):
            hit, _ = _best(rom, table)
            if hit:
                out.append((tok, rom, hit))
                break
        else:
            if rom in _JOINERS:
                out.append((tok, rom, rom))
    return out


def read_duration(text: str) -> str:
    """A spoken English duration, as a printable string.

    "রেস্ট ফর থ্রি টু ফাইভ দইস" -> "3-5 days". The range matters: a doctor
    saying three to five days has said something a single number cannot
    carry, and rounding it would be inventing precision.
    """
    # "টু" is TWO in "টু উইক" and TO in "থ্রি টু ফাইভ" - the same sound
    # doing both jobs, which is why "three to five days" first read as
    # 3-2. It is the joiner when a number sits on BOTH sides of it, and the
    # number otherwise. Position decides, so nothing has to be guessed.
    seq = [(eng, rom) for _o, rom, eng in read_tokens(text)]
    nums, unit = [], ""
    for i, (eng, rom) in enumerate(seq):
        if eng in NUMBERS:
            if rom in ("tu", "to"):
                prev_num = i > 0 and seq[i - 1][0] in NUMBERS
                next_num = i + 1 < len(seq) and seq[i + 1][0] in NUMBERS
                if prev_num and next_num:
                    continue                # it is the word "to"
            nums.append(NUMBERS[eng])
        elif eng in UNITS:
            unit = UNITS[eng]
    if not nums or not unit:
        return ""
    if len(nums) >= 2 and nums[0] != nums[1]:
        return f"{nums[0]}-{nums[1]} {unit}"
    return f"{nums[0]} {unit}"


def read_instructions(text: str) -> list[str]:
    """Instruction words spoken in English, as plain-English advice."""
    seen, out = set(), []
    for _orig, _rom, eng in read_tokens(text):
        if eng in INSTRUCTIONS:
            w = INSTRUCTIONS[eng]
            if w not in seen:
                seen.add(w); out.append(w)
    return out
