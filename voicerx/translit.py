# -*- coding: utf-8 -*-
"""Latin -> Bengali transliteration for the Indian brand register.

WHY RULES AND NOT A MODEL
-------------------------
The output of this is a LOOKUP KEY, never display text, and it is consumed
by fold() - which already collapses all three sibilants, folds the nasals,
strips spacing and drops doubled letters. So the generated spelling does not
have to be correct Bengali. It has to fold to the same key the ASR produces.

That is a far lower bar than transliteration, and it is why a rule table is
enough. Evidence from the real audio: the ASR wrote "পান্ডি" where the
dictionary form is "প্যান ডি" - the textbook spelling was WRONG and matching
still worked at 0.91, because folding absorbed the difference.

A neural transliterator would be a second model to host, to version and to
be surprised by, in a file whose whole premise is that the answer can be
looked up rather than inferred.
"""
from __future__ import annotations

# Longest first - digraphs must win over their own first letter.
_RULES: list[tuple[str, str, str]] = [
    # (latin, initial-form, post-consonant form)
    ("sch", "স", "স"), ("tch", "চ", "চ"),
    ("sh", "শ", "শ"), ("ch", "চ", "চ"), ("th", "থ", "থ"), ("ph", "ফ", "ফ"),
    ("kh", "খ", "খ"), ("gh", "ঘ", "ঘ"), ("dh", "ধ", "ধ"), ("bh", "ভ", "ভ"),
    ("jh", "ঝ", "ঝ"), ("ng", "ং", "ং"), ("ck", "ক", "ক"), ("qu", "ক", "ক"),
    ("x",  "ক্স", "ক্স"), ("z", "জ", "জ"), ("f", "ফ", "ফ"), ("v", "ভ", "ভ"),
    ("w",  "ও", "ও"), ("y", "য়", "য়"), ("q", "ক", "ক"),
    ("k", "ক", "ক"), ("g", "গ", "গ"), ("j", "জ", "জ"), ("t", "ট", "ট"),
    ("d", "ড", "ড"), ("n", "ন", "ন"), ("p", "প", "প"), ("b", "ব", "ব"),
    ("m", "ম", "ম"), ("r", "র", "র"), ("l", "ল", "ল"), ("s", "স", "স"),
    ("h", "হ", "হ"), ("c", "ক", "ক"),
]
_CONS = {r[0] for r in _RULES}

# Independent vowel (word-initial) and the sign that hangs off a consonant.
_VOWELS: list[tuple[str, str, str]] = [
    ("ee", "ঈ", "ী"), ("oo", "উ", "ু"), ("ai", "আই", "াই"), ("au", "আউ", "াউ"),
    ("ou", "আউ", "াউ"), ("ea", "ই", "ি"), ("ie", "আই", "াই"),
    ("y", "ই", "াই"),       # thyro -> থাইরো. Word-internal y is a vowel.
    ("a", "আ", "া"),         # ASR writes the sign explicitly, so do we
    ("i", "ই", "ি"), ("e", "এ", "ে"), ("o", "ও", "ো"), ("u", "উ", "ু"),
]
_VSET = {v[0] for v in _VOWELS}
_HASANTA = "্"


def translit(word: str) -> str:
    """One Latin token to one Bengali token."""
    w = "".join(c for c in word.lower() if c.isalnum())
    if not w:
        return ""
    out: list[str] = []
    i, at_start = 0, True
    prev_was_cons = False
    while i < len(w):
        if w[i].isdigit():
            i += 1
            continue
        matched = False
        # vowels first when we are not mid-cluster
        for lat, indep, sign in _VOWELS:
            if w.startswith(lat, i):
                out.append(indep if not prev_was_cons else sign)
                i += len(lat); at_start = False; prev_was_cons = False
                matched = True
                break
        if matched:
            continue
        for lat, init, post in _RULES:
            if w.startswith(lat, i):
                # consonant meeting consonant = conjunct, joined by hasanta
                if prev_was_cons:
                    out.append(_HASANTA)
                out.append(init if at_start else post)
                i += len(lat); at_start = False; prev_was_cons = True
                matched = True
                break
        if not matched:
            i += 1
    return "".join(out)


def transliterate_brand(name: str) -> str:
    """A whole brand name as ONE token.

    Joined, not split. Transliterating token by token shattered hyphenated
    and multi-word brands into single characters - "A-ARTI L" became
    আ / আর্টি / ল - and one-character keys are the collision-prone junk this
    gazetteer has been burned by repeatedly (টোবা matching "to the child").

    Joining is not a compromise, it is the correct form: fold() strips
    spacing before any lookup, so a joined key matches however the ASR chose
    to space what it heard. That is the same property that already makes
    "সি বি সি" and "সিবিসি" the same key.
    """
    return translit("".join(name.split()))
