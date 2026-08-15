"""Fuzzy drug-name recovery for Bengali-transliterated pharmaceutical names.

The exact-match tables in correct.py only fix confusions we have literally
seen before. But the underlying failure is systematic: the ASR mangles
English drug names written in Bengali script, and it mangles them
*differently each time* depending on the audio. Real examples from one
consultation, all meaning "Norflox":

    নলাক্স  /  নরফ্লক্স  /  ড লখেিচ্ছিি

An exact-match table can never keep up with that. This module instead does
similarity matching against a curated list of drug names in their Bengali
transliterated forms, so near-misses get recovered even when the exact
garbling has never been seen before.

DELIBERATELY CONSERVATIVE. This never silently rewrites a drug name - it
only proposes a candidate, which the pipeline records for human
confirmation. Given a previous version of this system once turned a garbled
fragment into "Naloxone" (a real, wrong, dangerous drug), automatic
rewriting of drug names is exactly the behaviour we do not want.
"""
from __future__ import annotations

import dataclasses
from difflib import SequenceMatcher

# Bengali transliterations of drugs actually seen in these consultations,
# plus the common OPD formulary. Extend as more corrections arrive.
_SEED_FORMS: dict[str, str] = {
    "নরফ্লক্স": "Norflox",
    "নরফ্লক্স টি জেড": "Norflox-TZ",
    "প্যারাসিটামল": "Paracetamol",
    "ক্ল্যাভাম": "Clavam",
    "এস্কোরিল": "Ascoril",
    "ভ্যালিয়াম": "Valium",
    "মন্টিকুলাষ্ট": "Montelukast",
    "মন্টিকুলাস": "Montelukast",
    "অ্যাজিথ্রোমাইসিন": "Azithromycin",
    "অ্যামোক্সিসিলিন": "Amoxicillin",
    "সেফিক্সিম": "Cefixime",
    "প্যান্টোপ্রাজল": "Pantoprazole",
    "ওমিপ্রাজল": "Omeprazole",
    "ডমপেরিডন": "Domperidone",
    "অনডানসেট্রন": "Ondansetron",
    "সেটিরিজিন": "Cetirizine",
    "লেভোসেটিরিজিন": "Levocetirizine",
    "মেটফরমিন": "Metformin",
    "অ্যামলোডিপিন": "Amlodipine",
    "টেলমিসারটান": "Telmisartan",
    "অ্যাটোরভাস্ট্যাটিন": "Atorvastatin",
    "আইবুপ্রোফেন": "Ibuprofen",
    "ডাইক্লোফেনাক": "Diclofenac",
    "মেট্রোনিডাজল": "Metronidazole",
    "সিপ্রোফ্লক্সাসিন": "Ciprofloxacin",
    "সালবুটামল": "Salbutamol",
    "বুডেসোনাইড": "Budesonide",
    "প্রেডনিসোলন": "Prednisolone",
    "ওআরএস": "ORS",
}

# EVERY Bengali form the gazetteer knows, not a hand-listed few.
#
# This table is what find_drug_candidates() searches, and it held 29 drugs
# while the gazetteer held 234 with 583 Bengali spellings between them. A
# drug outside those 29 that the model also missed was therefore INVISIBLE -
# no layer could propose it, so the gate never got to judge it.
#
# Measured on a live gastro consultation: the ASR heard "পান্ডি", which is
# Pan-D (pantoprazole + domperidone) and exactly right for the complaint.
# judge_medication resolves it at 0.91. Qwen had read the line as advice -
# "take one tablet in the morning" - and dropped the name; scan_drugs_spoken
# needs an exact or skeleton hit and "পান্ডি" is neither. This table was the
# last net, and against 29 forms the best score was 0.40. The prescription
# came back EMPTY with the drug sitting in the transcript.
#
# The caution this list was kept short for is real but aimed elsewhere: it
# is the 174k IMPORTED brand register that must never be fished out of raw
# text, because much of it is ordinary words and none of it is clinically
# reviewed. These 583 forms are the curated tables. A candidate is also
# still only a PROPOSAL - it goes to the gate, and now past the specialty
# guard, the general floor and the grounding check as well.
def _all_bengali_forms() -> dict[str, str]:
    forms = dict(_SEED_FORMS)
    try:
        from .glossary import ALL_DRUGS
    except Exception:                                  # noqa: BLE001
        return forms
    for drug in ALL_DRUGS:
        for form in (drug.bengali or ()):
            f = (form or "").strip()
            # One or two characters cannot be garbled evidence of anything;
            # they would match half the transcript.
            if len(f) < 4:
                continue
            # First writer wins: the seed spellings above are what the ASR
            # actually emits, which is a better target than the dictionary
            # spelling when the two differ.
            forms.setdefault(f, drug.generic)
    return forms


BENGALI_DRUG_FORMS: dict[str, str] = _all_bengali_forms()


# Below this, a "match" is more likely coincidence than a real garbling.
#
# Raised from 0.62 to match gate.SIMILARITY_FLOOR. Scores are computed in
# folded space now, which makes them tighter, and 0.62 was low enough to
# surface cross-class suggestions - Metoprolol proposed as Metronidazole at
# 0.64. Proposing the wrong drug is worse than proposing none: the reviewer
# sees a plausible-looking name and may accept it.
SIMILARITY_FLOOR = 0.65

# Common Bengali words that happen to sit near drug names in edit distance.
# Never propose a drug for these.
STOPWORDS = {
    "বলুন", "আছে", "হচ্ছে", "করে", "খাবেন", "দিচ্ছি", "একটু", "ঠিক",
    "আপনি", "আমি", "আমার", "আপনার", "কিছু", "ওষুধ", "ভালো", "বেশি",
    "দিন", "রাতে", "সকালে", "খুব", "একদম", "তারপর", "হয়েছে", "কেমন",
    "বুকে", "পেটে", "মাথা", "শরীর", "জ্বর", "কাশি", "ব্যথা",
}


@dataclasses.dataclass
class DrugCandidate:
    observed: str          # what the ASR actually produced
    bengali_form: str      # the canonical Bengali spelling it resembles
    english_name: str      # the English drug name
    similarity: float

    def __str__(self) -> str:
        return (f"{self.observed!r} may be {self.english_name} "
                f"({self.bengali_form}, similarity {self.similarity:.2f}) "
                f"- NEEDS HUMAN CONFIRMATION")


def _sim(a: str, b: str) -> float:
    """Similarity in FOLDED space.

    Comparing raw strings scored badly and produced dangerous proposals: a
    live consultation had "মেটো প্রোল" (Metoprolol, a beta blocker) matched
    to Metronidazole, an antibiotic, at 0.64 - a suggestion that crosses
    therapeutic class entirely.

    Folding first removes the spacing, dialect and aspiration noise that
    was dominating the score, so what remains is real garbling. It is the
    same normalisation the gate uses, and the two should agree.
    """
    from .glossary import fold
    return SequenceMatcher(a=fold(a), b=fold(b)).ratio()


def find_drug_candidates(text: str, floor: float = SIMILARITY_FLOOR) -> list[DrugCandidate]:
    """Propose - never apply - drug names for tokens that look mangled.

    Returns candidates sorted best-first. The caller is expected to surface
    these for human confirmation, not to substitute them into the record.
    """
    candidates: list[DrugCandidate] = []
    seen: set[str] = set()

    tokens = text.split()
    # check single tokens and adjacent pairs (many drug names are 2-3 words
    # once transliterated, e.g. "নরফ্লক্স টি জেড")
    spans = [(t,) for t in tokens] + [
        tuple(tokens[i:i + 2]) for i in range(len(tokens) - 1)
    ] + [tuple(tokens[i:i + 3]) for i in range(len(tokens) - 2)]

    for span in spans:
        observed = " ".join(span)
        if observed in seen or len(observed) < 4:
            continue
        if any(tok in STOPWORDS for tok in span):
            continue
        # if any token in a multi-word span is ALREADY a correct drug
        # spelling, the span is just that drug plus surrounding words -
        # proposing a "correction" for it would be noise.
        if len(span) > 1 and any(tok in BENGALI_DRUG_FORMS for tok in span):
            continue

        # If the gazetteer already resolves this span EXACTLY, do not guess.
        # Without this the two layers contradict each other: "রসু ভাস্টাটিন"
        # resolves to Rosuvastatin by lookup while fuzzy simultaneously
        # proposed Atorvastatin at 0.70 - a different statin, shown to the
        # reviewer as if it were the better answer.
        from .glossary import lookup_drug
        if lookup_drug(observed):
            continue

        best: tuple[float, str, str] | None = None
        for bn_form, en_name in BENGALI_DRUG_FORMS.items():
            s = _sim(observed, bn_form)
            if best is None or s > best[0]:
                best = (s, bn_form, en_name)

        if best and best[0] >= floor:
            # an exact match needs no confirmation - it's already correct
            if best[0] < 0.999:
                seen.add(observed)
                candidates.append(DrugCandidate(
                    observed=observed, bengali_form=best[1],
                    english_name=best[2], similarity=round(best[0], 2),
                ))

    candidates.sort(key=lambda c: -c.similarity)
    return candidates
