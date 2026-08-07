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
BENGALI_DRUG_FORMS: dict[str, str] = {
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

# Below this, a "match" is more likely coincidence than a real garbling.
SIMILARITY_FLOOR = 0.62

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
    return SequenceMatcher(a=a, b=b).ratio()


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
