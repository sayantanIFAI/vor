"""Node 3a: the medication gate. Decides what may be called a drug.

THE PROBLEM THIS SOLVES
-----------------------
The SLM proposed these as medications on real consultations:

    "সেবেন করে জল খাবেন এবং বিশ্রাম নেবেন"   (= "drink water and rest")
    "Antibiotic"   "মেডিসিন"   "Powder Pulotis"   "Boot"   "Kujer"   null

None are drugs. Meanwhile the previous check, drugs.is_known_drug(), made
it worse rather than better: it tested `n in known` as well as
`known in n`, so ANY short fragment was a substring of some drug name and
came back verified. Measured: 'a', 'in', 'or', 'no' and 'Nala' were all
reported as known drugs. That check is what let garbage reach the
medications array wearing a verified flag.

THE MODEL PROPOSES, THE GAZETTEER DECIDES
-----------------------------------------
"Is this a drug?" is a closed-set question - drug names are enumerable - so
it is looked up, not reasoned about. Three outcomes, and the middle one is
the point:

    VERIFIED   exact hit in the gazetteer (after phonetic folding, so
               spelling/dialect/accent variants all land here).
    PROBABLE   close to a real drug but not exact - "Montuculast" for
               Montelukast. KEPT, but never silently rewritten: the
               canonical name is attached as a PROPOSAL for the reviewer.
    REJECTED   not a drug. Demoted out of medications[] - but recorded in
               rejected_terms, never discarded, so nothing vanishes
               silently.

A strict allowlist alone was tested and was too harsh: it correctly
rejected 13 of 18 false positives but also threw away "Montuculast" and
"মন ডেগুলাস্ট", which are real prescriptions of Montelukast. The PROBABLE
tier exists precisely for those, and is why this is a three-way decision
rather than a filter.

WHY PROBABLE IS NEVER AUTO-APPLIED
----------------------------------
An earlier version of this system turned a garbled fragment into
"Naloxone" - a real drug, wrong patient, dangerous. Resolving a drug name
automatically is the one thing this pipeline must not do. Everything here
proposes; a human disposes.
"""
from __future__ import annotations

import dataclasses
import re
from difflib import SequenceMatcher

from .glossary import (Drug, fold, is_clinical_term, is_lab_test,
                       lookup_drug, _DRUG_LOOKUP)

# 179,002 Indian brand -> generic names, machine-imported. Optional so a
# fresh checkout works before anyone runs the importer.
try:
    from .brands_india import GENERIC_NAMES, INDIA_BRANDS
except ImportError:                              # pragma: no cover
    INDIA_BRANDS, GENERIC_NAMES = {}, frozenset()

# Folded once at import; a dict lookup keeps this O(1) per candidate.
_BRAND_LOOKUP: dict[str, str] = {}
for _b, _g in INDIA_BRANDS.items():
    _k = fold(_b)
    # curated table always wins - it carries Bengali and a department
    if _k and _k not in _DRUG_LOOKUP:
        _BRAND_LOOKUP.setdefault(_k, _g)

# Generic names, indexed separately from brands.
#
# Added after testing against 200 real cardiology prescriptions, where only
# 20/48 drugs verified: a brand table maps brand -> generic, so generics are
# VALUES and were never looked up. Clinicians write generics constantly, and
# "Flecainide", "Ivabradine", "Dronedarone" and "Sacubitril/Valsartan" were
# all rejected while sitting in the composition column the whole time.
for _g in GENERIC_NAMES:
    _k = fold(_g)
    if _k and _k not in _DRUG_LOOKUP:
        _BRAND_LOOKUP.setdefault(_k, _g.title())

VERIFIED = "verified"
PROBABLE = "probable"
REJECTED = "rejected"

# Similarity floor in FOLDED space. Folding already absorbs dialect and
# accent variation, so what fuzzy matching still has to handle is genuine
# garbling - inserted or dropped syllables.
#
# NOT a guessed constant. Scored against every medication the SLM proposed
# across the 10 real consultations, the two classes separate cleanly:
#
#     0.818  Montuculast     -> Montelukast     REAL DRUG
#     0.700  মন ডেগুলাস্ট     -> Montelukast     REAL DRUG
#     ---------------------------------------- empty band
#     0.588  মেডিসিন          ~ Prednisolone     false positive
#     0.588  এক্সিস্টিং        ~ Aspirin          false positive
#     0.571  জিন টাকে         ~ Insulin glargine  false positive
#     0.526  Powder Pulotis  ~ Clopidogrel      false positive
#     0.500  Antibiotic      ~ Pantoprazole     false positive
#
# 0.65 sits in the gap, ~0.06 clear of the nearest error on either side.
#
# CAVEAT: tuned on 13 non-exact samples from 10 consultations. That is the
# best evidence available, not a large one. Re-derive this from the score
# distribution when a bigger reviewed set exists - do not nudge it to fix a
# single case.
SIMILARITY_FLOOR = 0.65

# Never propose a drug for these, whatever the edit distance says. A term
# that is a known symptom or a known lab test has already been positively
# identified as something else.
_MIN_FUZZY_LEN = 5


@dataclasses.dataclass
class Verdict:
    tier: str
    canonical: str = ""       # resolved generic name, when known
    department: str = ""
    indication: str = ""
    reason: str = ""
    similarity: float = 0.0

    @property
    def keep(self) -> bool:
        """Whether this term may stay in medications[]."""
        return self.tier in (VERIFIED, PROBABLE)


def _fuzzy(text: str) -> tuple[Drug | None, float]:
    """Closest gazetteer drug in folded space. Proposes only."""
    t = fold(text)
    if len(t) < _MIN_FUZZY_LEN:
        return None, 0.0
    best: Drug | None = None
    best_score = 0.0
    for key, drug in _DRUG_LOOKUP.items():
        if len(key) < _MIN_FUZZY_LEN:
            continue
        score = SequenceMatcher(a=t, b=key).ratio()
        if score > best_score:
            best, best_score = drug, score
    return best, best_score


def judge_medication(name: str) -> Verdict:
    """Classify one SLM-proposed medication name."""
    raw = (name or "").strip()
    if not raw or raw.lower() in ("null", "none", "n/a"):
        return Verdict(REJECTED, reason="empty or null drug name")

    # 1. exact gazetteer hit, post-fold
    hit = lookup_drug(raw)
    if hit is not None:
        return Verdict(VERIFIED, canonical=hit.generic,
                       department=hit.department, indication=hit.indication,
                       similarity=1.0,
                       reason="exact gazetteer match")

    # 1b. Indian brand table (179k entries).
    #
    # EXACT folded match only, and deliberately NOT used to scan free text.
    # The SLM has already asserted "this is the drug"; this validates that
    # claim. Fishing 179,000 brand names out of a raw transcript would be a
    # different and much more dangerous operation - a great many of them are
    # ordinary words, and the "hair loss -> Lactulose" failure would come
    # back at scale. Restricting it to candidate validation keeps the
    # coverage without the exposure.
    brand = _BRAND_LOOKUP.get(fold(raw))
    if brand:
        return Verdict(VERIFIED, canonical=brand, similarity=1.0,
                       reason="Indian brand register (imported, not clinically reviewed)")

    # 1c. Combination products written as "A/B" or "A+B".
    #     "Sacubitril/Valsartan" is one prescription line but two molecules,
    #     and neither the brand nor the generic index holds the joined form.
    #     Resolving on a component is enough to confirm it IS a drug, which
    #     is the only question this gate answers.
    if any(sep in raw for sep in ("/", "+")):
        parts = [p.strip() for p in re.split(r"[/+]", raw) if p.strip()]
        resolved = [p for p in parts
                    if lookup_drug(p) or _BRAND_LOOKUP.get(fold(p))]
        if resolved and len(resolved) == len(parts):
            names = []
            for p in parts:
                d = lookup_drug(p)
                names.append(d.generic if d else _BRAND_LOOKUP[fold(p)])
            return Verdict(VERIFIED, canonical=" + ".join(names),
                           similarity=1.0,
                           reason="combination product, all components resolved")

    # 2. positively identified as something that is NOT a drug. Checked
    #    BEFORE fuzzy, so "sugar" can never be fuzzy-matched onto a drug.
    term = is_clinical_term(raw)
    if term:
        return Verdict(REJECTED, canonical=term,
                       reason=f"clinical term, not a drug: {term}")
    lab = is_lab_test(raw)
    if lab:
        return Verdict(REJECTED, canonical=lab,
                       reason=f"lab test, not a drug: {lab}")

    # 3. close to a real drug - propose, never apply
    cand, score = _fuzzy(raw)
    if cand is not None and score >= SIMILARITY_FLOOR:
        return Verdict(PROBABLE, canonical=cand.generic,
                       department=cand.department, indication=cand.indication,
                       similarity=round(score, 2),
                       reason=(f"not an exact match; resembles {cand.generic} "
                               f"(similarity {score:.2f}) - CONFIRM, not applied"))

    return Verdict(REJECTED, similarity=round(score, 2),
                   reason="not in the clinical gazetteer")


def judge_all(names: list[str]) -> list[tuple[str, Verdict]]:
    return [(n, judge_medication(n)) for n in names]
