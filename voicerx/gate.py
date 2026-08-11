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
                       lookup_drug, matched_exactly, _DRUG_LOOKUP,
                       _LAB_LOOKUP, _TERM_LOOKUP)

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

# A prescription names the FORM alongside the drug - "অ্যাম্ব্রুডিল সিরাপ",
# "Pan ট্যাবলেট", "Asthalin ইনহেলার". The form is not part of the name.
#
# Leaving it attached was not merely untidy, it lost the drug: the clinical
# term check saw "সিরাপ" inside the phrase, matched the non-drug term
# "syrup", and REJECTED the whole thing - so Ambrodil cough syrup was
# demoted out of the medications list while "অ্যাম্ব্রুডিল" on its own
# resolves perfectly well. The form has to come off before the name is
# judged, not turn into a verdict about it.
_FORM_WORDS = (
    "সিরাপ", "ট্যাবলেট", "ট্যাব", "ক্যাপসুল", "ক্যাপ", "ইনজেকশন",
    "ইনহেলার", "ড্রপ", "ড্রপস", "ক্রিম", "মলম", "জেল", "স্প্রে",
    "পাউডার", "সাসপেনশন", "লোশন", "অয়েন্টমেন্ট",
    "syrup", "tablet", "tab", "capsule", "cap", "injection", "inhaler",
    "drops", "drop", "cream", "ointment", "gel", "spray", "powder",
    "suspension", "lotion",
)


# Ordinary words that are NEVER a drug, whatever the edit distance says.
#
# The gate had no defence against everyday vocabulary. Fuzzy matching always
# returns its nearest neighbour, so a common word does not have to look like
# a drug - it only has to look more like one drug than any other. On a food
# poisoning consultation "খাবার" (food) resolved to Amoxicillin+Clavulanate
# and was proposed as a medication. The ASR heard it correctly and the model
# passed it through correctly; nothing downstream was willing to say that an
# ordinary noun is not a pharmaceutical.
#
# These are words a consultation says constantly BECAUSE it is a
# consultation - meals, water, timing, the body. Registering them as
# clinical terms would be wrong (they are not findings), so they are refused
# here instead, before any similarity is computed.
_NEVER_A_DRUG = frozenset(fold(w) for w in (
    # food and drink - the commonest source, and the one that failed
    "খাবার", "খাবেন", "খাওয়া", "খাদ্য", "ভাত", "রুটি", "জল", "পানি",
    "দুধ", "চা", "তেল", "নুন", "চিনি হীন", "ফাস্ট ফুড", "ঘুগনি", "চপ",
    "food", "water", "milk", "tea", "rice", "meal", "diet",
    # time and routine
    "সকাল", "দুপুর", "বিকেল", "রাত", "রোজ", "দিন", "সপ্তাহ", "মাস",
    "morning", "night", "daily", "week", "month",
    # everyday verbs and fillers that reach the gate as "names"
    "বাড়ি", "দোকান", "রাস্তা", "বাইরে", "বাসি", "ধন্যবাদ", "নার্স",
    "ডাক্তার", "রোগী", "হাসপাতাল",
    "doctor", "nurse", "patient", "hospital", "shop", "outside",
) if fold(w))


def _strip_form(name: str) -> str:
    """Remove a leading/trailing dosage form. Returns "" if nothing is left."""
    parts = [p for p in name.split() if p]
    changed = True
    while changed and parts:
        changed = False
        for end in (0, -1):
            if parts and fold(parts[end]) in _FORM_FOLDED:
                parts.pop(end)
                changed = True
    return " ".join(parts)


_FORM_FOLDED = frozenset(fold(w) for w in _FORM_WORDS if fold(w))


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


# How much a drug from the consultation's own department is favoured when
# choosing between similar-sounding candidates.
#
# THIS REORDERS CANDIDATES; IT NEVER LOWERS THE BAR. The raw similarity
# still has to clear SIMILARITY_FLOOR on its own - context decides WHICH
# real drug a garbled name is, never WHETHER the garble is a drug at all.
# That distinction is the whole safety argument: a neurology consultation
# must not start turning noise into antiepileptics.
#
# This is the general form of what was otherwise being done by hand. A
# seizure consultation returned "লোভা পল" and "ভাল পারিং" - Levipil and
# Valparin, garbled - and each needed its own alias added. Knowing the
# consultation is neurological is what should resolve them, and the
# gazetteer already carries a department on every drug.
_CONTEXT_BONUS = 0.12


def _fuzzy(text: str, department: str = "") -> tuple[Drug | None, float]:
    """Closest gazetteer drug in folded space. Proposes only.

    Returns the RAW similarity, not the context-adjusted one, so callers
    threshold on evidence rather than on a preference.
    """
    t = fold(text)
    if len(t) < _MIN_FUZZY_LEN:
        return None, 0.0
    best: Drug | None = None
    best_score = 0.0
    best_raw = 0.0
    for key, drug in _DRUG_LOOKUP.items():
        if len(key) < _MIN_FUZZY_LEN:
            continue
        raw = SequenceMatcher(a=t, b=key).ratio()
        score = raw + (_CONTEXT_BONUS
                       if department and drug.department == department else 0.0)
        if score > best_score:
            best, best_score, best_raw = drug, score, raw
    return best, best_raw


def judge_medication(name: str, department: str = "") -> Verdict:
    """Classify one SLM-proposed medication name.

    `department` is the consultation's own specialty, when known. It only
    breaks ties between similar-sounding real drugs - see _CONTEXT_BONUS.
    """
    raw = (name or "").strip()
    if not raw or raw.lower() in ("null", "none", "n/a"):
        return Verdict(REJECTED, reason="empty or null drug name")

    # Ordinary vocabulary, refused before any similarity is computed. This
    # runs FIRST because every later step - skeleton, brand register, fuzzy -
    # is a nearest-neighbour search that will happily return something.
    if fold(raw) in _NEVER_A_DRUG:
        return Verdict(REJECTED,
                       reason=f"'{raw}' is an ordinary word, not a medicine")

    # 1. exact gazetteer hit, post-fold
    #
    # The WHOLE name is tried before any form is stripped, because some
    # generics legitimately contain a form word - "Tobramycin eye drops"
    # is the entry's own name, not a drug plus a form.
    hit = lookup_drug(raw)
    if hit is not None and matched_exactly(raw):
        return Verdict(VERIFIED, canonical=hit.generic,
                       department=hit.department, indication=hit.indication,
                       similarity=1.0,
                       reason="exact gazetteer match")
    # A skeleton-only hit is NOT decided here. It is a guess - vowels were
    # dropped - and it must not outrank positive identification as
    # something that is not a drug. See step 2b below for where it lands
    # and why the order was moved.

    # 2. Positively identified as something that is NOT a drug.
    #
    # ORDER MATTERS, and this used to be wrong. This check sat AFTER the
    # imported brand table, so "electrolytes" - a lab test - matched some
    # product in the brand register and came back VERIFIED as a medication
    # before the lab check ever ran. Found on a 500-transcript dry run,
    # where it fired 227 times.
    #
    # The rule now: CURATED beats IMPORTED. The curated drug table above is
    # highest trust and wins outright; the curated lab and clinical-term
    # tables outrank the machine-imported brand register, which is a public
    # dataset nobody clinically reviewed. Only then is the brand table
    # consulted.
    #
    # This also keeps fuzzy matching from ever reaching "sugar".
    # The whole name did not resolve. If it carries a dosage form, take the
    # form off and judge the NAME - before the clinical-term check, which
    # is what turned the form into a rejection: "অ্যাম্ব্রুডিল সিরাপ" matched
    # the non-drug term "syrup" and the drug was demoted out of the
    # prescription, though "অ্যাম্ব্রুডিল" alone resolves.
    bare = _strip_form(raw)
    if bare and bare != raw:
        inner = judge_medication(bare, department)
        if inner.keep:
            return dataclasses.replace(
                inner, reason=f"{inner.reason} (dosage form dropped)")

    # When a consonant candidate exists, a term or lab hit only outranks it
    # if that hit covers the WHOLE name. Both tables match inside a string,
    # and a sub-span match is not an identification of the thing itself.
    #
    # "ডেক্সা মিথোসেন" is Dexamethasone. The lab table matches its first
    # word alone - ডেক্সা - and reports a DEXA bone-density scan, so an
    # allergy patient given a steroid injection was recorded as having had
    # a bone scan. The drug skeleton covers the entire name; the lab hit
    # covers five characters of it. Longest span wins, which is the same
    # rule _drop_overlapped already applies when scanning free text.
    #
    # "স্ট্রেন" is different and must still be rejected: the term table
    # matches the whole word, so it is an identification, not a fragment.
    whole = fold(raw)
    term = is_clinical_term(raw)
    if term and (hit is None or whole in _TERM_LOOKUP):
        return Verdict(REJECTED, canonical=term,
                       reason=f"clinical term, not a drug: {term}")
    lab = is_lab_test(raw)
    if lab and (hit is None or whole in _LAB_LOOKUP):
        return Verdict(REJECTED, canonical=lab,
                       reason=f"lab test, not a drug: {lab}")

    # 2b. Skeleton-only hit - vowels dropped.
    #
    # THIS USED TO RUN FIRST, and that was the same ordering mistake the
    # imported brand register made above: a GUESS was outranking positive
    # identification.
    #
    # Dropping vowels makes this index far more collision-prone than the
    # folded one, so it needs MORE protection, not less. It had less. On a
    # sports-injury consultation the patient said স্ট্রেন - "strain" - and
    # its skeleton "strn" is also the skeleton of Isotroin, a real
    # Isotretinoin brand. An acne drug was therefore proposed on a torn
    # muscle, from an ordinary English word the doctor never used as a
    # drug name.
    #
    # An exact fold match is evidence the name was said, so it still wins
    # outright above. A consonant match is a hypothesis, and any term the
    # curated tables already recognise as a symptom, condition or lab test
    # has a better claim than a hypothesis does.
    if hit is not None:
        return Verdict(PROBABLE, canonical=hit.generic,
                       department=hit.department, indication=hit.indication,
                       similarity=0.9,
                       reason="matched on consonants only (vowel-level ASR "
                              "variation) - CONFIRM")

    # 3. Indian brand / generic register (174k entries).
    #
    # EXACT folded match only, and deliberately NOT used to scan free text.
    # The SLM has already asserted "this is the drug"; this validates that
    # claim. Fishing 174,000 names out of a raw transcript would be a
    # different and much more dangerous operation - a great many of them are
    # ordinary words, and the "hair loss -> Lactulose" failure would come
    # back at scale. Restricting it to candidate validation keeps the
    # coverage without the exposure.
    brand = _BRAND_LOOKUP.get(fold(raw))
    if brand:
        return Verdict(VERIFIED, canonical=brand, similarity=1.0,
                       reason="Indian brand register (imported, not clinically reviewed)")

    # 4. Combination products written as "A/B" or "A+B".
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

    # 5. close to a real drug - propose, never apply
    cand, score = _fuzzy(raw, department)
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
