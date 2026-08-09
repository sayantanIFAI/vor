"""Node 3: safety validation.

Independent of whatever the extraction LLM flagged itself. This layer
exists because the LLM's own self-flagging was proven inconsistent this
session: on the same response, it correctly flagged one garbled term as
uncertain but confidently (and wrongly) resolved a different garbled term
into a named drug. This node does not trust the LLM's confidence - it
re-derives review flags from hard rules.
"""
from __future__ import annotations

import re

from .gate import PROBABLE, REJECTED, VERIFIED, judge_medication
from .glossary import (display_name, is_lab_test, lookup_drug,
                        scan_conditions, scan_symptoms)
from .schema import ExtractedRx, Medication

# Below this CTC/RNNT agreement a segment is treated as too garbled to
# support a symptom claim. Chosen against real data: the segment that
# produced hallucinated eye symptoms scored 0.29, the median segment 0.56.
LOW_AGREEMENT = 0.5


def _is_latin(text: str) -> bool:
    """Whether the text is written in the Latin alphabet."""
    letters = [c for c in text if c.isalpha()]
    return bool(letters) and all(c.isascii() for c in letters)


def _set_prescribed_name(med) -> None:
    """Decide the name that goes on the prescription. See schema.Medication.

    The tier already carries the distinction this needs, so no new
    heuristic is invented here:

      VERIFIED  the spoken string is itself a name the gazetteer knows -
                a brand or a generic. The doctor said a real drug name;
                print it. Rewriting "Ecosprin" to "Aspirin" is correct
                pharmacology but reads as a substitution, and it was
                reported as one.

      PROBABLE  the spoken string matched nothing and was resolved by
                similarity - which is what ASR garble looks like.
                "Rasu Basta Tin" is not a drug, a brand, or a word;
                printing it verbatim puts a non-existent medicine on the
                script. Print the resolved name.

    NO SEPARATE, HIGHER FLOOR FOR SUBSTITUTING THE NAME. It was considered:
    printing a specific name is a stronger claim than merely keeping the
    row, so a stricter threshold would be defensible. The scored data in
    gate.py does not support one - real drugs land at 0.700 and 0.818,
    false positives at 0.588 and below, so any second floor would sit in
    the same single gap as SIMILARITY_FLOOR and separate nothing. Adding a
    constant that no measurement distinguishes would be false precision.
    Revisit only if a scored set ever shows two distinct bands.

    This does NOT contradict "PROBABLE is never auto-applied". That rule
    exists so a resolution can never happen SILENTLY - the failure behind
    it was a garbled fragment becoming "Naloxone" with nothing to show a
    reviewer. Here verified stays False, review_reason stays set, and
    heard_as preserves the original, so the substitution is visible in
    exactly the place a human is already being asked to confirm.
    """
    spoken = med.drug.strip()
    if med.tier == VERIFIED and _is_latin(spoken):
        med.prescribed_name = spoken
        med.heard_as = ""
        return

    # Said in BENGALI. The text cannot go on an English prescription, but
    # the molecule is not the right answer either - a doctor who said
    # সর্বিট্রেট should read "Sorbitrate", not "Nitroglycerin". Resolve it
    # to whichever of that drug's own names was spoken.
    if med.tier == VERIFIED:
        entry = lookup_drug(spoken)
        if entry is not None:
            med.prescribed_name = display_name(entry, spoken)
            med.heard_as = spoken
            return

    # Garbled, or a name the gazetteer holds no entry for. Fall back to the
    # resolved name and keep the original visible.
    med.prescribed_name = med.canonical or spoken
    med.heard_as = spoken if med.prescribed_name != spoken else ""


# Values that carry no information. The model writes these instead of
# leaving a field empty, and they were printed verbatim: a real
# prescription line read "Nitrofurantoin [Not specified Not specified
# Not specified]". A blank field is the honest representation - it prompts
# the question, where filler reads as though something was recorded.
_NO_INFORMATION = frozenset({
    "not specified", "unspecified", "not mentioned", "not stated",
    "not given", "unknown", "none", "null", "n/a", "na", "-", "--",
})

# A dosage FORM is not a dose. "Tablet" and "gel" arrived in the dosage
# column, where a reader looking for a quantity finds a noun.
_DOSAGE_FORMS = frozenset({
    "tablet", "tab", "tabs", "capsule", "cap", "caps", "gel", "cream",
    "ointment", "syrup", "drops", "drop", "injection", "inhaler",
    "spray", "lotion", "powder", "sachet",
})

# Spelled-out quantities the ASR produced instead of digits.
_NUMBER_WORDS = frozenset({
    "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "fifteen", "twenty", "thirty",
    "forty", "fifty", "sixty", "eighty", "hundred", "half",
})

# Ordinary dosing English. A spelled-out number is NOT itself a problem -
# "one per day" and "Five Days" are perfectly clear instructions. What
# signals an unresolved dose is a number sitting next to a token that is
# not a word at all: "One Eightti EMI" is a mangled "180 mg".
_DOSING_VOCAB = _NUMBER_WORDS | _DOSAGE_FORMS | frozenset({
    "per", "a", "an", "the", "day", "days", "daily", "week", "weeks",
    "month", "months", "night", "nights", "morning", "evening",
    "afternoon", "noon", "time", "times", "hourly", "hours", "hour",
    "before", "after", "with", "without", "food", "meal", "meals",
    "breakfast", "lunch", "dinner", "bed", "bedtime", "empty", "stomach",
    "and", "at", "in", "on", "every", "once", "twice", "thrice",
    "od", "bd", "tds", "qid", "hs", "sos", "prn", "stat",
    "mg", "ml", "mcg", "gm", "g", "iu", "unit", "units", "drop", "puff",
    "puffs", "week's", "as", "needed", "required", "continue", "continued",
})


def _strip_no_information(value: str) -> str:
    """Remove filler phrases; return "" if nothing of substance remains."""
    out = value.strip()
    low = out.lower().strip(" .,;:")
    if low in _NO_INFORMATION:
        return ""
    # Also strip a filler phrase trailing real content: the model wrote
    # "daily not specified" into a duration field.
    for filler in sorted(_NO_INFORMATION, key=len, reverse=True):
        for pattern in (f" {filler}", f"{filler} "):
            while pattern in f" {out.lower()} ":
                idx = out.lower().find(filler)
                if idx < 0:
                    break
                out = (out[:idx] + out[idx + len(filler):]).strip(" .,;:")
                if not out:
                    return ""
    return out.strip(" .,;:")


def _clean_dosing(med) -> bool:
    """Normalise the dosing fields. Returns True if a dose needs confirming.

    DELIBERATELY DOES NOT RESOLVE NUMBERS. "One Eightti EMI Tab Five Days"
    is plainly an attempt at "180mg tab, 5 days", and turning it into that
    would be inventing a specific dose out of garble - the same move that
    once produced "Naloxone" from a garbled fragment, in the one field
    where being wrong is a dosing error. A tenfold mistake here is not
    recoverable by a reader who trusts the number.

    So the text is kept exactly as heard and the row is FLAGGED. The doctor
    reads "as heard" and types the dose; that is a five-second correction,
    where a confidently wrong "18mg" is not a correction at all because
    nothing signals it is wrong.
    """
    for field in ("dosage", "frequency", "duration"):
        setattr(med, field, _strip_no_information(getattr(med, field, "") or ""))

    # A form alone is not a dose. Kept only if it is all we have, moved out
    # of the quantity column either way.
    if med.dosage and med.dosage.strip().lower() in _DOSAGE_FORMS:
        med.dosage = ""

    words = [w.strip(" .,;:()").lower()
             for f in ("dosage", "frequency", "duration")
             for w in (getattr(med, f, "") or "").split()]
    words = [w for w in words if w]
    if not any(w in _NUMBER_WORDS for w in words):
        return False
    # A number is present. Flag only if something beside it is not a word
    # this vocabulary recognises and is not itself a digit.
    return any(w not in _DOSING_VOCAB and not any(c.isdigit() for c in w)
               for w in words)


def validate(rx: ExtractedRx) -> ExtractedRx:
    reasons: list[str] = []

    # --- medication gate -------------------------------------------------
    # Replaces drugs.is_known_drug(), which was actively harmful: it tested
    # `n in known` as well as `known in n`, so any short fragment was a
    # substring of some drug name and came back verified. Measured: 'a',
    # 'in', 'or', 'no' and 'Nala' all reported as known drugs, which is how
    # garbage reached medications[] wearing a verified flag.
    kept = []
    for med in rx.medications:
        if not med.drug.strip():
            continue
        v = judge_medication(med.drug)
        med.tier = v.tier
        med.canonical = v.canonical
        med.department = v.department
        med.indication = v.indication
        med.match_similarity = v.similarity

        if v.tier == VERIFIED:
            med.verified = True
            _set_prescribed_name(med)
            if _clean_dosing(med):
                med.review_reason = (
                    "dose heard as words, not digits - CONFIRM the amount")
                reasons.append(
                    f'"{med.prescribed_name}" dose was spoken as words '
                    f'("{med.dosage or med.frequency or med.duration}") '
                    f'and was NOT resolved to a number - CONFIRM'
                )
            kept.append(med)
        elif v.tier == PROBABLE:
            # Real drug, garbled name. Kept, but the canonical name stays a
            # proposal - see gate.py on why this is never auto-applied.
            med.verified = False
            med.review_reason = v.reason
            _set_prescribed_name(med)
            _clean_dosing(med)      # already flagged by tier
            reasons.append(
                f'medication heard as "{med.drug}" is not an exact match - '
                f'shown as {v.canonical} ({v.similarity:.2f}) - CONFIRM'
            )
            kept.append(med)
        else:
            # Not a drug. Recorded, not discarded.
            rx.rejected_terms.append(f"{med.drug} — {v.reason}")
            reasons.append(
                f'"{med.drug}" was proposed as a medication but rejected: {v.reason}'
            )
    rx.medications = kept

    # --- symptom grounding ------------------------------------------------
    # Symptoms extracted from a segment the two decoders disagree about are
    # demoted rather than asserted.
    #
    # A live consultation produced "eye redness" and "itchy eyes" from:
    #     কেলেঙ্কারি খুব জোটে চোখ লেগেছিল
    # which is nonsense - but it contains চোখ ("eye"), so the model
    # "grounded" one garbled word and invented two specific findings from
    # it. The patient had never mentioned their eyes.
    #
    # The signal was already there and unused: that segment scored 0.29
    # decoder agreement against a median of 0.56. Low agreement means the
    # decoders could not even agree what was SAID, so anything built on top
    # of it is speculation.
    #
    # Demoted, not deleted - they move to raw_uncertain_terms where a
    # reviewer still sees them. Dropping a real symptom would be worse.
    if (rx.symptoms and rx.decoder_agreement < LOW_AGREEMENT
            and len(rx.source_transcript.split()) >= 3):
        for s in rx.symptoms:
            note = (f"{s} (from a garbled segment, decoder agreement "
                    f"{rx.decoder_agreement:.2f} - not confirmed)")
            if note not in rx.raw_uncertain_terms:
                rx.raw_uncertain_terms.append(note)
        reasons.append(
            f"{len(rx.symptoms)} symptom(s) discarded: decoders disagreed "
            f"({rx.decoder_agreement:.2f}) on what was said"
        )
        rx.symptoms = []

    # --- symptom corroboration --------------------------------------------
    # The model proposes a symptom; the TRANSCRIPT decides whether it was
    # said. Same principle as the drug gate, and needed for the same
    # reason: on a cataract consultation the model reported "loose stools"
    # and a patient "wiping their eye with a finger". Neither was spoken.
    #
    # Corroboration is against the source text, not merely against the term
    # table - "loose stools" IS a valid clinical term, which is exactly why
    # checking the table alone would have confirmed a hallucination.
    #
    # Uncorroborated symptoms are MOVED, not deleted. The gazetteer cannot
    # know every way a patient describes something, so an unmatched symptom
    # is often real; it just cannot be asserted.
    if rx.symptoms and rx.source_transcript:
        supported = {t.lower() for t in scan_symptoms(rx.source_transcript)}
        supported |= {t.lower() for t in scan_conditions(rx.source_transcript)}
        confirmed, unconfirmed = [], []
        for s in rx.symptoms:
            sl = s.lower()
            if any(sl == t or sl in t or t in sl for t in supported):
                confirmed.append(s)
            else:
                unconfirmed.append(s)
        rx.symptoms = confirmed
        for s in unconfirmed:
            if s not in rx.symptoms_unconfirmed:
                rx.symptoms_unconfirmed.append(s)
        if unconfirmed:
            reasons.append(
                f"{len(unconfirmed)} symptom(s) not corroborated by the "
                f"transcript - confirm before use"
            )

    # --- lab gate --------------------------------------------------------
    # Same principle as the medication gate, and added because a live run
    # exposed the gap: labs_ordered came back as ['রেজ', 'সুগার', 'টেস্ট ক্ব']
    # - raw SLM output, where "রেজ" and "টেস্ট ক্ব" are ASR garble. The
    # pipeline was only APPENDING gazetteer hits to this list, never
    # filtering what the model proposed, so garbage was presented as a
    # confirmed lab order.
    #
    # Resolved names are canonicalised ("ইসিজি" -> "ECG"). Unresolved ones
    # are not deleted - they move to raw_uncertain_terms, because an
    # unrecognised lab name may still be a real order the gazetteer is
    # missing, and that has to stay visible.
    gated_labs: list[str] = []
    for lab in rx.labs_ordered:
        canon = is_lab_test(lab)
        if canon:
            if canon not in gated_labs:
                gated_labs.append(canon)
        else:
            note = f"{lab} (proposed as a lab test, not recognised)"
            if note not in rx.raw_uncertain_terms:
                rx.raw_uncertain_terms.append(note)
            reasons.append(f'lab "{lab}" not in the clinical gazetteer - confirm')
    rx.labs_ordered = gated_labs

    # --- recover what the model set aside ------------------------------------
    # The model files anything it is unsure of into raw_uncertain_terms, and
    # nothing ever looked at that list again - so a term the gazetteer can
    # resolve perfectly well sat there unexamined.
    #
    # Measured on a diabetes consultation: "অ্যামোরাল (possible medication
    # name, ASR unclear)". The gate resolves it to Glimepiride at 0.75, and
    # Glimepiride alongside the Metformin in the same sentence is the
    # standard pairing. The drug was recoverable the whole time; it was
    # never offered to the gate, because the model had already decided it
    # was uncertain and uncertainty was treated as final.
    #
    # THE MODEL'S DOUBT IS NOT A VERDICT. It is the same principle as the
    # rest of this file - the model proposes, the gazetteer decides. Doubt
    # is a reason to check, not a reason to discard.
    #
    # Nothing here is asserted: a recovery lands as PROBABLE at best, keeps
    # its original text, and is flagged for confirmation. The failure this
    # closes is the opposite of the "Naloxone" one - not a garbled fragment
    # becoming a confident drug, but a real drug staying invisible.
    recovered: list[str] = []
    for term in rx.raw_uncertain_terms:
        # Strip the parenthetical note the model or this file appended.
        candidate = re.sub(r"\s*\([^)]*\)\s*$", "", term).strip()
        n_tokens = len(candidate.split())
        if not candidate or n_tokens > 5:
            continue                      # a sentence is not a name

        # Labs tolerate more tokens than drugs: an acronym spelled out
        # letter by letter is legitimately five ("এইচ বি এ ওয়ান সি"),
        # where a drug name is one to three.
        lab = is_lab_test(candidate)
        if lab:
            if lab not in rx.labs_ordered:
                rx.labs_ordered.append(lab)
                reasons.append(
                    f'lab "{candidate}" recovered from uncertain terms - CONFIRM')
            recovered.append(term)
            continue

        if n_tokens > 3:
            continue
        v = judge_medication(candidate)
        if v.tier not in (VERIFIED, PROBABLE):
            continue
        key = (v.canonical or candidate).strip().lower()
        if any((m.canonical or m.drug).strip().lower() == key
               for m in rx.medications):
            recovered.append(term)        # already present under another form
            continue

        med = Medication(drug=candidate, canonical=v.canonical, tier=v.tier,
                         department=v.department, indication=v.indication,
                         match_similarity=v.similarity, verified=False,
                         review_reason="recovered from uncertain terms - CONFIRM")
        _set_prescribed_name(med)
        rx.medications.append(med)
        recovered.append(term)
        reasons.append(
            f'medication "{candidate}" recovered from uncertain terms - '
            f'resembles {v.canonical} ({v.similarity:.2f}) - CONFIRM'
        )

    rx.raw_uncertain_terms = [t for t in rx.raw_uncertain_terms
                              if t not in recovered]

    if rx.raw_uncertain_terms:
        reasons.append(f"{len(rx.raw_uncertain_terms)} uncertain term(s) flagged by extraction")

    if not rx.symptoms and not rx.medications and not rx.raw_uncertain_terms:
        reasons.append("no clinical content extracted - verify this segment was worth processing")

    # any medication that made it through without dosage/frequency at all is
    # incomplete enough to warrant a look, even if the drug name is known
    for med in rx.medications:
        if med.verified and not med.dosage and not med.frequency and not med.duration:
            reasons.append(f'medication "{med.drug}" has no dosage/frequency/duration - verify')

    # CTC and RNNT share the same encoder but decode independently. Low
    # agreement between them on a segment with real content is a signal
    # neither decoder's own confidence can give you - only checked on
    # segments with enough words that the measure isn't just noise from a
    # single spelling variant (e.g. "আমই" vs "আমি" on a 2-word segment).
    if (rx.decoder_agreement < 0.5
            and len(rx.source_transcript.split()) >= 4
            and (rx.medications or rx.symptoms)):
        reasons.append(
            f"low CTC/RNNT agreement ({rx.decoder_agreement:.2f}) on a segment "
            f"with extracted content - decoders disagree on what was said"
        )

    # medium-confidence auto-corrections changed the text the LLM saw - a
    # human should confirm the swap was right, especially on drug names.
    if rx.correction_needs_confirmation:
        swaps = [c for c in rx.corrections_applied if c.endswith("(medium)")]
        reasons.append(
            f"medium-confidence auto-correction applied ({', '.join(swaps)}) - confirm"
        )

    # a mangled token that closely resembles a real drug name is exactly the
    # case that once produced "Naloxone" from a garbled fragment. Always
    # surface it, never resolve it automatically.
    if rx.drug_candidates:
        reasons.append(
            f"{len(rx.drug_candidates)} possible drug name(s) detected in garbled text - confirm"
        )

    rx.needs_human_review = bool(reasons)
    rx.review_reasons = reasons
    return rx
