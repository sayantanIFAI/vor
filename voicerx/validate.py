"""Node 3: safety validation.

Independent of whatever the extraction LLM flagged itself. This layer
exists because the LLM's own self-flagging was proven inconsistent this
session: on the same response, it correctly flagged one garbled term as
uncertain but confidently (and wrongly) resolved a different garbled term
into a named drug. This node does not trust the LLM's confidence - it
re-derives review flags from hard rules.
"""
from __future__ import annotations

from .gate import PROBABLE, REJECTED, VERIFIED, judge_medication
from .glossary import is_lab_test
from .schema import ExtractedRx

# Below this CTC/RNNT agreement a segment is treated as too garbled to
# support a symptom claim. Chosen against real data: the segment that
# produced hallucinated eye symptoms scored 0.29, the median segment 0.56.
LOW_AGREEMENT = 0.5


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
            kept.append(med)
        elif v.tier == PROBABLE:
            # Real drug, garbled name. Kept, but the canonical name stays a
            # proposal - see gate.py on why this is never auto-applied.
            med.verified = False
            med.review_reason = v.reason
            reasons.append(
                f'medication "{med.drug}" is not an exact match - '
                f'possibly {v.canonical} ({v.similarity:.2f}) - CONFIRM'
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
