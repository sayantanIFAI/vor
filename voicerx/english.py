"""Node 3c: force the prescription into English.

WHY
---
A live consultation produced this in the SUMMARY field:

    医生提到患者有多年的糖尿病，并进行了冠状动脉支架植入手术。

and this among the symptoms:

    糖尿病        右脚疼痛        শ্বাসকষ্ট        লক্ষণগুলো

Two separate failures, needing two different treatments.

CHINESE - always wrong, always dropped.
Qwen2.5 is a Chinese-origin model. Given Bengali input it intermittently
falls back to Chinese, which no reviewer here can read. There is no
legitimate Chinese in a Bengali consultation, so any CJK text is a model
failure and is removed rather than shown.

BENGALI - usually right, just untranslated.
"শ্বাসকষ্ট" IS the symptom; the model simply failed to translate it. So it
is translated using the gazetteer, which already maps Bengali clinical
terms to English canonical names. That turns "শ্বাসকষ্ট" into
"breathlessness" rather than discarding real clinical content.

Bengali that the gazetteer cannot translate is NOT silently dropped - it
moves to raw_uncertain_terms, where a reviewer sees it. Deleting a symptom
because we could not translate it would be losing patient data to a
formatting rule.

This is applied to the OUTPUT fields only. source_transcript stays Bengali:
it is evidence, and a reviewer must be able to check the original.
"""
from __future__ import annotations

import re

from .glossary import is_clinical_term, is_lab_test, lookup_drug

# CJK Unified Ideographs, plus the Japanese kana blocks for completeness.
_CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿豈-﫿]")
_BENGALI = re.compile(r"[ঀ-৿]")
_LATIN = re.compile(r"[A-Za-z]")


def has_cjk(text: str) -> bool:
    return bool(_CJK.search(text or ""))


def has_bengali(text: str) -> bool:
    return bool(_BENGALI.search(text or ""))


def to_english(text: str) -> tuple[str | None, str]:
    """Return (english_text, reason).

    reason is "" when the text was already usable. Otherwise it explains
    what happened, for the review trail.
    """
    t = (text or "").strip()
    if not t:
        return None, ""

    if has_cjk(t):
        # Never attempt to salvage. A Chinese fragment in a Bengali
        # consultation is a model failure, not content.
        return None, "dropped: model emitted Chinese"

    if not has_bengali(t):
        return t, ""

    # Bengali present. Try the gazetteer, which knows the clinical terms.
    for lookup in (is_clinical_term, is_lab_test):
        hit = lookup(t)
        if hit:
            return hit, ""
    drug = lookup_drug(t)
    if drug:
        return drug.generic, ""

    # Mixed script that partly translated - keep the English part if it can
    # stand alone, otherwise hand the whole thing to the reviewer.
    if _LATIN.search(t):
        english_only = _BENGALI.sub("", t).strip(" ,.;:-—()")
        if len(english_only) >= 4:
            return english_only, ""

    return None, "untranslated Bengali - moved for review"


def englishise(rx) -> None:
    """Rewrite an ExtractedRx's output fields into English, in place.

    Anything removed is recorded in raw_uncertain_terms with a reason, so
    the reviewer can see that something was dropped and why. Nothing
    disappears without trace.
    """
    def _note(original: str, reason: str) -> None:
        # A CJK drop must NOT quote the offending text. The final sweep
        # below removes any entry containing CJK, so echoing it here made
        # the audit note delete itself - the summary disappeared with no
        # record at all, which is exactly what this module promises not to
        # do. Bengali is quoted, because a reviewer can read it and it is
        # often real clinical content.
        if has_cjk(original):
            entry = f"[{len(original)} chars of Chinese removed] ({reason})"
        else:
            entry = f"{original} ({reason})"
        if entry not in rx.raw_uncertain_terms:
            rx.raw_uncertain_terms.append(entry)

    clean_symptoms: list[str] = []
    for s in rx.symptoms:
        eng, reason = to_english(s)
        if eng and eng not in clean_symptoms:
            clean_symptoms.append(eng)
        elif reason:
            _note(s, reason)
    rx.symptoms = clean_symptoms

    for field in ("diagnosis", "summary", "follow_up"):
        value = getattr(rx, field, None)
        if not value:
            continue
        eng, reason = to_english(value)
        setattr(rx, field, eng)
        if not eng and reason:
            _note(value, f"{field} {reason}")

    # MEDICATION FIELDS. Missed in the first version, and Chinese duly
    # appeared in the frequency column of a live prescription. These are
    # the highest-stakes strings in the document - a dosing instruction
    # nobody can read is worse than a blank one, because a blank prompts a
    # question and a garbled one may be copied.
    for med in rx.medications:
        for field in ("dosage", "frequency", "duration"):
            value = getattr(med, field, "") or ""
            if not value:
                continue
            eng, reason = to_english(value)
            setattr(med, field, eng or "")
            if not eng and reason:
                _note(value, f"{med.drug} {field} {reason}")
        # The drug NAME keeps Bengali - it is what was actually said, and
        # the canonical English name already sits alongside it. Only CJK is
        # stripped, since that can only be a model failure.
        if has_cjk(med.drug):
            _note(med.drug, "drug name dropped: model emitted Chinese")
            med.drug = med.canonical or ""
            # prescribed_name may have been derived from the dropped text.
            med.prescribed_name = med.canonical or ""
            med.heard_as = ""

    # A patient NAME is not translated - it is a proper noun, and
    # "translating" it would be inventing a different person. Only CJK is
    # removed, because that can only be a model failure.
    if rx.patient_name and has_cjk(rx.patient_name):
        _note(rx.patient_name, "patient_name dropped: model emitted Chinese")
        rx.patient_name = None

    # Uncertain terms are Bengali BY DESIGN - they are the raw text a human
    # must read - so they are left alone apart from Chinese.
    rx.raw_uncertain_terms = [t for t in rx.raw_uncertain_terms if not has_cjk(t)]
