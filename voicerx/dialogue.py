# -*- coding: utf-8 -*-
"""Node 6: read the consultation as a DIALOGUE, not as a bag of segments.

WHY THIS EXISTS
---------------
Extraction runs per segment, and a segment is roughly one breath. That is
the right unit for grounding a drug name - it keeps every claim next to the
audio that produced it - but it is the wrong unit for MEANING, because the
meaning of a clinical question lives across two turns, not inside one:

    32.7s   বমি হচ্ছে                        doctor asks: vomiting?
    34.6s   না ডাক্তারবাবু বমি তো হয়নি        patient: no, there has been none

Segment-local negation (glossary._span_is_negated) reads each of those
correctly ON ITS OWN. The first has no negator, so "vomiting" is a finding;
the second has one, so it yields nothing. Merging the two gives a
prescription that records vomiting for a patient who just denied it - and
the denial is INVISIBLE, because a suppressed mention produces no output at
all. Nothing downstream can recover what was never emitted.

THE MODEL
---------
A consultation is a sequence of turns in which a later turn may revoke an
earlier one. The same property that makes multi-turn conversation work:
what was said most recently about a topic supersedes what was said before,
and an answer is interpreted against the question that opened it.

So findings are treated as PROVISIONAL and resolved at the end:

  1. Every turn is read for three things, not one - what it ASSERTS, what
     it DENIES, and what it merely RAISES (a question about a symptom is
     not a report of one).
  2. Each symptom accumulates a timeline of those events.
  3. The LAST event wins. A symptom asserted and then denied is denied. A
     symptom denied and then reported is present - patients correct
     themselves, and a rule that let the first mention win would be just as
     wrong in the other direction.

  4. A bare answer - "না", "হ্যাঁ" - carries no symptom of its own, so it
     resolves against the OPEN TOPIC: whatever the previous turn raised.
     This is ordinary anaphora, and it is why the window for a bare answer
     is tight (_ANAPHORA_TURNS). "না" four turns later is answering
     something else.

WHY NOT PUNCTUATION
-------------------
"বমি হচ্ছে" is a question in the audio and a statement on the page. The
obvious repair is to punctuate the transcript, and it is not available:
IndicConformer's vocabulary contains NO punctuation at all - not ।, not ?,
not even a comma - so the model cannot emit a question mark under any
setting. Verified against the checkpoint's own vocab files.

A separate punctuation-restoration model would add nothing but risk. It
would infer the question mark from the same text already in hand, so it
carries no new information, costs a model in memory and a pass per segment,
and introduces one more thing that can be confidently wrong in a
safety-critical path.

Punctuation is only ever a PROXY for what is actually wanted here: who is
speaking, and where the turn changed. Both are already present and free:

  * a PAUSE between segments marks a turn boundary - the timings are
    already carried on every segment (see _TURN_GAP_S);
  * a VOCATIVE marks the patient - one addresses the doctor, not oneself,
    so "না ডাক্তারবাবু বমি তো হয়নি" is an ANSWER;
  * a second-person pronoun marks the doctor - "বমি হচ্ছে আপনার" is asking
    the patient about the patient.

That is strictly more than a question mark would give: it separates a
symptom being ASKED ABOUT from one being REPORTED, which is the actual
distinction the record needs.

WHAT IT DELIBERATELY DOES NOT DO
--------------------------------
It does not guess intonation, and it never invents a finding. Speaker
inference only ever WEAKENS a claim - a symptom the doctor raised is
demoted to "asked" - so a wrong guess costs a confirmation prompt, never a
fabricated symptom. The resolution below does not depend on it: the denial
in the real case is explicit, and recency alone settles it.

Nothing is deleted. A dropped symptom is returned with its reason so the
record can show that the question was asked and answered, which is a
clinical fact in its own right.
"""
from __future__ import annotations

import dataclasses
from typing import Iterable

# How many turns a BARE answer ("না") may reach back to find its topic.
# Deliberately short: an unattached denial is only meaningful next to the
# question it answers, and a wider window starts revoking unrelated findings.
_ANAPHORA_TURNS = 2

# Answers that carry no content of their own.
_BARE_NO = {"না", "নাহ", "নাহি", "নেই", "নয়", "নাই", "no"}
_BARE_YES = {"হ্যাঁ", "হ্যা", "হা", "হুম", "আছে", "yes"}
# Address terms and fillers that may sit beside a bare answer without
# making it a statement: "না ডাক্তারবাবু" is still just "no".
_VOCATIVES = {"ডাক্তারবাবু", "ডাক্তার", "স্যার", "ম্যাডাম", "দাদা", "দিদি",
              "বাবু", "আচ্ছা", "তো", "ঠিক", "আছে"}

# Naming the doctor means one is TALKING to the doctor - so the patient is
# speaking, and what they say about a symptom is a report.
_ADDRESSES_DOCTOR = {"ডাক্তারবাবু", "ডাক্তার", "স্যার", "ম্যাডাম"}
# Second person about the patient's own body: the doctor asking.
_ADDRESSES_PATIENT = {"আপনার", "আপনি", "আপনাকে", "তোমার", "তুমি"}

# A gap this long between segments is a turn boundary rather than one
# speaker drawing breath. Taken from the real consultations: continuation
# runs to about 0.3s, while an answer follows its question by ~0.9s.
_TURN_GAP_S = 0.6

DOCTOR, PATIENT, UNKNOWN = "doctor", "patient", "unknown"


def infer_speaker(text: str) -> str:
    """Who is talking, from address terms alone.

    Deliberately conservative - it returns UNKNOWN unless the sentence
    addresses someone, because the cost of guessing wrong is asymmetric.
    A wrong PATIENT would let a doctor's question stand as a finding.
    """
    toks = set(_tokens(text))
    if toks & _ADDRESSES_DOCTOR:
        return PATIENT
    if toks & _ADDRESSES_PATIENT:
        return DOCTOR
    return UNKNOWN


@dataclasses.dataclass
class Turn:
    """One segment, read for polarity rather than for content alone."""
    idx: int
    text: str
    asserted: tuple[str, ...] = ()
    denied: tuple[str, ...] = ()
    queried: tuple[str, ...] = ()
    bare_no: bool = False
    bare_yes: bool = False
    speaker: str = "unknown"


@dataclasses.dataclass
class Resolution:
    symptoms: list[str]                     # what survives
    denied: list[tuple[str, str]]           # (symptom, why it was dropped)
    notes: list[str]                        # for review_reasons


def _tokens(text: str) -> list[str]:
    return [t.strip("।,?!.…") for t in (text or "").split() if t.strip("।,?!.…")]


# Bengali negates a verb with a suffix, not with a separate word, so the
# denial hides inside the verb: হয়নি ("did not happen"), করেনি ("did not
# do"), খাইনি ("did not eat"). None of these is a negator TOKEN, which is
# why a word-list check misses the plainest refusal a patient can give.
#
# Matched on the verbal endings only. A blanket "ends in নি" would catch
# ordinary nouns - পানি is water - so the vowel sign before it is required.
_NEG_VERB_ENDINGS = ("য়নি", "েনি", "িনি", "ইনি", "োনি", "ননি", "ায়নি")


def _is_negated_verb(token: str) -> bool:
    t = token.strip("।,?!.…")
    return len(t) >= 4 and t.endswith(_NEG_VERB_ENDINGS)


def _is_bare(text: str, vocab: set[str]) -> bool:
    """True when the turn is only an answer - a yes/no plus address terms."""
    toks = [t for t in _tokens(text)]
    if not toks:
        return False
    hit = False
    for t in toks:
        if t in vocab:
            hit = True
        elif t not in _VOCATIVES:
            return False                    # carries content of its own
    return hit


def read_turn(idx: int, text: str) -> Turn:
    """Split one segment into what it asserts, denies and merely raises.

    A mention SUPPRESSED by glossary's span rule is the interesting part -
    it is exactly the information the rest of the pipeline throws away. It
    is suppressed for one of two reasons, and they mean opposite things: a
    negator is the patient saying no, an interrogative is the doctor asking.
    """
    from .glossary import (CONDITIONS, NON_CLINICAL_TERMS, _CURATED_TERM_LOOKUP,
                           _NEGATORS, _INTERROGATIVES, _ngram_scan_all,
                           scan_symptoms)

    def _clinical(items: Iterable[str]) -> list[str]:
        return [t for t in items
                if t not in CONDITIONS and t not in NON_CLINICAL_TERMS]

    # Every symptom the turn NAMES, with the span-level polarity rule turned
    # off, because polarity is decided here at turn level instead.
    #
    # The span rule scans forward from the matched words, which is right for
    # a paragraph and wrong for a spoken turn. The patient's actual denial -
    # "না ডাক্তারবাবু বমি তো হয়নি" - defeats it twice over: the "না" sits
    # BEFORE the symptom where a forward scan never reaches it, and "হয়নি"
    # is not a negator word at all but a verb carrying its negation as a
    # suffix. So the sentence read as a REPORT of vomiting.
    #
    # A turn is one breath. At that length a negator anywhere in it governs
    # everything in it, and the precision the span rule needs is unnecessary.
    mentions = _clinical(_ngram_scan_all(text, _CURATED_TERM_LOOKUP,
                                          honour_negation=False))

    toks = set(_tokens(text))
    has_neg = bool(toks & _NEGATORS) or any(_is_negated_verb(t) for t in toks)
    has_q = bool(toks & _INTERROGATIVES)

    asserted: list[str] = []
    denied: list[str] = []
    queried: list[str] = []
    for term in mentions:
        if has_neg:
            # A turn carrying both ("বমি হচ্ছে কি না") is a question, not an
            # answer - the negator belongs to the question's own form.
            queried.append(term) if has_q else denied.append(term)
        elif has_q:
            queried.append(term)
        else:
            asserted.append(term)

    # A symptom the DOCTOR names about the patient is being asked about, not
    # reported. "বমি হচ্ছে আপনার" reads as a statement on the page and is a
    # question in the room; the second-person pronoun is what gives it away
    # where the missing question mark cannot. Demotion only - the symptom
    # stays on the record as "asked", and a later confirmation restores it.
    speaker = infer_speaker(text)
    if speaker == DOCTOR and asserted:
        queried.extend(asserted)
        asserted = []

    return Turn(idx=idx, text=text or "",
                asserted=tuple(asserted), denied=tuple(denied),
                queried=tuple(queried),
                bare_no=_is_bare(text, _BARE_NO),
                bare_yes=_is_bare(text, _BARE_YES),
                speaker=speaker)


def resolve(texts: list[str], merged_symptoms: list[str],
            spans: list[tuple[float, float]] | None = None) -> Resolution:
    """Decide which symptoms survive the whole conversation.

    `merged_symptoms` is what the pipeline collected across every segment;
    this returns the subset the dialogue actually supports, plus the ones it
    contradicts and why.

    `spans` are the (start, end) times already carried on every segment. A
    gap between them is the only turn boundary available once punctuation is
    ruled out, and it is what stops a bare "না" from answering a question
    nobody asked - see _TURN_GAP_S.
    """
    turns = [read_turn(i, t) for i, t in enumerate(texts)]

    def followed_a_turn(i: int) -> bool:
        """True when this segment begins after a real pause."""
        if not spans or i == 0 or i >= len(spans):
            return True                     # no timings: do not block on it
        gap = (spans[i][0] or 0) - (spans[i - 1][1] or 0)
        return gap >= _TURN_GAP_S

    # (symptom -> [(turn index, polarity)]) in spoken order.
    timeline: dict[str, list[tuple[int, str]]] = {}

    def record(sym: str, i: int, pol: str) -> None:
        timeline.setdefault(sym, []).append((i, pol))

    open_topic: tuple[int, tuple[str, ...]] | None = None
    for t in turns:
        raised = t.asserted + t.queried
        for s in t.asserted:
            record(s, t.idx, "yes")
        for s in t.queried:
            record(s, t.idx, "asked")
        for s in t.denied:
            record(s, t.idx, "no")

        # A bare answer has no symptom of its own - attach it to whatever
        # the previous turn put on the table.
        if (t.bare_no or t.bare_yes) and not raised and not t.denied:
            # An answer has to follow a pause. Without that check a "না"
            # that is merely the tail of the same breath revokes the
            # speaker's own sentence.
            if (open_topic and t.idx - open_topic[0] <= _ANAPHORA_TURNS
                    and followed_a_turn(t.idx)):
                for s in open_topic[1]:
                    record(s, t.idx, "no" if t.bare_no else "yes")
        elif raised:
            open_topic = (t.idx, raised)

    kept: list[str] = []
    dropped: list[tuple[str, str]] = []
    notes: list[str] = []

    for sym in merged_symptoms:
        events = timeline.get(sym)
        if not events:
            kept.append(sym)                # nothing to say about it
            continue
        last_idx, last = events[-1]
        if last == "no":
            # Was it ever independently reported AFTER the denial? No - the
            # last word is a denial, so it stands.
            dropped.append((sym, f"denied by the patient at turn {last_idx + 1}"))
            notes.append(
                f'"{sym}" was raised and then DENIED later in the '
                f'consultation - removed from symptoms, confirm if intended')
        elif last == "asked" and not any(p == "yes" for _, p in events):
            # KEPT. This used to delete the symptom, and it was wrong: in a
            # consultation the DOCTOR names most findings while taking the
            # history, and the patient answers with a nod or a word the
            # gazetteer does not carry. Demoting on the speaker heuristic and
            # then deleting for want of a verbal confirmation emptied four
            # consultations of correct symptoms - cough, breathlessness and
            # wheezing off a dust-allergy consultation among them.
            #
            # Only an EXPLICIT denial removes a finding now. That is what
            # settled the vomiting case; this rule never contributed to it.
            kept.append(sym)
            notes.append(
                f'"{sym}" was raised but not confirmed aloud - kept, confirm '
                f'it belongs')
        else:
            kept.append(sym)

    return Resolution(symptoms=kept, denied=dropped, notes=notes)
