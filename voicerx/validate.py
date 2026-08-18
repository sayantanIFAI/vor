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
from difflib import SequenceMatcher

from .gate import PROBABLE, REJECTED, VERIFIED, judge_medication
from .glossary import (_skeleton, department_for, display_name, fold,
                        is_lab_test, lookup_drug, scan_advice,
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


# A drug name must be TRACEABLE TO THE AUDIO.
#
# "Erythromycin" was printed on a urology prescription that never mentions
# it. The path: the model invented the name, and the gate VERIFIED it out
# of the 179,002-entry imported brand register - a machine-imported list
# nobody clinically reviewed - so a fabrication arrived wearing the same
# badge as a drug the doctor actually said. The gazetteer answers "is this
# a real drug"; it was never asked "was this one said".
#
# Checking the model's raw string against the transcript does NOT work and
# was measured failing: garbled-but-real names score no better than
# invented ones. What separates them is checking every KNOWN FORM of the
# RESOLVED drug - generic, brands, Bengali - against the transcript.
# Scored over all 16 consultations:
#
#     0.62  Lignocaine+Hydrocortisone   not said
#     0.67  Erythromycin                not said
#     0.75  Linagliptin                 not said
#     ------------------------------------------ gap
#     0.80  Ecosprin                    said as ইকোস্পিডিন
#     0.80  Norethisterone              said as ট্রিমোলাট
#     0.82  Choline Salicylate          said as কোলন স্যালেসাইল
#     0.89+ everything else
#
# CAVEAT, and it is the same one SIMILARITY_FLOOR carries: 12 samples and
# a 0.05 gap. Ecosprin at 0.80 is the nearest true positive, so this is
# not a constant to nudge for a single case - re-derive it if a larger
# reviewed set ever exists. Demoted to rejected_terms, never deleted.
_GROUNDING_FLOOR = 0.78


def _grounding_score(name: str, canonical: str, transcript: str) -> float:
    """How well any known form of this drug matches something that was said."""
    if not transcript.strip():
        return 1.0                     # nothing to check against
    entry = lookup_drug(name) or lookup_drug(canonical)
    forms = [name]
    if canonical:
        forms.append(canonical)
    if entry is not None:
        forms += [entry.generic, *entry.brands, *entry.bengali]

    folded = fold(transcript)
    if any(fold(f) and fold(f) in folded for f in forms):
        return 1.0                     # a known spelling is literally there

    # Cross-script: the model romanises what the ASR wrote in Bengali, so
    # the two never meet under fold(). Consonant skeletons do.
    tokens = transcript.split()
    spans = tokens + [" ".join(tokens[i:i + 2]) for i in range(len(tokens) - 1)]                    + [" ".join(tokens[i:i + 3]) for i in range(len(tokens) - 2)]
    best = 0.0
    for f in forms:
        sf = _skeleton(f)
        if len(sf) < 4:
            continue
        for span in spans:
            best = max(best, SequenceMatcher(a=sf, b=_skeleton(span)).ratio())
    return best


def department_clash(drug_dept: str, consult_dept: str) -> bool:
    """Whether a fuzzy match landed in a specialty the consultation is not.

    Both sides must be SPECIFIC. "general" drugs - paracetamol, antacids -
    are prescribed in every clinic, so a general drug is never a clash, and
    a consultation with no diagnosis yields no department and no signal.

    Public because the same rule has to run TWICE, on the same terms. This
    function validates one SEGMENT, and a segment rarely states the
    diagnosis - it is a consultation-level fact that only exists after the
    merge. So the check below almost never had a consult_dept to work with
    and silently passed everything. Re-running it at merge time is what
    makes it fire; sharing one function is what stops the two copies from
    drifting apart. See server._merge_segments.
    """
    if not consult_dept or consult_dept == "general":
        return False
    drug_dept = (drug_dept or "").strip()
    if not drug_dept or drug_dept == "general":
        return False
    return drug_dept != consult_dept


# A drug whose department is "general" is never specialty-checked - it is
# prescribed in every clinic, so there is no clash to detect. That is 15% of
# the gazetteer (35 of 234) receiving NO second opinion at all, and it is
# where the survivors cluster: Paracetamol from "একটা ফল" (one fruit) at
# 0.73, Biotin from "hair" at 0.67.
#
# Grounding cannot help either - the garbled words ARE in the transcript, so
# _grounding_score passes them. Nothing downstream will catch these.
#
# So the bar on the FIRST check is raised for exactly the drugs that get no
# second one. Measured on the corpus: general-department guesses that were
# right sit at 0.90 (Dexamethasone), wrong ones at 0.67-0.73.
_GENERAL_FLOOR = 0.80


def _department_from_evidence(rx) -> str:
    """Specialty from what is already established, when no condition names it.

    The diagnosis is the explicit statement and is tried first by the caller.
    It is also the narrowest source in the system: ~50 curated conditions,
    against a gazetteer where all 234 drugs carry a department. So when no
    condition resolves, the record is asked instead of giving up.

    This runs a throwaway pass of the gate over the drug names purely to
    find the EXACT matches. An exact match means the doctor really said that
    name, which makes its department firmer evidence of the clinic than a
    diagnosis the model may have paraphrased - and it was being discarded.
    An eyelid consultation held a VERIFIED "Tobramycin eye drops" while the
    guard reported no department and let three out-of-specialty drugs stand.

    Only VERIFIED counts. A PROBABLE department would let one bad guess pick
    the specialty and then validate the rest of the guesses against itself.
    """
    from collections import Counter
    from .glossary import department_for_labs

    votes = Counter()
    for med in rx.medications:
        name = (med.drug or "").strip()
        if not name:
            continue
        v = judge_medication(name)
        if v.tier == VERIFIED and v.department and v.department != "general":
            votes[v.department] += 1
    if votes:
        return votes.most_common(1)[0][0]

    return department_for_labs(rx.labs_ordered or [])


def medication_decision(verdict, consult_dept: str,
                        strict_unknown: bool = False) -> tuple[str, str]:
    """Keep or demote one medication. THE single implementation.

    Returns ("keep", "") or ("demote", human-readable reason).

    This rule used to be written in three places - the main loop below, the
    uncertain-terms recovery further down, and again at merge in server.py -
    and the recovery copy simply omitted it. An EAR consultation kept
    Dorzolamide, a GLAUCOMA drop, because a guess too weak for the front
    door came in through a side one. Three copies of a safety rule is two
    copies too many.

    VERIFIED is never demoted here: an exact gazetteer hit means the name
    really was spoken, and a doctor may prescribe outside their specialty.
    Only a GUESS is second-guessed.
    """
    if verdict.tier != PROBABLE:
        return "keep", ""

    drug_dept = (verdict.department or "").strip()

    # A guess in the wrong specialty.
    if _department_clash(verdict, consult_dept):
        return "demote", (f"that is a {drug_dept} drug on a "
                          f"{consult_dept} consultation")

    # A guess that no specialty check can reach - see _GENERAL_FLOOR.
    if (not drug_dept or drug_dept == "general") and \
            (verdict.similarity or 0) < _GENERAL_FLOOR:
        return "demote", (f"a general-use drug matched only by similarity "
                          f"({verdict.similarity:.2f}), below the "
                          f"{_GENERAL_FLOOR:.2f} needed where no specialty "
                          f"check applies")

    # A guess on a consultation that could not name its own specialty.
    #
    # FAILING OPEN IS WHAT MADE THE GAZETTEER GAPS EXPENSIVE: an unknown
    # department returned "no clash" and waved everything through, so the
    # consultations the system understood LEAST were policed least. An
    # unrecognised condition must mean more scrutiny, not none.
    #
    # ONLY WHERE THE SPECIALTY IS ACTUALLY KNOWABLE - strict_unknown, set
    # at merge. Per SEGMENT an unknown department is the ordinary case, not
    # a warning: the doctor names the condition in one segment and
    # prescribes in another, so every prescribing segment looks
    # "unidentified" on its own. Enforcing it there deleted real drugs -
    # Glimepiride beside its Metformin on a diabetes consultation, and the
    # only medication on a mouth-ulcer one - both of which resolve
    # perfectly once the segments are merged.
    # "general" counts as unknown HERE, though not for the clash above.
    #
    # A general consultation cannot contradict any drug's specialty, so
    # _department_clash returns False for it - correctly, since paracetamol
    # is prescribed in every clinic. But that same absence of signal is
    # exactly the situation the floor below exists for, and it was being
    # read as "no problem" instead of "nothing checked this".
    #
    # Giving dengue a department of "general" therefore switched the guard
    # off for a whole consultation, and two drugs came through on pure ASR
    # noise at decoder agreement 0.0: "এ এবে" became Tobramycin EYE DROPS
    # and "অ্যান্ড" - the English word "and" - became Pantoprazole, both at
    # 0.73. A department that discriminates nothing must raise the bar, not
    # remove it.
    if strict_unknown and consult_dept in ("", "general") and \
            (verdict.similarity or 0) < _GENERAL_FLOOR:
        return "demote", (f"matched only by similarity "
                          f"({verdict.similarity:.2f}) on a consultation "
                          f"whose specialty could not be determined")

    return "keep", ""


def _department_clash(verdict, consult_dept: str) -> bool:
    """Segment-level wrapper: pulls the department off a gate Verdict."""
    return department_clash(getattr(verdict, "department", ""), consult_dept)


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
    # The consultation's own specialty, used ONLY to break ties between
    # similar-sounding drugs. Clobazam and Clonazepam are 0.78 alike and
    # differ by which clinic you are in; a garbled name should resolve the
    # way a clinician reading the same page would resolve it. It cannot
    # make a non-drug into a drug - see gate._CONTEXT_BONUS.
    dept = department_for(
        ([rx.diagnosis] if rx.diagnosis else []) +
        scan_conditions(rx.source_transcript or ""))
    if not dept:
        dept = _department_from_evidence(rx)

    kept = []
    for med in rx.medications:
        if not med.drug.strip():
            continue
        v = judge_medication(med.drug, dept)
        med.tier = v.tier
        med.canonical = v.canonical
        med.department = v.department
        med.indication = v.indication
        med.match_similarity = v.similarity

        # A name the gazetteer knows, that nobody said. See
        # _GROUNDING_FLOOR - this is the check that catches an INVENTED
        # drug, which the gate cannot: the gate answers "is this a real
        # drug", and a fabrication passes that question easily.
        if v.keep and _grounding_score(
                med.drug, v.canonical, rx.source_transcript) < _GROUNDING_FLOOR:
            rx.rejected_terms.append(
                f"{med.drug} — resolves to {v.canonical or 'a drug'}, but no "
                f"form of that name appears in the transcript")
            reasons.append(
                f'"{med.drug}" is a real drug name that does NOT appear in '
                f'this consultation - NOT included, confirm if intended'
            )
            continue

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
        elif v.tier == PROBABLE and medication_decision(v, dept)[0] == "demote":
            # A GUESS that also lands in the wrong specialty.
            #
            # An exact match is evidence in its own right - the name was
            # said. A fuzzy match is a guess, and this one had two things
            # against it: on a menopause consultation "Traject" resolved by
            # edit distance alone to Linagliptin, a DIABETES drug, and was
            # printed as a medication. The doctor had said "ট্রাফিক" once;
            # the model romanised that same word twice, so one spoken word
            # became two prescription lines.
            #
            # Grounding the name in the transcript was tried first and does
            # not work: scored against the real consultations, the good and
            # bad names overlap - "Colonsalicyl" is a true reading of
            # "কোলন স্যালেসাইল" at 0.80 while the bogus "Traject" scores
            # 0.89 against an unrelated word. No threshold separates them.
            #
            # The specialty does separate them, on every real case
            # available: it clears Trimolat, ট্রাফিক, অ্যামোরাল, Adapaline,
            # Colonsalicyl and Montuculast, and catches only Traject and
            # Roxatodil. Demoted, never deleted - it goes where a human
            # still sees it.
            _why = medication_decision(v, dept)[1]
            rx.rejected_terms.append(
                f"{med.drug} — resembles {v.canonical} ({v.similarity:.2f}), "
                f"but {_why}")
            reasons.append(
                f'"{med.drug}" was resolved only by similarity to '
                f'{v.canonical} - {_why} - NOT included, confirm if intended'
            )
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

    # --- one utterance cannot be two different drugs -------------------------
    # The model romanises what the ASR wrote in Bengali, and BOTH strings
    # reach this list. When they resolve to the same drug, dedup by canonical
    # name already handles it. When they resolve to DIFFERENT drugs, nothing
    # did, and both were printed.
    #
    # Measured on a sports-injury consultation:
    #     আসিক্লোফেন্ক   -> Aceclofenac   0.90  (consonant match, correct)
    #     Asiklofenken  -> Diclofenac    0.73  (edit distance, wrong)
    # one spoken word, two prescription lines, one of them a drug nobody said.
    #
    # Skeleton similarity alone does NOT separate this safely: the true
    # duplicate scores 0.923 while genuinely distinct pairs reach 0.909
    # (Ecosprin / Ecosprin AV) - a 0.014 gap is not a threshold.
    #
    # The second condition is what makes it safe: only a FUZZY resolution is
    # ever demoted. An exact or skeleton hit is evidence the name was said,
    # so two real drugs that merely sound alike both survive; only a guess
    # loses, and only to a better-supported claim on the same sound.
    _SAME_UTTERANCE = 0.88
    survivors: list = []
    for med in rx.medications:
        sk = _skeleton(med.drug)
        beaten = False
        if len(sk) >= 4 and med.match_similarity < 0.9:
            for other in rx.medications:
                if other is med or other.match_similarity <= med.match_similarity:
                    continue
                osk = _skeleton(other.drug)
                if len(osk) < 4:
                    continue
                if SequenceMatcher(a=sk, b=osk).ratio() >= _SAME_UTTERANCE:
                    beaten = True
                    rx.rejected_terms.append(
                        f"{med.drug} — resolved to {med.canonical} by similarity "
                        f"({med.match_similarity:.2f}), but the same spoken word "
                        f"already resolved to {other.canonical} on stronger evidence")
                    reasons.append(
                        f'"{med.drug}" and "{other.drug}" are the same spoken '
                        f'word; kept {other.canonical}, dropped {med.canonical}'
                    )
                    break
        if not beaten:
            survivors.append(med)
    rx.medications = survivors

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
    # A catch-all is not an order. "টেস্ট করবেন" ("you'll do tests") resolves
    # to the placeholder entry "test (unspecified)", which names no
    # investigation - and it was printed on the prescription beside the real
    # ones, where it reads as though something specific was requested.
    #
    # Same judgement as _NO_INFORMATION in the dosing fields: a blank prompts
    # the question, filler pretends it was answered. Dropped only when a REAL
    # lab was also found, because on a transcript where the catch-all is all
    # there is, it is the only evidence that tests were ordered at all.
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
    _named = [l for l in gated_labs if "unspecified" not in l.lower()]
    if _named:
        gated_labs = _named
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
    # ADVICE AND SUMMARY MUST BE ENGLISH, OR THEY ARE NOT ADVICE.
    #
    # The model is asked for "short plain-English instructions" and on
    # degraded audio it stops obeying, echoing the transcript back instead.
    # A real prescription came out carrying "বাচ্চাদের বলছে দুদিন দেখে নে না
    # হলে তুই অযাট লিস্ট ডেঙ্গু টা রুলে কর তারপরে যাবি" as clinical advice -
    # a fragment of conversation, printed for a patient to follow.
    #
    # This is not answered by more prompt-tuning; this file's own history
    # records two rounds of it producing under-extraction rather than fewer
    # false positives. The model PROPOSES and the rules DECIDE, and the rule
    # here is simple: untranslated Bengali in a plain-English field means the
    # model did not understand the line well enough to render it. Its own
    # doubt, expressed by echoing.
    #
    # Curated advice is unaffected - scan_advice returns English canonicals
    # ("use less soap") because that is what the table holds. Nothing is
    # deleted: the text moves to raw_uncertain_terms, where a human can read
    # it beside the audio.
    _bengali = lambda t: any("ঀ" <= c <= "৿" for c in (t or ""))
    _kept_advice = []
    for _a in (rx.advice or []):
        if _bengali(_a):
            rx.raw_uncertain_terms.append(
                f"{_a} (advice left untranslated by the model - not printed)")
            reasons.append(
                "a line of advice came back in Bengali rather than as an "
                "instruction - moved to uncertain terms, not printed")
        else:
            _kept_advice.append(_a)
    rx.advice = _kept_advice

    if _bengali(rx.summary):
        rx.summary = None
        reasons.append("the summary came back untranslated - dropped")

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
        v = judge_medication(candidate, dept)
        if v.tier not in (VERIFIED, PROBABLE):
            continue

        # The SAME rule the main loop applies. This branch used to append
        # straight to rx.medications, which is how Dorzolamide reached an
        # ear consultation. A recovered term is the weakest evidence in the
        # file - the model already said it was unsure - so it does not get
        # a weaker check than everything else.
        # THE BRAND REGISTER VALIDATES A CLAIM, IT DOES NOT RESCUE A DOUBT.
        #
        # The 174k register is imported and unreviewed, and a great many of
        # its keys are ordinary words - "improve" is in it, mapped to
        # Rabeprazole. Reached from here it turned the English word the
        # doctor used about the patient getting better into a VERIFIED PPI
        # at 1.00 on a fever consultation.
        #
        # gate.py already restricts that table to validating a name the
        # model ASSERTED, precisely because fishing it out of free text is
        # dangerous. A term the model filed under raw_uncertain_terms is the
        # opposite of an assertion - it is the model saying it does not
        # know - so the weakest table in the system does not get to settle
        # it. Every curated route still applies.
        if "brand register" in (v.reason or ""):
            rx.rejected_terms.append(
                f"{candidate} — matches {v.canonical} only in the imported "
                f"brand register, and was a term the model was unsure of")
            recovered.append(term)
            continue

        _act, _why = medication_decision(v, dept)
        if _act == "demote":
            rx.rejected_terms.append(
                f"{candidate} — resembles {v.canonical} "
                f"({v.similarity:.2f}), but {_why}")
            reasons.append(
                f'uncertain term "{candidate}" resembles {v.canonical} - '
                f'{_why} - NOT included, confirm if intended')
            recovered.append(term)
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
