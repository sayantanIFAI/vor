"""Orchestrator: audio file -> VAD segments -> ASR -> extraction -> validation.

Extraction runs per VAD segment, not on the whole file concatenated. This
follows directly from the root-cause finding this session: unsegmented
audio is what caused the extraction LLM to hallucinate. Per-segment
extraction keeps each LLM call's context small, short, and grounded.
"""
from __future__ import annotations

import dataclasses
import time

from .asr import ASRNode, TranscribedSegment
from .correct import correct_transcript
from .fuzzy_drugs import find_drug_candidates
from .extract import extract_rx, ExtractionError
from .english import englishise
from .glossary import (scan_advice, scan_conditions, scan_dosing,
                        scan_drugs_spoken, scan_labs, scan_symptoms)
from .schema import ExtractedRx, Medication
from .translate import Translator
from .validate import validate


@dataclasses.dataclass
class PipelineResult:
    audio_file: str
    segments: list[TranscribedSegment]
    extractions: list[ExtractedRx]
    timing: dict


class VoiceToRxPipeline:
    def __init__(self, asr_node: ASRNode | None = None,
                  translator: Translator | None = None):
        """translator=None keeps the Bengali-only path that was actually
        verified. Pass a Translator to enable the bn->en bridge, so the two
        can be compared on the same audio instead of the change being
        assumed to help."""
        self.asr = asr_node or ASRNode()
        self.translator = translator

    def process_file(self, audio_path: str, skip_before_s: float = 0.0,
                      max_end_s: float | None = None) -> PipelineResult:
        """Transcribe and extract.

        skip_before_s / max_end_s are used by streaming capture to process
        only the newly-arrived audio on each chunk. Defaults reproduce the
        original whole-file behaviour exactly.
        """
        t0 = time.time()
        segments = self.asr.transcribe_file(audio_path,
                                             skip_before_s=skip_before_s,
                                             max_end_s=max_end_s)
        asr_time = time.time() - t0

        extractions: list[ExtractedRx] = []
        extract_time = 0.0
        extract_errors = []

        for seg in segments:
            t1 = time.time()
            try:
                # post-ASR correction of known systematic confusions, learned
                # from real human corrections (see correct.py). Runs BEFORE
                # extraction so the LLM sees "ক্ল্যাভাম" rather than "ক্লাব".
                cr = correct_transcript(seg.text)
                seg.corrected_text = cr.text
                seg.corrections_applied = (
                    [f"{a}->{b}" for a, b in cr.high_conf_applied]
                    + [f"{a}->{b}(medium)" for a, b in cr.medium_conf_applied]
                )

                # bn -> en bridge, when enabled. Falls back to the Bengali
                # text on any failure (see translate.py), so this can only
                # change output quality, never lose a segment.
                english = None
                if self.translator is not None:
                    english = self.translator.translate_one(cr.text)
                    if english == cr.text:      # pass-through == not translated
                        english = None

                rx, diag = extract_rx(cr.text, audio_file=seg.audio_file,
                                       seg_start=seg.start_s, seg_end=seg.end_s,
                                       transcript_en=english)
                rx.transcript_en = english or ""
                rx.decoder_used = seg.decoder_used
                rx.decoder_agreement = seg.decoder_agreement
                rx.corrections_applied = seg.corrections_applied
                rx.correction_needs_confirmation = cr.needs_confirmation
                # propose (never apply) drug names for still-mangled tokens
                rx.drug_candidates = [str(c) for c in find_drug_candidates(cr.text)]

                # Lab tests come from the gazetteer reading the ORIGINAL
                # Bengali, not from the SLM and not from the translation.
                # The SLM missed every lab order in the 10 consultations;
                # the gazetteer finds CBC even in the ASR's mangled
                # "সি ভিসিটা". Merged rather than replacing, so anything the
                # SLM did catch survives.
                for lab in scan_labs(cr.text):
                    if lab not in rx.labs_ordered:
                        rx.labs_ordered.append(lab)

                # Drugs, same principle as labs. The SLM misses drug names
                # the ASR split across words - a live consultation had
                # "মেট ফর্মিন", "রসু ভাস্টাটিন" and "মেটো প্রোল" all absent
                # from medications[] while sitting in the transcript. The
                # fold joins the pieces and resolves them exactly.
                # scan_drugs_spoken, not scan_drugs: the brand the doctor
                # actually named has to survive. Scanning for the generic
                # is what put "Nitroglycerin" on a prescription where the
                # word spoken was সরবিট্রেট (Sorbitrate).
                known = {(m.canonical or m.drug).lower() for m in rx.medications}
                for drug, printed, spoken, exact in scan_drugs_spoken(cr.text):
                    if drug.generic.lower() in known:
                        continue
                    known.add(drug.generic.lower())
                    rx.medications.append(Medication(
                        drug=spoken, canonical=drug.generic,
                        prescribed_name=printed,
                        heard_as=spoken if printed != spoken else "",
                        tier="verified" if exact else "probable",
                        verified=exact,
                        department=drug.department, indication=drug.indication,
                        match_similarity=1.0 if exact else 0.9,
                        review_reason="found in transcript by gazetteer, "
                                       "not proposed by the model",
                    ))

                # Symptoms and the diagnosis from the gazetteer, same
                # principle as drugs and labs. A live cataract consultation
                # transcribed "ক্যাটারাক্ট" and "ছানি" perfectly and the
                # gazetteer recognised both, yet the prescription came back
                # with a blank diagnosis and no mention of cataract -
                # nothing ever carried the term into an output field.
                for sym in scan_symptoms(cr.text):
                    if sym not in rx.symptoms:
                        rx.symptoms.append(sym)

                # Advice, same principle. It was recognised as non-clinical
                # and then dropped because nothing carried it anywhere.
                for adv in scan_advice(cr.text):
                    if adv not in rx.advice:
                        rx.advice.append(adv)

                # A named condition is a DIAGNOSIS, not a symptom. Only
                # filled when the model left it blank - the doctor's own
                # stated diagnosis always wins.
                conditions = scan_conditions(cr.text)
                if conditions and not rx.diagnosis:
                    rx.diagnosis = ", ".join(conditions)

                # Frequency and duration are spoken in plain Bengali
                # ("দুপুরে খাওয়ার পর"), which the model returns as blank
                # because it expects clinical shorthand. Filled only where
                # the model left them empty - never overwriting it.
                # ONLY when the segment prescribes ONE medicine. A segment
                # timing is a segment fact; attributing it to a particular
                # drug is a guess, and copying it to every drug in the
                # segment made that guess silently, several times over.
                #
                # It produced a clinically wrong instruction on a real
                # cardiology consultation:
                #
                #   "রোজ সকালে খাওয়ার পর ... ইকোস্পিডিন আর রসু ভাস্টা টিন ...
                #    আর বুকে ব্যাথা উঠলে ... জিভের তলায় একটা সর্বিট্রেট"
                #
                # One sentence, two schedules: a daily tablet and an
                # as-needed sublingual. "after breakfast" was copied onto
                # the Sorbitrate, which is taken WHEN THE PAIN STARTS.
                # Nitrate timing is not cosmetic - a patient following that
                # takes it at breakfast and has none during angina.
                #
                # With several drugs in one segment the model's own
                # attribution is the only one with the sentence structure
                # to go on, so nothing is filled in behind it. A blank
                # prompts a question; a wrong schedule does not.
                if len(rx.medications) == 1:
                    freq, dur = scan_dosing(cr.text)
                    med = rx.medications[0]
                    if freq and not med.frequency:
                        med.frequency = freq
                    if dur and not med.duration:
                        med.duration = dur

                # Force output fields to English. Chinese is dropped
                # outright (Qwen falls back to it on Bengali input);
                # Bengali clinical terms are translated via the gazetteer.
                englishise(rx)

                rx = validate(rx)
                extractions.append(rx)
            except ExtractionError as e:
                # NEVER silently drop a segment. Doing so lost real clinical
                # content (a segment containing "পাতলা পায়খানা" vanished
                # entirely) AND misaligned segments[] against extractions[]
                # for everything downstream, because callers zip() them.
                # Emit a placeholder that is impossible to miss instead.
                extract_errors.append({"segment": f"{seg.start_s}-{seg.end_s}", "error": str(e)})
                placeholder = ExtractedRx(
                    source_transcript=seg.corrected_text or seg.text,
                    audio_file=seg.audio_file,
                    segment_start_s=seg.start_s,
                    segment_end_s=seg.end_s,
                    decoder_used=seg.decoder_used,
                    decoder_agreement=seg.decoder_agreement,
                    corrections_applied=seg.corrections_applied,
                    confidence_note="EXTRACTION FAILED - transcript preserved, not interpreted",
                    needs_human_review=True,
                    review_reasons=[f"extraction failed after retries: {e}"],
                )
                extractions.append(placeholder)
            extract_time += time.time() - t1

        return PipelineResult(
            audio_file=audio_path,
            segments=segments,
            extractions=extractions,
            timing={
                "asr_s": round(asr_time, 2),
                "extract_s": round(extract_time, 2),
                "total_s": round(asr_time + extract_time, 2),
                "extract_errors": extract_errors,
            },
        )
