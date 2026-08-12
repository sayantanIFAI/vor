"""Orchestrator: audio file -> VAD segments -> ASR -> extraction -> validation.

Extraction runs per VAD segment, not on the whole file concatenated. This
follows directly from the root-cause finding this session: unsegmented
audio is what caused the extraction LLM to hallucinate. Per-segment
extraction keeps each LLM call's context small, short, and grounded.

PHASE 3: Concurrent segment extraction within a chunk using ThreadPoolExecutor.
Segments are processed in parallel; results are merged back in order.
"""
from __future__ import annotations

import dataclasses
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from .asr import ASRNode, TranscribedSegment
from .correct import correct_transcript
from .fuzzy_drugs import find_drug_candidates
from .extract import extract_rx, ExtractionError
from .english import englishise
from .extraction_cache import reset_cache as reset_extraction_cache
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
                  translator: Translator | None = None,
                  max_workers: int = 4):
        """translator=None keeps the Bengali-only path that was actually
        verified. Pass a Translator to enable the bn->en bridge, so the two
        can be compared on the same audio instead of the change being
        assumed to help.

        max_workers: number of concurrent threads for segment extraction
        (default 4, safe for multi-segment consultations without overwhelming
        VRAM or network connection).

        Optimizations:
        - Prompt caching: reuses extraction template across segments
        - Structured outputs: JSON validation + repair
        - Constitutional AI: safety guardrails catch hallucinations
        """
        self.asr = asr_node or ASRNode()
        self.translator = translator
        self.max_workers = max_workers
        reset_extraction_cache()  # Fresh cache per pipeline instance

    def _extract_segment(self, seg: TranscribedSegment) -> tuple[TranscribedSegment, ExtractedRx | None, dict]:
        """Extract one segment's prescription. Returns (segment, extraction, error_dict).

        This is the per-segment extraction logic, factored out so it can run
        in parallel via ThreadPoolExecutor.
        """
        try:
            # post-ASR correction
            cr = correct_transcript(seg.text)
            seg.corrected_text = cr.text
            seg.corrections_applied = (
                [f"{a}->{b}" for a, b in cr.high_conf_applied]
                + [f"{a}->{b}(medium)" for a, b in cr.medium_conf_applied]
            )

            # bn -> en bridge
            english = None
            if self.translator is not None:
                english = self.translator.translate_one(cr.text)
                if english == cr.text:
                    english = None

            rx, diag = extract_rx(cr.text, audio_file=seg.audio_file,
                                   seg_start=seg.start_s, seg_end=seg.end_s,
                                   transcript_en=english)
            rx.transcript_en = english or ""
            rx.decoder_used = seg.decoder_used
            rx.decoder_agreement = seg.decoder_agreement
            rx.corrections_applied = seg.corrections_applied
            rx.correction_needs_confirmation = cr.needs_confirmation
            rx.drug_candidates = [str(c) for c in find_drug_candidates(cr.text)]

            self._gazetteer_fill(rx, cr.text)

            englishise(rx)
            rx = validate(rx)

            return (seg, rx, {})

        except ExtractionError as e:
            error_info = {"segment": f"{seg.start_s}-{seg.end_s}", "error": str(e)}
            placeholder = ExtractedRx(
                source_transcript=seg.corrected_text or seg.text,
                audio_file=seg.audio_file,
                segment_start_s=seg.start_s,
                segment_end_s=seg.end_s,
                decoder_used=seg.decoder_used,
                decoder_agreement=seg.decoder_agreement,
                corrections_applied=seg.corrections_applied,
                confidence_note="EXTRACTION FAILED - gazetteer only, "
                                 "model output not interpreted",
                needs_human_review=True,
                review_reasons=[f"extraction failed after retries: {e}"],
            )
            # THE MODEL FAILING MUST NOT DISCARD WHAT THE GAZETTEER KNOWS.
            #
            # This path used to return the transcript and nothing else. The
            # scans in _gazetteer_fill are deterministic - they never needed
            # the model at all. On a carpal tunnel consultation the ONE
            # segment carrying the diagnosis - "আপনার কার্পেল টানেল সিন্ড্রোম
            # হয়েছে" - was the segment where Qwen returned unparseable JSON
            # three times running. The consultation came back with
            # diagnosis=None.
            #
            # A blank diagnosis is not cosmetic. It yields no department,
            # and department_for() returning "" SWITCHES OFF the specialty
            # guard in server._merge_segments for every drug in the
            # consultation - the guard that stops a dermatology steroid
            # being prescribed on a nerve complaint. One model failure
            # silently disarmed the drug protection for the whole record.
            #
            # Degrading to gazetteer-only keeps the labs, symptoms, advice,
            # diagnosis and any drug actually named in the transcript. It is
            # the same scan, the same gate and the same validate() as the
            # success path, so nothing is trusted more here than there, and
            # the segment stays flagged for review regardless.
            try:
                self._gazetteer_fill(placeholder, seg.corrected_text or seg.text)
                englishise(placeholder)
                placeholder = validate(placeholder)
                placeholder.needs_human_review = True
                if not any("extraction failed" in r
                           for r in placeholder.review_reasons):
                    placeholder.review_reasons.append(
                        f"extraction failed after retries: {e}")
            except Exception as scan_exc:               # noqa: BLE001
                # A recovery attempt must never turn a failed segment into
                # a lost one.
                placeholder.review_reasons.append(
                    f"gazetteer recovery also failed: "
                    f"{type(scan_exc).__name__}: {scan_exc}")
            return (seg, placeholder, error_info)

    def _gazetteer_fill(self, rx, text: str) -> None:
        """Add everything the curated tables can find in the transcript.

        Deterministic and model-independent, which is why it runs on BOTH
        the success path and the extraction-failure path - see the comment
        in _extract_segment. Shared rather than copied so the two cannot
        drift apart.
        """
        for lab in scan_labs(text):
            if lab not in rx.labs_ordered:
                rx.labs_ordered.append(lab)

        known = {(m.canonical or m.drug).lower() for m in rx.medications}
        for drug, printed, spoken, exact in scan_drugs_spoken(text):
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

        for sym in scan_symptoms(text):
            if sym not in rx.symptoms:
                rx.symptoms.append(sym)

        for adv in scan_advice(text):
            if adv not in rx.advice:
                rx.advice.append(adv)

        conditions = scan_conditions(text)
        if conditions and not rx.diagnosis:
            rx.diagnosis = ", ".join(conditions)

        if len(rx.medications) == 1:
            freq, dur = scan_dosing(text)
            med = rx.medications[0]
            if freq and not med.frequency:
                med.frequency = freq
            if dur and not med.duration:
                med.duration = dur

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

        # PHASE 3: Process segments concurrently
        extractions: list[ExtractedRx] = []
        extract_errors = []
        extract_time = 0.0

        if len(segments) > 1:
            # Parallel extraction using ThreadPoolExecutor
            t1 = time.time()
            with ThreadPoolExecutor(max_workers=min(self.max_workers, len(segments))) as executor:
                futures = {executor.submit(self._extract_segment, seg): i
                          for i, seg in enumerate(segments)}

                # Collect results in order (maintains segments[] ↔ extractions[] alignment)
                results = [None] * len(segments)
                for future in as_completed(futures):
                    idx = futures[future]
                    seg, rx, error = future.result()
                    results[idx] = rx
                    if error:
                        extract_errors.append(error)

            extractions = results
            extract_time = time.time() - t1
        else:
            # Single segment: no threading overhead
            for seg in segments:
                t1 = time.time()
                _, rx, error = self._extract_segment(seg)
                extractions.append(rx)
                if error:
                    extract_errors.append(error)
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
