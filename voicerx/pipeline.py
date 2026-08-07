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
from .schema import ExtractedRx
from .validate import validate


@dataclasses.dataclass
class PipelineResult:
    audio_file: str
    segments: list[TranscribedSegment]
    extractions: list[ExtractedRx]
    timing: dict


class VoiceToRxPipeline:
    def __init__(self, asr_node: ASRNode | None = None):
        self.asr = asr_node or ASRNode()

    def process_file(self, audio_path: str) -> PipelineResult:
        t0 = time.time()
        segments = self.asr.transcribe_file(audio_path)
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

                rx, diag = extract_rx(cr.text, audio_file=seg.audio_file,
                                       seg_start=seg.start_s, seg_end=seg.end_s)
                rx.decoder_used = seg.decoder_used
                rx.decoder_agreement = seg.decoder_agreement
                rx.corrections_applied = seg.corrections_applied
                rx.correction_needs_confirmation = cr.needs_confirmation
                # propose (never apply) drug names for still-mangled tokens
                rx.drug_candidates = [str(c) for c in find_drug_candidates(cr.text)]
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
