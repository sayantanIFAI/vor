"""Voice-activity segmentation.

This exists because of a concrete, reproduced failure: IndicConformer's RNNT
decoder silently drops content on long unsegmented audio, and feeding a full
unsegmented consultation to the extraction LLM measurably increases
hallucination risk (a run-on 49s block produced an invented symptom and a
dangerous invented drug name; the same content, isolated, did not).
Segmenting into natural utterances before ASR fixes the cause, not just the
symptom.
"""
from __future__ import annotations

import dataclasses

import torch
import torchaudio


@dataclasses.dataclass
class Segment:
    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


class VoiceActivityDetector:
    """Thin wrapper around Silero VAD with utterance merging.

    Silero returns raw speech spans; merge_gap_s stitches spans separated by
    a short pause into one utterance (so "um... four or five times" stays
    one segment), and max_segment_s hard-splits anything that would still
    exceed IndicConformer's safe length (empirically: RNNT starts dropping
    content well before 49s; CTC has no such ceiling, but keeping segments
    short also keeps the LLM's extraction context small and grounded).
    """

    def __init__(self, max_segment_s: float = 25.0, merge_gap_s: float = 0.6,
                 min_segment_s: float = 0.3):
        self.max_segment_s = max_segment_s
        self.merge_gap_s = merge_gap_s
        self.min_segment_s = min_segment_s
        self.model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad", model="silero_vad",
            force_reload=False, onnx=False, trust_repo=True,
        )
        self._get_speech_timestamps = utils[0]

    def segment(self, audio_path: str) -> list[Segment]:
        wav, sr = torchaudio.load(audio_path)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
            sr = 16000
        wav = wav.squeeze(0)

        raw = self._get_speech_timestamps(
            wav, self.model, sampling_rate=sr, return_seconds=True,
        )
        if not raw:
            total_s = wav.shape[0] / sr
            return [Segment(0.0, total_s)]

        merged: list[Segment] = []
        for r in raw:
            s, e = float(r["start"]), float(r["end"])
            if merged and s - merged[-1].end_s <= self.merge_gap_s:
                merged[-1].end_s = e
            else:
                merged.append(Segment(s, e))

        final: list[Segment] = []
        for seg in merged:
            if seg.duration_s <= self.max_segment_s:
                final.append(seg)
                continue
            # hard-split an over-long merged utterance into fixed windows
            n_parts = int(seg.duration_s // self.max_segment_s) + 1
            part_len = seg.duration_s / n_parts
            for i in range(n_parts):
                final.append(Segment(seg.start_s + i * part_len,
                                      min(seg.end_s, seg.start_s + (i + 1) * part_len)))

        return [s for s in final if s.duration_s >= self.min_segment_s]
