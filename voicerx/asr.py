"""Node 1: ASR - dual-decoder (CTC + RNNT) with cross-validation.

History that matters for anyone touching this file:

RNNT's default "greedy_batch" strategy silently drops content on long
unsegmented audio - confirmed via frame-level alignment inspection (it
emitted zero non-blank tokens for the first 78% of a 49s file), then proven
to be a *sequence-length* bug, not a model-quality one, by slicing the same
audio and re-running: the "missing" content decodes correctly in isolation.
Switching to the non-batched "greedy" strategy fixes the empty-output
failure mode outright.

Once VAD segmentation (this pipeline's Node 1 preprocessing) keeps segments
short - which it does independent of this fix, for extraction-quality
reasons - RNNT with the "greedy" strategy was measured at 55/57 (96.5%)
non-empty on real VAD-length segments (<=25s), and on several segments
produced text that was more accurate than CTC's on the same audio (e.g.
correctly splitting "চার্জিনারস" into "চার্জ নার্স" / "charge nurse").

So: run both decoders per segment. Prefer RNNT's output when it produced
something; fall back to CTC when RNNT is empty (the two observed empty
cases were both very short clips, <=3.1s - a normal edge case, not the
catastrophic failure mode). Record decoder agreement so Node 3 can flag
segments where the two decoders substantially disagree, independent of
whatever either decoder or the extraction LLM is confident about.
"""
from __future__ import annotations

import dataclasses
import os
import tempfile

import torch
import torchaudio
import nemo.collections.asr as nemo_asr
from omegaconf import OmegaConf
from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTDecodingConfig

from .vad import VoiceActivityDetector, Segment

def _resolve_nemo_file() -> str:
    """Locate the .nemo checkpoint without hardcoding a cache path.

    An earlier version hardcoded the full HF cache path including its
    content-hash directory. That path is machine-specific - it broke the
    moment the pipeline moved to a different pod. Resolution order:
      1. VOICERX_NEMO_FILE env var (explicit override)
      2. model_path.txt written by setup_pod.sh
      3. glob the HF cache
    """
    import glob as _glob

    explicit = os.environ.get("VOICERX_NEMO_FILE")
    if explicit and os.path.exists(explicit):
        return explicit

    for marker in ("/workspace/voicerx/model_path.txt", "model_path.txt"):
        if os.path.exists(marker):
            with open(marker) as f:
                snapshot_dir = f.read().strip()
            hits = _glob.glob(os.path.join(snapshot_dir, "*.nemo"))
            if hits:
                return hits[0]

    search_roots = [
        os.environ.get("HF_HOME", "/workspace/.cache/huggingface"),
        "/workspace/.cache",
        os.path.expanduser("~/.cache"),
    ]
    for root in search_roots:
        hits = _glob.glob(
            os.path.join(root, "**", "*indicconformer*", "**", "*.nemo"),
            recursive=True,
        )
        if hits:
            return hits[0]

    raise FileNotFoundError(
        "Could not locate the IndicConformer .nemo checkpoint. Set "
        "VOICERX_NEMO_FILE, or re-run setup_pod.sh to download it."
    )


NEMO_FILE = None  # resolved lazily at ASRNode construction


def _first_text(texts) -> str:
    """model.transcribe() has been observed to return either a flat list of
    strings or a list of single-item lists depending on internal batching
    state - unwrap defensively rather than assume one shape."""
    if not texts:
        return ""
    item = texts[0]
    while isinstance(item, (list, tuple)):
        if not item:
            return ""
        item = item[0]
    return (item or "").strip()


def _word_agreement(a: str, b: str) -> float:
    """Jaccard overlap of the two decoders' word sets. Cheap, symmetric,
    good enough to distinguish "same utterance" from "very different
    output" without needing a real alignment/edit-distance computation."""
    wa, wb = set(a.split()), set(b.split())
    if not wa and not wb:
        return 1.0
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


@dataclasses.dataclass
class TranscribedSegment:
    audio_file: str
    start_s: float
    end_s: float
    text: str                  # raw ASR output (pre-correction)
    decoder_used: str          # "rnnt" or "ctc_fallback"
    ctc_text: str = ""
    rnnt_text: str = ""
    decoder_agreement: float = 1.0
    # filled in by the pipeline's correction pass (see correct.py)
    corrected_text: str = ""
    corrections_applied: list = dataclasses.field(default_factory=list)


class ASRNode:
    def __init__(self, nemo_file: str | None = None, language_id: str = "bn",
                 device: str | None = None):
        self.language_id = language_id
        nemo_file = nemo_file or _resolve_nemo_file()
        print(f"[asr] using checkpoint: {nemo_file}", flush=True)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = nemo_asr.models.ASRModel.restore_from(restore_path=nemo_file)
        self.model = self.model.to(self.device)
        self.model.freeze()

        # non-batched greedy - see module docstring for why "greedy_batch"
        # (the library default) is not used here.
        rnnt_cfg = OmegaConf.structured(RNNTDecodingConfig(strategy="greedy"))
        self.model.change_decoding_strategy(rnnt_cfg, decoder_type="rnnt")

        self.vad = VoiceActivityDetector()

    def _transcribe_clip(self, clip_path: str) -> tuple[str, str]:
        self.model.cur_decoder = "ctc"
        ctc_texts = self.model.transcribe(
            [clip_path], batch_size=1, logprobs=False, language_id=self.language_id,
        )
        ctc_text = _first_text(ctc_texts)

        self.model.cur_decoder = "rnnt"
        rnnt_texts = self.model.transcribe(
            [clip_path], batch_size=1, language_id=self.language_id,
        )
        rnnt_text = _first_text(rnnt_texts)

        return ctc_text, rnnt_text

    def transcribe_file(self, audio_path: str) -> list[TranscribedSegment]:
        segments = self.vad.segment(audio_path)
        wav, sr = torchaudio.load(audio_path)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
            sr = 16000

        results: list[TranscribedSegment] = []
        audio_name = os.path.basename(audio_path)

        with tempfile.TemporaryDirectory() as tmpdir:
            clip_paths = []
            for i, seg in enumerate(segments):
                a = int(seg.start_s * sr)
                b = int(seg.end_s * sr)
                clip = wav[:, a:b]
                clip_path = os.path.join(tmpdir, f"seg_{i:04d}.wav")
                torchaudio.save(clip_path, clip, sr)
                clip_paths.append((seg, clip_path))

            for seg, clip_path in clip_paths:
                ctc_text, rnnt_text = self._transcribe_clip(clip_path)

                if rnnt_text:
                    text, decoder_used = rnnt_text, "rnnt"
                elif ctc_text:
                    text, decoder_used = ctc_text, "ctc_fallback"
                else:
                    continue

                results.append(TranscribedSegment(
                    audio_file=audio_name,
                    start_s=round(seg.start_s, 2),
                    end_s=round(seg.end_s, 2),
                    text=text,
                    decoder_used=decoder_used,
                    ctc_text=ctc_text,
                    rnnt_text=rnnt_text,
                    decoder_agreement=round(_word_agreement(ctc_text, rnnt_text), 2),
                ))

        return results
