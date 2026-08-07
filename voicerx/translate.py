"""Node 2a: Bengali -> English bridge, so the SLM never has to read Bengali.

WHY
---
Qwen2.5-7B is weak at Bengali clinical text. Its Bengali came from a small
slice of pretraining, and no practical amount of fine-tuning fixes that:
teaching a model a language is a continued-pretraining problem measured in
billions of tokens, not a LoRA over a few thousand consultations. Worse,
fine-tuning it on ASR output at 25.7% WER would mostly teach it to guess
confidently on garbled input - the worst possible failure for a
prescription.

IndicTrans2 is purpose-built for exactly this direction, runs offline, and
is small (200M distilled / 1B). So the SLM stops doing Bengali:

    Bengali ASR ──> IndicTrans2 (bn->en) ──> Qwen (English extraction)
                └─> glossary.py (Bengali) ──> medications / labs

WHAT THIS DELIBERATELY DOES *NOT* TOUCH
---------------------------------------
Drug and lab names never pass through translation. glossary.py reads the
original Bengali directly and, since the phonetic fold landed, survives
spacing, half-letters, dialect and accent - it recovers CBC even from the
ASR's mangled "সি ভিসিটা". Routing drug names through an MT model could only
add a second place for them to be corrupted, and MT is unreliable on
transliterated brand names precisely because they are not really Bengali
words.

So translation serves the NARRATIVE fields only - symptoms, diagnosis,
summary, follow-up. The safety-critical fields stay on a deterministic
lookup over the original audio's own language. A translation failure
degrades readability; it cannot invent a drug.

The Bengali source is always retained alongside the English so a reviewer
can check the translation rather than trust it.
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)

# Distilled 200M is the default: it runs on CPU at usable speed, which
# matters because this must stay deployable offline in a clinic. The 1B is
# more accurate if a GPU is present.
DEFAULT_MODEL = "ai4bharat/indictrans2-indic-en-dist-200M"
LARGE_MODEL = "ai4bharat/indictrans2-indic-en-1B"

SRC_LANG = "ben_Beng"
TGT_LANG = "eng_Latn"


class Translator:
    """Lazy-loading bn->en translator.

    Degrades to a pass-through if the model cannot be loaded, rather than
    taking the pipeline down. A consultation processed with untranslated
    Bengali narrative is still useful - the medications and labs come from
    the gazetteer and are unaffected - whereas a hard failure loses the
    whole record. `available` records which happened, so the caller can
    surface it instead of silently shipping Bengali in an English field.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._tok = None
        self._proc = None
        self.available = False
        self.load_error = ""

    def load(self) -> bool:
        if self._model is not None:
            return True
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            # IndicTrans2 requires its own preprocessor for the language
            # tags and script normalisation; plain tokenizer use produces
            # silently degraded output rather than an error.
            from IndicTransToolkit.processor import IndicProcessor

            if self.device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"

            self._tok = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name, trust_remote_code=True).to(self.device).eval()
            self._proc = IndicProcessor(inference=True)
            self.available = True
            log.info("IndicTrans2 loaded (%s) on %s", self.model_name, self.device)
        except Exception as exc:                      # noqa: BLE001
            self.load_error = f"{type(exc).__name__}: {exc}"
            self.available = False
            log.warning("IndicTrans2 unavailable, passing Bengali through: %s",
                        self.load_error)
        return self.available

    def translate(self, sentences: list[str], batch_size: int = 8) -> list[str]:
        """bn -> en. Returns the input unchanged if the model is unavailable."""
        if not sentences:
            return []
        if not self.load():
            return list(sentences)

        import torch

        out: list[str] = []
        for i in range(0, len(sentences), batch_size):
            chunk = [s for s in sentences[i:i + batch_size]]
            try:
                prepped = self._proc.preprocess_batch(
                    chunk, src_lang=SRC_LANG, tgt_lang=TGT_LANG)
                enc = self._tok(prepped, truncation=True, padding=True,
                                max_length=256, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    gen = self._model.generate(
                        **enc, max_length=256, num_beams=5,
                        num_return_sequences=1)
                dec = self._tok.batch_decode(gen, skip_special_tokens=True)
                out.extend(self._proc.postprocess_batch(dec, lang=TGT_LANG))
            except Exception as exc:                  # noqa: BLE001
                # Per-batch fallback: one bad batch must not lose the rest.
                log.warning("translation batch failed, keeping Bengali: %s", exc)
                out.extend(chunk)
        return out

    def translate_one(self, text: str) -> str:
        return self.translate([text])[0] if text else ""
