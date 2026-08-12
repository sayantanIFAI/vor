"""Node 2c: gazetteer-biased ASR decoding (CTC-WS context biasing).

STATUS: BLOCKED. NOT USABLE WITH THIS MODEL. DO NOT ENABLE. DISABLE IMPORTS."""
=========================================================
NeMo's CTC-WS cannot run against IndicConformer, for an architectural
reason rather than a fixable API mismatch. Recorded here so nobody spends
another day rediscovering it.

Five incompatibilities were found and the first four were fixed:

  1. text_to_ids() needs a lang_id      -> fixed (_BoundTokenizer)
  2. transcribe(logprobs=True) returns  -> fixed (ctc_logprobs(), via
     hypothesis STRING lists, not logprobs   encoder -> ctc_decoder)
  3. add_to_graph's type hint says      -> fixed (plain int ids)
     tuple[str,int]; the code needs ints
  4. NeMo internally calls              -> fixed (_ModelProxy)
     ids_to_tokens() without lang_id

  5. KeyError on a token id, inside NOT FIXABLE from outside NeMo:
         get_ctc_word_alignment
           -> ids_to_tokens(ids, "bn")
             -> MultilingualTokenizer.ids_to_tokens
               -> SentencePieceTokenizer.ids_to_tokens
                 -> KeyError: 270

     CTC emits ids in the GLOBAL aggregate vocabulary (0..5632, spanning
     22 languages). ids_to_tokens(ids, "bn") delegates to the Bengali
     SUB-tokenizer, whose local vocabulary does not contain global id 270.
     The two id spaces differ, and CTC-WS has no notion of that because it
     was written for models with one flat vocabulary.

     Fixing it means reverse-engineering the aggregate tokenizer's
     global<->local id mapping and rewriting NeMo's alignment code around
     it - in a fork that is re-cloned on every pod rebuild, so the patch
     would not survive.

WHAT TO DO INSTEAD
------------------
The same goal - pull garbled drug names toward real ones - is already met
at the TEXT level by correct.py, fuzzy_drugs.py and gate.py, which took
WER from 28.8% to 25.7% and recover "Montuculast" -> Montelukast. That
path needs no NeMo internals and is measurable with the same harness.

If decode-time biasing is revisited, the tractable routes are:
  * a flat-vocabulary Bengali CTC model, where CTC-WS works as shipped; or
  * pyctcdecode with `hotwords=`, which does its own beam search over the
    logprobs and never calls the model's tokenizer.

The code below is kept because it is correct up to bug 5, and because
ctc_logprobs() is independently useful for any logprob-level work.

WHY
---
The 25.7% WER on this audio is not spread evenly. It concentrates on
English drug names spoken in Bengali - exactly the prescription-critical
vocabulary. Real examples of what the ASR produced for *Norflox*:

    নলাক্স   /   নরফ্লক্স   /   ড লখেিচ্ছিি

Everything downstream (correction, fuzzy matching, the gate) is repair work
on text that is already damaged. This node attacks it earlier: it tells the
DECODER which words are plausible before it commits to a transcript.

The vocabulary already exists - the gazetteer's Bengali transliterations
are precisely the forms the ASR emits. No new dataset is needed, which is
why this is cheaper than any fine-tuning option.

HOW
---
NeMo's CTC-based Word Spotter (CTC-WS), which ships in AI4Bharat's fork:

  1. Hotwords are compiled into a prefix graph over ASR tokens.
  2. A Token Passing Algorithm scans the CTC logprobs for those token
     sequences, scoring them with `cb_weight` added.
  3. Spotted words are merged into the base transcription where they beat
     the greedy alignment.

Only BENGALI forms are used. The ASR emits Bengali script, so Latin brand
names could never match a token sequence, and loading all 174k of them
would bloat the graph without helping.

THE RISK, STATED PLAINLY
------------------------
Context biasing makes drug names *easier* to output. Pushed too hard, it
will insert a drug name that was never spoken - which is precisely the
"Naloxone" failure this project already had once, arriving by a new route.

`cb_weight` is the dial. Higher recovers more real names AND invents more
fake ones. It is therefore:

  * DISABLED BY DEFAULT. Opt in with --bias.
  * Required to be measured against human-corrected ground truth before
    being trusted. See tools/measure_biasing.py - if WER does not improve,
    leave it off.
  * Never a substitute for the gate. A biased-in drug name is still just a
    proposal; glossary.py still decides whether it may be called a drug.

An insertion is worse than a deletion here. A missing drug shows up as an
obviously incomplete prescription; an invented one looks correct.
"""
from __future__ import annotations

import logging
import re

log = logging.getLogger(__name__)

_BENGALI = re.compile(r"[ঀ-৿]")

# NeMo's default is 3.0. Deliberately lower here: this vocabulary is
# phonetically dense (many drug names differ by one syllable), so a strong
# boost makes confusable names MORE likely to be swapped for each other,
# not less. Raise only with measurements in hand.
DEFAULT_CB_WEIGHT = 2.0


def gazetteer_hotwords(include_labs: bool = True) -> list[str]:
    """Bengali surface forms worth biasing the decoder toward.

    Single tokens only. Multi-word entries ("সি বি সি") are split, because
    the word spotter matches word-level token sequences - a phrase would
    simply never be spotted as one unit.
    """
    # from .glossary import ALL_DRUGS, LAB_TESTS  # BLOCKED - not available

    words: set[str] = set()
    for drug in ALL_DRUGS:
        for form in drug.bengali:
            for part in form.split():
                if len(part) >= 3 and _BENGALI.search(part):
                    words.add(part)
    if include_labs:
        for _canon, alts in LAB_TESTS.items():
            for alt in alts:
                for part in alt.split():
                    # Single Bengali letters ("সি", "বি") are spelled-out
                    # acronym fragments. Biasing them would boost extremely
                    # common syllables all over the audio.
                    if len(part) >= 3 and _BENGALI.search(part):
                        words.add(part)
    return sorted(words)


class _BoundTokenizer:
    """Presents IndicConformer's multilingual tokenizer as a plain one.

    NeMo's context-biasing code was written for a standard SentencePiece
    tokenizer and calls `tokenizer.ids_to_tokens(ids)` internally. This
    model's MultilingualTokenizer requires a lang_id on every call, so the
    spotter dies inside NeMo with:
        TypeError: MultilingualTokenizer.ids_to_tokens() missing 1 required
                   positional argument: 'lang_id'

    Binding the language here is preferable to patching NeMo: the fork is a
    dependency we re-clone on every pod, so a local patch would silently
    disappear on the next rebuild.
    """

    def __init__(self, tokenizer, lang_id: str):
        self._tok = tokenizer
        self._lang = lang_id

    def ids_to_tokens(self, ids, lang_id=None):
        try:
            return self._tok.ids_to_tokens(ids, lang_id or self._lang)
        except TypeError:
            return self._tok.ids_to_tokens(ids)

    def text_to_ids(self, text, lang_id=None):
        try:
            return self._tok.text_to_ids(text, lang_id or self._lang)
        except TypeError:
            return self._tok.text_to_ids(text)

    def tokens_to_text(self, tokens, lang_id=None):
        try:
            return self._tok.tokens_to_text(tokens, lang_id or self._lang)
        except TypeError:
            return self._tok.tokens_to_text(tokens)

    def __getattr__(self, name):
        return getattr(self._tok, name)


class _ModelProxy:
    """The ASR model with its tokenizer language pre-bound."""

    def __init__(self, model, lang_id: str):
        object.__setattr__(self, "_model", model)
        object.__setattr__(self, "tokenizer", _BoundTokenizer(model.tokenizer, lang_id))

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_model"), name)


def ctc_logprobs(model, audio_path: str):
    """CTC logprobs [T, V] for one file.

    NOT via transcribe(logprobs=True). In AI4Bharat's fork that flag does
    not return logprobs at all - it returns a 2-tuple of hypothesis STRING
    lists, so feeding its output to the word spotter yields
    "IndexError: tuple index out of range" on logprobs.shape[0].

    The encoder -> ctc_decoder path is used instead, which is what the
    spotter actually needs. Verified shape on this model: [1, 93, 5633] for
    a short clip, i.e. blank_idx = V-1 = 5632.
    """
    import numpy as np
    import soundfile as sf
    import torch

    audio, sr = sf.read(audio_path)
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1)
    device = next(model.parameters()).device
    sig = torch.tensor(np.asarray(audio), dtype=torch.float32, device=device).unsqueeze(0)
    length = torch.tensor([sig.shape[1]], device=device)

    model.eval()
    with torch.no_grad():
        proc, proc_len = model.preprocessor(input_signal=sig, length=length)
        encoded, _ = model.encoder(audio_signal=proc, length=proc_len)
        decoder = getattr(model, "ctc_decoder", None) or model.decoder
        logprobs = decoder(encoder_output=encoded)
    return logprobs[0].cpu().numpy()


class ContextBiaser:
    """Wraps NeMo CTC-WS. Degrades to a no-op rather than failing.

    If the fork lacks context_biasing, or graph construction fails, this
    reports unavailable and the caller keeps the unbiased transcript. A
    consultation transcribed without biasing is the current, known-good
    behaviour; a crashed pipeline is not.
    """

    def __init__(self, asr_model, hotwords: list[str] | None = None,
                  cb_weight: float = DEFAULT_CB_WEIGHT, debug: bool = False):
        self.model = asr_model
        self.hotwords = hotwords if hotwords is not None else gazetteer_hotwords()
        self.cb_weight = cb_weight
        self.debug = debug
        self.failures = 0
        self.graph = None
        self._proxy = _ModelProxy(asr_model, self.LANG_ID)
        self.available = False
        self.error = ""
        self.blank_idx = 0

    # IndicConformer uses a multilingual AGGREGATE tokenizer, so
    # text_to_ids() takes a mandatory lang_id that a normal SentencePiece
    # tokenizer does not:
    #     TypeError: MultilingualTokenizer.text_to_ids() missing 1 required
    #                positional argument: 'lang_id'
    # This is the same aggregate-tokenizer quirk that stops mainline NeMo
    # loading this model at all.
    LANG_ID = "bn"

    def _encode(self, tok, word: str):
        """Token ids + token strings for one word, across tokenizer flavours."""
        try:
            ids = tok.text_to_ids(word, self.LANG_ID)
        except TypeError:
            ids = tok.text_to_ids(word)          # plain SentencePiece
        try:
            toks = tok.ids_to_tokens(ids, self.LANG_ID)
        except TypeError:
            toks = tok.ids_to_tokens(ids)
        return ids, toks

    def build(self) -> bool:
        try:
            from nemo.collections.asr.parts.context_biasing import (  # noqa: F401
                merge_alignment_with_ws_hyps,
            )
            from nemo.collections.asr.parts.context_biasing.context_graph_ctc import (
                ContextGraphCTC,
            )

            tok = self.model.tokenizer
            vocab_size = len(tok.vocab) if hasattr(tok, "vocab") else self.model.decoder.num_classes_with_blank - 1
            self.blank_idx = vocab_size

            # ContextGraphCTC wants, per word, a list of ALTERNATIVE
            # tokenisations: [(word, [[id, id, ...], ...]), ...]
            #
            # NOTE: the signature is annotated
            #     List[tuple[str, List[List[tuple[str, int]]]]]
            # which reads as (token_string, token_id) pairs. That annotation
            # is WRONG. add_to_graph keys the graph on the token object
            # directly (`prev_node.next[token] = node`), and the word
            # spotter then evaluates `logprobs[frame][int(transition_state)]`
            # - so a (str, int) pair produces:
            #     TypeError: int() argument must be ... not 'tuple'
            # Plain integer ids are what both sides actually require.
            word_items = []
            skipped = 0
            for word in self.hotwords:
                try:
                    ids, _toks = self._encode(tok, word)
                except Exception:
                    skipped += 1
                    continue
                if not ids or len(ids) > 20:
                    skipped += 1
                    continue
                word_items.append((word, [list(int(i) for i in ids)]))

            if not word_items:
                self.error = "no hotword tokenised successfully"
                return False

            self.graph = ContextGraphCTC(blank_id=self.blank_idx)
            self.graph.add_to_graph(word_items)
            self.available = True
            log.info("context biasing: %d hotwords (%d skipped), cb_weight=%.1f",
                     len(word_items), skipped, self.cb_weight)
        except Exception as exc:                       # noqa: BLE001
            self.error = f"{type(exc).__name__}: {exc}"
            self.available = False
            log.warning("context biasing unavailable, decoding unbiased: %s",
                        self.error)
        return self.available

    def apply(self, logprobs, base_text: str) -> tuple[str, list[str]]:
        """Return (possibly biased text, list of words the spotter injected).

        The injected list is returned rather than discarded so the caller can
        flag it for review. A word the decoder did not produce on its own,
        which appeared only because we boosted it, is exactly the kind of
        change a human should see.
        """
        if not self.available:
            return base_text, []
        try:
            from nemo.collections.asr.parts.context_biasing import (
                merge_alignment_with_ws_hyps,
            )
            from nemo.collections.asr.parts.context_biasing.ctc_based_word_spotter import (
                run_word_spotter,
            )

            hyps = run_word_spotter(
                logprobs=logprobs,
                context_graph=self.graph,
                asr_model=self._proxy,
                blank_idx=self.blank_idx,
                cb_weight=self.cb_weight,
            )
            if not hyps:
                return base_text, []

            biased = merge_alignment_with_ws_hyps(
                raw_text=base_text,
                asr_model=self._proxy,
                ws_hyps=hyps,
                decoder_type="ctc",
                blank_idx=self.blank_idx,
            )
            before = set(base_text.split())
            injected = [w for w in str(biased).split() if w not in before]
            return str(biased), injected
        except Exception as exc:                       # noqa: BLE001
            # Per-segment fallback: one failure must not lose the segment.
            # The type is logged as well as the message because NeMo raises
            # bare KeyErrors whose str() is just a token id - "263" alone
            # tells you nothing about where it came from.
            self.failures += 1
            if self.debug and self.failures <= 3:
                import traceback
                log.warning("biasing failed (%s: %s)\n%s",
                            type(exc).__name__, exc, traceback.format_exc())
            else:
                log.warning("biasing failed on a segment (%s: %s), keeping unbiased",
                            type(exc).__name__, exc)
            return base_text, []
