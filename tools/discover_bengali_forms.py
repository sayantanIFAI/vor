"""Discover the Bengali spellings the ASR ACTUALLY produces for drug names.

THE PROBLEM
-----------
The gazetteer knows 1,509 generic molecules in Latin and 179,002 brand
names, but only ~145 have Bengali surface forms. That gap is what decides
recognition: the ASR emits Bengali script, so a Latin-only entry can never
match a spoken drug name.

Hand-writing the rest is the wrong answer - it was tried, one alias per
failed consultation, and each was found only after a real prescription had
already lost data.

WHY TTS RATHER THAN TRANSLITERATION
-----------------------------------
A transliterator produces textbook Bengali. What is needed is what THIS
ASR emits for THIS drug name, which is a different and messier thing -
"মেট ফর্মিন" split across two words, "ফ্লক্সাসিম" with an m, "সি ভিসিটা"
with a ভ for a ব. Those are properties of the acoustic model, not of
Bengali orthography, and no transliterator can predict them.

So: speak the name, transcribe it, and record the result. The output is
correct by construction, because it IS the model's own output.

This is the legitimate use of TTS here. It is useless for TRAINING the ASR
(synthetic audio does not match clinic acoustics) but ideal for probing
what the ASR does with a known input.

VARIATION MATTERS
-----------------
Each name is spoken with several speaker descriptions, because the ASR
transcribes the same word differently by voice and pace - and every
distinct output is a real surface form worth having. The phonetic fold
then absorbs the remainder.

SAFETY
------
Discovered forms are written to a SEPARATE generated file, never merged
into the curated table, and every one is checked against collisions()
before being kept. A discovered form that collides with an existing drug,
lab or clinical term is dropped and reported - a wrong Bengali form is
worse than a missing one, because it makes a real drug resolve to the
wrong molecule.

Usage (on the pod, needs GPU):
    python3 tools/discover_bengali_forms.py 300 voicerx/forms_discovered.py
"""
from __future__ import annotations

import collections
import sys

sys.path.insert(0, "/workspace/voicerx")
sys.path.insert(0, ".")

# Speaker descriptions for Indic Parler-TTS. Varied deliberately - a single
# voice yields a single spelling, and the point is to collect the spread.
VOICES = [
    "Aditi speaks in a clear, moderate-paced voice with minimal background noise.",
    "Sunita speaks quickly in a slightly noisy environment.",
    "A male speaker with a low-pitched voice speaks slowly and clearly.",
]

TTS_MODEL = "ai4bharat/indic-parler-tts"


def top_molecules(n: int) -> list[str]:
    """Most-prescribed molecules lacking a Bengali form.

    Brand count is the prevalence proxy: a molecule sold under 4,000 brand
    names is prescribed far more often than one sold under three. The top
    300 cover ~92% of all brand products, so this is where coverage is won.
    """
    from voicerx.brands_india import INDIA_BRANDS
    from voicerx.glossary import lookup_drug

    counts = collections.Counter(v.lower().strip() for v in INDIA_BRANDS.values())
    out = []
    for molecule, _cnt in counts.most_common():
        if len(molecule) < 5 or not molecule.replace(" ", "").isalpha():
            continue
        if lookup_drug(molecule):          # already has a Bengali form
            continue
        out.append(molecule)
        if len(out) >= n:
            break
    return out


def _write(dest, molecules, discovered, dropped) -> None:
    with open(dest, "w", encoding="utf-8") as out:
        out.write('"""AUTO-GENERATED - do not edit by hand.\n\n')
        out.write("Regenerate with:\n")
        out.write("    python3 tools/discover_bengali_forms.py 300 "
                  "voicerx/forms_discovered.py\n\n")
        out.write("Bengali spellings the ASR ACTUALLY produces for each drug\n")
        out.write("name, obtained by speaking the name with Indic Parler-TTS and\n")
        out.write("transcribing it with the same IndicConformer the pipeline uses.\n")
        out.write("Correct by construction: these are the model's own outputs, not\n")
        out.write("a transliterator's guess at how the name ought to be spelled.\n\n")
        out.write("MACHINE-DISCOVERED, NOT CLINICALLY REVIEWED.\n\n")
        out.write(f"molecules probed        : {len(molecules)}\n")
        out.write(f"molecules with forms    : {len(discovered)}\n")
        out.write(f"total surface forms     : {sum(len(v) for v in discovered.values())}\n")
        out.write(f"dropped for collisions  : {len(dropped)}\n\n")
        out.write("dropped (each would have made a real drug resolve wrongly):\n")
        for m, f, why in dropped[:40]:
            out.write(f"    {m} -> {f} : {why}\n")
        out.write('"""\n\n')
        out.write("DISCOVERED_FORMS: dict[str, tuple[str, ...]] = {\n")
        for m in sorted(discovered):
            forms = ", ".join(f'"{f}"' for f in sorted(discovered[m]))
            out.write(f'    "{m}": ({forms},),\n')
        out.write("}\n")


def main() -> None:
    import torch
    from transformers import AutoTokenizer
    from parler_tts import ParlerTTSForConditionalGeneration
    import soundfile as sf

    from voicerx.asr import ASRNode
    from voicerx.biasing import ctc_logprobs  # noqa: F401  (import check)
    from voicerx.glossary import (fold, is_clinical_term, is_lab_test,
                                   lookup_drug)

    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    dest = sys.argv[2] if len(sys.argv) > 2 else "voicerx/forms_discovered.py"

    molecules = top_molecules(limit)
    print(f"molecules to probe: {len(molecules)}", flush=True)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("loading TTS...", flush=True)
    tts = ParlerTTSForConditionalGeneration.from_pretrained(TTS_MODEL).to(device)
    desc_tok = AutoTokenizer.from_pretrained(TTS_MODEL)
    prompt_tok = AutoTokenizer.from_pretrained(tts.config.text_encoder._name_or_path)

    print("loading ASR...", flush=True)
    node = ASRNode()

    discovered: dict[str, set[str]] = {}
    dropped: list[tuple[str, str, str]] = []

    # Written after EVERY molecule, not at the end. Two earlier runs died
    # around molecule 25 and discarded everything, because the file was
    # only produced on clean exit. Probing is expensive; losing it to a
    # crash is inexcusable.
    def flush() -> None:
        _write(dest, molecules, discovered, dropped)

    for i, molecule in enumerate(molecules, 1):
        forms: set[str] = set()
        for voice in VOICES:
            try:
                d = desc_tok(voice, return_tensors="pt").to(device)
                p = prompt_tok(molecule, return_tensors="pt").to(device)
                with torch.no_grad():
                    audio = tts.generate(input_ids=d.input_ids,
                                          attention_mask=d.attention_mask,
                                          prompt_input_ids=p.input_ids,
                                          prompt_attention_mask=p.attention_mask)
                wav = audio.cpu().numpy().squeeze()
                clip = "/tmp/probe.wav"
                sf.write(clip, wav, tts.config.sampling_rate)
                ctc, rnnt = node._transcribe_clip(clip)
                for text in (rnnt, ctc):
                    t = (text or "").strip()
                    if t and len(t) >= 3:
                        forms.add(t)
            except Exception as exc:                    # noqa: BLE001
                # A CUDA error leaves the context unusable, and every later
                # call then dies WITHOUT a traceback - which is how two runs
                # vanished mid-transcription with no error recorded. Clear
                # the cache and carry on; if the context is truly gone the
                # next molecule fails loudly rather than silently.
                print(f"  [{i}] {molecule}: TTS/ASR failed: "
                      f"{type(exc).__name__}: {exc}", flush=True)
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

        keep = set()
        for f in forms:
            # Never let a discovered form shadow something already known.
            hit = lookup_drug(f)
            if hit and hit.generic.lower() != molecule:
                dropped.append((molecule, f, f"collides with drug {hit.generic}"))
                continue
            lab = is_lab_test(f)
            if lab:
                dropped.append((molecule, f, f"collides with lab {lab}"))
                continue
            term = is_clinical_term(f)
            if term:
                dropped.append((molecule, f, f"collides with term {term}"))
                continue
            if len(fold(f)) < 5:
                dropped.append((molecule, f, "too short to be safe"))
                continue
            keep.add(f)

        if keep:
            discovered[molecule] = keep
        flush()
        if i % 25 == 0:
            print(f"  ...{i}/{len(molecules)}  kept={len(discovered)}", flush=True)

    flush()
    print(f"DONE probed={len(molecules)} kept={len(discovered)} "
          f"forms={sum(len(v) for v in discovered.values())} dropped={len(dropped)}",
          flush=True)


if __name__ == "__main__":
    main()
