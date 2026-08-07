# Voice-to-Rx

Bengali doctor–patient consultation audio → structured English prescription.
Runs fully offline: no API calls, no data leaves the machine.

```
audio ──▶ VAD ──▶ ASR (IndicConformer) ──▶ correction ──▶ SLM (Qwen2.5) ──▶ validator ──▶ JSON + UI
          split    CTC + RNNT dual         learned from   extraction        safety gate
          on       decode, cross-checked   real human     to structured     (never trusts
          silence                          corrections    fields            the model)
```

---

## ⚠️ Read this before you run anything

This produces **drafts, not prescriptions.** A clinician must confirm every
output before it reaches a patient. The system is built around that
assumption — it flags aggressively and refuses to guess drug names, because
an earlier, looser version once turned a garbled audio fragment into
**"Naloxone"**, a real and dangerous drug that was never said. The audio
actually contained *Norflox-TZ*.

Measured on 10 real consultations against 67 human-corrected segments:

| Metric | Value |
|---|---|
| Word Error Rate | **25.7%** (was 28.8% before the correction layer) |
| Segments auto-flagged for review | ~82% |
| Real errors caught by the validator | 87% |
| Extraction failures | 0 |

**25.7% WER means roughly one word in four is wrong.** That is normal for
zero-shot multilingual ASR on accented, code-switched clinical speech, and
it is why the human review step is not optional.

---

## Quick start (laptop, no GPU)

You can run everything except the ASR model on CPU. The ASR needs ~4GB VRAM
for comfort; it will run on CPU but slowly (~40× real time).

### 1. Prerequisites

- Python 3.10+
- `ffmpeg` on PATH
- [Ollama](https://ollama.com) installed and running
- A HuggingFace account with access to the (gated) ASR model

### 2. Get the ASR model

The model is **gated** — you must accept its licence once, in a browser:

👉 https://huggingface.co/ai4bharat/indicconformer_stt_bn_hybrid_ctc_rnnt_large

Click **"Agree and access repository"** (auto-approves instantly), then
create a read token at https://huggingface.co/settings/tokens

```bash
export HF_TOKEN=hf_your_token_here
```

### 3. Install

⚠️ **The install order below is load-bearing.** Each pin fixes a specific,
reproducible failure. See [Why these pins](#why-these-pins) before changing
any of them.

```bash
# AI4Bharat's NeMo fork - mainline NeMo CANNOT load this model
git clone --depth 1 -b nemo-v2 https://github.com/AI4Bharat/NeMo.git
cd NeMo
pip install --editable ".[asr]"        # .[asr] NOT .[all] - see notes
cd ..

pip install "pytorch-lightning==2.2.1"
pip install "datasets>=2.19,<3" --no-deps
pip install fastapi "uvicorn[standard]" python-multipart silero-vad soundfile

# numpy 2.0 removed np.sctypes, which the fork still uses
SEG=NeMo/nemo/collections/asr/parts/preprocessing/segment.py
sed -i "s/samples\.dtype in np\.sctypes\['int'\]/np.issubdtype(samples.dtype, np.signedinteger)/" $SEG
sed -i "s/samples\.dtype in np\.sctypes\['float'\]/np.issubdtype(samples.dtype, np.floating)/" $SEG

python -c "import nemo.collections.asr; print('OK')"
```

### 4. The language model

```bash
ollama pull qwen2.5:7b
ollama serve          # leave running
```

### 5. Run

```bash
export HF_HUB_ENABLE_HF_TRANSFER=0    # the env var is often set but the package isn't
uvicorn server:app --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000** — record or upload audio, get a structured
prescription plus a review queue.

Batch mode, no UI:

```bash
python run_pipeline.py consultation1.wav consultation2.wav
```

To enable the Bengali→English bridge (see `translate.py`), add `--translate`.
Run it both ways on the same audio and compare — that is what the flag is
for. If the model is unavailable the run says so loudly and continues in
Bengali, rather than silently scoring an untested configuration:

```bash
python run_pipeline.py --translate consultation1.wav
```

---

## How it works

### `voicerx/vad.py` — split on silence
Silero VAD, segments capped at 25s. Not cosmetic: feeding a whole
unsegmented consultation to the SLM measurably increased hallucination, and
the ASR's RNNT decoder silently drops content on long audio.

### `voicerx/asr.py` — dual decoder
Runs **both** CTC and RNNT on every segment, prefers RNNT, falls back to CTC
when RNNT returns empty (~3% of segments, all very short clips). Their
disagreement is recorded and used as an independent review signal — two
decoders sharing an encoder but disagreeing on the words is information
neither one's own confidence gives you.

**RNNT must use `strategy="greedy"`, not the library default
`"greedy_batch"`.** The batched path mishandles the `language_ids` argument
this multilingual model requires and silently returns empty strings.

### `voicerx/correct.py` — learned corrections
Built from 67 real human-corrected segments. The key finding: **the ASR's
errors are not random.** They cluster on English medical loanwords written
in Bengali script — exactly the prescription-critical vocabulary.

| ASR produces | Actually | |
|---|---|---|
| ক্লাব | ক্ল্যাভাম | Clavam |
| আস্কোরিল | এস্কোরিল | Ascoril |
| ইনফেকটআন (3×) | ইনফেকশন | infection |
| প্রেস্কৃপ্তিন (2×) | প্রেসক্রিপশন | prescription |

Two tiers: repeated confusions are applied silently; single-observation ones
are applied but flagged for confirmation.

### `voicerx/fuzzy_drugs.py` — propose, never apply
Exact-match tables can't keep up: the same drug gets mangled differently
every time (`নলাক্স`, `নরফ্লক্স`, `ড লখেিচ্ছিি` were all *Norflox*). This
does similarity matching against Bengali drug transliterations and
**proposes** candidates for human confirmation. It never rewrites a drug
name automatically — that behaviour is what produced "Naloxone".

### `voicerx/glossary.py` — the clinical gazetteer, matched phonetically
69 drugs / 25 lab tests / 36 clinical terms, organised by department.

Matching is **not** string equality. Both the gazetteer and the ASR text go
through the same lossy phonetic fold, because real audio breaks exact
matching four ways at once and the variants multiply combinatorially:

| | example | why |
|---|---|---|
| spacing | `সি বি সি` vs `সিবিসি` | clinicians spell acronyms out, the ASR writes each letter separately |
| half-letters | `মন্টিকুলাষ্ট` vs `মনটিকুলাসট` | conjuncts (যুক্তাক্ষর) carry a hasant the ASR drops or adds |
| dialect | `শ` `ষ` `স` | one sound in spoken Bengali; also `ণ/ন`, `ড়/র`, `য/জ` |
| accent | `ব` vs `ভ` | aspiration is the least stable feature across speakers |

The spacing case alone caused **zero** lab tests to match across 255 real
segments. After folding, CBC is recovered even from the ASR's mangled
`সি ভিসিটা`.

The fold is deliberately lossy, so `collisions()` checks every entry against
every other at import — if two distinct entries ever fold together, that is
a bug, not a tuning knob.

Matching is on token **n-grams**, never substrings: substring matching let
keys straddle word boundaries, and the EEG key matched inside `এই জিভটা বার
কর` ("stick your tongue out"). Negation and interrogative scope suppress
`নতুন কোনো টেস্ট দিচ্ছি না` ("I am **not** giving a new test") while keeping
the conditional order `যদি ... তাহলে রক্ত পরীক্ষা`.

### `voicerx/gate.py` — the SLM proposes, the gazetteer decides
"Is this a drug?" is a closed-set question, so it is looked up, not reasoned
about. Three outcomes, and the middle one is the point:

- **verified** — exact gazetteer hit after folding
- **probable** — close to a real drug (`Montuculast` → Montelukast, 0.82).
  Kept, with the canonical name attached as a *proposal* in a separate
  field. Never written over what was actually said.
- **rejected** — moved to `rejected_terms`, **recorded rather than dropped**,
  because a silent deletion looks identical to a term that was never
  extracted, and if the gate is ever wrong that list is the only evidence.

A strict allowlist was tried first and was too harsh: it rejected 13 of 18
false positives but also discarded two genuine Montelukast prescriptions.
Hence three tiers, not a filter.

`SIMILARITY_FLOOR` is derived, not guessed — scored on real audio, the
classes separate cleanly (real drugs 0.818/0.700, worst false positive
0.588) and the threshold sits in the empty band.

Result on all 188 real extractions: 22 proposed → 7 verified, 2 probable,
13 rejected.

### `voicerx/translate.py` — bn→en bridge (optional, `--translate`)
Qwen2.5-7B is weak at Bengali clinical text, and fine-tuning does not fix
that: teaching a model a language is a continued-pretraining problem, not a
LoRA over a few thousand consultations — and training it on 25.7%-WER
transcripts mostly teaches confident guessing.

So the SLM stops doing Bengali. IndicTrans2 (offline, 200M distilled)
carries the **narrative**; drug and lab names never pass through translation
at all, because the gazetteer reads the original Bengali and MT is least
reliable exactly on transliterated brand names. The Bengali is always kept
alongside so a reviewer can *check* the translation rather than trust it.

**Off by default.** The Bengali-only prompt is the path that was actually
verified, so the bridge is opt-in and meant to be A/B'd on the same audio.

### `voicerx/validate.py` — the safety gate
Independent of what the SLM claims. Re-derives review flags from hard rules:
gate verdict, decoder disagreement, uncertain terms, missing dosage,
extraction failure. **The SLM's own confidence is not trusted** — it was
observed correctly flagging one garbled term while confidently mis-resolving
another in the same response.

> The previous check here, `drugs.is_known_drug()`, was worse than useless:
> it tested substring containment in *both* directions, so `'a'`, `'in'`,
> `'or'` and `'Nala'` all came back as known drugs. That is how garbage
> reached `medications[]` wearing a verified flag. It has been deleted.

---

## Why these pins

Every one of these cost real debugging time. Changing them will break the build.

| Pin | Without it |
|---|---|
| AI4Bharat NeMo fork (not mainline) | `KeyError: 'dir'` — mainline can't read this model's multilingual aggregate tokenizer |
| `.[asr]` not `.[all]` | `.[all]` pulls tensorstore → a full Bazel/protobuf C++ compile, 10+ minutes, for packages never used |
| `pytorch-lightning==2.2.1` | Upstream says `>=2.2.1` with no ceiling → pip takes 2.6.5 → `NeptuneLogger` was removed → ImportError |
| `datasets>=2.19 --no-deps` | 2.14.x calls `pa.PyExtensionType`, removed in modern pyarrow. `--no-deps` stops it dragging `huggingface_hub` off the required `0.23.2` |
| numpy `sctypes` patch | `np.sctypes` removed in numpy 2.0; every `transcribe()` dies in the audio loader |
| `HF_HUB_ENABLE_HF_TRANSFER=0` | The env var ships set in many images but `hf_transfer` isn't installed → every download fails |

---

## Known limitations

**Honest list. None of these are hidden in the output.**

1. **25.7% WER.** Roughly one word in four is wrong. Human review is required, not advisory.
2. **~13% of real errors slip past the validator.** These are the cases where the ASR produces *plausible but wrong* Bengali — no garbling to detect, so no heuristic fires. Both decoders sometimes agree on the same wrong words.
3. **Non-medications can appear in the medications table.** The SLM sometimes classifies dietary advice ("ORS", "light food") as a drug. The validator marks them *unverified*, but they shouldn't be there at all. Fixing this properly needs a curated drug gazetteer, not prompt tuning.
4. **RNNT drops content on audio longer than ~35s.** Worked around by VAD segmentation; the underlying bug in the fork is unfixed.
5. **Patient name extraction is deliberately conservative** — only populated when a name is explicitly spoken, never inferred.

---

## Deploying on a GPU pod

`setup_pod.sh` builds the whole environment in ~8 minutes on a fresh
RunPod/Vast pod.

```bash
export HF_TOKEN=hf_xxx
bash setup_pod.sh 2>&1 | tee setup.log
```

**Persistence gotcha, learned the hard way:** put your code and models on a
network volume (`/workspace`) — but note that `pip install` writes to the
container's *ephemeral* disk. A pod restart wipes the Python environment
even though your data survives. After any restart, re-run `setup_pod.sh`.

To reach the UI from a browser, expose HTTP port **8000** in the pod config.
RunPod's SSH proxy does **not** forward TCP, so `ssh -L` tunnelling will not
work, and neither will `scp`.

---

## Repository layout

```
voicerx/
  vad.py           Silero VAD segmentation
  asr.py           IndicConformer, dual CTC+RNNT decode
  correct.py       corrections learned from human data
  fuzzy_drugs.py   drug-name candidates (proposed, never applied)
  extract.py       Qwen2.5 prompt + retry/validation
  schema.py        strict Pydantic output contract
  validate.py      independent safety gate
  drugs.py         known-drug list
  pipeline.py      orchestrator
server.py          FastAPI + REST API
run_pipeline.py    batch CLI
ui/dist/           React UI (single file, no build step)
setup_pod.sh       GPU pod provisioning
```

## Licence / data

Code: choose your own licence before publishing.
**No patient audio or derived transcripts are included in this repository,
and `.gitignore` is configured to keep it that way.** Keep it that way.
