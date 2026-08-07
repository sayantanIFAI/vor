# Voice-to-Rx — Local Laptop Guide

**Audience:** a developer who has never seen this repo. Follow top to bottom.
Every command is copy-pasteable. Every error we actually hit is in
[Troubleshooting](#7-troubleshooting), with the fix.

**What you will have at the end:** a browser page on your own laptop where
you press **Record**, speak a Bengali consultation, press **Stop**, and get a
structured English prescription — symptoms, diagnosis, medications, lab
tests, follow-up — on screen.

---

## 1. Read this first (it will save you a day)

Three things about this system are unusual. Skipping them is how people
waste hours.

**It is a safety-gated pipeline, not a chatbot.** The language model
*proposes* a prescription; a curated gazetteer *decides* what is allowed
through. If a real drug is missing from the gazetteer it gets **rejected**
and shown in a "Rejected as medications" panel. That is intended. A missing
drug that a human can see beats a wrong drug that nobody notices.

**Mainline NeMo cannot load the ASR model.** IndicConformer uses a
multilingual *aggregate* tokenizer. You must use AI4Bharat's fork. Every
"just pip install nemo_toolkit" attempt fails with
`KeyError: 'dir'`.

**The version pins are load-bearing.** Each one in
[section 3](#3-install) cost real debugging time. Do not "clean them up".

---

## 2. Hardware and prerequisites

### Will it run on your laptop?

| Setup | ASR | LLM | Verdict |
|---|---|---|---|
| NVIDIA GPU, 8 GB+ VRAM | GPU | `qwen2.5:7b` | Full quality. Recommended. |
| Apple Silicon, 16 GB+ | CPU | `qwen2.5:7b` | Works. LLM is the slow part. |
| CPU only, 16 GB RAM | CPU | `qwen2.5:3b` | Works, reduced extraction quality. |
| CPU only, 8 GB RAM | — | — | Not enough. Use the GPU-pod path instead. |

Disk: **~12 GB** (IndicConformer 0.5 GB, Qwen 4.7 GB, Python deps ~5 GB,
brand register 6 MB).

### Prerequisites

- **Python 3.10–3.12.** 3.13 is not supported by some pinned deps.
- **ffmpeg** on `PATH`.
- **git**.
- A **Hugging Face account** — the ASR model is *gated* and you must accept
  its licence in a browser before any download will work.

```bash
python --version
ffmpeg -version
```

---

## 3. Install

Run these **in order**. The order matters — see the comments.

### 3.1 Get the code

```bash
git clone <your-repo-url> voice-to-rx && cd voice-to-rx
```

```bash
python -m venv .venv
```

Activate it — Windows (Git Bash): `source .venv/Scripts/activate`,
macOS/Linux: `source .venv/bin/activate`

### 3.2 Accept the model licence (do this before anything downloads)

1. Sign in at huggingface.co
2. Open `ai4bharat/indicconformer_stt_bn_hybrid_ctc_rnnt_large`
3. Click **Agree and access repository**
4. Create a token: Settings → Access Tokens → *read* scope

```bash
export HF_TOKEN=hf_your_token_here
```

> **Skipping this gives a 401 during download that does not mention
> licensing.** It is the single most common first failure.

### 3.3 Disable hf_transfer

```bash
export HF_HUB_ENABLE_HF_TRANSFER=0
```

> Some environments set this to `1` but do not ship the `hf_transfer`
> package, so every download dies with an unhelpful `ValueError`.

### 3.4 Install the ASR stack

```bash
git clone -b nemo-v2 https://github.com/AI4Bharat/NeMo.git AI4Bharat_NeMo
```

```bash
cd AI4Bharat_NeMo && pip install --editable ".[asr]" && cd ..
```

> Use `.[asr]`, **not** their `reinstall.sh`. That script defaults to
> `.[all]`, which pulls multimodal extras → tensorstore → a full Bazel C++
> compile that runs 10+ minutes and often never finishes.

Now the pins:

```bash
pip install "pytorch-lightning==2.2.1"
```

> Their requirements say `>=2.2.1` with no upper bound. pip resolves to
> 2.6.x, where `NeptuneLogger` was removed → `ImportError` on
> `import nemo.collections.asr`.

```bash
pip install "datasets>=2.19,<3" --no-deps
```

> NeMo pins `huggingface_hub==0.23.2` exactly. `datasets` is unpinned and
> resolves to 2.14.4, which calls `pa.PyExtensionType` — removed in modern
> pyarrow. `--no-deps` is required so this does **not** drag
> `huggingface_hub` off 0.23.2 and break `ModelFilter`.

### 3.5 Patch numpy 2.0 incompatibility

`np.sctypes` was removed in numpy 2.0 but AI4Bharat's audio loader still
uses it, so **every** `transcribe()` call dies inside
`_convert_samples_to_float32`.

```bash
SEG=AI4Bharat_NeMo/nemo/collections/asr/parts/preprocessing/segment.py
sed -i "s/samples\.dtype in np\.sctypes\['int'\]/np.issubdtype(samples.dtype, np.signedinteger)/" "$SEG"
sed -i "s/samples\.dtype in np\.sctypes\['float'\]/np.issubdtype(samples.dtype, np.floating)/" "$SEG"
grep -q issubdtype "$SEG" && echo "patch OK" || echo "PATCH FAILED - stop here"
```

### 3.6 Remaining dependencies

```bash
pip install silero-vad soundfile fastapi "uvicorn[standard]" python-multipart pydantic
```

> `python-multipart` is **not** a FastAPI dependency but *is* required for
> file uploads. Without it the server starts fine and then fails only when
> the first upload arrives — a confusing failure we hit in production.

### 3.7 Install the language model

Install Ollama from [ollama.com](https://ollama.com), then:

```bash
ollama pull qwen2.5:7b
```

CPU-only laptop? Use `ollama pull qwen2.5:3b` and set
`OLLAMA_MODEL = "qwen2.5:3b"` in `voicerx/extract.py`.

### 3.8 Download the ASR model

```bash
python -c "
import os
from huggingface_hub import snapshot_download
p = snapshot_download('ai4bharat/indicconformer_stt_bn_hybrid_ctc_rnnt_large', token=os.environ['HF_TOKEN'])
open('model_path.txt','w').write(p)
print('model at', p)
"
```

### 3.9 Build the drug register (optional but strongly recommended)

Without this the gate knows ~69 drugs. With it, ~174,000.

Download *Extensive A-Z Medicines Dataset of India* (Kaggle) and
*MedER Bengali/English*, put the CSVs somewhere, fix the paths at the top of
each tool, then:

```bash
python tools/import_india_brands.py voicerx/brands_india.py
```

```bash
python tools/import_meder.py voicerx/terms_imported.py
```

Both are safe to skip — the code falls back to the curated tables.

---

## 4. Run

```bash
python -m uvicorn server:app --host 127.0.0.1 --port 8000
```

Wait for `Application startup complete`, then open **http://127.0.0.1:8000**

Model loading takes **30–60 s on first start**. This is normal.

---

## 5. Test

Do these in order. Stop at the first failure and go to
[Troubleshooting](#7-troubleshooting).

**5.1 — Health**

```bash
curl -s http://127.0.0.1:8000/api/health
```

Expect `{"status":"ok",...,"model_loaded":true}`.

**5.2 — The gate (no audio needed)**

```bash
python -c "
from voicerx.gate import judge_medication as j
for t in ['Nitrocontin','Montuculast','Antibiotic','hair loss']:
    v=j(t); print(f'{t:14} {v.tier:9} {v.canonical}')
"
```

Expected — and each line is a real regression guard:

```
Nitrocontin    verified  Nitroglycerin     brand resolves
Montuculast    probable  Montelukast       garbled name recovered, NOT auto-applied
Antibiotic     rejected                    a drug class is not a drug
hair loss      rejected                    a symptom is not a drug
```

**5.3 — Bengali phonetic matching**

```bash
python -c "
from voicerx.glossary import scan_labs, is_clinical_term
print(scan_labs('আপনি এই সি বি সি টা করবেন'))
print(is_clinical_term('প্রেশারটা'))
"
```

Expect `['CBC']` and `blood pressure`. The first proves spelled-out
acronyms match; the second proves Bengali suffixes are handled.

**5.4 — End to end**

Open the UI, click **Record**, speak Bengali for ~20 s, click **Stop**.
You should get a prescription card. Then verify the safety behaviour: say a
drug name that is *not* in the gazetteer and confirm it appears under
**Rejected as medications** rather than in the medication table.

---

## 6. Hitting the 3-second target

### The naive design cannot do it

Measured on an RTX 4090, a 32-segment consultation took **61.8 s** end to
end. Almost all of it is the LLM: ~2 s per segment × 32. On a laptop it is
worse. No amount of tuning makes 32 sequential LLM calls finish in 3 s.

### The fix is architectural, not an optimisation

**Do the work during the consultation, not after it.** A consultation takes
minutes; the click happens at the end. So:

```
  ── during recording ───────────────────────►  ── on click ──►
  VAD closes a segment
      → ASR that segment          (~0.3 s)
      → gazetteer scan            (0.76 ms)
      → LLM extract that segment  (~2 s)          merge segments
      …repeat, overlapped with the doctor           validate
        still talking                               render
                                                  ≈ 50 ms
```

Every segment is already extracted by the time the doctor stops speaking.
The click only merges, validates and renders.

### Measured component costs

| Step | Cost | Notes |
|---|---|---|
| `judge_medication` | **0.007 ms** | dict lookup over 174k entries |
| `scan_labs` per segment | **0.76 ms** | n-gram scan |
| Gazetteer import | 1.3 s | one-time at startup |
| ASR per segment | ~0.3 s GPU | |
| **LLM per segment** | **~2 s** | the only real cost |
| Merge + validate + render | ~50 ms | this is the click path |

The gate and glossary are **free** at these timescales. Do not optimise
them; optimise the LLM call count.

### Rules to hold the budget

1. **Never batch LLM calls to the end.** That is the 61 s design.
2. **Cap concurrency at 1–2** on a laptop. Parallel LLM calls thrash memory
   and get slower, not faster.
3. **Load models once at startup**, never per request. `server.py` already
   does this — keep it that way.
4. **Keep the model warm.** First call after idle pays a reload. Ping
   `/api/health` periodically.
5. **If still slow, shrink the model before shrinking the pipeline.**
   `qwen2.5:3b` roughly halves latency. Losing an extraction node loses a
   safety property; losing model size only loses some fluency, because the
   gazetteer — not the model — decides what counts as a drug.

### Measured, not predicted

Streaming capture is implemented (`/api/session/*`) and measured on a real
2-minute consultation on an RTX 4090:

| | |
|---|---|
| audio | 120.4 s |
| compute during recording | 59.4 s — **0.49x real time** |
| segments processed live | 31 |
| segments held back by the tail guard | 1 (drained at finalize) |
| **click to result** | **2.11 s** |

The 0.49x ratio is what makes it work: processing consumes audio about
twice as fast as it arrives, so it cannot fall behind. (An earlier estimate
of 0.25x was optimistic by 2x - the margin is real but smaller than
predicted.)

> **Caveat:** this was measured by posting one 120 s chunk, not by a
> browser posting 10 s chunks. Real chunks are smaller and therefore
> easier; end-to-end browser capture has not been timed.

---

## 7. Troubleshooting

Every one of these was hit for real.

| Symptom | Cause | Fix |
|---|---|---|
| `KeyError: 'dir'` loading the model | Mainline NeMo | Use the AI4Bharat `nemo-v2` fork (§3.4) |
| `401` on model download | Gated licence not accepted | §3.2 — accept in a browser first |
| `ValueError` about hf_transfer | Env var set, package absent | `export HF_HUB_ENABLE_HF_TRANSFER=0` |
| `ImportError: NeptuneLogger` | pytorch-lightning too new | `pip install "pytorch-lightning==2.2.1"` |
| `AttributeError: pa.PyExtensionType` | `datasets` too old | `pip install "datasets>=2.19,<3" --no-deps` |
| `AttributeError: np.sctypes` | numpy 2.0 | Apply the patch in §3.5 |
| RNNT returns empty text | Broken default `greedy_batch` | `asr.py` sets `strategy="greedy"` — don't change it |
| Upload fails, server logs nothing useful | `python-multipart` missing | `pip install python-multipart` |
| `Connection refused` on port 11434 | Ollama not running | Start Ollama, then `ollama list` |
| Install takes 10+ min on tensorstore | You used `reinstall.sh` (`.[all]`) | Kill it; use `.[asr]` |
| Everything loads, transcription is garbage | Wrong sample rate | Model expects **16 kHz mono**; ffmpeg must resample |
| A real drug shows as "Rejected" | Not in the gazetteer | Expected. Add it to `glossary.py` — see §8 |

---

## 8. Adding a drug the gate rejected

This will happen and it is the normal way to extend the system.

Add an entry to the right department list in `voicerx/glossary.py`:

```python
Drug("Ivabradine", ("Ivabrad", "Inapure"),
     ("আইভাব্রাডিন",),                 # how the ASR actually writes it
     "heart rate control", "cardiac"),
```

The `bengali` field matters most — it decides whether a *spoken* name is
recognised at all. Then re-run test 5.2 and check for collisions:

```bash
python -c "from voicerx.glossary import collisions; print(collisions() or 'CLEAN')"
```

**If `collisions()` is not empty, stop.** Two entries have folded onto the
same key, meaning one will shadow the other. Make the entry more specific
rather than forcing it through.

---

## 9. Before any clinical use

Not optional, and not things code can settle.

- [ ] **`glossary.py` reviewed by a pharmacist or clinician.** It is marked
      `PENDING CLINICAL REVIEW`. It was drafted by an AI, and the Bengali
      transliterations — which decide what gets recognised — have not been
      verified by anyone qualified.
- [ ] **The 174k brand register is machine-imported, unreviewed.** It comes
      from a public dataset, not a validated formulary.
- [ ] **Measure accuracy on held-out audio you corrected yourself.** Current
      figures come from 255 segments, far too few to claim a percentage.
- [ ] **Every prescription is reviewed by the clinician before use.** The
      system is a drafting aid. The `probable` tier and the rejected-terms
      panel exist because the model is expected to be wrong sometimes.
- [ ] **Patient audio never leaves the machine.** Voice is biometric. The
      `.gitignore` blocks `*.wav` and result JSON — keep it that way.
