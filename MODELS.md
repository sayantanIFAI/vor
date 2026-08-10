# Model artifacts

Three models run in sequence. Losing any one of them degrades the output in a
different way, and only one of the three fails loudly.

```
Bengali audio
   │
   ├─ 1. IndicConformer  (ASR)          Bengali speech -> Bengali text
   ├─ 2. IndicTrans2     (translation)  Bengali text   -> English text
   └─ 3. Qwen2.5-7B      (extraction)   English text   -> structured Rx
                                            │
                                            └─ gate + gazetteer (code, not a model)
```

Qwen2.5 reads English clinical text far better than Bengali, which is why step 2
exists. See `voicerx/translate.py:16`.

---

## 1. IndicConformer — ASR

| | |
|---|---|
| Hugging Face id | `ai4bharat/indicconformer_stt_bn_hybrid_ctc_rnnt_large` |
| Gated | **Yes** — requires `HF_TOKEN` and an accepted licence |
| On-disk size | 503 MB |
| File | `indicconformer_stt_bn_hybrid_rnnt_large.nemo` |
| Cache path | `$HF_HOME/hub/models--ai4bharat--indicconformer_stt_bn_hybrid_ctc_rnnt_large/` |
| Located by | `voicerx/asr.py` via `/workspace/voicerx/model_path.txt`, then by glob |

**Mainline NeMo cannot load this checkpoint.** IndicConformer uses a multilingual
*aggregate* tokenizer; mainline's `_setup_monolingual_tokenizer` raises
`KeyError: 'dir'`. Only the AI4Bharat fork works:

```
git clone --depth 1 -b nemo-v2 https://github.com/AI4Bharat/NeMo.git
```

Pinned at `8dce88c` ("added shuffling") on branch `nemo-v2`.

**Failure mode:** loud. The server will not start.

### The numpy patch (mandatory)

numpy 2.0 removed `np.sctypes`; the fork's audio loader still calls it, so every
`transcribe()` dies inside `_convert_samples_to_float32`. Patch after cloning:

```bash
SEG=<NEMO>/nemo/collections/asr/parts/preprocessing/segment.py
sed -i "s/samples\.dtype in np\.sctypes\['int'\]/np.issubdtype(samples.dtype, np.signedinteger)/" "$SEG"
sed -i "s/samples\.dtype in np\.sctypes\['float'\]/np.issubdtype(samples.dtype, np.floating)/" "$SEG"
grep -q issubdtype "$SEG" || { echo "patch failed"; exit 1; }
```

**Failure mode if skipped:** imports fine, dies on the first audio file.

---

## 2. IndicTrans2 — translation

| | |
|---|---|
| Hugging Face id | `ai4bharat/indictrans2-indic-en-dist-200M` |
| Gated | **Yes** — requires `HF_TOKEN` |
| On-disk size | ~800 MB when complete |
| Cache path | `$HF_HOME/hub/models--ai4bharat--indictrans2-indic-en-dist-200M/` |
| Extra package | `IndicTransToolkit` (provides `IndicProcessor`) |
| Larger variant | `ai4bharat/indictrans2-indic-en-1B` (`LARGE_MODEL` in `translate.py`) |

**Failure mode: SILENT, and this is the dangerous one.** `translate.py:97` catches
any load error and passes Bengali through untranslated. No exception, no warning
in the response — Qwen simply receives Bengali it reads poorly, and extraction
quality drops. You will see raw Bengali tokens appear in `medications`.

Verify it is really present — a directory containing only `LICENSE` and
`README.md` is the signature of an interrupted or unauthorised download:

```bash
python3 -c "
from transformers import AutoModelForSeq2SeqLM
AutoModelForSeq2SeqLM.from_pretrained('ai4bharat/indictrans2-indic-en-dist-200M', trust_remote_code=True)
print('IT2 OK')"
```

---

## 3. Qwen2.5-7B — extraction

| | |
|---|---|
| Ollama tag | `qwen2.5:7b` |
| Gated | No |
| Size | 4.68 GB (4,683,087,332 bytes) |
| Digest | `845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e` |
| Format | gguf, qwen2 family, 7.6B params |
| Endpoint | `http://localhost:11434/api/generate` (`voicerx/extract.py:19`) |
| Ollama version | 0.32.7 |

Ollama needs **both** the binary and its runtime libraries. Copying only
`bin/ollama` produces `llama-server binary not found` and every generate call
returns HTTP 500. Keep the extracted tree and make sure `../lib/ollama` resolves
relative to the binary:

```bash
ln -sfn <dist>/lib /workspace/lib     # so /workspace/bin/ollama finds /workspace/lib/ollama
```

Release assets are `.tar.zst` (not `.tgz`) as of v0.32.7; the old
`ollama.com/download/ollama-linux-amd64.tgz` path returns 404.

**Failure mode:** loud in the log, but *looks* like a content bug from the API —
extraction returns empty with `segments_flagged == segments_total`. Check
`ollama.log` before suspecting the gazetteer.

---

## Storage summary

| Asset | Size | Must live on |
|---|---|---|
| Python packages | 8.1 GB | persistent volume |
| Qwen model | 4.4 GB | persistent volume |
| Ollama distribution | 2.2 GB | persistent volume |
| IndicConformer | 503 MB | persistent volume |
| IndicTrans2 | ~800 MB | persistent volume |

On RunPod, `/` `/root` `/usr/local` are rebuilt from the image on every restart;
only `/workspace` (the network volume) survives. Anything above installed to the
container disk is lost, which is what `boot.sh` exists to prevent.

Environment variables that decide where downloads land — set these *before*
anything downloads, or it goes to the ephemeral tier:

```bash
export HF_HOME=/workspace/.cache/huggingface   # HF models
export TORCH_HOME=/workspace/.cache/torch      # silero-vad
export OLLAMA_MODELS=/workspace/ollama/models  # the 4.4GB
export HF_HUB_ENABLE_HF_TRANSFER=0             # set in RunPod images, package absent
```
