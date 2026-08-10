#!/usr/bin/env bash
# ============================================================================
# Build a self-contained bundle for machines with NO internet access.
#
# Run this ON A CONNECTED MACHINE with the same OS/Python/CPU architecture as
# the target. Wheels are platform-specific: a bundle built on Windows or on
# python3.11 will not install on a python3.12 Linux box.
#
#   bash bundle.sh /path/to/output
#
# Then move the directory to the target and:
#   WHEELS=/path/to/bundle/wheels bash setup_offline.sh --offline
# ============================================================================
set -euo pipefail

OUT="${1:?usage: bundle.sh <output-dir>}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$OUT/wheels" "$OUT/models"

echo "== 1/4  python wheels =="
# --no-deps: requirements-lock.txt is already a complete closure. Letting pip
# re-resolve here re-introduces the torch/CUDA pull that the lock exists to avoid.
python3 -m pip download --no-deps -r "$REPO/requirements-lock.txt" -d "$OUT/wheels"

echo "== 2/4  ollama =="
curl -fsSL -o "$OUT/wheels/ollama-linux-amd64.tar.zst" \
    https://github.com/ollama/ollama/releases/latest/download/ollama-linux-amd64.tar.zst

echo "== 3/4  NeMo fork =="
git clone --depth 1 -b nemo-v2 https://github.com/AI4Bharat/NeMo.git "$OUT/AI4Bharat_NeMo"
rm -rf "$OUT/AI4Bharat_NeMo/.git"

echo "== 4/4  models (gated - needs HF_TOKEN) =="
: "${HF_TOKEN:?export HF_TOKEN=hf_xxx - both AI4Bharat models are gated}"
HF_HOME="$OUT/models" python3 - <<'PY'
import os
from huggingface_hub import snapshot_download
tok = os.environ["HF_TOKEN"]
for rid in ("ai4bharat/indicconformer_stt_bn_hybrid_ctc_rnnt_large",
            "ai4bharat/indictrans2-indic-en-dist-200M"):
    print(" ", rid, "->", snapshot_download(rid, token=tok))
PY

# Qwen: pull it, then copy the blob store. `ollama pull` on the target needs
# internet; copying OLLAMA_MODELS does not.
echo "== qwen2.5:7b (4.7GB) =="
if command -v ollama >/dev/null; then
    OLLAMA_MODELS="$OUT/ollama-models" ollama pull qwen2.5:7b 2>&1 | cat
else
    echo "  SKIP: ollama not installed here. On the target, run:"
    echo "    OLLAMA_MODELS=<prefix>/ollama/models ollama pull qwen2.5:7b"
fi

cat > "$OUT/README-OFFLINE.txt" <<'EOF'
Offline install
===============
Copy this whole directory to the target machine, then:

  export PREFIX=/workspace            # or wherever persistent storage is
  mkdir -p $PREFIX
  cp -r AI4Bharat_NeMo    $PREFIX/
  cp -r models/hub        $PREFIX/.cache/huggingface/hub
  cp -r ollama-models/*   $PREFIX/ollama/models/
  WHEELS=$PWD/wheels bash setup_offline.sh --offline
  bash boot.sh

Verify before trusting it (see MODELS.md for what each failure looks like):

  curl -s localhost:8000/api/health         # cuda:true, model_loaded:true
  curl -s localhost:11434/api/tags          # qwen2.5:7b listed
  python3 -c "from transformers import AutoModelForSeq2SeqLM as M; \
      M.from_pretrained('ai4bharat/indictrans2-indic-en-dist-200M', \
      trust_remote_code=True); print('IT2 OK')"

The last one matters most: IndicTrans2 fails SILENTLY. Without it the pipeline
still answers, but Bengali reaches Qwen untranslated and extraction quality
drops. Raw Bengali tokens appearing in `medications` is the tell.
EOF

du -sh "$OUT"
echo "bundle ready: $OUT"
