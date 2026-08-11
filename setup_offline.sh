#!/usr/bin/env bash
# ============================================================================
# Voice-to-Rx : one-shot environment setup
#
#   ONLINE  : bash setup_offline.sh
#   OFFLINE : bash setup_offline.sh --offline    (needs a bundle, see bundle.sh)
#
# Every step below exists because it failed at least once. The comments say
# which failure, so nobody re-derives it.
#
# Requires: python3.12, an NVIDIA GPU, ~18GB free, and HF_TOKEN for the two
# gated AI4Bharat models.
# ============================================================================
set -euo pipefail

PREFIX="${PREFIX:-/workspace}"          # must be persistent storage
PYLIBS="$PREFIX/pylibs"
NEMO_DIR="$PREFIX/AI4Bharat_NeMo"
WHEELS="${WHEELS:-$PREFIX/wheels}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OFFLINE=0
[ "${1:-}" = "--offline" ] && OFFLINE=1

# ---------------------------------------------------------------------------
# 0. Where things go
#
# On RunPod (and most container hosts) / /root /usr/local are rebuilt from the
# image on restart. Only the mounted volume survives. Installing to the wrong
# tier is what turns a restart into a two-hour rebuild.
# ---------------------------------------------------------------------------
export PYTHONPATH="$PYLIBS:$NEMO_DIR"
export HF_HOME="$PREFIX/.cache/huggingface"
export TORCH_HOME="$PREFIX/.cache/torch"
export OLLAMA_MODELS="$PREFIX/ollama/models"
export HF_HUB_ENABLE_HF_TRANSFER=0    # var is set in RunPod images, package is not
export PATH="$PREFIX/bin:$PATH"

# TASK 4: Hardening step for truly offline clinics
# HuggingFace libraries attempt a revision check on model load, which hangs or
# errors on isolated machines. These flags disable network checks entirely.
# Safe to set always; with network, they're just ignored.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
mkdir -p "$PYLIBS" "$HF_HOME" "$TORCH_HOME" "$OLLAMA_MODELS" "$PREFIX/bin"

command -v nvidia-smi >/dev/null || { echo "FATAL: no GPU visible"; exit 1; }
command -v ffmpeg >/dev/null || apt-get update -qq && apt-get install -y -qq ffmpeg zstd

PIP_ARGS="--no-cache-dir --target $PYLIBS"
[ "$OFFLINE" = 1 ] && PIP_ARGS="$PIP_ARGS --no-index --find-links $WHEELS"

# ---------------------------------------------------------------------------
# 1. Keep pip away from torch
#
# --target makes pip blind to the container's site-packages, so any package
# declaring `torch` triggers a fresh ~10GB torch + CUDA download. If that torch
# differs from the container's, it shadows it and desynchronises the ABI with
# torchvision:  "RuntimeError: operator torchvision::nms does not exist".
# The constraints file forces the version already present.
# ---------------------------------------------------------------------------
TORCH_V=$(python3 -c "import torch;print(torch.__version__.split('+')[0])" 2>/dev/null || echo "2.8.0")
cat > "$PREFIX/constraints.txt" <<C
torch==$TORCH_V
numpy==2.4.6
huggingface_hub==0.23.2
pytorch-lightning==2.2.1
tokenizers==0.19.1
transformers==4.44.2
C

# ---------------------------------------------------------------------------
# 2. NeMo - the AI4Bharat fork, not mainline
#
# IndicConformer uses a multilingual AGGREGATE tokenizer. Mainline NeMo's
# _setup_monolingual_tokenizer raises KeyError: 'dir'. This is not optional.
# ---------------------------------------------------------------------------
if [ ! -d "$NEMO_DIR" ]; then
    [ "$OFFLINE" = 1 ] && { echo "FATAL: offline mode needs $NEMO_DIR pre-staged"; exit 1; }
    git clone --depth 1 -b nemo-v2 https://github.com/AI4Bharat/NeMo.git "$NEMO_DIR"
fi

# numpy 2.0 removed np.sctypes; the fork's loader still uses it, so every
# transcribe() dies in _convert_samples_to_float32. Imports fine without this -
# fails only when audio arrives.
SEG="$NEMO_DIR/nemo/collections/asr/parts/preprocessing/segment.py"
sed -i "s/samples\.dtype in np\.sctypes\['int'\]/np.issubdtype(samples.dtype, np.signedinteger)/" "$SEG"
sed -i "s/samples\.dtype in np\.sctypes\['float'\]/np.issubdtype(samples.dtype, np.floating)/" "$SEG"
grep -q issubdtype "$SEG" || { echo "FATAL: numpy patch failed"; exit 1; }
echo "  numpy 2.0 patch applied"

# ---------------------------------------------------------------------------
# 3. Python packages
#
# Installed from requirements-lock.txt - the freeze of an environment that is
# known to work end to end. Resolving from scratch re-runs a dependency hunt
# whose answer is already known.
#
# NO --upgrade: with it, pip re-fetches satisfied packages and drags torch back.
# ---------------------------------------------------------------------------
python3 -m pip install $PIP_ARGS -c "$PREFIX/constraints.txt" -r "$REPO/requirements-lock.txt"

# these must not be shadowed by a --target copy: the container's are correct
rm -rf "$PYLIBS/torch" "$PYLIBS/torchvision" "$PYLIBS/torchaudio" \
       "$PYLIBS/nvidia" "$PYLIBS/cuda" "$PYLIBS/triton" "$PYLIBS/sympy" \
       "$PYLIBS"/sympy-*.dist-info

python3 -c "import nemo.collections.asr" && echo "  NEMO IMPORT OK"

# ---------------------------------------------------------------------------
# 4. Models  (see MODELS.md)
# ---------------------------------------------------------------------------
if [ "$OFFLINE" = 0 ]; then
    : "${HF_TOKEN:?Both AI4Bharat models are gated. export HF_TOKEN=hf_xxx}"
    python3 - <<PY
import os
from huggingface_hub import snapshot_download
tok = os.environ["HF_TOKEN"]
p = snapshot_download("ai4bharat/indicconformer_stt_bn_hybrid_ctc_rnnt_large", token=tok)
os.makedirs("$PREFIX/voicerx", exist_ok=True)
open("$PREFIX/voicerx/model_path.txt", "w").write(p)
print("  ASR   :", p)
print("  TRANS :", snapshot_download("ai4bharat/indictrans2-indic-en-dist-200M", token=tok))
PY
fi

# ---------------------------------------------------------------------------
# 5. Ollama + Qwen
#
# v0.32.7 ships .tar.zst; ollama.com/download/*.tgz is a 404. The binary alone
# is not enough - it needs ../lib/ollama/llama-server or every generate call
# returns HTTP 500.
# ---------------------------------------------------------------------------
if [ ! -x "$PREFIX/bin/ollama" ]; then
    if [ "$OFFLINE" = 1 ]; then
        [ -f "$WHEELS/ollama-linux-amd64.tar.zst" ] || { echo "FATAL: stage ollama tarball"; exit 1; }
        tar --zstd -xf "$WHEELS/ollama-linux-amd64.tar.zst" -C "$PREFIX/ollama-dist"
    else
        mkdir -p "$PREFIX/ollama-dist"
        curl -fsSL -o /tmp/ollama.tar.zst \
          https://github.com/ollama/ollama/releases/latest/download/ollama-linux-amd64.tar.zst
        tar --zstd -xf /tmp/ollama.tar.zst -C "$PREFIX/ollama-dist"
    fi
    cp "$(find "$PREFIX/ollama-dist" -type f -name ollama | head -1)" "$PREFIX/bin/ollama"
    chmod +x "$PREFIX/bin/ollama"
fi
ln -sfn "$PREFIX/ollama-dist/lib" "$PREFIX/lib"

pgrep -f "ollama serve" >/dev/null || \
    (nohup setsid "$PREFIX/bin/ollama" serve > "$PREFIX/ollama.log" 2>&1 < /dev/null &)
for i in $(seq 1 30); do curl -s -m 2 localhost:11434/api/tags >/dev/null && break; sleep 2; done

if [ "$OFFLINE" = 0 ]; then
    # piping to cat: the live progress bar's ANSI redraw has killed SSH tunnels
    "$PREFIX/bin/ollama" pull qwen2.5:7b 2>&1 | cat
fi
"$PREFIX/bin/ollama" list

echo "=============================================================="
echo " DONE - now run:  bash boot.sh"
echo "=============================================================="
