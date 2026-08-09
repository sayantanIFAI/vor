#!/bin/bash
# ============================================================================
# Make the Python environment survive a pod restart.
#
# THE PROBLEM
# RunPod wipes the CONTAINER disk on every restart and keeps only the
# network volume at /workspace. pip installs into
# /usr/local/lib/python3.x/site-packages, which is container disk - so nemo,
# fastapi and parler_tts vanish every time, and the environment has to be
# rebuilt from scratch. That happened four times in one day.
#
# THE FIX
# Install into /workspace/pylibs with `pip --target`, and put that on
# PYTHONPATH. The packages then live on the network volume and survive.
#
# NeMo needs no copy at all: it is an editable install whose SOURCE is
# already at /workspace/AI4Bharat_NeMo. Only the site-packages link is
# lost, so putting that directory on PYTHONPATH restores it directly.
#
# Torch, CUDA and the rest stay in the base image - they are reinstalled by
# RunPod for free and are far too large to duplicate on the volume.
#
# Run ONCE:        bash persist_env.sh
# After restart:   source /workspace/activate.sh     (seconds, no downloads)
# ============================================================================
set -u

PYLIBS=/workspace/pylibs
NEMO_SRC=/workspace/AI4Bharat_NeMo

echo "=== installing into $PYLIBS (persistent) ==="
mkdir -p "$PYLIBS"
export PYTHONPATH="$PYLIBS:$NEMO_SRC:${PYTHONPATH:-}"

# --target does not resolve already-satisfied deps against the base image,
# so --no-deps is used where the base image already provides them. Anything
# genuinely missing is installed normally.
pip install --quiet --target="$PYLIBS" --upgrade \
    "pytorch-lightning==2.2.1" \
    silero-vad soundfile \
    fastapi "uvicorn[standard]" python-multipart \
    indic-transliteration 2>&1 | tail -3

pip install --quiet --target="$PYLIBS" --no-deps "datasets>=2.19,<3" 2>&1 | tail -2
pip install --quiet --target="$PYLIBS" --no-deps \
    git+https://github.com/huggingface/parler-tts.git 2>&1 | tail -3

# parler-tts's own requirements that the base image lacks
pip install --quiet --target="$PYLIBS" --no-deps \
    descript-audio-codec descript-audiotools 2>&1 | tail -2

cat > /workspace/activate.sh <<'EOS'
# Restore the Voice-to-Rx environment after a pod restart.
#   source /workspace/activate.sh
export PYTHONPATH=/workspace/pylibs:/workspace/AI4Bharat_NeMo:${PYTHONPATH:-}
export HF_HOME=/workspace/.cache/huggingface
export HF_HUB_ENABLE_HF_TRANSFER=0
export OLLAMA_MODELS=/workspace/.ollama/models
EOS

# Load it automatically in every new shell, without duplicating the line.
grep -q 'source /workspace/activate.sh' ~/.bashrc 2>/dev/null \
  || echo 'source /workspace/activate.sh' >> ~/.bashrc

echo "=== verifying ==="
source /workspace/activate.sh
python3 - <<'PY'
mods = ["nemo.collections.asr", "fastapi", "parler_tts", "silero_vad", "soundfile"]
bad = []
for m in mods:
    try:
        __import__(m)
        print(f"  ok   {m}")
    except Exception as e:
        bad.append(m)
        print(f"  FAIL {m}: {type(e).__name__}: {e}")
print("PERSIST_OK" if not bad else f"PERSIST_INCOMPLETE: {bad}")
PY

echo
echo "After any future restart, just run:"
echo "    source /workspace/activate.sh"
echo "No pip, no downloads. Ollama still needs: ollama serve &"
