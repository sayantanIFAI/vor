#!/bin/bash
# ============================================================================
# Make the runtime survive a pod restart.
#
# THE PROBLEM
# RunPod wipes the CONTAINER disk on every restart and keeps only the
# network volume at /workspace. Measured on this pod, what lives where:
#
#   SAFE, already on /workspace
#     nemo          /workspace/AI4Bharat_NeMo   (editable install)
#     ASR weights   /workspace/.cache/huggingface        4.1G
#     ollama models /workspace/.ollama                   4.4G
#
#   LOST on every restart
#     fastapi, uvicorn, silero_vad, soundfile, pytorch_lightning
#                   /usr/local/lib/python3.12/dist-packages
#     ollama binary /usr/local/bin/ollama                 38M
#
# So the two things to move are the pip packages and the ollama binary.
# The 8.5G of weights and models are already safe and are NOT copied - at
# that size a copy would be the slowest and most fragile part of this.
#
# THE FIX
# Install into /workspace/pylibs with `pip --target`, copy the ollama
# binary to /workspace/bin, and have activate.sh put both on the path
# along with the caches that point at the volume.
#
# Run ONCE:        bash persist_env.sh
# After restart:   source /workspace/activate.sh     (seconds, no downloads)
# ============================================================================
set -u

PYLIBS=/workspace/pylibs
PYBIN=/workspace/bin
NEMO_SRC=/workspace/AI4Bharat_NeMo

echo "=== installing into $PYLIBS (persistent) ==="
mkdir -p "$PYLIBS" "$PYBIN"
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

# The ollama BINARY is on container disk and goes with it. Only 38M, and
# without it the extraction model cannot run at all - the 4.4G of models
# already on the volume are useless on their own.
if [ -x /usr/local/bin/ollama ] && [ ! -x "$PYBIN/ollama" ]; then
    echo "=== copying ollama binary to $PYBIN ==="
    cp /usr/local/bin/ollama "$PYBIN/ollama"
fi

cat > /workspace/activate.sh <<'EOS'
# Restore the Voice-to-Rx environment after a pod restart.
#   source /workspace/activate.sh
export PYTHONPATH=/workspace/pylibs:/workspace/AI4Bharat_NeMo:${PYTHONPATH:-}
export PATH=/workspace/bin:${PATH}
export HF_HOME=/workspace/.cache/huggingface
export HF_HUB_ENABLE_HF_TRANSFER=0
export OLLAMA_MODELS=/workspace/.ollama/models
EOS

# Load it automatically in every new shell, without duplicating the line.
grep -q 'source /workspace/activate.sh' ~/.bashrc 2>/dev/null \
  || echo 'source /workspace/activate.sh' >> ~/.bashrc

# ---------------------------------------------------------------------------
# VERIFY
#
# The previous version of this check only asserted that each module
# imported. That is worth very little: right after running this script the
# container disk is still intact, so every import succeeds whether or not
# anything was persisted, and the check passes on an environment that will
# be empty after the next restart - the exact failure it exists to catch.
#
# So the assertion is on the RESOLVED PATH, not on importability. A module
# only counts as persisted if Python loads it from /workspace.
# ---------------------------------------------------------------------------
echo "=== verifying (path, not just importability) ==="
source /workspace/activate.sh
python3 - <<'PY'
import importlib, sys

# Import from the persistent locations FIRST, the way a restarted pod
# will, rather than letting container-disk copies answer.
sys.path.insert(0, "/workspace/AI4Bharat_NeMo")
sys.path.insert(0, "/workspace/pylibs")

MODULES = ["nemo.collections.asr", "fastapi", "uvicorn", "parler_tts",
           "silero_vad", "soundfile", "pytorch_lightning"]

bad = []
for name in MODULES:
    try:
        mod = importlib.import_module(name)
    except Exception as e:
        bad.append(f"{name}: cannot import ({type(e).__name__})")
        print(f"  FAIL  {name}: {type(e).__name__}")
        continue
    path = getattr(mod, "__file__", "") or ""
    if path.startswith("/workspace/"):
        print(f"  ok    {name}")
    else:
        bad.append(f"{name}: loads from {path}")
        print(f"  LOST  {name} -> {path}")
        print(f"        on container disk; will vanish on restart")

import shutil
if shutil.which("ollama", path="/workspace/bin"):
    print("  ok    ollama binary")
else:
    bad.append("ollama binary not in /workspace/bin")
    print("  LOST  ollama binary")

print()
print("PERSIST_OK" if not bad else "PERSIST_INCOMPLETE:")
for b in bad:
    print("   ", b)
PY

echo
echo "After any future restart:"
echo "    source /workspace/activate.sh    # then ollama serve &"
