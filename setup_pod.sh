#!/bin/bash
# ============================================================================
# Voice-to-Rx : one-shot pod setup
#
# Every step here was learned the hard way in a previous session that took
# ~45 minutes of trial and error across 5 sequential dependency conflicts.
# The ORDER and the PINS matter. Do not "simplify" them without reading the
# comments - each one is load-bearing.
#
# Run:  bash setup_pod.sh 2>&1 | tee /workspace/setup.log
# ============================================================================
set -euo pipefail

WORKDIR="${WORKDIR:-/workspace/voicerx}"
HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN before running: export HF_TOKEN=hf_xxx}"

echo "=============================================================="
echo " 0. Preflight"
echo "=============================================================="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || {
    echo "FATAL: no NVIDIA GPU visible."; exit 1; }

# LESSON: last time the pod had only a 30GB container disk and no network
# volume, so everything was lost when the pod's SSH died. Refuse to install
# into ephemeral storage.
if ! mountpoint -q /workspace 2>/dev/null && [ ! -d /workspace ]; then
    echo "FATAL: /workspace does not exist. Attach a Network Volume."; exit 1
fi
df -h /workspace
echo "Installing into $WORKDIR (persistent)."
mkdir -p "$WORKDIR"

# LESSON: this env var is set in RunPod's pytorch images but the hf_transfer
# package is NOT installed, so every HF download dies with an unhelpful
# ValueError. Turn it off globally.
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HOME=/workspace/.cache/huggingface
mkdir -p "$HF_HOME"

# persist the env for future shells (pod restarts drop exported vars)
{
  echo "export HF_TOKEN=\"$HF_TOKEN\""
  echo "export HF_HUB_ENABLE_HF_TRANSFER=0"
  echo "export HF_HOME=/workspace/.cache/huggingface"
} >> ~/.bashrc

echo "=============================================================="
echo " 1. System packages"
echo "=============================================================="
apt-get update -qq
apt-get install -y -qq ffmpeg git tmux >/dev/null

echo "=============================================================="
echo " 2. AI4Bharat NeMo fork"
echo "=============================================================="
# LESSON: vanilla `pip install nemo_toolkit[asr]` CANNOT load this model.
# IndicConformer uses a multilingual AGGREGATE tokenizer; mainline NeMo's
# _setup_monolingual_tokenizer throws KeyError: 'dir'. Only AI4Bharat's fork
# handles it. This is not optional.
if [ ! -d /workspace/AI4Bharat_NeMo ]; then
    git clone --depth 1 -b nemo-v2 https://github.com/AI4Bharat/NeMo.git \
        /workspace/AI4Bharat_NeMo
fi
cd /workspace/AI4Bharat_NeMo

# LESSON: their reinstall.sh defaults to `.[all]`, which pulls the multimodal
# extras -> tensorstore -> a full Bazel/protobuf C++ compile that ran 10+
# minutes and was still going when we killed it. We need ASR only.
pip install --quiet --editable ".[asr]"

echo "=============================================================="
echo " 3. Dependency pins (order matters)"
echo "=============================================================="
# LESSON: requirements_lightning.txt says `pytorch-lightning>=2.2.1` with NO
# upper bound, so pip takes 2.6.5, where NeptuneLogger was removed from
# pytorch_lightning.loggers -> ImportError on `import nemo.collections.asr`.
pip install --quiet "pytorch-lightning==2.2.1"

# LESSON: nemo pins huggingface_hub==0.23.2 exactly. `datasets` is unpinned,
# so pip resolves it to 2.14.4, which calls pa.PyExtensionType - removed in
# modern pyarrow -> AttributeError. Upgrade datasets, but --no-deps so it
# does NOT drag huggingface_hub off 0.23.2 (which would break ModelFilter).
pip install --quiet "datasets>=2.19,<3" --no-deps

# LESSON: numpy 2.0 removed np.sctypes, which AI4Bharat's audio loader still
# uses -> every transcribe() call dies in _convert_samples_to_float32.
SEG=/workspace/AI4Bharat_NeMo/nemo/collections/asr/parts/preprocessing/segment.py
sed -i "s/samples\.dtype in np\.sctypes\['int'\]/np.issubdtype(samples.dtype, np.signedinteger)/" "$SEG"
sed -i "s/samples\.dtype in np\.sctypes\['float'\]/np.issubdtype(samples.dtype, np.floating)/" "$SEG"
grep -q "issubdtype" "$SEG" && echo "  numpy 2.0 patch applied" || { echo "FATAL: patch failed"; exit 1; }

pip install --quiet silero-vad soundfile

echo "  verifying NeMo imports..."
python3 -c "import nemo.collections.asr as m; print('  NEMO_ASR_IMPORT_OK')"

echo "=============================================================="
echo " 4. Download IndicConformer (gated - needs accepted licence)"
echo "=============================================================="
python3 - <<'PY'
import os
from huggingface_hub import snapshot_download
p = snapshot_download(
    "ai4bharat/indicconformer_stt_bn_hybrid_ctc_rnnt_large",
    token=os.environ["HF_TOKEN"],
)
print("  model at:", p)
with open("/workspace/voicerx/model_path.txt", "w") as f:
    f.write(p)
PY

echo "=============================================================="
echo " 5. Ollama + Qwen2.5"
echo "=============================================================="
if ! command -v ollama >/dev/null; then
    curl -fsSL https://ollama.com/install.sh | sh
fi
export OLLAMA_MODELS=/workspace/.ollama/models
echo "export OLLAMA_MODELS=/workspace/.ollama/models" >> ~/.bashrc
mkdir -p "$OLLAMA_MODELS"
(ollama serve > /workspace/ollama.log 2>&1 &)
sleep 5
# LESSON: `ollama pull` renders a live progress bar whose ANSI redraw spam
# killed our SSH tunnel mid-download. Piping to cat makes it emit plain text.
ollama pull qwen2.5:7b 2>&1 | cat
ollama list

echo "=============================================================="
echo " DONE"
echo "=============================================================="
echo "Everything is under /workspace and survives pod termination."
echo "Next: upload the voicerx/ package and run_pipeline.py, then:"
echo "  cd $WORKDIR && python3 run_pipeline.py *.wav"
