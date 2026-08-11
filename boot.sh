#!/usr/bin/env bash
# One command to bring the pod back after a restart:  bash /workspace/boot.sh
#
# RunPod rebuilds the container disk (/, /root, /usr/local) from the image on
# every start. Only /workspace survives - it is a MooseFS network volume. So
# every heavy asset lives there and this script re-points the environment at
# them. Nothing here downloads anything; if it starts downloading, something
# was installed to the wrong tier.

export PYTHONPATH=/workspace/pylibs:/workspace/AI4Bharat_NeMo
export HF_HOME=/workspace/.cache/huggingface      # else models land on container disk
export TORCH_HOME=/workspace/.cache/torch         # silero-vad cache - same reason
export OLLAMA_MODELS=/workspace/ollama/models     # else the 4.7GB qwen pull repeats
export HF_HUB_ENABLE_HF_TRANSFER=0                # set in RunPod images, package absent
export PATH=/workspace/bin:$PATH

# Offline hardening: disable HF revision checks on truly isolated machines
# Safe with network (just ignored); essential without network (prevents hangs)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# ollama looks for its runtime libs at ../lib/ollama relative to the binary;
# we keep the extracted dist on the volume and link it into place.
ln -sfn /workspace/ollama-dist/lib /workspace/lib

# direct SSH: /root does not survive, the key does
if [ -f /workspace/ssh/authorized_keys ]; then
    mkdir -p /root/.ssh && chmod 700 /root/.ssh
    cp /workspace/ssh/authorized_keys /root/.ssh/authorized_keys
    chmod 600 /root/.ssh/authorized_keys
fi

# ffmpeg lives on the container disk and is needed to transcode browser audio
command -v ffmpeg >/dev/null || (apt-get update -qq && apt-get install -y -qq ffmpeg)

echo "== ollama =="
if ! pgrep -f "ollama serve" >/dev/null; then
    nohup setsid /workspace/bin/ollama serve > /workspace/ollama.log 2>&1 < /dev/null &
    for i in $(seq 1 30); do
        curl -s -m 2 localhost:11434/api/tags >/dev/null 2>&1 && break
        sleep 2
    done
fi
curl -s -m 5 localhost:11434/api/tags | head -c 120; echo

echo "== dependencies =="
pip install -q pydantic soundfile 2>/dev/null || echo "pip install skipped"

echo "== server =="
cd /workspace/voice-to-rx-repo || { echo "MISSING REPO"; exit 1; }
if ! pgrep -f "uvicorn server:app" >/dev/null; then
    nohup setsid python3 -m uvicorn server:app --host 0.0.0.0 --port 8000 \
        > /workspace/server.log 2>&1 < /dev/null &
fi

for i in $(seq 1 60); do
    curl -s -m 3 localhost:8000/api/health && break
    sleep 5
done
echo
echo "== ready =="
echo "local  : http://localhost:8000"
echo "public : https://<POD_ID>-8000.proxy.runpod.net"
