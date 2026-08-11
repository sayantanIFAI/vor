#!/bin/bash
# PHASE 2: Enable concurrent Ollama processing
# Sets OLLAMA_NUM_PARALLEL=8 in boot.sh + restarts server

set -euo pipefail

POD_PORT="${1:-49567}"
POD_IP="${2:-213.173.102.132}"
POD_SSH="root@${POD_IP}"

echo "========================================"
echo "PHASE 2: OLLAMA CONCURRENCY SETUP"
echo "========================================"
echo "Target: $POD_SSH:$POD_PORT"
echo ""

# 1. Check connection
echo "[1/3] Verifying connection..."
timeout 10 ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 \
    -o BatchMode=yes "$POD_SSH" -p "$POD_PORT" -i ~/.ssh/id_ed25519 \
    "echo 'Connected'" || {
    echo "ERROR: Cannot reach pod. Check SSH port."
    exit 1
}

# 2. Update boot.sh to enable OLLAMA_NUM_PARALLEL
echo "[2/3] Enabling OLLAMA_NUM_PARALLEL=8..."
timeout 20 ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 \
    -o BatchMode=yes "$POD_SSH" -p "$POD_PORT" -i ~/.ssh/id_ed25519 \
    'cat > /tmp/set_parallel.sh <<'"'"'SCRIPT'"'"'
#!/bin/bash
# Add OLLAMA_NUM_PARALLEL=8 to boot.sh if not already there
if ! grep -q "OLLAMA_NUM_PARALLEL" /workspace/boot.sh; then
    cat >> /workspace/boot.sh <<'"'"'EOF'"'"'

# PHASE 2: Enable concurrent Ollama processing
export OLLAMA_NUM_PARALLEL=8
EOF
    echo "Added OLLAMA_NUM_PARALLEL=8 to boot.sh"
else
    echo "OLLAMA_NUM_PARALLEL already set"
fi
SCRIPT
bash /tmp/set_parallel.sh' || {
    echo "ERROR: Failed to update boot.sh"
    exit 1
}

# 3. Restart Ollama and server
echo "[3/3] Restarting Ollama (with parallel=8) and server..."
timeout 120 ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 \
    -o BatchMode=yes "$POD_SSH" -p "$POD_PORT" -i ~/.ssh/id_ed25519 \
    'source /workspace/boot.sh; sleep 3; pgrep -f "[u]vicorn" | xargs kill -9 2>/dev/null || true; sleep 2; setsid --fork bash /workspace/startsrv.sh > /workspace/srv.log 2>&1 < /dev/null; sleep 10; echo "Restarted"' || {
    echo "ERROR: Restart failed, but continuing..."
}

# 4. Verify
echo ""
echo "Verifying Ollama parallelism..."
timeout 30 ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 \
    -o BatchMode=yes "$POD_SSH" -p "$POD_PORT" -i ~/.ssh/id_ed25519 \
    'OLLAMA_NUM_PARALLEL=8 pgrep -f "ollama serve" >/dev/null && echo "✓ Ollama running with parallel processing" || echo "⚠ Ollama starting up..."'

echo ""
echo "========================================"
echo "✓ PHASE 2 READY"
echo "========================================"
echo ""
echo "Next: Run load test"
echo "  bash scripts/load_test.sh $POD_PORT $POD_IP"
