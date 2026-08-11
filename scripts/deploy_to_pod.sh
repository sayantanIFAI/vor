#!/bin/bash
# Deploy PHASE 1 to RunPod and restart server
# Usage: bash scripts/deploy_to_pod.sh <POD_SSH_PORT> <POD_IP>

set -euo pipefail

POD_PORT="${1:-49567}"
POD_IP="${2:-213.173.102.132}"
POD_SSH="root@${POD_IP}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "========================================"
echo "PHASE 1 DEPLOYMENT"
echo "========================================"
echo "Target: $POD_SSH:$POD_PORT"

# 1. Create deployment bundle
echo ""
echo "[1/5] Creating deployment bundle..."
cd "$REPO_DIR"
git archive --format=tar.gz -o /tmp/vor_phase1.tgz HEAD

# 2. Upload to pod
echo "[2/5] Uploading to pod..."
timeout 120 scp -o StrictHostKeyChecking=no -o ConnectTimeout=20 \
    -P "$POD_PORT" -i ~/.ssh/id_ed25519 \
    /tmp/vor_phase1.tgz "$POD_SSH:/workspace/vor_phase1.tgz" || {
    echo "ERROR: SCP failed. Check pod is running and SSH accessible."
    exit 1
}

# 3. Extract and deploy
echo "[3/5] Extracting and deploying..."
timeout 120 ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 \
    -o BatchMode=yes "$POD_SSH" -p "$POD_PORT" -i ~/.ssh/id_ed25519 \
    "tar -xzf /workspace/vor_phase1.tgz -C /workspace/voice-to-rx-repo && echo 'Extracted OK'" || {
    echo "ERROR: SSH extraction failed."
    exit 1
}

# 4. Restart server
echo "[4/5] Restarting server..."
timeout 120 ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 \
    -o BatchMode=yes "$POD_SSH" -p "$POD_PORT" -i ~/.ssh/id_ed25519 \
    'pkill -9 -f "[u]vicorn server:app" 2>/dev/null; sleep 3; setsid --fork bash /workspace/startsrv.sh > /workspace/srv.log 2>&1 < /dev/null; sleep 5; ps -eo pid,args | grep "[u]vicorn" | head -1' || {
    echo "ERROR: Server restart failed."
    exit 1
}

# 5. Verify health
echo "[5/5] Verifying server health..."
for i in $(seq 1 30); do
    if timeout 5 ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 \
        -o BatchMode=yes "$POD_SSH" -p "$POD_PORT" -i ~/.ssh/id_ed25519 \
        'curl -s -m 3 localhost:8000/api/health >/dev/null 2>&1' 2>/dev/null; then
        echo "✓ Server healthy"
        break
    fi
    echo "  Waiting for server (attempt $i/30)..."
    sleep 2
done

echo ""
echo "========================================"
echo "✓ PHASE 1 DEPLOYED"
echo "========================================"
echo ""
echo "Next: Run load test"
echo "  bash scripts/load_test.sh $POD_PORT $POD_IP"
