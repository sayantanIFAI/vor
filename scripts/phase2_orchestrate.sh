#!/bin/bash
# PHASE 2 Orchestration: Deploy → Enable Concurrency → Load Test
# Usage: bash scripts/phase2_orchestrate.sh <POD_SSH_PORT> <POD_IP>

set -euo pipefail

POD_PORT="${1:-}"
POD_IP="${2:-}"

if [ -z "$POD_PORT" ] || [ -z "$POD_IP" ]; then
    echo "Usage: bash scripts/phase2_orchestrate.sh <POD_SSH_PORT> <POD_IP>"
    echo ""
    echo "Example:"
    echo "  bash scripts/phase2_orchestrate.sh 49567 213.173.102.132"
    echo ""
    echo "To find your pod's port and IP:"
    echo "  1. Go to RunPod console"
    echo "  2. Find your pod"
    echo "  3. SSH shows: ssh user@host -p PORT"
    exit 1
fi

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║              PHASE 2: CONCURRENCY ORCHESTRATION                ║"
echo "║                                                                ║"
echo "║  1. Deploy PHASE 1 code to pod                                ║"
echo "║  2. Enable OLLAMA_NUM_PARALLEL=8                              ║"
echo "║  3. Run load test (10 concurrent doctors)                     ║"
echo "║                                                                ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Step 1: Deploy
echo "STEP 1: Deploying PHASE 1 code..."
echo "=========================================="
bash "$REPO_DIR/scripts/deploy_to_pod.sh" "$POD_PORT" "$POD_IP" || {
    echo ""
    echo "ERROR: Deployment failed. Check SSH connectivity."
    exit 1
}

echo ""
echo "STEP 2: Enabling OLLAMA_NUM_PARALLEL=8..."
echo "=========================================="
bash "$REPO_DIR/scripts/phase2_setup.sh" "$POD_PORT" "$POD_IP" || {
    echo ""
    echo "⚠️  Setup encountered issues, but continuing to load test..."
}

echo ""
echo "STEP 3: Running load test (10 concurrent doctors × 16 recordings)..."
echo "=========================================="
bash "$REPO_DIR/scripts/load_test.sh" "$POD_PORT" "$POD_IP"

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                     PHASE 2 COMPLETE                           ║"
echo "║                                                                ║"
echo "║  Check results above for:                                     ║"
echo "║  • P95/P99 latency                                            ║"
echo "║  • % of jobs under 3s SLA                                     ║"
echo "║                                                                ║"
echo "║  Next steps:                                                  ║"
echo "║  • If >95% under 3s: PHASE 2 PASS → proceed to PHASE 3        ║"
echo "║  • If <95% under 3s: Consider vLLM → escalate                 ║"
echo "║                                                                ║"
echo "╚════════════════════════════════════════════════════════════════╝"
