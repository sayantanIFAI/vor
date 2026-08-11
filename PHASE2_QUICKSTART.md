# PHASE 2: Concurrency Baseline — Quick Start

## Pod Connection Info

When your pod is running, RunPod shows:

```
SSH: ssh beo4eprep05rpp-XXXXXX@ssh.runpod.io -i ~/.ssh/id_ed25519
```

**Extract the port and IP:**
- Run this in your terminal:
  ```bash
  ssh beo4eprep05rpp-XXXXXX@ssh.runpod.io -p 22 -i ~/.ssh/id_ed25519 "echo IP=$(hostname -I | awk '{print $1}'); netstat -tuln | grep LISTEN | grep ssh"
  ```
- Or use the direct TCP endpoint (faster):
  - SSH shows a public IP like `213.173.XXX.XXX` with port like `49567`

**For direct SSH (skip the relay):**
```bash
ssh root@213.173.XXX.XXX -p 49567 -i ~/.ssh/id_ed25519 "bash /workspace/boot.sh"
```

---

## One Command: Deploy + Enable Concurrency + Load Test

```bash
bash scripts/phase2_orchestrate.sh <PORT> <IP>
```

**Example:**
```bash
bash scripts/phase2_orchestrate.sh 49567 213.173.102.132
```

This will:
1. Deploy PHASE 1 code (chunk_seconds=4, offline hardening, collision audit)
2. Enable OLLAMA_NUM_PARALLEL=8
3. Restart Ollama + server
4. Run load test: 16 recordings × 10 concurrent doctors (160 total jobs)
5. Report latency percentiles + SLA pass/fail

---

## What PHASE 2 Measures

| Metric | Target | Interpretation |
|--------|--------|-----------------|
| **P50 (median)** | <2.5s | Typical doctor experience |
| **P95** | <3.5s | 95% of doctors get result in <3.5s |
| **P99** | <4.5s | Even tail cases are fast |
| **% under 3s SLA** | >95% | Clinical acceptable threshold |

---

## Expected Results (with OLLAMA_NUM_PARALLEL=8)

### Best case (if batching is effective):
```
✓ P95: 2.2s
✓ P99: 3.1s
✓ 98% jobs under 3s → PASS
→ Stop here. Concurrency is solved.
```

### Moderate case (partial batching benefit):
```
⚠ P95: 3.5s
⚠ P99: 4.2s
⚠ 88% jobs under 3s → FAIL
→ Escalate to vLLM (2-hour change, better throughput)
```

### Worst case (no improvement):
```
✗ P95: 5+ s
✗ P99: 8+ s
✗ 40% jobs under 3s → FAIL
→ Serious bottleneck. Investigate:
   - Ollama not actually using OLLAMA_NUM_PARALLEL
   - GPU memory contention
   - Network latency on RunPod
```

---

## Troubleshooting

### Load test fails with "Could not start session"
- Server is down: `ssh root@IP -p PORT bash /workspace/boot.sh`
- Server crashed: Check `/workspace/srv.log`
- Network issue: Try direct IP instead of relay

### Load test takes forever (>30 min)
- Ollama is bottlenecked. Try increasing OLLAMA_NUM_PARALLEL to 16, or switch to vLLM.

### Some jobs timeout, some pass
- Indicates queue depth issues. PHASE 2 is failing. Escalate to vLLM.

---

## Quick Manual Test (before load test)

If you want to verify deployment worked first:

```bash
# SSH into pod
ssh root@213.173.102.132 -p 49567 -i ~/.ssh/id_ed25519

# Check server is up
curl -s localhost:8000/api/health | jq .

# Check chunk_seconds=4 (PHASE 1)
curl -s -X POST localhost:8000/api/session/start | jq '.chunk_seconds'
# Should return: 4

# Check OLLAMA_NUM_PARALLEL is set
env | grep OLLAMA
# Should show: OLLAMA_NUM_PARALLEL=8
```

---

## After Results

- **If PASS (>95% under 3s):** Commit to main, proceed to PHASE 3 (concurrent segment extraction)
- **If FAIL:** I'll prepare vLLM migration (alternative to Ollama with native parallelism)

---

## Commands Summary

```bash
# Deploy PHASE 1
bash scripts/deploy_to_pod.sh 49567 213.173.102.132

# Enable parallelism
bash scripts/phase2_setup.sh 49567 213.173.102.132

# Run load test
bash scripts/load_test.sh 49567 213.173.102.132

# All three at once
bash scripts/phase2_orchestrate.sh 49567 213.173.102.132
```
