#!/bin/bash
# Load test: Replay 16 recordings 10 times concurrently (10 concurrent doctors)
# Usage: bash scripts/load_test.sh <POD_SSH_PORT> <POD_IP>

set -euo pipefail

POD_PORT="${1:-49567}"
POD_IP="${2:-213.173.102.132}"
POD_SSH="root@${POD_IP}"
POD_URL="https://cynxpkzxph7xkv-8000.proxy.runpod.net"

RECORDINGS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/recordings"

if [ ! -d "$RECORDINGS_DIR" ]; then
    echo "ERROR: $RECORDINGS_DIR not found"
    echo "You need 16 test recordings (recording_32.wav .. recording_47.wav)"
    exit 1
fi

RECORDINGS=()
for i in $(seq 32 47); do
    if [ -f "$RECORDINGS_DIR/recording_$i.wav" ]; then
        RECORDINGS+=("$RECORDINGS_DIR/recording_$i.wav")
    fi
done

if [ ${#RECORDINGS[@]} -eq 0 ]; then
    echo "ERROR: No recordings found in $RECORDINGS_DIR"
    exit 1
fi

NUM_RECORDINGS=${#RECORDINGS[@]}
NUM_CONCURRENT=10

echo "========================================"
echo "PHASE 2: CONCURRENCY LOAD TEST"
echo "========================================"
echo "Recordings: $NUM_RECORDINGS (will replay $NUM_CONCURRENT×, total $((NUM_RECORDINGS * NUM_CONCURRENT)) jobs)"
echo "Target: $POD_URL"
echo ""

# Verify server is up
echo "Checking server health..."
if ! timeout 10 curl -s -m 5 "$POD_URL/api/health" >/dev/null 2>&1; then
    echo "ERROR: Server not responding at $POD_URL"
    exit 1
fi
echo "✓ Server is up"
echo ""

# Create results directory
RESULTS_DIR="/tmp/load_test_results_$(date +%s)"
mkdir -p "$RESULTS_DIR"

echo "Running $NUM_RECORDINGS recordings × $NUM_CONCURRENT concurrent (total $((NUM_RECORDINGS * NUM_CONCURRENT)) jobs)..."
echo "Results saved to: $RESULTS_DIR"
echo ""

# Launch concurrent jobs
# Each job:
#   1. Start a session
#   2. Upload the entire recording
#   3. Finalize and get prescription
#   4. Record timing

JOB_COUNT=0
BATCH_SIZE=$NUM_CONCURRENT

for recording in "${RECORDINGS[@]}"; do
    for job in $(seq 1 $NUM_CONCURRENT); do
        (
            JOB_ID="$(basename "$recording" .wav)_doctor_$job"
            REC_BASENAME="$(basename "$recording")"

            # Start session
            SESSION_RESPONSE=$(curl -s -m 10 -X POST "$POD_URL/api/session/start")
            SESSION_ID=$(echo "$SESSION_RESPONSE" | grep -o '"session_id":"[^"]*' | cut -d'"' -f4)

            if [ -z "$SESSION_ID" ]; then
                echo "[$JOB_ID] FAIL: Could not start session"
                echo "FAIL" > "$RESULTS_DIR/$JOB_ID.result"
                exit 1
            fi

            START_TIME=$(date +%s.%N)

            # Upload entire recording at once
            curl -s -m 120 -X POST "$POD_URL/api/session/$SESSION_ID/chunk" \
                -F "file=@$recording" >/dev/null 2>&1

            # Finalize
            FINAL_RESPONSE=$(curl -s -m 30 -X POST "$POD_URL/api/session/$SESSION_ID/finalize")

            END_TIME=$(date +%s.%N)

            # Extract results
            TOTAL_TIME=$(echo "$END_TIME - $START_TIME" | bc -l 2>/dev/null || echo "0")
            CLICK_TO_RESULT=$(echo "$FINAL_RESPONSE" | grep -o '"click_to_result_s":[0-9.]*' | cut -d':' -f2)
            PROCESSING_TIME=$(echo "$FINAL_RESPONSE" | grep -o '"processing_s":[0-9.]*' | cut -d':' -f2)
            NUM_MEDS=$(echo "$FINAL_RESPONSE" | grep -o '"medications":\[' | wc -l)

            if [ -z "$CLICK_TO_RESULT" ]; then
                echo "[$JOB_ID] FAIL: No result"
                echo "FAIL" > "$RESULTS_DIR/$JOB_ID.result"
                exit 1
            fi

            # Save result
            {
                echo "job_id=$JOB_ID"
                echo "recording=$REC_BASENAME"
                echo "total_time=$TOTAL_TIME"
                echo "click_to_result=$CLICK_TO_RESULT"
                echo "processing_time=$PROCESSING_TIME"
                echo "num_medications=$NUM_MEDS"
                echo "status=OK"
            } > "$RESULTS_DIR/$JOB_ID.result"

            echo "[$JOB_ID] OK in ${TOTAL_TIME}s (${CLICK_TO_RESULT}s post-stop)"

        ) &

        JOB_COUNT=$((JOB_COUNT + 1))

        # Limit concurrency to avoid overwhelming server
        if [ $((JOB_COUNT % BATCH_SIZE)) -eq 0 ]; then
            echo "  ... launched $JOB_COUNT jobs, waiting for batch to complete ..."
            wait
        fi
    done
done

# Wait for all jobs
echo ""
echo "Waiting for all jobs to complete..."
wait

echo ""
echo "========================================"
echo "LOAD TEST RESULTS"
echo "========================================"

# Analyze results
TOTAL_JOBS=$(ls "$RESULTS_DIR"/*.result 2>/dev/null | wc -l)
SUCCESSFUL=0
FAILED=0

CLICK_TO_RESULT_TIMES=()
PROCESSING_TIMES=()

for result_file in "$RESULTS_DIR"/*.result; do
    if [ -f "$result_file" ]; then
        source "$result_file"

        if [ "$status" = "OK" ]; then
            SUCCESSFUL=$((SUCCESSFUL + 1))
            CLICK_TO_RESULT_TIMES+=("$click_to_result")
            PROCESSING_TIMES+=("$processing_time")
        else
            FAILED=$((FAILED + 1))
        fi
    fi
done

echo ""
echo "Summary:"
echo "  Total jobs: $TOTAL_JOBS"
echo "  Successful: $SUCCESSFUL"
echo "  Failed: $FAILED"

if [ $SUCCESSFUL -gt 0 ]; then
    echo ""
    echo "Post-Stop Latency (click_to_result_s):"

    # Sort times and calculate percentiles
    SORTED_CLICK_TO_RESULT=($(printf '%s\n' "${CLICK_TO_RESULT_TIMES[@]}" | sort -n))

    MIN=${SORTED_CLICK_TO_RESULT[0]}
    MAX=${SORTED_CLICK_TO_RESULT[-1]}
    P50=${SORTED_CLICK_TO_RESULT[$((SUCCESSFUL / 2))]}
    P95=${SORTED_CLICK_TO_RESULT[$((SUCCESSFUL * 95 / 100))]}
    P99=${SORTED_CLICK_TO_RESULT[$((SUCCESSFUL * 99 / 100))]}

    echo "  Min: ${MIN}s"
    echo "  P50: ${P50}s"
    echo "  P95: ${P95}s"
    echo "  P99: ${P99}s"
    echo "  Max: ${MAX}s"

    # Check SLA
    echo ""
    echo "SLA Analysis (target: ≤3s):"
    UNDER_3S=0
    for t in "${CLICK_TO_RESULT_TIMES[@]}"; do
        if (( $(echo "$t <= 3.0" | bc -l) )); then
            UNDER_3S=$((UNDER_3S + 1))
        fi
    done
    UNDER_3S_PCT=$((UNDER_3S * 100 / SUCCESSFUL))
    echo "  ${UNDER_3S}/${SUCCESSFUL} jobs under 3s ($UNDER_3S_PCT%)"

    if [ "$UNDER_3S_PCT" -ge 95 ]; then
        echo "  ✓ PASS: >95% within SLA"
    else
        echo "  ✗ FAIL: <95% within SLA"
    fi
fi

echo ""
echo "========================================"
echo "Detailed results: $RESULTS_DIR"
echo "========================================"
