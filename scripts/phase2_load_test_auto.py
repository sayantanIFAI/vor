#!/usr/bin/env python3
"""
PHASE 2: Auto load test runner
Connects via SSH with password, checks recordings, runs load test
"""

import paramiko
import time
import json
from pathlib import Path

def ssh_exec(client, cmd):
    """Execute command via SSH, return output"""
    stdin, stdout, stderr = client.exec_command(cmd)
    return stdout.read().decode('utf-8', errors='ignore')

def main():
    HOST = "213.173.102.235"
    PORT = 35353
    USER = "root"
    PASSWORD = "Uncertain999!"

    print("=" * 70)
    print("PHASE 2: LOAD TEST (AUTO)")
    print("=" * 70)
    print(f"Connecting to {USER}@{HOST}:{PORT}...")

    # Connect
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        client.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=20)
        print("✓ Connected")
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        return False

    # Check recordings
    print("\nChecking recordings on pod...")
    output = ssh_exec(client, "ls -1 /workspace/voice-to-rx-repo/recordings/recording_*.wav 2>/dev/null | wc -l")
    num_recordings = int(output.strip())
    print(f"Found {num_recordings} recordings")

    if num_recordings < 16:
        print(f"✗ Need 16 recordings, found {num_recordings}")
        print("\nTrying alternative paths...")
        output = ssh_exec(client, "find /workspace -name 'recording_*.wav' 2>/dev/null | head -5")
        print(f"Found: {output}")
        client.close()
        return False

    # List recordings
    output = ssh_exec(client, "ls -1 /workspace/voice-to-rx-repo/recordings/recording_*.wav")
    recordings = output.strip().split('\n')[:16]
    print(f"Recordings: {len(recordings)}")
    for rec in recordings[:3]:
        print(f"  {Path(rec).name}")
    print("  ...")

    # Run load test
    print("\n" + "=" * 70)
    print("Running load test: 10 concurrent doctors × 16 recordings")
    print("=" * 70)
    print()

    # Create load test script on pod
    load_test_script = '''#!/bin/bash
set -euo pipefail

POD_URL="http://localhost:8000"
RECORDINGS_DIR="/workspace/voice-to-rx-repo/recordings"
NUM_CONCURRENT=10
RESULTS_DIR="/tmp/load_test_$(date +%s)"
mkdir -p "$RESULTS_DIR"

echo "Running $NUM_CONCURRENT concurrent jobs per recording..."

RECORDINGS=($(ls "$RECORDINGS_DIR"/recording_*.wav | head -16))
JOB_COUNT=0

for recording in "${RECORDINGS[@]}"; do
    for job in $(seq 1 $NUM_CONCURRENT); do
        (
            JOB_ID="$(basename "$recording" .wav)_doctor_$job"

            # Start session
            SESSION_RESPONSE=$(curl -s -m 10 -X POST "$POD_URL/api/session/start")
            SESSION_ID=$(echo "$SESSION_RESPONSE" | grep -o '"session_id":"[^"]*' | cut -d'"' -f4)

            if [ -z "$SESSION_ID" ]; then
                echo "[$JOB_ID] FAIL: No session"
                echo "FAIL" > "$RESULTS_DIR/$JOB_ID.result"
                exit 1
            fi

            START_TIME=$(date +%s.%N)

            # Upload
            curl -s -m 120 -X POST "$POD_URL/api/session/$SESSION_ID/chunk" \\
                -F "file=@$recording" >/dev/null 2>&1

            # Finalize
            FINAL=$(curl -s -m 30 -X POST "$POD_URL/api/session/$SESSION_ID/finalize")

            END_TIME=$(date +%s.%N)
            TOTAL_TIME=$(echo "$END_TIME - $START_TIME" | bc -l 2>/dev/null || echo "0")
            CLICK_TO_RESULT=$(echo "$FINAL" | grep -o '"click_to_result_s":[0-9.]*' | cut -d':' -f2)

            if [ -z "$CLICK_TO_RESULT" ]; then
                echo "[$JOB_ID] FAIL"
                echo "FAIL" > "$RESULTS_DIR/$JOB_ID.result"
            else
                echo "[$JOB_ID] OK ${CLICK_TO_RESULT}s"
                echo "OK:$CLICK_TO_RESULT" > "$RESULTS_DIR/$JOB_ID.result"
            fi
        ) &

        JOB_COUNT=$((JOB_COUNT + 1))
        if [ $((JOB_COUNT % $NUM_CONCURRENT)) -eq 0 ]; then
            wait
        fi
    done
done

wait

# Analyze
echo ""
echo "RESULTS:"
TOTAL=$(ls "$RESULTS_DIR"/*.result 2>/dev/null | wc -l)
PASSED=$(grep -l "^OK:" "$RESULTS_DIR"/*.result 2>/dev/null | wc -l)
echo "  Total: $TOTAL"
echo "  Passed: $PASSED"

if [ $PASSED -gt 0 ]; then
    echo ""
    echo "Latency:"
    for f in "$RESULTS_DIR"/*.result; do
        grep "^OK:" "$f" | cut -d':' -f2
    done | sort -n | awk '{
        arr[NR]=$1;
    }
    END {
        n=length(arr);
        print "  Min: " arr[1];
        print "  P50: " arr[int(n*0.5)];
        print "  P95: " arr[int(n*0.95)];
        print "  P99: " arr[int(n*0.99)];
        print "  Max: " arr[n];
    }'
fi
'''

    # Write script to pod
    print("Writing load test script to pod...")
    sftp = client.open_sftp()
    with sftp.file('/tmp/load_test.sh', 'w') as f:
        f.write(load_test_script)
    sftp.close()

    # Run it
    print("Executing load test (this will take ~10 minutes)...\n")
    output = ssh_exec(client, "bash /tmp/load_test.sh 2>&1")
    print(output)

    client.close()
    return True

if __name__ == "__main__":
    main()
