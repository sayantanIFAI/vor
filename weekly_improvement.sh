#!/bin/bash
# Weekly Improvement Cycle
# Run every Monday morning to analyze errors and recalibrate thresholds
# Usage: bash weekly_improvement.sh

set -e

echo "==========================================="
echo "WEEKLY IMPROVEMENT CYCLE"
echo "==========================================="
echo "Week of: $(date +%Y-%m-%d)"
echo ""

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/scripts" && pwd)"
LOG_DIR="/workspace/reports"
mkdir -p "$LOG_DIR"

echo "[1/3] Analyzing doctor corrections..."
python "$SCRIPTS_DIR/analyze_errors.py" \
    --days 7 \
    --log-file /workspace/error_log.jsonl \
    > "$LOG_DIR/weekly_errors.txt" 2>&1

if [ -f "$LOG_DIR/weekly_errors.txt" ]; then
    echo "✓ Error analysis complete"
    echo ""
    echo "=== ERROR REPORT ==="
    tail -30 "$LOG_DIR/weekly_errors.txt"
    echo ""
else
    echo "✗ Error analysis failed"
fi

echo "[2/3] Recalibrating thresholds..."
python "$SCRIPTS_DIR/calibrate_thresholds.py" \
    --use-production-logs \
    --log-file /workspace/threshold_scores.jsonl \
    --output "$LOG_DIR/threshold_calibration.json" \
    > "$LOG_DIR/threshold_report.txt" 2>&1

if [ -f "$LOG_DIR/threshold_report.txt" ]; then
    echo "✓ Threshold calibration complete"
    echo ""
    echo "=== THRESHOLD REPORT ==="
    tail -30 "$LOG_DIR/threshold_report.txt"
    echo ""
else
    echo "✗ Threshold calibration failed"
fi

echo "[3/3] Summary and recommendations..."
echo ""
echo "==========================================="
echo "WEEKLY SUMMARY"
echo "==========================================="
echo ""
echo "Report files:"
echo "  - $LOG_DIR/weekly_errors.txt"
echo "  - $LOG_DIR/threshold_report.txt"
echo "  - $LOG_DIR/weekly_error_analysis.json"
echo "  - $LOG_DIR/threshold_calibration.json"
echo ""
echo "Next steps:"
echo "  1. Review weekly_errors.txt for patterns"
echo "  2. Check threshold_report.txt for confidence improvements"
echo "  3. If MEDIUM+ confidence, update gate.py with new thresholds"
echo "  4. Monitor production metrics"
echo ""
echo "==========================================="
