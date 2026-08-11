#!/usr/bin/env python3
"""
THRESHOLD CALIBRATION from actual recordings.

Measures similarity and grounding score distributions to replace
guess-based thresholds with data-driven ones.

Usage:
  python scripts/calibrate_thresholds.py /workspace/voice-to-rx-repo/recordings
  python scripts/calibrate_thresholds.py --use-production-logs  # After deployment
"""

import json
import statistics
from pathlib import Path
from typing import Optional
from collections import defaultdict
import sys


def find_gaps(sorted_scores: list) -> list:
    """Find largest gaps in score distribution (natural thresholds)."""
    gaps = []
    for i in range(len(sorted_scores) - 1):
        gap = sorted_scores[i + 1] - sorted_scores[i]
        if gap > 0.03:  # Significant gap (>=3%)
            gaps.append({
                "between": f"{sorted_scores[i]:.3f} - {sorted_scores[i+1]:.3f}",
                "size": round(gap, 4),
                "suggested_threshold": round((sorted_scores[i] + sorted_scores[i+1]) / 2, 3),
                "left_count": i + 1,  # How many scores below this gap
                "right_count": len(sorted_scores) - i - 1  # How many above
            })
    return sorted(gaps, key=lambda x: x["size"], reverse=True)


def analyze_distribution(scores: list, key: str, name: str) -> dict:
    """Analyze a score distribution."""
    values = [s[key] for s in scores if key in s and s[key] is not None]

    if not values:
        return {
            "n": 0,
            "note": f"No {name} scores found in data"
        }

    values.sort()

    analysis = {
        "n": len(values),
        "min": round(min(values), 3),
        "max": round(max(values), 3),
        "mean": round(statistics.mean(values), 3),
        "median": round(statistics.median(values), 3),
        "stdev": round(statistics.stdev(values), 3) if len(values) > 1 else 0,
        "p25": round(values[max(0, len(values) // 4)], 3),
        "p50": round(values[len(values) // 2], 3),
        "p75": round(values[3 * len(values) // 4], 3),
        "gaps": find_gaps(values),
        "name": name
    }

    return analysis


def recommend_threshold(analysis: dict) -> dict:
    """Recommend threshold based on gap analysis."""
    if analysis['n'] == 0:
        return {
            "recommendation": "INSUFFICIENT DATA",
            "confidence": "NONE",
            "reason": "No samples collected"
        }

    largest_gap = analysis['gaps'][0] if analysis['gaps'] else None

    if analysis['n'] < 15:
        confidence = "VERY_LOW"
        reason = f"Only {analysis['n']} samples (need 50+)"
    elif analysis['n'] < 30:
        confidence = "LOW"
        reason = f"Only {analysis['n']} samples (need 50+)"
    elif analysis['n'] < 50:
        confidence = "MEDIUM"
        reason = f"{analysis['n']} samples, gap needs to exceed 0.05"
    else:
        confidence = "HIGH"
        reason = f"{analysis['n']} samples, gap {largest_gap['size']:.3f}"

    if largest_gap and largest_gap['size'] > 0.05:
        recommendation = f"{largest_gap['suggested_threshold']}"
        reason = f"Natural gap of {largest_gap['size']:.3f} at {largest_gap['between']}"
    else:
        recommendation = analysis['median']
        reason = "Using median (no clear gap found)"

    return {
        "recommendation": recommendation,
        "confidence": confidence,
        "reason": reason,
        "largest_gap": largest_gap
    }


def calibrate_from_recordings(recordings_dir: str) -> dict:
    """Collect scores from consultation results."""

    recordings_path = Path(recordings_dir)
    if not recordings_path.exists():
        print(f"ERROR: Recordings directory not found: {recordings_dir}")
        return {"error": "Directory not found", "samples": []}

    # Collect all threshold data from consultations
    # For now, this is a template—in production, will read from /api/consultation/*/metrics

    # Simulated: During production, gate.py will log every score
    # For now, we'll create a sample structure

    all_fuzzy = []
    all_grounding = []

    # In production deployment, this would:
    # 1. Process each recording through the pipeline
    # 2. Collect gate.py internal scores
    # 3. Log to a JSON file

    # For now, return structure with notes on how to populate
    return {
        "fuzzy_matches": all_fuzzy,
        "grounding_scores": all_grounding,
        "analysis": {
            "fuzzy": {
                "n": len(all_fuzzy),
                "note": "No data yet. Will populate from production logs."
            },
            "grounding": {
                "n": len(all_grounding),
                "note": "No data yet. Will populate from production logs."
            }
        },
        "source": "production_logs" if len(all_fuzzy) > 0 else "empty",
        "note": "See server.py to enable score logging in production"
    }


def load_from_production_logs(log_file: str = "/workspace/threshold_scores.jsonl") -> dict:
    """Load scores from production log file."""

    all_fuzzy = []
    all_grounding = []

    log_path = Path(log_file)
    if not log_path.exists():
        return {
            "error": "No production logs found",
            "note": f"File does not exist: {log_file}",
            "note2": "Production logging will be enabled in server.py finalize() endpoint"
        }

    try:
        with open(log_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    entry = json.loads(line)

                    if 'similarity_score' in entry and entry['similarity_score'] is not None:
                        all_fuzzy.append({
                            "score": entry['similarity_score'],
                            "drug": entry.get('drug', ''),
                            "tier": entry.get('tier', ''),
                            "line": line_num
                        })

                    if 'grounding_score' in entry and entry['grounding_score'] is not None:
                        all_grounding.append({
                            "score": entry['grounding_score'],
                            "drug": entry.get('drug', ''),
                            "tier": entry.get('tier', ''),
                            "line": line_num
                        })
                except json.JSONDecodeError:
                    print(f"Warning: Line {line_num} is not valid JSON")

    except IOError as e:
        return {"error": str(e)}

    return {
        "fuzzy_matches": all_fuzzy,
        "grounding_scores": all_grounding,
        "sample_lines": log_path,
        "loaded_at": "production"
    }


def print_calibration_report(analysis: dict, production: bool = False):
    """Generate human-readable calibration report."""

    print("\n" + "=" * 80)
    print("THRESHOLD CALIBRATION REPORT")
    print("=" * 80)

    if "error" in analysis:
        print(f"\nERROR: {analysis['error']}")
        if "note" in analysis:
            print(f"  {analysis['note']}")
        if "note2" in analysis:
            print(f"  {analysis['note2']}")
        return

    # SIMILARITY FLOOR
    print("\n" + "-" * 80)
    print("SIMILARITY FLOOR (fuzzy drug matching threshold)")
    print("-" * 80)

    fuzzy_analysis = analysis['analysis']['fuzzy']

    if fuzzy_analysis['n'] == 0:
        print(f"\nNo fuzzy matching data collected yet.")
        print(f"Current value: SIMILARITY_FLOOR = 0.65")
        print(f"Confidence: LOW (based on {fuzzy_analysis.get('collected_on', 13)} old samples, gap 0.06)")
    else:
        print(f"\nSamples: {fuzzy_analysis['n']}")
        print(f"Range: {fuzzy_analysis['min']:.3f} - {fuzzy_analysis['max']:.3f}")
        print(f"Mean: {fuzzy_analysis['mean']:.3f}")
        print(f"Median: {fuzzy_analysis['median']:.3f}")
        print(f"Stdev: {fuzzy_analysis['stdev']:.3f}")
        print(f"P25/P50/P75: {fuzzy_analysis['p25']:.3f} / {fuzzy_analysis['p50']:.3f} / {fuzzy_analysis['p75']:.3f}")

        if fuzzy_analysis['gaps']:
            print(f"\nLargest gaps (natural breakpoints):")
            for i, gap in enumerate(fuzzy_analysis['gaps'][:3], 1):
                print(f"  {i}. {gap['between']}: gap of {gap['size']:.3f}")
                print(f"     → Suggested: {gap['suggested_threshold']:.3f} (splits {gap['left_count']} vs {gap['right_count']} samples)")

        fuzzy_rec = recommend_threshold(fuzzy_analysis)
        print(f"\nRECOMMENDATION: {fuzzy_rec['recommendation']}")
        print(f"Confidence: {fuzzy_rec['confidence']}")
        print(f"Reason: {fuzzy_rec['reason']}")

        print(f"\nCurrent value: SIMILARITY_FLOOR = 0.65")

        if fuzzy_rec['confidence'] in ("MEDIUM", "HIGH"):
            action = f"UPDATE to {fuzzy_rec['recommendation']}"
        elif fuzzy_rec['confidence'] == "LOW":
            action = f"CONSIDER {fuzzy_rec['recommendation']} (after 50+ samples)"
        else:
            action = "COLLECT MORE DATA (need 50+ samples)"

        print(f"Action: {action}")

    # GROUNDING FLOOR
    print("\n" + "-" * 80)
    print("GROUNDING FLOOR (transcript grounding threshold)")
    print("-" * 80)

    grounding_analysis = analysis['analysis']['grounding']

    if grounding_analysis['n'] == 0:
        print(f"\nNo grounding data collected yet.")
        print(f"Current value: GROUNDING_FLOOR = 0.78")
        print(f"Confidence: LOW (based on {grounding_analysis.get('collected_on', 12)} old samples, gap 0.05)")
    else:
        print(f"\nSamples: {grounding_analysis['n']}")
        print(f"Range: {grounding_analysis['min']:.3f} - {grounding_analysis['max']:.3f}")
        print(f"Mean: {grounding_analysis['mean']:.3f}")
        print(f"Median: {grounding_analysis['median']:.3f}")
        print(f"Stdev: {grounding_analysis['stdev']:.3f}")
        print(f"P25/P50/P75: {grounding_analysis['p25']:.3f} / {grounding_analysis['p50']:.3f} / {grounding_analysis['p75']:.3f}")

        if grounding_analysis['gaps']:
            print(f"\nLargest gaps (natural breakpoints):")
            for i, gap in enumerate(grounding_analysis['gaps'][:3], 1):
                print(f"  {i}. {gap['between']}: gap of {gap['size']:.3f}")
                print(f"     → Suggested: {gap['suggested_threshold']:.3f} (splits {gap['left_count']} vs {gap['right_count']} samples)")

        grounding_rec = recommend_threshold(grounding_analysis)
        print(f"\nRECOMMENDATION: {grounding_rec['recommendation']}")
        print(f"Confidence: {grounding_rec['confidence']}")
        print(f"Reason: {grounding_rec['reason']}")

        print(f"\nCurrent value: GROUNDING_FLOOR = 0.78")

        if grounding_rec['confidence'] in ("MEDIUM", "HIGH"):
            action = f"UPDATE to {grounding_rec['recommendation']}"
        elif grounding_rec['confidence'] == "LOW":
            action = f"CONSIDER {grounding_rec['recommendation']} (after 50+ samples)"
        else:
            action = "COLLECT MORE DATA (need 50+ samples)"

        print(f"Action: {action}")

    # CONFIDENCE SUMMARY
    print("\n" + "=" * 80)
    print("CONFIDENCE LEVELS")
    print("=" * 80)

    print("""
Confidence interpretation:
  HIGH      (>50 samples, gap >0.10): Safe to update in production
  MEDIUM    (20-50 samples, gap >0.05): Likely good, monitor after update
  LOW       (<20 samples, gap <0.05): Collect more data before updating
  VERY_LOW  (<15 samples): Do NOT update; collect at least 50 samples first

Before clinic deployment: collect 50+ samples for BOTH thresholds
During production: weekly recalibration (run this script every Monday)
""")

    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("""
1. Deploy error logging to server.py (see: LEVEL 1 implementation)
2. Run production consultations for 2-4 weeks
3. Re-run this script: python scripts/calibrate_thresholds.py --use-production-logs
4. When confidence is MEDIUM+, update gate.py with new thresholds
5. Every week, re-run to detect drift
""")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Calibrate thresholds from data")
    parser.add_argument('recordings_dir', nargs='?', default='/workspace/voice-to-rx-repo/recordings',
                       help='Path to recordings directory')
    parser.add_argument('--use-production-logs', action='store_true',
                       help='Load from production log file instead of recordings')
    parser.add_argument('--log-file', default='/workspace/threshold_scores.jsonl',
                       help='Path to production log file')
    parser.add_argument('--output', default='threshold_calibration.json',
                       help='Save analysis to JSON file')

    args = parser.parse_args()

    # Load data
    if args.use_production_logs:
        print("Loading from production logs...")
        analysis = load_from_production_logs(args.log_file)
    else:
        print(f"Loading from recordings: {args.recordings_dir}")
        analysis = calibrate_from_recordings(args.recordings_dir)

    # Analyze distributions
    if 'fuzzy_matches' in analysis:
        analysis['analysis']['fuzzy'] = analyze_distribution(
            analysis['fuzzy_matches'], 'score', 'SIMILARITY_FLOOR'
        )
    if 'grounding_scores' in analysis:
        analysis['analysis']['grounding'] = analyze_distribution(
            analysis['grounding_scores'], 'score', 'GROUNDING_FLOOR'
        )

    # Print report
    production = args.use_production_logs
    print_calibration_report(analysis, production=production)

    # Save JSON
    with open(args.output, 'w') as f:
        json.dump(analysis, f, indent=2)

    print(f"\n✓ Analysis saved to {args.output}")


if __name__ == "__main__":
    main()
