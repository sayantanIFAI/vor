#!/usr/bin/env python3
"""
WEEKLY ERROR ANALYSIS: Categorize corrections, find patterns.

Run: python scripts/analyze_errors.py

Reads /workspace/error_log.jsonl (doctor corrections) and categorizes each error:
- ASR: IndicConformer misheared (both are real drugs, different names)
- GAZETTEER: System missed a spelling variant
- GATE: Threshold or logic error
- QWEN: Extraction model error

Generates a report with patterns and recommendations.
"""

import json
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Optional

# Try to import glossary, but don't fail if not available
try:
    from voicerx.glossary import DRUG_LOOKUP, fold
    HAVE_GLOSSARY = True
except ImportError:
    HAVE_GLOSSARY = False
    DRUG_LOOKUP = {}


def categorize_error(error: dict) -> str:
    """Determine WHERE the error came from.

    ASR: System said drug X, should have been Y (both are real drugs)
         → IndicConformer misheared two similar drug names

    Gazetteer: System didn't recognize the drug, or recognized it wrong
               → Missing spelling variant or alias

    Gate: System rejected a real drug, or accepted a non-drug
          → Threshold or logic error in gate.py

    Qwen: System extracted wrong field or hallucinated
          → Extraction model error
    """

    system_drug = error.get("what_system_said", "").strip().lower()
    correct_drug = error.get("what_doctor_said", "").strip().lower()

    if not system_drug or not correct_drug:
        return "UNKNOWN"

    if not HAVE_GLOSSARY:
        # Without glossary, guess based on field presence
        if error.get("dose_correction"):
            return "GATE"
        return "ASR"

    # Check if drugs are in lookup
    system_in_lookup = system_drug in DRUG_LOOKUP
    correct_in_lookup = correct_drug in DRUG_LOOKUP

    if system_in_lookup and correct_in_lookup:
        # Both real drugs; this is ASR confusion
        return "ASR"

    if not system_in_lookup and correct_in_lookup:
        # System invented/hallucinated something
        if error.get("dose_correction"):
            # Dose was wrong = extraction issue
            return "QWEN"
        else:
            # Medicine was wrong = gate issue (should have rejected)
            return "GATE"

    if system_in_lookup and not correct_in_lookup:
        # System said a real drug, but doctor says it's wrong
        # This is odd; likely user error in the correction
        return "UNKNOWN"

    # Neither in lookup
    if correct_in_lookup:
        # Doctor's correction IS a real drug, system missed it
        return "GAZETTEER"

    return "UNKNOWN"


def load_errors_since(days: int = 7, log_file: str = "/workspace/error_log.jsonl") -> list:
    """Load all errors from past N days."""

    log_path = Path(log_file)
    if not log_path.exists():
        return []

    errors = []
    cutoff = datetime.now() - timedelta(days=days)

    try:
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    error = json.loads(line)
                    ts_str = error.get("timestamp", "")
                    if ts_str:
                        ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                        if ts > cutoff:
                            error["category"] = categorize_error(error)
                            errors.append(error)
                except (json.JSONDecodeError, ValueError):
                    pass
    except IOError:
        return []

    return errors


def analyze_weekly_errors(days: int = 7) -> dict:
    """Analyze all errors from past N days."""

    errors = load_errors_since(days=days)

    if not errors:
        return {
            "total_errors": 0,
            "by_category": {},
            "details": {},
            "week_of": (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        }

    by_category = defaultdict(list)
    for error in errors:
        category = error.get("category", "UNKNOWN")
        by_category[category].append(error)

    return {
        "total_errors": len(errors),
        "by_category": {k: len(v) for k, v in by_category.items()},
        "details": dict(by_category),
        "week_of": (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d"),
        "cutoff_date": datetime.now().isoformat()
    }


def find_patterns(errors: list) -> dict:
    """Find repeated error pairs (same drug confused multiple times)."""

    pairs = defaultdict(int)
    for err in errors:
        pair = (
            err.get("what_system_said", "").strip().lower(),
            err.get("what_doctor_said", "").strip().lower()
        )
        if pair[0] and pair[1]:
            pairs[pair] += 1

    # Filter to patterns (2+ occurrences)
    patterns = {
        pair: count for pair, count in pairs.items() if count >= 2
    }

    return {
        "total_unique_pairs": len(pairs),
        "pattern_count": len(patterns),
        "patterns": sorted(
            [{"system": p[0], "correct": p[1], "count": c}
             for p, c in patterns.items()],
            key=lambda x: x["count"],
            reverse=True
        )
    }


def recommend_actions(analysis: dict) -> list:
    """Generate actionable recommendations."""

    recommendations = []

    for category, errors in analysis["details"].items():
        if len(errors) < 1:
            continue

        if category == "GAZETTEER" and len(errors) >= 1:
            recommendations.append({
                "action": "ADD_TO_GAZETTEER",
                "priority": "HIGH" if len(errors) >= 3 else "MEDIUM",
                "count": len(errors),
                "description": f"Add {len(errors)} new drug spellings to glossary",
                "examples": [
                    e.get("what_doctor_said") for e in errors[:3]
                ]
            })

        if category == "ASR" and len(errors) >= 2:
            patterns = find_patterns(errors)
            recommendations.append({
                "action": "ASR_RETRAIN_CANDIDATE",
                "priority": "HIGH" if patterns["pattern_count"] >= 3 else "MEDIUM",
                "count": len(errors),
                "description": f"IndicConformer confused these {len(errors)} times",
                "patterns": patterns.get("patterns", [])[:3]
            })

        if category == "GATE" and len(errors) >= 2:
            recommendations.append({
                "action": "RECALIBRATE_THRESHOLDS",
                "priority": "HIGH",
                "count": len(errors),
                "description": "Recalibrate SIMILARITY_FLOOR or GROUNDING_FLOOR",
                "note": "Run: python scripts/calibrate_thresholds.py --use-production-logs"
            })

        if category == "QWEN" and len(errors) >= 3:
            recommendations.append({
                "action": "QWEN_EXTRACTION_REVIEW",
                "priority": "MEDIUM",
                "count": len(errors),
                "description": f"Qwen extraction errors ({len(errors)} times)",
                "note": "May warrant fine-tuning after 30+ examples"
            })

    return sorted(recommendations, key=lambda x: x["priority"] == "HIGH", reverse=True)


def print_report(analysis: dict, patterns: Optional[dict] = None, recommendations: Optional[list] = None):
    """Generate human-readable weekly report."""

    print("\n" + "=" * 80)
    print(f"WEEKLY ERROR REPORT ({analysis['week_of']})")
    print("=" * 80)

    total = analysis["total_errors"]
    print(f"\nTotal corrections: {total}")

    if total == 0:
        print("No errors logged this week. Great job!")
        print("=" * 80)
        return

    print("\nBy category:")
    for category, count in sorted(
        analysis["by_category"].items(), key=lambda x: x[1], reverse=True
    ):
        errors = analysis["details"][category]
        pct = f" ({100*count//total}%)" if total else ""
        print(f"  {category}: {count}{pct}")

        # Show top 3 examples
        for i, err in enumerate(errors[:3], 1):
            system = err.get("what_system_said", "?").strip()
            correct = err.get("what_doctor_said", "?").strip()
            print(f"    {i}. '{system}' → '{correct}'")

        if len(errors) > 3:
            print(f"    ... and {len(errors) - 3} more")

    # Patterns
    if patterns and patterns.get("patterns"):
        print("\n" + "-" * 80)
        print("REPEATED ERRORS (Pattern Detection)")
        print("-" * 80)
        for pat in patterns["patterns"][:5]:
            print(f"  {pat['count']}× '{pat['system']}' → '{pat['correct']}'")

    # Recommendations
    if recommendations:
        print("\n" + "-" * 80)
        print("RECOMMENDED ACTIONS")
        print("-" * 80)
        for rec in recommendations[:5]:
            action = rec["action"]
            priority = rec["priority"]
            count = rec["count"]
            desc = rec["description"]
            print(f"\n  [{priority}] {action}")
            print(f"      {desc} ({count} errors)")

            if "examples" in rec:
                for ex in rec["examples"][:2]:
                    if ex:
                        print(f"        - {ex}")

            if "note" in rec:
                print(f"      Note: {rec['note']}")

    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("""
1. Review recommendations above
2. Address HIGH priority items this week
3. Next Monday: Re-run this script to track progress
4. Monthly: Decide on model retraining (if 30+ Qwen errors)

Keep error_log.jsonl for long-term trend analysis.
Every error is a data point for improvement.
    """)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Weekly error analysis")
    parser.add_argument("--days", type=int, default=7, help="Analyze past N days")
    parser.add_argument("--log-file", default="/workspace/error_log.jsonl")

    args = parser.parse_args()

    # Load and analyze
    analysis = analyze_weekly_errors(days=args.days)

    # Find patterns and recommend actions
    patterns = find_patterns(analysis["details"].get("ASR", []))
    recommendations = recommend_actions(analysis)

    # Print report
    print_report(analysis, patterns, recommendations)

    # Save JSON for programmatic use
    output_file = "weekly_error_analysis.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "analysis": analysis,
            "patterns": patterns,
            "recommendations": recommendations,
            "generated_at": datetime.now().isoformat()
        }, f, indent=2, ensure_ascii=False)

    print(f"✓ Analysis saved to {output_file}")


if __name__ == "__main__":
    main()
