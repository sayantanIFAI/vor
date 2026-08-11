# VoiceToRx Continuous Learning System

**Purpose:** Convert clinic corrections into data for continuous improvement. Every doctor flag becomes a calibration point.

---

## Overview: 5 Levels of Learning

| Level | Timeline | Effort | Method | Payoff |
|-------|----------|--------|--------|--------|
| **1: Capture** | Week 1 | 2h | Doctor correction UI | Baseline data |
| **2: Analyze** | Week 2 | 2h | Weekly error script | Pattern visibility |
| **3: Fast Fixes** | Weekly | 15m | Gazetteer + recalibrate | 50-70% error reduction |
| **4: Deep Analysis** | Monthly | 4h | Systematic pattern search | Identify root causes |
| **5: Retrain** | Month 6+ | 2-4w | Model fine-tuning | Permanent improvement |

---

## Level 1: Doctor Correction Capture ✓

**Status:** Implemented in server.py

### API Endpoint

```http
POST /api/log-correction
Content-Type: application/json

{
  "consultation_id": "1692864000_a3f5b2",
  "medication_id": "0",
  "what_system_said": "Amoxycillin",
  "what_doctor_said": "Amoxicillin",
  "dose_correction": null,
  "timestamp": "2026-08-11T14:30:00Z"
}
```

**Response:**
```json
{
  "status": "logged",
  "message": "Correction recorded. Thank you for helping us improve."
}
```

### Storage

All corrections logged to `/workspace/error_log.jsonl` (one JSON per line).

Each line:
```json
{
  "consultation_id": "...",
  "medication_id": "0",
  "what_system_said": "Amoxycillin",
  "what_doctor_said": "Amoxicillin",
  "dose_correction": null,
  "timestamp": "2026-08-11T14:30:00Z",
  "error_type": "NEEDS_ANALYSIS"
}
```

---

## Level 2: Weekly Error Analysis ✓

**Status:** Implemented in scripts/analyze_errors.py

### Running the Report

```bash
python scripts/analyze_errors.py --days 7
```

**Output:** Human-readable report + JSON analysis file

### Error Categories

| Category | Cause | Fix |
|----------|-------|-----|
| **ASR** | IndicConformer mishears (both are real drugs, wrong one) | Consider fine-tuning after 10+ instances of same pair |
| **GAZETTEER** | System missed a spelling variant | Add new spelling to drug aliases (< 1 hour) |
| **GATE** | Threshold or logic error | Recalibrate SIMILARITY_FLOOR or GROUNDING_FLOOR |
| **QWEN** | Extraction model error (dose/field extracted wrong) | Review; fine-tune after 30+ instances |
| **UNKNOWN** | Unclear categorization | Manual review |

### Report Sections

1. **Summary:** Total errors by category
2. **Patterns:** Drug pairs that repeat 2+ times
3. **Recommendations:** Actionable fixes ranked by priority

---

## Level 3: Fast Fixes ✓

**Status:** Scripting implemented; manual action on recommendations

### Action 3a: Recalibrate Thresholds

When error count in GATE category exceeds 2:

```bash
python scripts/calibrate_thresholds.py --use-production-logs
```

**Output:** Recommends new SIMILARITY_FLOOR or GROUNDING_FLOOR if data confidence is MEDIUM+.

**Apply change (if recommended):**

1. Update `gate.py`:
   ```python
   SIMILARITY_FLOOR = 0.68  # Was 0.65 (13 samples, 0.06 gap)
   SIMILARITY_FLOOR_CONFIDENCE = "MEDIUM"  # Now 50 samples, 0.08 gap
   ```

2. Restart server:
   ```bash
   bash boot.sh
   ```

### Action 3b: Add to Gazetteer

When GAZETTEER errors appear:

1. Find the error in weekly_errors.txt
2. Extract the doctor's spelling: `what_doctor_said`
3. Add to `voicerx/glossary.py`:
   ```python
   Drug(
       generic="Amoxicillin",
       bengali=["অ্যামক্সিসিলিন", "অ্যামক্সিলিন"],  # ← ADD NEW VARIANT
       ...
   )
   ```
4. Restart server

### Action 3c: Add to _NEVER_A_DRUG

If non-drug terms keep getting proposed:

```python
_NEVER_A_DRUG = frozenset(fold(w) for w in (
    # existing words...
    "নতুন_শব্দ",  # ← ADD HERE
))
```

---

## Level 4: Deep Analysis (Monthly)

**Status:** Framework ready, manual application needed

### Pattern Detection

Run after 50+ errors in a single category:

```python
python scripts/analyze_errors.py --days 30 | grep "PATTERN"
```

**Decision tree:**

- **ASR patterns 10+:** Justify fine-tuning (needs 100+ hours audio)
- **GATE patterns 5+:** Systematic threshold drift
- **QWEN patterns 20+:** Consider extraction model retrain (200+ examples)

---

## Level 5: Model Retraining (Month 6+)

**Status:** Decision framework in place

### Trigger Criteria

After 6 months of production:

1. **IndicConformer fine-tuning:** If any single drug pair confused 10+ times
   - Effort: 2-4 weeks (needs 100+ hours labeled audio)
   - ROI: Permanent ASR improvement

2. **Qwen2.5 fine-tuning:** If 30+ structured extraction errors
   - Effort: 2-3 weeks (200+ labeled consultations)
   - ROI: Better extraction accuracy

3. **Gazetteer growth:** If >50 new spellings added
   - Effort: ~1 week (review, test, deploy)
   - ROI: Immediate error reduction

---

## Threshold Confidence Levels

### Current State

| Threshold | Value | Samples | Gap | Confidence | Action |
|-----------|-------|---------|-----|------------|--------|
| SIMILARITY_FLOOR | 0.65 | 13 | 0.06 | **LOW** | Monitor in production |
| GROUNDING_FLOOR | 0.72 | 12 | 0.05 | **LOW** | Monitor in production |
| MIN_SKELETON | 5 | 7 | ~1 char | **MEDIUM** | Safe to use |

### Confidence Definition

- **HIGH** (>50 samples, gap >0.10): Safe to deploy without monitoring
- **MEDIUM** (20-50 samples, gap >0.05): Deploy with weekly review
- **LOW** (<20 samples, gap <0.05): Collect more data; monitor closely
- **VERY_LOW** (<15 samples): Do NOT deploy; collect to 50+ first

### Recalibration Timeline

- **Week 1-4:** Collect 50 samples (SIMILARITY_FLOOR only)
- **Week 5-8:** Improve to MEDIUM confidence; consider update
- **Month 2+:** HIGH confidence; safe to deploy

---

## Automatic Production Logging ✓

**Status:** Implemented in server.py

### What Gets Logged

**To `/workspace/threshold_scores.jsonl`:**

Every medication decision:
```json
{
  "consultation_id": "...",
  "timestamp": "2026-08-11T14:35:22Z",
  "drug": "Amoxicillin",
  "tier": "probable",
  "similarity_score": 0.68,
  "verified": false,
  "review_reason": "not an exact match; ..."
}
```

**To `/workspace/error_log.jsonl`:**

Every doctor correction (via /api/log-correction):
```json
{
  "consultation_id": "...",
  "timestamp": "2026-08-11T14:36:00Z",
  "what_system_said": "Amoxycillin",
  "what_doctor_said": "Amoxicillin",
  "error_type": "NEEDS_ANALYSIS"
}
```

---

## Weekly Workflow

**Every Monday morning:**

```bash
bash weekly_improvement.sh
```

**Steps:**

1. Analyze error_log.jsonl (last 7 days)
2. Recalibrate thresholds using threshold_scores.jsonl
3. Generate human-readable report
4. Print recommendations

**Output files:**

- `reports/weekly_errors.txt` — Error summary
- `reports/threshold_report.txt` — Threshold recalibration
- `reports/weekly_error_analysis.json` — Structured data
- `reports/threshold_calibration.json` — Detailed threshold stats

**Time:** ~3 minutes

---

## Before Clinic Deployment

**Checklist:**

- [ ] Run Level 1 test (doctor correction UI works)
- [ ] Run Level 2 test (weekly analysis produces report)
- [ ] Verify Level 3 process (thresholds can be updated)
- [ ] Document Level 3 SOP (who reviews, who deploys)
- [ ] Set up Level 4 monitoring (script runs automatically)
- [ ] Establish Level 5 decision criteria (when to retrain)

**Timeline:** Must complete before first consultations.

---

## In Production: Monitoring Dashboard (Optional)

**If implementing monitoring UI, track:**

1. Total errors per week
2. Errors by category (ASR, GATE, etc.)
3. Confidence levels of active thresholds
4. Time since last threshold update
5. Model performance metrics (accuracy, false positives)

---

## Troubleshooting

### No errors being logged

**Check:**
```bash
ls -la /workspace/error_log.jsonl
```

If doesn't exist, doctor correction UI hasn't been used yet. Test manually:
```bash
curl -X POST http://localhost:8000/api/log-correction \
  -H "Content-Type: application/json" \
  -d '{"consultation_id": "test", "what_system_said": "X", "what_doctor_said": "Y"}'
```

### Threshold scores empty

**Check:**
```bash
ls -la /workspace/threshold_scores.jsonl
```

Verify server.py has logging enabled (line: `_log_threshold_scores(...)`).

### Script fails with "No such file"

Ensure scripts are run from repo root:
```bash
cd /workspace/voice-to-rx-repo
python scripts/calibrate_thresholds.py --use-production-logs
```

---

## References

- **Calibration Script:** `scripts/calibrate_thresholds.py`
- **Error Analysis:** `scripts/analyze_errors.py`
- **Weekly Automation:** `weekly_improvement.sh`
- **Threshold Documentation:** `voicerx/gate.py` (constants at top)
- **Logging Code:** `server.py` (functions: `_log_threshold_scores`, `/api/log-correction`)

---

## Questions?

This system is designed to be self-documenting. Every script has:
- Help text (pass `--help`)
- Docstrings explaining logic
- Inline comments for non-obvious decisions
- JSON output for programmatic use

Start with the weekly script, read the report, and follow its recommendations.
