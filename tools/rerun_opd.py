"""Re-process the real OPD recordings and build a correction worksheet.

RUNS ENTIRELY ON THE POD. It reads audio from disk and calls the pipeline
in-process - no HTTP, no upload, no local compute. Point it at the audio
directory and it does ASR, extraction, the gate and the gazetteer on the
pod's GPU, then writes one worksheet for a clinician to mark up.

    python tools/rerun_opd.py /workspace/opd_audio /workspace/opd_out

WHAT THE WORKSHEET IS FOR
-------------------------
The previous round of this worksheet is what caught every defect fixed
since: "Aspirin" shown for a spoken "Ecosprin", "Rasu Basta Tin" printed
as a drug, symptoms marked NOT SAID that were said. So the columns are
built around being CORRECTABLE, not around looking finished:

  *_FOUND            what the system produced
  *_MISSING_or_WRONG blank, for the reviewer to fill
  transcript         the ASR text the answer must be checkable against

WHY IT SHOWS THE PRINTED NAME
-----------------------------
An earlier worksheet listed `canonical`, so a correct result - spoken
"Ecosprin", molecule Aspirin - appeared as a drug substitution and was
reported as a bug that did not exist. The worksheet must show exactly
what a doctor would see on the prescription, which is prescribed_name,
with the spoken form beside it when the two differ.
"""
from __future__ import annotations

import io
import json
import os
import sys
import time
import traceback

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

AUDIO_EXT = (".m4a", ".mp3", ".wav", ".ogg", ".flac")


def _sort_key(name: str):
    """'My recording 32.m4a' -> 32, so output is in recording order."""
    digits = "".join(c if c.isdigit() else " " for c in name).split()
    return (int(digits[-1]) if digits else 0, name)


def main() -> None:
    audio_dir = sys.argv[1] if len(sys.argv) > 1 else "/workspace/opd_audio"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "/workspace/opd_out"
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(
        (f for f in os.listdir(audio_dir) if f.lower().endswith(AUDIO_EXT)),
        key=_sort_key,
    )
    if not files:
        print(f"no audio in {audio_dir}")
        sys.exit(1)

    # Imported here, after the path is set, so a missing GPU dependency
    # reports against this script rather than at module import.
    from voicerx.pipeline import VoiceToRxPipeline

    # Built ONCE. The ASR model is ~2GB on the GPU and takes tens of
    # seconds to load; constructing it per file would dominate the run.
    print("loading ASR...", flush=True)
    pipeline = VoiceToRxPipeline()

    print(f"{len(files)} recordings from {audio_dir}\n")
    worksheet: list[dict] = []
    t_start = time.time()

    for idx, fname in enumerate(files, 1):
        path = os.path.join(audio_dir, fname)
        print(f"[{idx}/{len(files)}] {fname}", flush=True)
        try:
            result = pipeline.process_file(path)
        except Exception:                      # noqa: BLE001
            # One bad recording must not lose the other fifteen.
            print(f"    FAILED\n{traceback.format_exc()}", flush=True)
            worksheet.append({"recording": fname, "ERROR": traceback.format_exc(limit=3)})
            continue

        rx = _merge(result)
        stem = os.path.splitext(fname)[0].replace(" ", "_")
        with open(os.path.join(out_dir, f"{stem}.json"), "w", encoding="utf-8") as f:
            json.dump(rx, f, ensure_ascii=False, indent=2)

        row = _worksheet_row(fname, rx)
        worksheet.append(row)
        print(f"    dx={row['diagnosis_FOUND'] or '-'}  "
              f"meds={len(row['medications_FOUND'])}  "
              f"labs={len(row['labs_FOUND'])}  "
              f"sx={len(row['symptoms_FOUND'])}", flush=True)

    out = os.path.join(out_dir, "CORRECTION_WORKSHEET.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(worksheet, f, ensure_ascii=False, indent=2)

    print(f"\n{len(worksheet)} recordings in {time.time() - t_start:.0f}s")
    print(f"worksheet: {out}")


def _merge(result) -> dict:
    """Flatten per-segment extractions into one consultation record.

    Mirrors server.py's merge so the worksheet shows what the UI shows -
    including the medication de-duplication, without which one molecule
    appears twice and reads as a double dose.
    """
    meds: list[dict] = []
    symptoms: list[str] = []
    labs: list[str] = []
    uncertain: list[str] = []
    advice: list[str] = []
    rejected: list[str] = []
    diagnosis = None
    transcript_parts: list[str] = []

    for seg in getattr(result, "segments", []) or []:
        text = getattr(seg, "corrected_text", "") or getattr(seg, "text", "")
        if text:
            transcript_parts.append(text)

    for rx in result.extractions:
        for m in rx.medications:
            key = (m.canonical or m.drug).strip().lower()
            existing = next(
                (x for x in meds
                 if (x["canonical"] or x["printed"]).strip().lower() == key), None)
            if existing is not None:
                if m.tier == "verified" and not existing["verified"]:
                    existing["verified"] = True
                    existing["tier"] = m.tier
                    if m.prescribed_name:
                        existing["printed"] = m.prescribed_name
                        existing["heard_as"] = m.heard_as
                for fld in ("dosage", "frequency", "duration"):
                    if not existing[fld] and getattr(m, fld, ""):
                        existing[fld] = getattr(m, fld)
                continue
            if m.drug:
                meds.append({
                    "printed": m.prescribed_name or m.drug,
                    "heard_as": m.heard_as,
                    "canonical": m.canonical,
                    "dosage": m.dosage, "frequency": m.frequency,
                    "duration": m.duration,
                    "route": m.route, "instructions": m.instructions,
                    "verified": m.verified, "tier": m.tier,
                    "review_reason": m.review_reason,
                })
        for s in rx.symptoms:
            if s not in symptoms:
                symptoms.append(s)
        for lb in rx.labs_ordered:
            if lb not in labs:
                labs.append(lb)
        for a in rx.advice:
            if a not in advice:
                advice.append(a)
        for u in rx.raw_uncertain_terms:
            if u not in uncertain:
                uncertain.append(u)
        for r in rx.rejected_terms:
            if r not in rejected:
                rejected.append(r)
        if rx.diagnosis and not diagnosis:
            diagnosis = rx.diagnosis

    return {
        "diagnosis": diagnosis,
        "medications": meds,
        "labs_ordered": labs,
        "advice": advice,
        "symptoms": symptoms,
        "raw_uncertain_terms": uncertain,
        "rejected_terms": rejected,
        "transcript": " ".join(transcript_parts),
    }


def _worksheet_row(fname: str, rx: dict) -> dict:
    """One reviewable row. Medications carry provenance inline, because a
    name shown without what was actually said cannot be checked."""
    med_lines = []
    for m in rx["medications"]:
        line = m["printed"]
        dosing = " ".join(x for x in (m["dosage"], m["frequency"], m["duration"]) if x)
        if dosing:
            line += f"  [{dosing}]"
        # Route and as-needed instructions are part of the prescription -
        # sublingual nitrate is taken WHEN the pain starts, which no
        # schedule field can express.
        if m.get("route"):
            line += f"  ({m['route']})"
        if m.get("instructions"):
            line += f"  — {m['instructions']}"
        if m["heard_as"]:
            line += f"   (heard: {m['heard_as']})"
        if not m["verified"]:
            line += "   ** CONFIRM **"
        elif "CONFIRM the amount" in (m.get("review_reason") or ""):
            line += "   ** CONFIRM DOSE **"
        med_lines.append(line)

    return {
        "recording": fname,
        "diagnosis_FOUND": rx["diagnosis"] or "",
        "diagnosis_CORRECT": "",
        "medications_FOUND": med_lines,
        "medications_MISSING_or_WRONG": "",
        "labs_FOUND": rx["labs_ordered"],
        "advice_FOUND": rx.get("advice", []),
        "advice_MISSING_or_WRONG": "",
        "labs_MISSING_or_WRONG": "",
        "symptoms_FOUND": rx["symptoms"],
        "symptoms_MISSING_or_WRONG": "",
        "uncertain_terms": rx["raw_uncertain_terms"],
        "rejected_as_meds": rx["rejected_terms"],
        "transcript": rx["transcript"],
    }


if __name__ == "__main__":
    main()
