"""Voice-to-Rx backend API.

Serves the React UI and runs the pipeline on uploaded audio.

Endpoints:
  GET  /                      -> the React app
  GET  /api/health            -> model/GPU status
  POST /api/transcribe        -> audio file in, structured prescription out
  GET  /api/results           -> previously processed consultations
  GET  /api/review/pending    -> segments flagged for human review
  POST /api/review/{seg_id}   -> save a human decision

The ASR model is loaded ONCE at startup (~10s) and reused. Loading it per
request would add 10s to every upload.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from voicerx.pipeline import VoiceToRxPipeline

APP_DIR = Path(__file__).parent
STATIC_DIR = APP_DIR / "ui" / "dist"
RESULTS_DIR = Path(os.environ.get("VOICERX_RESULTS", "/workspace/voicerx/results"))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
REVIEW_FILE = RESULTS_DIR / "human_review.json"

app = FastAPI(title="Voice-to-Rx")

# The UI may be served from a different origin during development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_pipeline: VoiceToRxPipeline | None = None


def get_pipeline() -> VoiceToRxPipeline:
    global _pipeline
    if _pipeline is None:
        print("[server] loading ASR model (one time, ~10s)...", flush=True)
        t0 = time.time()
        _pipeline = VoiceToRxPipeline()
        print(f"[server] model ready in {time.time()-t0:.1f}s", flush=True)
    return _pipeline


@app.on_event("startup")
def _warm():
    get_pipeline()


@app.get("/api/health")
def health():
    import torch
    return {
        "status": "ok",
        "cuda": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "model_loaded": _pipeline is not None,
    }


def _merge_segments(result) -> dict:
    """Roll per-segment extractions into one consultation-level record.

    Extraction runs per VAD segment (short context = less hallucination),
    but a clinician wants one prescription, not 20 fragments. Merge with
    de-duplication, and carry every review flag up so nothing gets hidden
    by the aggregation.
    """
    meds: list[dict] = []
    symptoms: list[str] = []
    labs: list[str] = []
    uncertain: list[str] = []
    candidates: list[str] = []
    rejected: list[str] = []
    summaries: list[str] = []
    diagnosis = None
    follow_up = None
    patient_name = None
    review_reasons: list[str] = []
    n_flagged = 0

    for rx in result.extractions:
        for m in rx.medications:
            if m.drug and not any(x["drug"] == m.drug for x in meds):
                meds.append({
                    "drug": m.drug, "dosage": m.dosage, "frequency": m.frequency,
                    "duration": m.duration, "verified": m.verified,
                    "review_reason": m.review_reason,
                    # The gate's verdict MUST survive the merge. This dict
                    # was written before gate.py existed and rebuilt each
                    # medication from a fixed field list, so tier and
                    # canonical were silently dropped - the gate ran per
                    # segment and its entire decision was thrown away here.
                    # A live run showed every medication arriving at the UI
                    # with tier=None, which is exactly the information the
                    # reviewer needs most.
                    "tier": m.tier, "canonical": m.canonical,
                    "department": m.department, "indication": m.indication,
                    "match_similarity": m.match_similarity,
                })
        # Terms the gate refused. Merged rather than dropped for the same
        # reason they are recorded at all: a silent deletion is
        # indistinguishable from a term that was never extracted, and if
        # the gate is WRONG this list is the only surviving evidence.
        for t in rx.rejected_terms:
            if t not in rejected:
                rejected.append(t)
        for s in rx.symptoms:
            if s not in symptoms:
                symptoms.append(s)
        for l in rx.labs_ordered:
            if l not in labs:
                labs.append(l)
        for u in rx.raw_uncertain_terms:
            if u not in uncertain:
                uncertain.append(u)
        for c in rx.drug_candidates:
            if c not in candidates:
                candidates.append(c)
        if rx.summary:
            summaries.append(rx.summary)
        if rx.diagnosis and not diagnosis:
            diagnosis = rx.diagnosis
        if rx.follow_up and not follow_up:
            follow_up = rx.follow_up
        if rx.patient_name and not patient_name:
            patient_name = rx.patient_name
        if rx.needs_human_review:
            n_flagged += 1
            review_reasons.extend(rx.review_reasons)

    return {
        "patient_name": patient_name,
        "symptoms": symptoms,
        "diagnosis": diagnosis,
        "labs_ordered": labs,
        "medications": meds,
        "follow_up": follow_up,
        "summary": " ".join(summaries) if summaries else None,
        "raw_uncertain_terms": uncertain,
        "drug_candidates": candidates,
        "rejected_terms": rejected,
        "segments_flagged": n_flagged,
        "segments_total": len(result.extractions),
        "review_reasons": sorted(set(review_reasons)),
    }


@app.post("/api/transcribe")
async def transcribe(file: UploadFile = File(...)):
    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    consult_id = f"{int(time.time())}_{uuid.uuid4().hex[:6]}"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    # Browsers record webm/ogg; the ASR needs 16k mono WAV.
    wav_path = tmp_path
    if suffix.lower() != ".wav":
        wav_path = tmp_path + ".wav"
        rc = os.system(f'ffmpeg -y -loglevel error -i "{tmp_path}" '
                       f'-ac 1 -ar 16000 -c:a pcm_s16le "{wav_path}"')
        if rc != 0 or not os.path.exists(wav_path):
            raise HTTPException(500, "audio conversion failed (ffmpeg)")

    try:
        t0 = time.time()
        result = get_pipeline().process_file(wav_path)
        merged = _merge_segments(result)
        merged["consult_id"] = consult_id
        merged["processing_s"] = round(time.time() - t0, 1)
        merged["timing"] = result.timing
        merged["segments"] = [
            {
                "start_s": s.start_s, "end_s": s.end_s,
                "raw_text": s.text, "corrected_text": s.corrected_text or s.text,
                "decoder_used": s.decoder_used,
                "decoder_agreement": s.decoder_agreement,
                "corrections_applied": s.corrections_applied,
            }
            for s in result.segments
        ]
        merged["per_segment_rx"] = [rx.model_dump() for rx in result.extractions]

        with open(RESULTS_DIR / f"{consult_id}.json", "w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)

        return JSONResponse(merged)
    finally:
        for p in {tmp_path, wav_path}:
            try:
                os.unlink(p)
            except OSError:
                pass


@app.get("/api/results")
def list_results():
    out = []
    for p in sorted(RESULTS_DIR.glob("*.json"), reverse=True):
        if p.name == "human_review.json":
            continue
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            out.append({
                "consult_id": d.get("consult_id", p.stem),
                "patient_name": d.get("patient_name"),
                "diagnosis": d.get("diagnosis"),
                "n_medications": len(d.get("medications", [])),
                "segments_flagged": d.get("segments_flagged", 0),
                "segments_total": d.get("segments_total", 0),
            })
        except Exception:
            continue
    return {"results": out}


@app.get("/api/results/{consult_id}")
def get_result(consult_id: str):
    p = RESULTS_DIR / f"{consult_id}.json"
    if not p.exists():
        raise HTTPException(404, "not found")
    return JSONResponse(json.loads(p.read_text(encoding="utf-8")))


@app.get("/api/review/pending")
def review_pending():
    """Every segment still awaiting a human decision, newest first."""
    decisions = {}
    if REVIEW_FILE.exists():
        decisions = json.loads(REVIEW_FILE.read_text(encoding="utf-8"))

    pending = []
    for p in sorted(RESULTS_DIR.glob("*.json"), reverse=True):
        if p.name == "human_review.json":
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        cid = d.get("consult_id", p.stem)
        for i, rx in enumerate(d.get("per_segment_rx", [])):
            if not rx.get("needs_human_review"):
                continue
            seg_id = f"{cid}#{i}"
            if seg_id in decisions:
                continue
            pending.append({
                "segment_id": seg_id,
                "consult_id": cid,
                "start_s": rx.get("segment_start_s"),
                "end_s": rx.get("segment_end_s"),
                "transcript": rx.get("source_transcript", ""),
                "review_reasons": rx.get("review_reasons", []),
                "medications": rx.get("medications", []),
                "symptoms": rx.get("symptoms", []),
                "drug_candidates": rx.get("drug_candidates", []),
                "raw_uncertain_terms": rx.get("raw_uncertain_terms", []),
            })
    return {"pending": pending, "reviewed": len(decisions)}


@app.post("/api/review/{segment_id:path}")
async def save_review(segment_id: str, decision: dict):
    decisions = {}
    if REVIEW_FILE.exists():
        decisions = json.loads(REVIEW_FILE.read_text(encoding="utf-8"))
    decisions[segment_id] = {
        **decision,
        "reviewed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    REVIEW_FILE.write_text(json.dumps(decisions, ensure_ascii=False, indent=2),
                            encoding="utf-8")
    return {"ok": True, "total_reviewed": len(decisions)}


# React app last so /api/* wins
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="ui")


@app.get("/")
def root():
    index = STATIC_DIR / "index.html"
    if index.exists():
        return FileResponse(index)
    return {"status": "backend running", "ui": "not built yet",
            "docs": "/docs", "health": "/api/health"}
