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
    unconfirmed: list[str] = []
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
        # Symptoms the transcript does not support. Carried up separately
        # so the merge cannot quietly promote a hallucination into the
        # confirmed list.
        for s in rx.symptoms_unconfirmed:
            if s not in unconfirmed and s not in symptoms:
                unconfirmed.append(s)
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
        "symptoms_unconfirmed": unconfirmed,
        "segments_flagged": n_flagged,
        "segments_total": len(result.extractions),
        "review_reasons": sorted(set(review_reasons)),
    }


# ===========================================================================
# STREAMING CAPTURE
# ===========================================================================
# /api/transcribe processes everything AFTER the upload, so a 32-segment
# consultation takes ~62s of staring at a spinner - the LLM is ~2s a segment
# and none of it starts until the doctor stops talking.
#
# The consultation itself lasts minutes, so the work is moved into that
# window. Chunks arrive during recording and are transcribed and extracted
# immediately; by the time the doctor stops, only the final chunk is
# outstanding. Measured budget: ~2.5s of compute per 10s of audio, a ~4x
# margin, so processing never falls behind real time.
#
# TWO THINGS THAT MAKE THIS NON-TRIVIAL, both handled below:
#
# 1. MediaRecorder chunks are NOT independently decodable. Only the first
#    carries the container header, so running ffmpeg on chunk 2 alone
#    fails. Bytes are therefore appended to one growing file which is
#    re-decoded whole each time. Re-decoding is cheap next to ASR+LLM.
#
# 2. A chunk boundary can fall mid-word. Segments ending within
#    TAIL_GUARD_S of the end of the audio received so far are LEFT for the
#    next chunk, because a drug name sliced in half is precisely the error
#    this pipeline can least afford.
#
# 3. VAD boundaries MOVE as audio arrives, so each pass must see only fresh
#    audio - see _process_slice(). Re-running VAD over the whole growing
#    file orphaned any segment straddling the processed boundary, and a
#    real 3-minute recording yielded 2 segments instead of 37.

TAIL_GUARD_S = 1.5          # never process audio this close to the live edge
MIN_SLICE_S = 4.0           # don't bother VAD-ing a sliver
SESSION_TTL_S = 3600

_sessions: dict[str, dict] = {}


def _decode_to_wav(raw_path: str, wav_path: str) -> bool:
    rc = os.system(f'ffmpeg -y -loglevel error -i "{raw_path}" '
                   f'-ac 1 -ar 16000 -c:a pcm_s16le "{wav_path}"')
    return rc == 0 and os.path.exists(wav_path) and os.path.getsize(wav_path) > 44


def _wav_duration_s(wav_path: str) -> float:
    import wave
    try:
        with wave.open(wav_path, "rb") as w:
            return w.getnframes() / float(w.getframerate())
    except Exception:                                  # noqa: BLE001
        return 0.0


def _reap_sessions() -> None:
    now = time.time()
    for sid in [s for s, v in _sessions.items() if now - v["created"] > SESSION_TTL_S]:
        _drop_session(sid)


def _drop_session(sid: str) -> None:
    sess = _sessions.pop(sid, None)
    if not sess:
        return
    for key in ("raw_path", "wav_path"):
        try:
            os.unlink(sess[key])
        except OSError:
            pass


def _process_slice(sess: dict, start_s: float, end_s: float) -> int:
    """VAD + transcribe + extract ONLY the audio between start_s and end_s.

    THE BUG THIS FIXES
    ------------------
    The first version re-ran VAD over the whole growing file each chunk and
    kept segments with start_s >= processed_until_s. That silently dropped
    most of the consultation: Silero's boundaries MOVE as more audio
    arrives, so a segment that straddled processed_until_s was excluded on
    every subsequent pass and never processed at all. A 3-minute live
    recording produced 2 segments.

    Cutting the unprocessed region out first means VAD only ever sees fresh
    audio, so no segment can straddle the boundary. Timestamps are shifted
    back to consultation time afterwards, so the UI still shows real
    offsets.
    """
    import soundfile as sf

    audio, sr = sf.read(sess["wav_path"])
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1)
    a, b = int(start_s * sr), int(end_s * sr)
    if b - a < int(MIN_SLICE_S * sr):
        return 0

    slice_path = sess["wav_path"] + f".slice_{int(start_s * 100)}.wav"
    sf.write(slice_path, audio[a:b], sr)
    try:
        result = get_pipeline().process_file(slice_path)
    finally:
        try:
            os.unlink(slice_path)
        except OSError:
            pass

    # Shift back into consultation time.
    for seg in result.segments:
        seg.start_s = round(seg.start_s + start_s, 2)
        seg.end_s = round(seg.end_s + start_s, 2)
    for rx in result.extractions:
        if rx.segment_start_s is not None:
            rx.segment_start_s = round(rx.segment_start_s + start_s, 2)
        if rx.segment_end_s is not None:
            rx.segment_end_s = round(rx.segment_end_s + start_s, 2)

    sess["segments"].extend(result.segments)
    sess["extractions"].extend(result.extractions)
    # Advance by the slice we consumed, NOT by the last segment's end.
    # Trailing silence carries no segment, and advancing only to the last
    # segment would re-process that silence forever.
    sess["processed_until_s"] = end_s
    return len(result.segments)


class _Accum:
    """Quacks like PipelineResult so _merge_segments works unchanged."""

    def __init__(self, segments, extractions, timing):
        self.segments = segments
        self.extractions = extractions
        self.timing = timing


@app.post("/api/session/start")
def session_start():
    _reap_sessions()
    sid = f"{int(time.time())}_{uuid.uuid4().hex[:6]}"
    raw = tempfile.NamedTemporaryFile(delete=False, suffix=".webm")
    raw.close()
    _sessions[sid] = {
        "created": time.time(),
        "raw_path": raw.name,
        "wav_path": raw.name + ".wav",
        "processed_until_s": 0.0,
        "segments": [],
        "extractions": [],
        "chunks": 0,
        "compute_s": 0.0,
    }
    get_pipeline()                                     # warm before recording
    return {"session_id": sid, "chunk_seconds": 10}


@app.post("/api/session/{sid}/chunk")
async def session_chunk(sid: str, file: UploadFile = File(...)):
    sess = _sessions.get(sid)
    if not sess:
        raise HTTPException(404, "unknown or expired session")

    with open(sess["raw_path"], "ab") as f:
        f.write(await file.read())
    sess["chunks"] += 1

    if not _decode_to_wav(sess["raw_path"], sess["wav_path"]):
        # Early chunks can be too short to decode. Not an error - the bytes
        # are retained and will decode once more audio arrives.
        return {"ok": True, "segments_done": len(sess["segments"]), "pending": True}

    duration = _wav_duration_s(sess["wav_path"])
    safe_until = duration - TAIL_GUARD_S
    if safe_until - sess["processed_until_s"] < MIN_SLICE_S:
        return {"ok": True, "segments_done": len(sess["segments"]), "pending": True}

    t0 = time.time()
    processed = _process_slice(sess, sess["processed_until_s"], safe_until)
    sess["compute_s"] += time.time() - t0

    return {
        "ok": True,
        "segments_done": len(sess["segments"]),
        "audio_s": round(duration, 1),
        "compute_s": round(sess["compute_s"], 1),
        "pending": False,
    }


@app.post("/api/session/{sid}/finalize")
def session_finalize(sid: str):
    sess = _sessions.get(sid)
    if not sess:
        raise HTTPException(404, "unknown or expired session")

    t0 = time.time()
    # Drain whatever is left, including the tail that was held back.
    if _decode_to_wav(sess["raw_path"], sess["wav_path"]):
        duration = _wav_duration_s(sess["wav_path"])
        if duration - sess["processed_until_s"] > 0.3:
            _process_slice(sess, sess["processed_until_s"], duration)

    acc = _Accum(sess["segments"], sess["extractions"],
                  {"streamed": True, "chunks": sess["chunks"],
                   "compute_s": round(sess["compute_s"], 1)})
    merged = _merge_segments(acc)
    consult_id = sid
    merged["consult_id"] = consult_id
    # The number the clinician actually experiences: time from pressing
    # Stop to seeing a prescription. Everything before that was overlapped
    # with the consultation itself.
    merged["click_to_result_s"] = round(time.time() - t0, 2)
    merged["timing"] = acc.timing
    merged["segments"] = [
        {
            "start_s": s.start_s, "end_s": s.end_s,
            "raw_text": s.text, "corrected_text": s.corrected_text or s.text,
            "decoder_used": s.decoder_used,
            "decoder_agreement": s.decoder_agreement,
            "corrections_applied": s.corrections_applied,
        }
        for s in sess["segments"]
    ]
    merged["per_segment_rx"] = [rx.model_dump() for rx in sess["extractions"]]

    with open(RESULTS_DIR / f"{consult_id}.json", "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    _drop_session(sid)
    return JSONResponse(merged)


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
