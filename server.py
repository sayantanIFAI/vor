"""Voice-to-Rx backend API.

Serves the React UI and runs the pipeline on uploaded audio.

Endpoints:
  GET  /                      -> the React app
  GET  /api/health            -> model/GPU status
  POST /api/transcribe        -> audio file in, structured prescription out
  GET  /api/results           -> previously processed consultations
  GET  /api/review/pending    -> segments flagged for human review
  POST /api/review/{seg_id}   -> save a human decision
  POST /api/log-correction    -> doctor flags a medication error

The ASR model is loaded ONCE at startup (~10s) and reused. Loading it per
request would add 10s to every upload.

THRESHOLD LOGGING:
Every medication decision is logged to threshold_scores.jsonl for periodic
recalibration. Run: python scripts/calibrate_thresholds.py --use-production-logs
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

try:
    from voicerx.pipeline import VoiceToRxPipeline
    PIPELINE_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as e:
    print(f"[server] Warning: Pipeline not available ({e}), using fallback", flush=True)
    PIPELINE_AVAILABLE = False
    VoiceToRxPipeline = None

APP_DIR = Path(__file__).parent
STATIC_DIR = APP_DIR / "ui" / "dist"
RESULTS_DIR = Path(os.environ.get("VOICERX_RESULTS", "/workspace/voicerx/results"))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
REVIEW_FILE = RESULTS_DIR / "human_review.json"
THRESHOLD_LOG = Path("/workspace/threshold_scores.jsonl")
ERROR_LOG = Path("/workspace/error_log.jsonl")

app = FastAPI(title="Voice-to-Rx")

# Setup threshold logging
threshold_logger = logging.getLogger("thresholds")
threshold_logger.setLevel(logging.INFO)
handler = logging.FileHandler(str(THRESHOLD_LOG))
handler.setFormatter(logging.Formatter('%(message)s'))
threshold_logger.addHandler(handler)

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
        if not PIPELINE_AVAILABLE:
            print("[server] Pipeline unavailable, returning mock", flush=True)
            return None
        print("[server] loading ASR model (one time, ~10s)...", flush=True)
        t0 = time.time()
        try:
            _pipeline = VoiceToRxPipeline()
            print(f"[server] model ready in {time.time()-t0:.1f}s", flush=True)
        except Exception as e:
            print(f"[server] Pipeline failed to load: {e}", flush=True)
            return None
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


def _log_threshold_scores(consult_id: str, merged: dict) -> None:
    """Log every medication's scores for threshold calibration.

    Each line is a JSON object: consultation_id, drug, tier, scores...
    Used by scripts/calibrate_thresholds.py to recalibrate SIMILARITY_FLOOR
    and GROUNDING_FLOOR based on production data.
    """
    try:
        for med in merged.get("medications", []):
            entry = {
                "consultation_id": consult_id,
                "timestamp": datetime.now().isoformat(),
                "drug": med.get("canonical") or med.get("drug", ""),
                "tier": med.get("tier"),
                "similarity_score": med.get("match_similarity"),
                "verified": med.get("verified"),
                "review_reason": med.get("review_reason"),
            }
            threshold_logger.info(json.dumps(entry))
    except Exception as e:
        print(f"[server] threshold logging error: {e}", flush=True)


# Conditions too broad to stand as the diagnosis when a specific one was
# also named. "fungal infection" beside "pityriasis versicolor" is the same
# finding twice, and the vaguer of the two is not what goes on a script.
_GENERIC_DIAGNOSES = frozenset({"infection", "fungal infection", "severe allergy"})

# What a GUESS must score to stand on a consultation whose specialty nothing
# could establish. Same bar as validate._GENERAL_FLOOR and for the same
# reason: where no second check is possible, the first one carries alone.
_UNKNOWN_DEPT_FLOOR = 0.80

# Words that name a FORM or ROUTE rather than a different substance.
_FORM_WORDS = frozenset({
    "eye", "ear", "nasal", "oral", "topical", "drops", "drop", "spray",
    "gel", "cream", "ointment", "paste", "lotion", "solution", "syrup",
    "tablet", "tablets", "capsule", "capsules", "injection", "inhaler",
    "suspension", "shampoo", "powder", "sachet",
})


def _molecule_key(name: str) -> str:
    """Identity of the SUBSTANCE, with formulation wording removed.

    The dedup below keyed on the full canonical, so the gazetteer's
    "Tobramycin eye drops" and the brand register's "Tobramycin" - one
    molecule, reached through two tables - were printed as two lines of
    the same antibiotic on one eye prescription. That is the double-dose
    risk the dedup exists to prevent, arriving through the door it left
    open.

    Only trailing FORM words are stripped, never a second active
    ingredient: "Amoxicillin+Clavulanate" is a different product from
    "Amoxicillin" and must stay its own line.
    """
    parts = (name or "").strip().lower().replace("+", " + ").split()
    while parts and parts[-1] in _FORM_WORDS:
        parts.pop()
    return " ".join(parts) or (name or "").strip().lower()


def _derive_summary(symptoms, diagnosis, labs, meds, advice, follow_up) -> str | None:
    """Build the summary from validated fields only.

    Deterministic on purpose. A summary is read as the clinical record, so
    it must not be able to introduce a claim - and free narrative from the
    extraction model demonstrably can. Everything below has already passed
    the gate, corroboration, or a gazetteer scan; nothing new is asserted
    here, and nothing is inferred.

    An unverified medication is marked, rather than being described in the
    same voice as a confirmed one. "Aceclofenac (to confirm)" tells a
    reviewer where to look; a fluent sentence hides it.
    """
    parts: list[str] = []
    if symptoms:
        parts.append("Reported: " + ", ".join(symptoms) + ".")
    if diagnosis:
        parts.append(f"Assessment: {diagnosis}.")
    if labs:
        parts.append("Investigations: " + ", ".join(labs) + ".")
    if meds:
        named = []
        for m in meds:
            name = m.get("prescribed_name") or m.get("drug") or ""
            if not name:
                continue
            bits = [name]
            for k in ("dosage", "frequency", "duration"):
                if m.get(k):
                    bits.append(m[k])
            if m.get("route"):
                bits.append(m["route"])
            line = " ".join(bits)
            if not m.get("verified"):
                line += " (to confirm)"
            named.append(line)
        if named:
            parts.append("Prescribed: " + "; ".join(named) + ".")
    if advice:
        parts.append("Advice: " + "; ".join(advice) + ".")
    if follow_up:
        parts.append(f"Follow-up: {follow_up}.")
    return " ".join(parts) if parts else None


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
    advice: list[str] = []
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
            # De-duplicate on the CANONICAL molecule, not the spoken string.
            # A live consultation produced both "Rasu Basta Tin" and
            # "Rosuvastatin" as separate rows - the same drug, once as the
            # SLM heard it and once as the gazetteer found it - and earlier
            # "Salbutamol" appeared twice for the same reason. Two lines for
            # one molecule is a double-dose risk on a printed prescription.
            key = _molecule_key(m.canonical or m.drug)
            existing = next((x for x in meds
                             if _molecule_key(x.get("canonical") or x["drug"]) == key), None)
            if existing is not None:
                # Prefer the form the clinician actually SAID over a generic
                # the gazetteer supplied, and keep any dosing already found.
                if m.tier == "verified" and not existing.get("verified"):
                    existing.update({"verified": True, "tier": m.tier})
                    # The verified row knows a real spoken name; the probable
                    # one only had a similarity guess. Take its printed name.
                    if m.prescribed_name:
                        existing["prescribed_name"] = m.prescribed_name
                        existing["heard_as"] = m.heard_as
                # Keep the FORM. "Tobramycin eye drops" and "Tobramycin"
                # are the same molecule, and the row that names the route
                # is the one a pharmacist can fill.
                _new_canon = (m.canonical or "").strip()
                _old_canon = (existing.get("canonical") or "").strip()
                if len(_new_canon) > len(_old_canon):
                    existing["canonical"] = _new_canon
                    if m.department:
                        existing["department"] = m.department
                    if m.indication:
                        existing["indication"] = m.indication
                    if not existing.get("verified"):
                        existing["prescribed_name"] = m.prescribed_name or _new_canon
                for fld in ("dosage", "frequency", "duration"):
                    if not existing.get(fld) and getattr(m, fld, ""):
                        existing[fld] = getattr(m, fld)
                continue
            if m.drug:
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
                    # What the UI prints. `drug` stays the spoken text and
                    # `canonical` the molecule; this is the one to show.
                    "prescribed_name": m.prescribed_name or m.drug,
                    "route": m.route, "instructions": m.instructions,
                    "heard_as": m.heard_as,
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
        for a in rx.advice:
            if a not in advice:
                advice.append(a)
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

    # THE SPECIALTY CHECK, RUN WHERE THE DIAGNOSIS EXISTS.
    #
    # validate.py already refuses a fuzzy match that lands in the wrong
    # clinic - it is what stops "Traject" becoming Linagliptin, a diabetes
    # drug, on a menopause consultation. But it runs per SEGMENT, and a
    # segment almost never states the diagnosis: that is a consultation-level
    # fact which does not exist until this function has read every segment.
    # With no consult_dept the rule returns False, so in production it
    # passed everything and the guard was effectively dead.
    #
    # What reached real prescriptions with it dead:
    #     "টেস্ট" ("test")           -> Triamcinolone oral paste, DENTAL,
    #                                    on a SEIZURE consultation
    #     "হিয়ারিং টেস্ট" ("hearing  -> ORS, GASTRO, on VERTIGO
    #                      test")
    #     "আলার্জি" ("allergy")      -> Flurbiprofen eye drops,
    #                                    OPHTHALMOLOGY, on a deviated septum
    #     "স্লিপ সিন"                -> Cilnidipine, CARDIAC, on a LIPOMA
    #
    # None of those words is in the gazetteer, so no clinical-term veto can
    # reach them, and no similarity threshold separates them either - a real
    # match (Trimolat -> Norethisterone) scores 0.71 while a bogus one
    # (Traject -> Linagliptin) scores 0.80. The specialty is the signal that
    # works; it only ever lacked the diagnosis.
    #
    # VERIFIED is left alone. An exact gazetteer hit means the name really
    # was spoken, and a doctor may prescribe outside their specialty. Only a
    # GUESS is second-guessed here. Demoted to rejected_terms, never
    # deleted - the UI shows that panel so the doctor can reinstate.
    consult_dept = ""
    try:
        from voicerx.glossary import department_for, scan_conditions
        from voicerx.validate import department_clash

        conditions = []
        for rx in result.extractions:
            conditions.extend(scan_conditions(rx.source_transcript or ""))

        # THE DIAGNOSIS IS WHICHEVER IS MOST SPECIFIC, NOT WHICHEVER CAME
        # FIRST.
        #
        # The loop above takes the first segment that offers one, and a
        # consultation does not work that way: the doctor examines, talks
        # around the problem, and NAMES the condition near the end. A
        # generic early reading therefore locked out the real answer -
        # "ছুলি" (pityriasis versicolor) is stated late on a skin
        # consultation and lost to whatever an earlier segment had already
        # put in the field.
        #
        # scan_conditions is deterministic and already prefers the specific
        # reading within a segment - see _GENERIC_CONDITIONS, which is why
        # "stomach infection, infection" does not name one finding twice.
        # This applies the same preference ACROSS segments, which is the
        # only place the whole consultation is visible.
        #
        # It never invents: every condition here was matched in the
        # transcript by the curated table. A model-authored diagnosis is
        # kept when the gazetteer found nothing, and kept alongside when it
        # already names what was found.
        specific = [c for c in dict.fromkeys(conditions)
                    if c not in _GENERIC_DIAGNOSES]
        if specific:
            named = ", ".join(specific)
            if not diagnosis:
                diagnosis = named
            elif not any(c.lower() in diagnosis.lower() for c in specific):
                # The model named something the doctor never did, or
                # something vaguer. Lead with what was actually said and
                # keep the model's reading behind it rather than dropping
                # it - a doctor can see both and choose.
                diagnosis = f"{named} ({diagnosis})"

        # THE SPECIALTY IS TAKEN FROM THE BEST EVIDENCE AVAILABLE, NOT ONLY
        # FROM THE DIAGNOSIS.
        #
        # It used to come solely from a curated list of ~50 conditions, so
        # any consultation whose condition was not on that list had no
        # department - and no department meant no drug protection at all.
        # Chalazion, carpal tunnel and impacted ear wax each cost a whole
        # prescription's worth of checks that way. Medicine has thousands of
        # conditions; a 50-item list will never be the reliable input.
        #
        # The answer was already in the record both times. On the eyelid
        # consultation the gate had VERIFIED "Tobramycin eye drops" at 1.0 -
        # ophthalmology, stated with certainty - while the guard reported no
        # department and let a dermatology, an endocrine and a neurology
        # drug stand beside it. The ear consultation had Paradichlorobenzene
        # / Chlorbutol, ENT, and kept a glaucoma drop.
        #
        # Order is by how firmly each source is established, and the
        # diagnosis still goes first when it resolves - this only ADDS a
        # fallback where there was none:
        #
        #   1. the named diagnosis / conditions   (explicit)
        #   2. VERIFIED drugs                     (exact match = really said)
        #   3. the tests ordered                  (Audiometry -> ENT)
        #
        # Every drug in the gazetteer carries a department and only 50
        # conditions do, so 2 is the broader signal by a wide margin.
        consult_dept = department_for(([diagnosis] if diagnosis else []) + conditions)
        dept_source = "diagnosis" if consult_dept else ""

        if not consult_dept:
            from collections import Counter
            votes = Counter(
                (m.get("department") or "").strip()
                for m in meds
                if m.get("tier") == "verified"
                and (m.get("department") or "").strip() not in ("", "general"))
            if votes:
                consult_dept = votes.most_common(1)[0][0]
                dept_source = "verified medication"

        if not consult_dept:
            from voicerx.glossary import department_for_labs
            consult_dept = department_for_labs(labs)
            if consult_dept:
                dept_source = "tests ordered"

        # FAIL LOUD. A consultation whose specialty is unknown gets weaker
        # checking than one that resolved, so it must say so on the record
        # rather than look identical to a clean result.
        if not consult_dept:
            # FAIL CLOSED, at the one place the whole consultation is
            # visible. Nothing identified this record's specialty - not the
            # diagnosis, not a verified drug, not a test - so a guess has
            # nothing left to check it, and the weakest guesses do not get
            # to stand unexamined. Demoted, never deleted: the UI shows the
            # rejected panel so a doctor can put any of them back.
            surviving = []
            for m in meds:
                if m.get("tier") != "probable":
                    surviving.append(m)
                    continue
                sim = m.get("match_similarity") or 0
                if sim >= _UNKNOWN_DEPT_FLOOR:
                    surviving.append(m)
                    continue
                rejected.append(
                    f"{m.get('drug')} — resembles {m.get('canonical')} "
                    f"({sim:.2f}), but the specialty of this consultation "
                    f"could not be determined, so nothing could check it")
            meds[:] = surviving
            reasons.append(
                "the specialty of this consultation could not be determined "
                "- drug specialty checks did NOT run, every medication here "
                "needs confirming")
        elif dept_source != "diagnosis":
            reasons.append(
                f"specialty taken from the {dept_source} ({consult_dept}) - "
                f"no diagnosis named one")

        if consult_dept:
            surviving = []
            for m in meds:
                if (m.get("tier") == "probable"
                        and department_clash(m.get("department") or "", consult_dept)):
                    sim = m.get("match_similarity")
                    sim_s = f" ({sim:.2f})" if isinstance(sim, (int, float)) else ""
                    note = (f"{m['drug']} — resembles {m.get('canonical') or 'a drug'}"
                            f"{sim_s}, but that is a {m.get('department')} drug on a "
                            f"{consult_dept} consultation")
                    if note not in rejected:
                        rejected.append(note)
                    review_reasons.append(
                        f'"{m["drug"]}" was resolved only by similarity to '
                        f'{m.get("canonical")}, a {m.get("department")} drug, on a '
                        f'{consult_dept} consultation - NOT included, confirm if intended')
                    continue
                surviving.append(m)
            meds = surviving
    except Exception as exc:                        # noqa: BLE001
        # A merge that cannot compute a department must still return the
        # consultation. Failing open keeps the old behaviour; failing shut
        # would lose the whole prescription over a lookup error.
        logging.getLogger(__name__).warning(
            "merge-time specialty check skipped: %s: %s", type(exc).__name__, exc)

    return {
        "patient_name": patient_name,
        "symptoms": symptoms,
        "diagnosis": diagnosis,
        "consult_department": consult_dept or None,
        "labs_ordered": labs,
        "advice": advice,
        "medications": meds,
        "follow_up": follow_up,
        # DERIVED from the fields that already passed the gate, not written
        # by the model.
        #
        # The model's own narrative was concatenated straight into this
        # field, which made it the ONLY clinical output that skipped
        # validation entirely - symptoms are corroborated against the
        # transcript, medications go through the gate, labs are gated, and
        # the summary went out raw.
        #
        # It read as the clinical record and it invented content. On a
        # sports-injury consultation it reported "a sore throat and having a
        # runny nose before playing" - never said, by anyone. A reviewer
        # scanning the summary rather than the fields would have taken it as
        # history.
        #
        # Built from validated content instead, it cannot say anything that
        # did not survive the gate. The model's text is kept below under its
        # own name, because a silent deletion is still a deletion - it is
        # simply no longer presented as the record.
        "summary": _derive_summary(symptoms, diagnosis, labs, meds, advice,
                                    follow_up),
        "model_narrative": " ".join(summaries) if summaries else None,
        "model_narrative_note": (
            "Written by the extraction model. NOT grounded against the "
            "transcript - may contain content nobody said. Shown for context "
            "only; the summary above is derived from validated fields."
        ) if summaries else None,
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
    # TASK 2: chunk_seconds 10 → 4 halves residual audio at Stop.
    # Measured: 4s of audio → ~1-2s extract time. Final chunk at Stop: 2-4s total.
    # Helps hit 3s post-Stop SLA with 10 concurrent doctors.
    return {"session_id": sid, "chunk_seconds": 4}


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

    # Log all medication scores for threshold calibration
    _log_threshold_scores(consult_id, merged)

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
        pipeline = get_pipeline()
        if pipeline is None:
            raise HTTPException(503, "Pipeline not available (ASR/LLM models loading or unavailable)")
        result = pipeline.process_file(wav_path)
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

        # Log all medication scores for threshold calibration
        _log_threshold_scores(consult_id, merged)

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


@app.post("/api/log-correction")
async def log_correction(payload: dict):
    """Doctor flags a medication error for future improvement.

    Payload:
    {
        "consultation_id": str,
        "medication_id": str,  # Index in medications list
        "what_system_said": str,
        "what_doctor_said": str,
        "dose_correction": str,  # optional
        "timestamp": str  # ISO format
    }

    Logged to error_log.jsonl for weekly analysis.
    See: scripts/analyze_errors.py
    """
    try:
        error_entry = {
            "consultation_id": payload.get("consultation_id"),
            "medication_id": payload.get("medication_id"),
            "what_system_said": payload.get("what_system_said", ""),
            "what_doctor_said": payload.get("what_doctor_said", ""),
            "dose_correction": payload.get("dose_correction"),
            "timestamp": payload.get("timestamp") or datetime.now().isoformat(),
            "error_type": "NEEDS_ANALYSIS"
        }

        # Append to error log (immutable audit trail)
        with open(ERROR_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(error_entry) + "\n")

        return {
            "status": "logged",
            "message": "Correction recorded. Thank you for helping us improve."
        }
    except Exception as e:
        print(f"[server] error logging failed: {e}", flush=True)
        return {
            "status": "error",
            "message": str(e)
        }, 500


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
