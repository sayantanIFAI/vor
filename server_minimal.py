"""Minimal test server - no ASR, tests framework only."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import json
import time
from pathlib import Path

app = FastAPI(title="Voice-to-Rx (Test)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "mode": "test_minimal",
        "cuda": False,
        "model_loaded": False,
        "note": "ASR/LLM disabled for testing framework"
    }

@app.post("/api/session/start")
def session_start():
    import uuid
    sid = f"{int(time.time())}_{uuid.uuid4().hex[:6]}"
    return {"session_id": sid, "chunk_seconds": 4}

@app.post("/api/session/{sid}/finalize")
def session_finalize(sid: str):
    return {
        "consult_id": sid,
        "status": "test_mode",
        "medications": [],
        "symptoms": [],
        "diagnosis": None,
        "click_to_result_s": 0.1
    }

@app.get("/api/results")
def list_results():
    return {"results": []}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
