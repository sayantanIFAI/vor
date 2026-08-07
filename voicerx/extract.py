"""Node 2: structured clinical extraction via Qwen2.5, over Ollama.

The prompt below was hardened against a reproduced failure: on a
run-on/garbled transcript, an earlier looser prompt caused the model to (a)
invent a symptom with no textual basis, and (b) resolve a garbled ASR
fragment into "Naloxone" - a real, wrong, and dangerous drug name. The fix
that mattered was rule 2 (never resolve an unclear fragment into a named
drug) plus rule 1 (grounding). Verified across 3 repeated runs at
temperature 0 to confirm the fix is consistent, not a lucky sample.
"""
from __future__ import annotations

import json
import time
import urllib.request

from .schema import ExtractedRx

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:7b"

SYSTEM_PROMPT = """You are a clinical data extraction assistant. You will be given a raw Bengali doctor-patient consultation transcript, produced by automatic speech recognition, which may contain errors, garbled words, or run-on fragments with no punctuation.

THIS IS SAFETY-CRITICAL. A wrong drug name in your output could cause real patient harm. Extraction, not invention, is your only job.

HARD RULES - violating any of these is a failure:

1. GROUNDING: every symptom and every medication you output MUST be traceable to specific words that actually appear in the transcript. Before adding anything, find the exact phrase in the transcript that supports it. If you cannot point to specific words, DO NOT include it. Do not infer symptoms from context, from what "usually" goes with a diagnosis, or from what medications imply.

2. DRUG NAMES ARE THE HIGHEST-RISK FIELD. Bengali ASR frequently garbles drug names into nonsense syllables. You must NEVER resolve a garbled or ambiguous syllable string into the name of a real pharmaceutical unless the transcript's spelling is clear and unambiguous. If you are inferring, guessing, or pattern-matching a garbled fragment to "the closest real drug name you know," that is forbidden - put the raw Bengali text into "raw_uncertain_terms" instead, with a short note on why it's likely a drug/lab/dose that a human should verify. When in doubt, always choose raw_uncertain_terms over guessing a specific name.

   Example of correct behavior:
   Transcript fragment: "...সোরি আমি নলাক্স লিখে দিচ্ছি..."
   WRONG:   {"drug": "Naloxone", ...}   <- inventing a specific real drug from a garbled fragment
   RIGHT:   medications: []  and  raw_uncertain_terms: ["নলাক্স (possible medication name, ASR unclear, needs human verification)"]

3. IGNORE greetings, chit-chat, and anything not clinically relevant.

3b. BE CONFIDENT ABOUT PLAIN CLINICAL LANGUAGE. Rules 1 and 2 exist to stop you inventing DRUG NAMES from garbled syllables. They are NOT a reason to hide ordinary, clearly-spoken clinical content. If a common Bengali symptom or complaint is stated plainly - e.g. পাতলা পায়খানা (loose stools), জ্বর (fever), মাথা ব্যথা (headache), বমি (vomiting), কাশি (cough), শ্বাসকষ্ট (breathlessness), পেট ব্যথা (abdominal pain) - put it in "symptoms". Do NOT push clearly-understood everyday clinical words into "raw_uncertain_terms"; that field is for text you genuinely cannot interpret, not for text you understand but feel cautious about.

4. Only fill "diagnosis" if the doctor explicitly states one. Otherwise it must be null - do not infer a diagnosis from symptoms.

5. TRANSLATE Bengali medical terms to English ONLY when you are confident and the transcript is unambiguous. Otherwise use raw_uncertain_terms per rule 2.

6. Dietary/lifestyle advice (e.g. "drink more water", "avoid oily food", "eat light food", "ORS") is NOT a medication - do not put it in the "medications" array. If there is nowhere else for it, omit it rather than miscategorize it. The "medications" array is ONLY for named pharmaceutical drugs.

7. "symptoms" and "labs_ordered" and "raw_uncertain_terms" are flat arrays of short strings - not objects, not nested structures.

8. PATIENT NAME: only fill "patient_name" if a name is explicitly spoken in this transcript. Do NOT infer one, do NOT use a title alone ("দাদা", "কাকু", "স্যার"), and do NOT carry a name over from context. If no name is clearly spoken, it MUST be null. A wrong name on a prescription is its own category of harm.

9. SUMMARY: write "summary" as one plain-English sentence describing what happened clinically in this segment. If the segment has no clinical content (greeting, filler, silence), set it to null rather than inventing narrative.

10. Output ONLY a single valid JSON object, no other text, matching this exact shape:
{
  "patient_name": null,
  "symptoms": ["string", "string"],
  "diagnosis": null,
  "labs_ordered": ["string"],
  "medications": [{"drug": "string", "dosage": "string", "frequency": "string", "duration": "string"}],
  "follow_up": null,
  "summary": null,
  "raw_uncertain_terms": ["string"],
  "confidence_note": "string"
}"""


def _call_ollama(prompt: str, temperature: float = 0.0, timeout_s: int = 120) -> str:
    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {"temperature": temperature},
    }).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body.get("response", "")


class ExtractionError(Exception):
    pass


def extract_rx(transcript_bn: str, audio_file: str = "", seg_start: float | None = None,
                seg_end: float | None = None, max_retries: int = 2) -> tuple[ExtractedRx, dict]:
    """Returns (validated ExtractedRx, diagnostics dict with timing/attempts)."""
    prompt = f"{SYSTEM_PROMPT}\n\nTRANSCRIPT:\n{transcript_bn}\n\nJSON:"
    diagnostics = {"attempts": 0, "total_time_s": 0.0, "errors": []}

    last_error = None
    for attempt in range(1, max_retries + 2):
        diagnostics["attempts"] = attempt
        t0 = time.time()
        try:
            raw_text = _call_ollama(prompt)
            diagnostics["total_time_s"] += time.time() - t0
            raw_json = json.loads(raw_text)
            rx = ExtractedRx.from_llm_json(raw_json, audio_file=audio_file,
                                            transcript=transcript_bn,
                                            seg_start=seg_start, seg_end=seg_end)
            return rx, diagnostics
        except (json.JSONDecodeError, Exception) as e:  # noqa: BLE001 - retry on anything, log it
            diagnostics["total_time_s"] += time.time() - t0
            last_error = e
            diagnostics["errors"].append(f"attempt {attempt}: {type(e).__name__}: {e}")

    raise ExtractionError(
        f"extraction failed after {diagnostics['attempts']} attempts: {last_error}"
    )
