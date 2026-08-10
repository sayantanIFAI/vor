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

3b. REPORT EVERY SYMPTOM THE PATIENT ACTUALLY STATES, and only those.

   Rules 1 and 2 exist to stop you inventing DRUG NAMES from garbled syllables. They are NOT a reason to omit ordinary, clearly-spoken clinical content. If the patient plainly describes a complaint - in any words, including colloquial ones - put it in "symptoms".

   Two failures matter equally, and the second is easy to miss:
     - Reporting a symptom that was NOT said. This is a fabricated clinical record.
     - Omitting a symptom that WAS said. On a cardiac consultation, sweating, palpitations and breathlessness were all spoken aloud and all left out, which loses the three findings that make the diagnosis.

   Do not restrict yourself to textbook symptom words. Patients describe things loosely - "my chest feels gripped", "I sweat heavily", "my heart races", "I run out of breath". Each of those is a symptom and belongs in the list, phrased in plain English.

   Do NOT push clearly-understood everyday clinical words into "raw_uncertain_terms"; that field is for text you genuinely cannot interpret, not for text you understand but feel cautious about.

3c. HOW AND WHEN A MEDICINE IS TAKEN IS PART OF THE PRESCRIPTION, not an unknown term.

   Doctors give ordinary spoken instructions, and they are ordinary Bengali. Record them:
     - "route"        how it is taken: জিভের তলায় (under the tongue) = sublingual,
                      খাবেন (oral), লাগাবেন (topical), ড্রপ (drops), ইনহেলার (inhaled)
     - "instructions" WHEN or under what condition, and any advice attached to it:
                      "ব্যাথা উঠলে" = when the pain starts, "পকেটে রাখবেন" = keep it in your pocket

   Example, and this is a real one that was got wrong:
     "বুকে ব্যাথা উঠলে সাথে সাথে জিভের তলায় একটা সর্বিট্রেট দিয়ে দেবেন"
     WRONG:  raw_uncertain_terms: ["জিভের তলায় না একটা সর্বিট্রেট"]
             <- this sentence is not unclear. দিয়ে দেবেন is an ordinary instruction.
     WRONG:  {"drug": "Sorbitrate", "frequency": "after breakfast"}
             <- that schedule belongs to a DIFFERENT medicine in the same sentence
     RIGHT:  {"drug": "সর্বিট্রেট", "dosage": "1", "route": "sublingual",
              "instructions": "when chest pain starts; keep in pocket"}

   Rule 2 is about not INVENTING A DRUG NAME from garbled syllables. It is not a reason
   to discard an instruction you understood. If you understood the sentence, write it down.

   ATTRIBUTE EACH INSTRUCTION TO THE RIGHT MEDICINE. One sentence often prescribes two
   things on different schedules - a daily tablet and an as-needed one. Do not copy one
   medicine's timing onto another. If you cannot tell which medicine a timing belongs to,
   leave that medicine's timing EMPTY rather than guessing.

4. Only fill "diagnosis" if the doctor explicitly states one. Otherwise it must be null - do not infer a diagnosis from symptoms.

5. TRANSLATE Bengali medical terms to English ONLY when you are confident and the transcript is unambiguous. Otherwise use raw_uncertain_terms per rule 2.

6. Dietary/lifestyle advice (e.g. "drink more water", "avoid oily food", "eat light food", "use less soap") is NOT a medication - do not put it in the "medications" array, which is ONLY for named pharmaceutical drugs. Put it in "advice" instead, as short plain-English instructions. Do NOT discard it: on a skin or diabetes consultation the advice is half the prescription. "বেশি সাবান মাখবেন না আর প্রচুর জল খাবেন" is two pieces of advice - "use less soap" and "drink plenty of water".

7. "symptoms" and "labs_ordered" and "raw_uncertain_terms" are flat arrays of short strings - not objects, not nested structures.

8. PATIENT NAME: only fill "patient_name" if a name is explicitly spoken in this transcript. Do NOT infer one, do NOT use a title alone ("দাদা", "কাকু", "স্যার"), and do NOT carry a name over from context. If no name is clearly spoken, it MUST be null. A wrong name on a prescription is its own category of harm.

9. SUMMARY: write "summary" as one plain-English sentence describing what happened clinically in this segment. If the segment has no clinical content (greeting, filler, silence), set it to null rather than inventing narrative.

10. Output ONLY a single valid JSON object, no other text, matching this exact shape:
{
  "patient_name": null,
  "symptoms": ["string", "string"],
  "diagnosis": null,
  "labs_ordered": ["string"],
  "medications": [{"drug": "string", "dosage": "string", "frequency": "string", "duration": "string", "route": "string", "instructions": "string"}],
  "follow_up": null,
  "advice": ["string"],
  "summary": null,
  "raw_uncertain_terms": ["string"],
  "confidence_note": "string"
}"""


# Appended only when an English translation is supplied (translate.py).
# Qwen2.5-7B reads English clinical text far better than Bengali, so the
# translation carries the NARRATIVE. Drug names are pinned to the Bengali
# original on purpose: machine translation is least reliable exactly where
# the stakes are highest, on transliterated brand names that are not really
# Bengali words. "নরফ্লক্স" can come out of an MT model as anything at all.
BILINGUAL_HEADER = """You are given the SAME consultation segment twice: an English translation, and the Bengali original it came from.

- Use the ENGLISH text to understand what happened - symptoms, diagnosis, summary, follow-up.
- Treat the BENGALI text as authoritative for DRUG NAMES, DOSAGES and LAB TEST NAMES. Copy those from the Bengali. The English translation may have corrupted them, because translation systems handle transliterated brand names badly.
- If the two disagree about a drug name, trust the Bengali and, if it is unclear there, follow rule 2 and use raw_uncertain_terms.
"""


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
                seg_end: float | None = None, max_retries: int = 2,
                transcript_en: str | None = None) -> tuple[ExtractedRx, dict]:
    """Returns (validated ExtractedRx, diagnostics dict with timing/attempts).

    transcript_en, when supplied, switches on the bilingual prompt - see
    translate.py for why. It is OPTIONAL and off by default: the
    Bengali-only prompt above is the path that was actually verified across
    repeated runs, and it stays the default until the bridge is measured
    against it on real audio rather than assumed to be better.
    """
    if transcript_en:
        prompt = (f"{SYSTEM_PROMPT}\n\n{BILINGUAL_HEADER}\n"
                  f"\nENGLISH TRANSLATION:\n{transcript_en}\n"
                  f"\nBENGALI ORIGINAL (authoritative for drug/lab names):\n{transcript_bn}\n"
                  f"\nJSON:")
    else:
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
