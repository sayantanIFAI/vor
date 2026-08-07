"""Regression suite. Every case here is a real failure from a real consultation.

Run:  python tests_regression.py

No test framework, deliberately - this must run on a fresh pod with nothing
installed but the pipeline's own dependencies.

Adding a case here is how a bug stays fixed. The rule used throughout this
project: when a live consultation exposes a defect, the transcript line that
caused it goes in as a test before the fix goes in as code.
"""
from __future__ import annotations

import io
import sys

# Windows consoles default to cp1252 and every Bengali test name would
# raise UnicodeEncodeError before a single assertion ran.
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, ".")

from voicerx.english import englishise
from voicerx.gate import judge_medication
from voicerx.glossary import (collisions, is_clinical_term, scan_conditions,
                               scan_dosing, scan_drugs, scan_labs,
                               scan_symptoms, stats)
from voicerx.schema import ExtractedRx, Medication
from voicerx.validate import validate

PASS = FAIL = 0
FAILURES: list[str] = []


def check(name: str, got, want) -> None:
    global PASS, FAIL
    ok = got == want
    if ok:
        PASS += 1
    else:
        FAIL += 1
        FAILURES.append(f"{name}\n      got:  {got!r}\n      want: {want!r}")
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}")


def contains(name: str, got, want) -> None:
    global PASS, FAIL
    ok = want in got
    if ok:
        PASS += 1
    else:
        FAIL += 1
        FAILURES.append(f"{name}\n      got:  {got!r}\n      want to contain: {want!r}")
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}")


print("=" * 70)
print("1. THE FOLD - spacing, half-letters, dialect, accent")
print("=" * 70)
check("spelled-out acronym", scan_labs("সি বি সি"), ["CBC"])
check("+ agglutinative suffix", scan_labs("আপনি এই সি বি সি টা করবেন"), ["CBC"])
check("ASR wrote ভ for ব", scan_labs("আপনি এই সি ভিসিটা করবেন"), ["CBC"])
check("two labs one segment",
      scan_labs("আপনাকে একটা টি এম টি আর ই সি জি করতে হবে"), ["TMT", "ECG"])
check("half-letter conjunct",
      [d.generic for d in scan_drugs("মন্টিকুলাষ্ট")], ["Montelukast"])
check("bengali suffix on a term", is_clinical_term("প্রেশারটা"), "blood pressure")

print()
print("=" * 70)
print("2. GAPPED MATCHING - an interposed word must not break a match")
print("=" * 70)
check("interposed 'blood'", scan_labs("ফাস্টিং ব্লাড সুগার"), ["Fasting sugar"])
check("interposed, PP", scan_labs("পিপি ব্লাড সুগার"), ["PP sugar"])
check("real transcript line, 4 labs",
      scan_labs("ওসিটি ফাস্টিং ব্লাড সুগার পিপি ইসিজি ব্লাড প্রেসার চেক"),
      ["OCT", "Fasting sugar", "PP sugar", "ECG"])

print()
print("=" * 70)
print("3. ASR-SPLIT DRUG NAMES - the ASR breaks names across words")
print("=" * 70)
for spoken, want in [("মেট ফর্মিন", "Metformin"),
                     ("রসু ভাস্টাটিন", "Rosuvastatin"),
                     ("মেটো প্রোল", "Metoprolol"),
                     ("মক্সি ফ্লক্সাসিম অ্যান্টিবায়োটিক", "Moxifloxacin eye drops")]:
    check(f"split name {spoken[:18]}",
          [d.generic for d in scan_drugs(spoken)][:1], [want])

print()
print("=" * 70)
print("4. THE GATE - what may be called a drug")
print("=" * 70)
for term, tier in [("Nitrocontin", "verified"), ("Alendronate", "verified"),
                   ("Levipil", "verified"), ("Montuculast", "probable"),
                   ("Antibiotic", "rejected"), ("painkiller", "rejected"),
                   ("hair loss", "rejected"), ("electrolytes", "rejected"),
                   ("Boot", "rejected"), ("null", "rejected")]:
    check(f"gate: {term}", judge_medication(term).tier, tier)

check("brand resolves to generic", judge_medication("Fosamax").canonical, "Alendronate")
check("combination product", judge_medication("Sacubitril/Valsartan").tier, "verified")

print()
print("=" * 70)
print("5. CONDITIONS vs SYMPTOMS vs ADVICE")
print("=" * 70)
t = "কমলবাবু আপনার ডান চোখে ক্যাটারাক্ট করেছে চোখে ব্যথা আছে বিশ্রাম নিন"
check("condition -> diagnosis", scan_conditions(t), ["cataract"])
contains("symptom found", scan_symptoms(t), "eye pain")
check("advice is neither", "rest" in scan_symptoms(t), False)

print()
print("=" * 70)
print("6. NEGATION - a refused order is not an order")
print("=" * 70)
check("negated test", scan_labs("নতুন কোনো টেস্ট দিচ্ছি না"), [])
check("negated angiogram", scan_labs("ওই এনজিওগ্রাম করতে চাই না এখন"), [])
contains("conditional order IS an order",
         scan_labs("যদি জ্বর না কমে তাহলে রক্ত পরীক্ষা করতে হবে"),
         "blood test (unspecified)")

print()
print("=" * 70)
print("7. DOSING - spoken Bengali, not clinical shorthand")
print("=" * 70)
check("after lunch + duration",
      scan_dosing("দুপুরে খাওয়ার পর একটা করে খাবেন সাত দিন"),
      ("after lunch", "7 days"))
check("night + twice daily",
      scan_dosing("রাতে খাবার পরে দিনে দুবার তিন মাস"),
      ("after dinner, BD", "3 months"))

print()
print("=" * 70)
print("8. ENGLISH ONLY - Chinese dropped, Bengali translated")
print("=" * 70)
rx = ExtractedRx(symptoms=["chest pain", "শ্বাসকষ্ট", "糖尿病", "হয়াট অ্যাটাক"],
                  summary="医生提到患者有多年的糖尿病", diagnosis="ডায়াবেটিজ")
englishise(rx)
check("bengali translated", rx.symptoms,
      ["chest pain", "breathlessness", "heart attack"])
check("chinese summary dropped", rx.summary, None)
check("bengali diagnosis translated", rx.diagnosis, "diabetes")
check("drop is audited", len(rx.raw_uncertain_terms), 2)

med_rx = ExtractedRx()
med_rx.medications.append(Medication(drug="Paracetamol", frequency="每日两次"))
englishise(med_rx)
check("chinese frequency cleared", med_rx.medications[0].frequency, "")

print()
print("=" * 70)
print("9. HALLUCINATION - the transcript decides, not the model")
print("=" * 70)
rx = ExtractedRx(symptoms=["eye pain", "loose stools", "wiping eye with finger"],
                  source_transcript="আপনার ডান চোখে ক্যাটারাক্ট করেছে চোখে ব্যথা আছে",
                  decoder_agreement=0.9)
rx = validate(rx)
check("supported symptom kept", rx.symptoms, ["eye pain"])
check("invented symptoms moved", len(rx.symptoms_unconfirmed), 2)

garbled = ExtractedRx(symptoms=["eye redness", "itchy eyes"],
                       decoder_agreement=0.29,
                       source_transcript="কেলেঙ্কারি খুব জোটে চোখ লেগেছিল")
garbled = validate(garbled)
check("garbled segment yields no symptoms", garbled.symptoms, [])

# The gapped matcher must not over-reach. Real false positive it caused
# before the stacked-relaxation fix: HbA1c spoken letter-by-letter matched
# "CT scan", because a gapped fragment merely STARTED with the CT key.
check("gapped match does not over-reach",
      scan_labs("এইচ ওয়ান বি এ সি তো ঠিক ই দেখছি আগের থেকে কমেছে"), [])

print()
print("=" * 70)
print("10. THE LAB GATE - ASR garble is not a lab order")
print("=" * 70)
rx = ExtractedRx(labs_ordered=["রেজ", "সুগার", "ইসিজি", "সি বি সি"])
rx = validate(rx)
check("garbage filtered, names canonical", rx.labs_ordered, ["ECG", "CBC"])

print()
print("=" * 70)
print("11. GAZETTEER INTEGRITY")
print("=" * 70)
check("no two entries fold together", collisions(), [])
st = stats()
print(f"        {st['drugs']} drugs / {st['drug_aliases']} aliases / "
      f"{st['lab_tests']} labs / {st['clinical_terms']} terms")
print(f"        departments: {', '.join(st['departments'])}")

print()
print("=" * 70)
print(f"  {PASS} passed, {FAIL} failed")
print("=" * 70)
for f in FAILURES:
    print(f"\n  FAILED: {f}")
sys.exit(1 if FAIL else 0)
