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
                               scan_dosing, scan_drugs, scan_drugs_spoken,
                               scan_labs,
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
print("11. THE PRINTED NAME - said vs garbled")
print("=" * 70)
# Reported from a real cardiology consultation: the medication list showed
# "Rasu Basta Tin", which is not a drug, not a brand, and not a word - just
# what the ASR made of রসু ভাস্টা টিন. It must print as Rosuvastatin.
#
# The same report also objected to the opposite error: "Ecosprin" being
# displayed as "Aspirin". Both are the SAME field getting it wrong in
# opposite directions, so both directions are pinned here.
_rx = validate(ExtractedRx(medications=[
    Medication(drug="Ecosprin"),        # real brand, verified
    Medication(drug="Rasu Basta Tin"),  # ASR garble, probable
    Medication(drug="Sorbitrate"),      # real brand whose generic differs
]))
_by = {m.drug: m for m in _rx.medications}
check("said a real brand -> print it",
      _by["Ecosprin"].prescribed_name, "Ecosprin")
check("  and its generic stays alongside",
      _by["Ecosprin"].canonical, "Aspirin")
check("  nothing to disclose as heard-as",
      _by["Ecosprin"].heard_as, "")
check("garbled -> print the real drug",
      _by["Rasu Basta Tin"].prescribed_name, "Rosuvastatin")
check("  substitution stays visible",
      _by["Rasu Basta Tin"].heard_as, "Rasu Basta Tin")
check("  and still flagged for a human",
      _by["Rasu Basta Tin"].verified, False)
check("brand kept, not swapped for generic",
      _by["Sorbitrate"].prescribed_name, "Sorbitrate")

# The same failure via the OTHER path into medications[]. The gazetteer
# scan of the transcript reported entry.generic, so a doctor who said
# "Sorbitrate" got "Nitroglycerin" printed - the substitution complaint,
# arriving from a route the gate never sees.
_m = {x.printed: x for x in scan_drugs_spoken("Sorbitrate ar Ecosprin cholbe")}
check("scan keeps the brand spoken", sorted(_m), ["Ecosprin", "Sorbitrate"])
check("  generic still resolved",
      _m["Sorbitrate"].drug.generic, "Nitroglycerin")
# A brand said in BENGALI must print as that brand, not as the molecule.
# Reported from recording 32: the doctor said সর্বিট্রেট and the script
# read "Nitroglycerin". Drug.brands and Drug.bengali are unpaired in the
# data, so this is derived by consonant skeleton - srbtrt == srbtrt.
_b = scan_drugs_spoken("সর্বিট্রেট টা চলবে")[0]
check("bengali brand -> that brand", _b.printed, "Sorbitrate")
check("  spoken form retained", _b.spoken, "সর্বিট্রেট")
check("bengali generic -> generic",
      scan_drugs_spoken("নাইট্রোগ্লিসারিন")[0].printed, "Nitroglycerin")
# A near-tie must fall to the generic: this is plainly Montelukast, and
# without the margin it printed "Montek", a brand nobody said.
check("near-tie prefers the generic",
      scan_drugs_spoken("মন্টিকুলাস")[0].printed, "Montelukast")

# Ordinary Bengali words must not become drugs. All three reached a real
# prescription: short keys (টোবা=Tobra, টেলমা=Telma) were being welded
# together out of separate function words.
for _txt in ["তো বাচ্চাকে", "তো মেনোপস বা", "তেল মাখলাম"]:
    check(f"not a drug: {_txt}", [m.printed for m in scan_drugs_spoken(_txt)], [])
# ...while the short brands themselves still match as whole tokens.
check("short brand as whole token",
      [m.printed for m in scan_drugs_spoken("ডোলো খাবেন")], ["Dolo"])
check("  with a bound suffix",
      [m.printed for m in scan_drugs_spoken("ডোলোটা খাবেন")], ["Dolo"])

# Bengali script cannot go on an English prescription, but the molecule is
# not the answer either - the brand the doctor said, romanised, is.
_rx = validate(ExtractedRx(medications=[
    Medication(drug="ইকোস্পিরিন"), Medication(drug="সর্বিট্রেট")]))
check("bengali brand via the gate",
      _rx.medications[0].prescribed_name, "Ecosprin")
check("  original preserved",
      _rx.medications[0].heard_as, "ইকোস্পিরিন")
check("  and the molecule alongside",
      _rx.medications[0].canonical, "Aspirin")
check("gate agrees with the scanner",
      _rx.medications[1].prescribed_name, "Sorbitrate")

print()
print("=" * 70)
print("12. GAZETTEER INTEGRITY")
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
