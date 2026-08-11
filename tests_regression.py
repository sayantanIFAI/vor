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
                               scan_advice, scan_labs,
                               scan_symptoms, stats)
from voicerx.schema import ExtractedRx, Medication
from voicerx.validate import validate, _department_clash

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
print("12. THE GAZETTEER MUST NOT INVENT FINDINGS")
print("=" * 70)
# The imported MedER vocabulary is pharmacology prose, and scanning
# transcripts with it put these on real prescriptions as symptoms.
for _junk in ("ডাক্তারবাবু", "রিপোর্ট", "বয়স"):
    check(f"not a symptom: {_junk}", scan_symptoms(_junk), [])
# ...while the curated table still RECOVERS what the model missed. These
# three were spoken aloud on a cardiac consultation and all left out.
check("still recovers sweating",
      scan_symptoms("খুব ঘাম হয় দরদর করে ঘাম হয়"), ["sweating"])
check("still recovers fever + vomiting",
      scan_symptoms("জ্বর আর বমি হচ্ছে"), ["fever", "vomiting"])
# Folding collides genuine clinical words with everyday Bengali. গা is
# "body" far more often than it is ঘা "sore" - measured 3 of 5.
check("body is not a wound", scan_symptoms("গা ব্যথা করছে"), ["body ache"])
# ...but the veto still knows it, so it can never become a medication.
check("veto still recognises it", is_clinical_term("ঘা"), "wound")

print()
print("=" * 70)
print("13. DOSING - normalise filler, never invent a number")
print("=" * 70)
_rx = validate(ExtractedRx(medications=[
    Medication(drug="Nitrofurantoin", dosage="Not specified",
               frequency="Not specified", duration="Not specified"),
    Medication(drug="Ecosprin", dosage="one per day",
               frequency="after breakfast", duration="daily not specified"),
    Medication(drug="Linagliptin", dosage="Tablet"),
    Medication(drug="Fexofenadine", dosage="One Eightti EMI",
               frequency="Tab", duration="Five Days"),
    Medication(drug="Metformin", dosage="500mg", frequency="BD",
               duration="10 days"),
]))
_m = {x.prescribed_name: x for x in _rx.medications}
check("filler becomes blank", _m["Nitrofurantoin"].dosage, "")
check("filler stripped from real content", _m["Ecosprin"].duration, "daily")
check("a form is not a dose", _m["Linagliptin"].dosage, "")
check("real dosing untouched", _m["Metformin"].dosage, "500mg")
# A spelled-out number is fine on its own - "one per day" is clear.
check("plain words are not flagged",
      "CONFIRM the amount" in _m["Ecosprin"].review_reason, False)
# A garbled strength is NOT resolved to a number. "One Eightti EMI" is
# plainly 180mg, and guessing it would be a dosing error nothing signals.
check("garbled dose kept verbatim",
      _m["Fexofenadine"].dosage, "One Eightti EMI")
check("  and flagged for a human",
      "CONFIRM the amount" in _m["Fexofenadine"].review_reason, True)

print()
print("=" * 70)
print("14. NOTHING SET ASIDE STAYS UNEXAMINED")
print("=" * 70)
# The model files what it doubts into raw_uncertain_terms, and nothing
# looked at that list again. "অ্যামোরাল" sat there while the gate could
# resolve it - Glimepiride, alongside the Metformin in the same sentence.
_rx = validate(ExtractedRx(
    medications=[Medication(drug="গ্লাইকোমেট")],
    raw_uncertain_terms=[
        "অ্যামোরাল (possible medication name, ASR unclear, needs human verification)",
        "এইচ বি এ ওয়ান (proposed as a lab test, not recognised)",
        "আর রাতে দু তিনবার কেন চারবার বাথরুম ছুটতে হয় (untranslated Bengali)",
    ]))
_names = [m.prescribed_name for m in _rx.medications]
check("drug recovered from uncertain", "Glimepiride" in _names, True)
# Resolved through the consonant-skeleton index (অজামোরাল differs from the
# stored spelling only in vowels). That is weaker evidence than a full
# match, so it stays PROBABLE and keeps its CONFIRM flag.
check("  never asserted, only proposed",
      [m.tier for m in _rx.medications if m.prescribed_name == "Glimepiride"],
      ["probable"])
check("  original text retained",
      [m.heard_as for m in _rx.medications if m.prescribed_name == "Glimepiride"],
      ["অ্যামোরাল"])
check("lab recovered from uncertain", _rx.labs_ordered, ["HbA1c"])
check("a sentence is not recovered", len(_rx.raw_uncertain_terms), 1)

# The classic diabetic triad. Only "weakness" was reported from a
# consultation that stated all three; polyuria and polydipsia are what
# make the picture diabetic.
check("diabetic triad",
      scan_symptoms("যাই খাই না কেন ওজন কমে যাচ্ছে গলাটাও কেমন শুকিয়ে কাঠ "
                    "হয়ে থাকে আর রাতে দু তিনবার কেন চারবার বাথরুম ছুটতে হয়"),
      ["weight loss", "excessive thirst", "frequent urination"])
# Ordered in one breath with the fasting sugar, which WAS caught.
check("post-meal sugar",
      scan_labs("খালি পেটে সুগার আর খাওয়ার পরে সুগার আর এইচ বি এ ওয়ান সি"),
      ["Fasting sugar", "PP sugar", "HbA1c"])

print()
print("=" * 70)
print("15. DOSING BELONGS TO ONE DRUG, NOT THE SEGMENT")
print("=" * 70)
# "রোজ সকালে খাওয়ার পর ... ইকোস্পিডিন আর রসু ভাস্টা টিন ... আর বুকে ব্যাথা
# উঠলে ... জিভের তলায় একটা সর্বিট্রেট" - one sentence, two schedules. The
# segment frequency was copied onto every drug, so the sublingual nitrate
# read "after breakfast" instead of when the pain starts.
_seg = ("আমি আপনাকে রোজ সকালে খাওয়ার পর একটা করে এখনই ইকোস্পিডিন আর রসু "
        "ভাস্টা টিন খেতে দিচ্ছি আর বুকে ব্যাথা উঠলে সাথে সাথে জিভের তলায় "
        "না একটা সর্বিট্রেট দিয়ে দেবেন")
check("segment names two drugs",
      len(scan_drugs_spoken(_seg)) > 1, True)
check("  so no timing is broadcast", scan_dosing(_seg)[0], "after breakfast")
# The schema must be able to HOLD the instruction the model understood.
_m = Medication(drug="Sorbitrate", route="sublingual",
                instructions="when chest pain starts")
check("route has a home", _m.route, "sublingual")
check("as-needed has a home", _m.instructions, "when chest pain starts")

print()
print("=" * 70)
print("16. A DOSAGE FORM IS NOT A VERDICT ON THE DRUG")
print("=" * 70)
# "অ্যাম্ব্রুডিল সিরাপ" was REJECTED as "clinical term, not a drug: syrup".
# The form matched the non-drug term and took the drug down with it, while
# the bare name resolves perfectly well. Ambrodil cough syrup was dropped
# from a real prescription this way.
check("drug named with its form",
      judge_medication("অ্যাম্ব্রুডিল সিরাপ").canonical, "Ambroxol")
check("  same in Latin", judge_medication("Ambrodil syrup").canonical, "Ambroxol")
check("  and kept, not rejected",
      judge_medication("অ্যাম্ব্রুডিল সিরাপ").keep, True)
# The whole name is tried FIRST, so a generic that legitimately contains a
# form word is not mangled by stripping it.
check("form word inside a real name",
      judge_medication("Tobramycin eye drops").canonical, "Tobramycin eye drops")
# A bare form is still not a drug.
check("a form alone is still rejected", judge_medication("সিরাপ").keep, False)

# The doctor stated a diagnosis and the consultation returned none: only
# "allergies" existed, which is a symptom entry, not a diagnosable one.
check("stated diagnosis is found",
      scan_conditions("আপনার ডাস্ট আলার্জি তাহলে বুঝতে পারবো"), ["dust allergy"])
# Region is the point of the order, and this is the ASR's real output for
# "চেস্ট এক্স-রে" on a chest consultation.
check("garbled chest film",
      scan_labs("একটা চেস টেক্সটে করিয়ে নেবেন"), ["Chest X-ray"])
# One order must not print as two - a patient sent for both pays twice.
check("generic dropped for the specific",
      scan_labs("চেস্ট এক্স রে করাবেন"), ["Chest X-ray"])
check("  but distinct orders both survive",
      scan_labs("খালি পেটে সুগার আর খাওয়ার পরে সুগার"),
      ["Fasting sugar", "PP sugar"])

print()
print("=" * 70)
print("17. A GUESS IN THE WRONG SPECIALTY IS NOT A PRESCRIPTION")
print("=" * 70)
# On a menopause consultation "Traject" resolved by edit distance alone to
# Linagliptin - a DIABETES drug - and was printed as a medication. The
# doctor said "ট্রাফিক" once; the model romanised that one word twice, so
# one utterance became two prescription lines.
_rx = validate(ExtractedRx(
    diagnosis="menopause",
    source_transcript="একটা ট্রাফিক ট্যাবলেট দিচ্ছি আর মাসিক নিয়মিত করতে ট্রিমোলাট দিলাম",
    medications=[Medication(drug="Traject"), Medication(drug="ট্রাফিক"),
                 Medication(drug="Trimolat")]))
_kept = [m.prescribed_name for m in _rx.medications]
check("wrong-specialty guess dropped", "Linagliptin" in _kept, False)
check("  but recorded, never deleted",
      any("Linagliptin" in t for t in _rx.rejected_terms), True)
check("same-specialty guess kept", "Norethisterone" in _kept, True)
check("exact match kept", "Trapic" in _kept, True)
# Grounding the name in the transcript was tried and does NOT separate
# these - the good and bad names overlap on every similarity measure.
# Only the specialty separates them, so only specific departments count.
check("general drugs never clash",
      _department_clash(judge_medication("Paracetamol"), "gynaecology"), False)
check("no diagnosis means no signal",
      _department_clash(judge_medication("Traject"), ""), False)

# The presenting complaint on that same consultation, and the symptom the
# doctor reasoned from - neither was in the vocabulary.
check("menopause symptoms",
      scan_symptoms("তলপেটটা খুব ব্যাথা করছে খুব গরম লাগে হঠাৎ হঠাৎ"),
      ["lower abdominal pain", "hot flushes"])
check("  গরম alone is not a symptom", scan_symptoms("গরম জল খাবেন"), [])

print()
print("=" * 70)
print("18. ADVICE IS PART OF THE PRESCRIPTION")
print("=" * 70)
# "বেশি সাবান মাখবেন না আর প্রচুর জল খাবেন" was lost entirely. The terms
# were classified NON_CLINICAL - correctly, they are not symptoms - but
# that only kept them OUT of the symptom list; nothing carried them
# anywhere, and the prompt said to omit advice if there was nowhere to
# put it. There was nowhere.
check("advice recovered",
      scan_advice("বেশি সাবান মাখবেন না আর প্রচুর জল খাবেন"),
      ["use less soap", "drink water"])
check("  and is not a symptom",
      scan_symptoms("বেশি সাবান মাখবেন না আর প্রচুর জল খাবেন"), [])
# Negation suppression exists so a REFUSED order is not recorded as an
# order. Advice is the opposite case: the না IS the instruction.
check("a prohibition is advice", scan_advice("নখ দিয়ে খুঁটবেন না"),
      ["do not scratch"])
check("  negation still guards orders",
      scan_labs("নতুন কোনো টেস্ট দিচ্ছি না"), [])

# The acne diagnosis was unreachable: ব্রণ folds onto বরন, which ভ্রণ
# ("embryo") also lands on, so that key is blocked. The INFLECTED forms
# fold to distinct keys and carry no such ambiguity.
check("acne from inflected forms",
      scan_conditions("ব্রোনো কমানোর জন্য অ্যাডাপ্যালিন দিচ্ছি"), ["acne"])
check("  and from একমির", scan_conditions("এটি একমির মতো নাকি"), ["acne"])
check("  ambiguous bare key still blocked", scan_conditions("ভ্রণ"), [])
# Isotretinoin is the standard acne drug and was rejected outright,
# because the model sent the dosage form along with the name.
check("isotretinoin through the form",
      judge_medication("আইশো ট্রোটন নয়েন ক্যাপসুল", "dermatology").canonical,
      "Isotretinoin")

print()
print("=" * 70)
print("19. THE SITE IS PART OF THE DIAGNOSIS")
print("=" * 70)
# "আপনার পেটে হয়তো ইনফেকশন হয়েছে" came back as a bare "infection", which
# tells a reader nothing about what was found.
check("stomach infection",
      scan_conditions("আপনার পেটে হয়তো ইনফেকশন হয়েছে ওই খাবার থেকে"),
      ["stomach infection"])
check("  generic survives on its own",
      scan_conditions("ইনফেকশন হয়ে যাবে"), ["infection"])
# Two drugs missed on the same consultation; Ofloxacin was absent from
# the gazetteer entirely, and it is one of the commonest oral antibiotics.
check("racecadotril, ASR-split",
      [m.printed for m in scan_drugs_spoken("একটা রোসকাডো ট্রিল দিচ্ছি")],
      ["Racecadotril"])
check("ofloxacin by brand",
      [m.printed for m in scan_drugs_spoken("অ্যান্টিবায়োটিক হিসাবে ওফ্লোম্যাক দিলাম")],
      ["Oflomac"])
# ORS is the single most important instruction on a diarrhoea
# consultation, and the ASR renders it "ওয়ারেস্ট".
check("ORS as the ASR hears it",
      scan_advice("বারবার ওয়ারেস্ট খাবেন"), ["ORS"])
# With the site known, the wrong drug is caught by specialty.
_rx = validate(ExtractedRx(
    diagnosis="stomach infection",
    source_transcript="একটা রোসকাডো ট্রিল দিচ্ছি আর ওফ্লোম্যাক দিলাম",
    medications=[Medication(drug="Roxatodil"), Medication(drug="রোসকাডো ট্রিল")]))
check("surgery drug off a gastro script",
      [m.prescribed_name for m in _rx.medications], ["Racecadotril"])

print()
print("=" * 70)
print("20. A DRUG NOBODY SAID IS NOT A PRESCRIPTION")
print("=" * 70)
# "Erythromycin" was printed on a urology consultation that never mentions
# it. The model invented the name and the gate VERIFIED it out of the
# 179k imported brand register, so a fabrication arrived wearing the same
# badge as a drug the doctor actually said. The gate answers "is this a
# real drug"; it was never asked "was this one said".
_t40 = ("আমি আপনাকে হ্যাঁ এরা ইউরিম্যাক্স দিচ্ছি রোজরাতে খাবেন পেচ্ছাপের "
        "নালিটা খুলে যাবে পোতসাবের আর সাথে ইনফেকশন যদি হয়ে থাকে তার জন্য "
        "নাইট্রো ফুরান্টোইন আর একটা পিএসএ করিয়ে রাখবেন আর এরা কিডনিতে "
        "আল্ট্রাসাউন্ড টেস্ট করিয়ে নেবেন")
_rx = validate(ExtractedRx(diagnosis="urine infection", source_transcript=_t40,
    medications=[Medication(drug="Erythromycin"),
                 Medication(drug="Nitrofurantoin"),
                 Medication(drug="ইউরিম্যাক্স")]))
check("invented drug dropped",
      [m.prescribed_name for m in _rx.medications], ["Nitrofurantoin", "Urimax"])
check("  and recorded, never deleted",
      any("Erythromycin" in t for t in _rx.rejected_terms), True)

# The site is the diagnosis, and it is named several words from the word
# "infection" - so the region is attached afterwards, as with imaging.
check("urine infection + prostate",
      scan_conditions(_t40 + " আপনার পোস্টার টা বড় হয়ে থাকতে পারে"),
      ["urine infection", "enlarged prostate"])
check("kidney ultrasound", scan_labs(_t40), ["PSA", "USG KUB"])
check("every night", scan_dosing("ইউরিম্যাক্স দিচ্ছি রোজরাতে খাবেন")[0],
      "OD (night)")

# An ENT consultation where the doctor states the diagnosis AND its cause,
# and the prescription came back with no diagnosis at all. Vertigo was
# only ever a symptom entry; in an ENT clinic it is the finding.
_t41 = ("আপনার ভার্টিকও হয়েছে কানের ব্যালেন্স নষ্ট হওয়ার কারণে আর কানের এই "
        "যে ব্যালেন্স নষ্ট হয়ে গেছে")
check("vertigo is the diagnosis",
      scan_conditions(_t41), ["vertigo", "inner ear balance disorder"])
check("  and stops double-reporting as a symptom",
      "vertigo" in scan_symptoms(_t41), False)

print()
print("=" * 70)
print("21. ONE SET OF WORDS, ONE CLAIM ON IT")
print("=" * 70)
# Each table scanned the same words independently and nothing arbitrated,
# so the lab table took "ডেক্সা" out of the middle of Dexamethasone: an
# allergy patient given a steroid injection was recorded as having a
# bone-density scan ordered. Longest span wins, across tables.
_t42 = "একটা অ্যাভল আর ডেক্সা মিথোসেন ইনজেকশন দিতে বলছি"
check("a drug is not a scan", scan_labs(_t42), [])
check("  the drug is found instead",
      [m.printed for m in scan_drugs_spoken(_t42)], ["Avil", "Dexamethasone"])
check("  and a real DEXA still works",
      scan_labs("একটা ডেক্সা স্কান করবেন"), ["DEXA scan"])

# Vowel-level ASR variation, which fold() does not absorb. Each of these
# was a separate hand-added spelling before the skeleton index.
for _spoken, _want in [("ডেক্সা মিথোসেন", "Dexamethasone"),
                       ("রোসকাডো ট্রিল", "Racecadotril")]:
    check(f"consonants carry it: {_spoken[:14]}",
          [m.drug.generic for m in scan_drugs_spoken(_spoken)], [_want])
# A skeleton match drops every vowel, so it is weaker evidence than a full
# match and must not be asserted as certain.
check("skeleton match stays flagged",
      judge_medication("ডেক্সা মিথোসেন").tier, "probable")

# Diagnoses and advice the doctors gave that were being discarded.
check("severe allergy + hives",
      scan_conditions("লাল লাল চাকা হয়ে ফুলে গেছে আমবাদ বলে এটাকে সিভিয়ার আলার্জি"),
      ["hives", "severe allergy"])
check("mastitis, not bare infection",
      scan_conditions("বুকে ইনফেকশন হয়েছে একটা ইনফেকশন হয়েছে মাস্টিটাইটিস"),
      ["mastitis"])
check("the whole treatment plan is advice",
      scan_advice("গরম সেক দিতে পারেন কিন্তু চাপাচাপি করবেন না বেস্ট প্রাম্প ব্যবহার করতে হবে"),
      ["warm compress", "do not press the breast", "use a breast pump"])
# "গায়ে কাটা দিয়ে জ্বর" is shivering; the wound alias was reading it as a
# cut. Measured 0 true positives to 1 false across the 16.
check("shivering is not a wound",
      scan_symptoms("কাল থেকে খুব জ্বর গায়ে কাটা দিয়ে জ্বর আসছে"),
      ["fever", "chills"])

# "ভিটামিন ব টেলভে" resolved to Vitamin B COMPLEX - a different product.
# B12 is what is given for nerve damage, so dropping the "12" changes the
# drug. Same qualifier rule as MRI -> MRI brain: the longer match wins,
# now applied WITHIN a table as well as across them.
check("B12 is not B complex",
      [m.printed for m in scan_drugs_spoken("ভিটামিন ব টেলভে আর প্রেগাবালিন দিচ্ছি")],
      ["Vitamin B12", "Pregabalin"])
check("  plain B complex still resolves",
      [m.printed for m in scan_drugs_spoken("ভিটামিন বি দিচ্ছি")],
      ["Vitamin B complex"])
# The splint is the treatment on a carpal tunnel consultation, not an aside.
check("a device can be the treatment",
      scan_advice("একটা রিস্ট প্লিন্ট মানে ওই কবজির একটা বেল্ট দিচ্ছি"),
      ["wear a wrist splint"])

# The ASR also DROPS whole consonants, which exact skeletons cannot
# bridge: "আল্টা সাউন্ড" is "আল্ট্রাসাউন্ড" without the র.
check("a dropped consonant", scan_labs("একটা আল্টা সাউন্ড করে"), ["USG"])
# That relaxation must not leak into ordinary speech. Catch-all entries
# are excluded from it - their aliases are ordinary verbs, so "সি টা করবেন"
# was drifting into "test (unspecified)".
check("  and does not over-reach",
      scan_labs("আপনি এই সি বি সি টা করবেন"), ["CBC"])
# ...nor into the gate, where is_lab_test VETOES a drug. A near match is
# not strong enough to do that.
check("  nor vetoes a real drug",
      judge_medication("আইশো ট্রোটন নয়েন ক্যাপসুল", "dermatology").canonical,
      "Isotretinoin")

# Diagnoses named outright with no gazetteer entry at all, so they rested
# on the model alone.
check("lipoma", scan_conditions("এটাকে আমরা লাইপোমা বলি"), ["lipoma"])
check("deviated nasal septum",
      scan_conditions("ডেভিয়েটেড নেজাল সেপ্টেম্বলি আমরা ডাক্তারি ভাষায় ডি এন এস"),
      ["deviated nasal septum"])
check("nasal spray + montelukast",
      [m.printed for m in scan_drugs_spoken(
          "একটা ফাল্টিকাসন নাসাল স্প্রাযে একটা মাল্টিকুলার স্ট্যাবলেট দিচ্ছে")],
      ["Fluticasone nasal spray", "Montelukast"])

# A denture edge rubbing the gum. Distinct from an aphthous mouth ulcer -
# the cause is the appliance, which is what the treatment addresses.
check("traumatic ulcer",
      scan_conditions("মাটিতে না ট্রমাটিক একটা আলসার হয়ে গেছে"),
      ["traumatic ulcer"])
check("  and the sore it caused",
      scan_symptoms("বাঁদিকের মাড়ির কাছে কেটে ঘা হয়ে গেছে"), ["gum sore"])
# "একদম পরাবন্ধ করবেন না" - do NOT stop wearing it. The instruction is the
# prohibition, which is why advice is exempt from negation suppression.
check("keep wearing it",
      scan_advice("ঘা শুকিয়ে গেলে তবেই পরবেন কিন্তু একদম পরাবন্ধ করবেন না"),
      ["keep wearing the denture"])

print()
print("=" * 70)
print("22. AN ORDINARY WORD IS NOT A DRUG (sports-injury consultation)")
print("=" * 70)
# The patient said স্ট্রেন - "strain". Its consonant skeleton "strn" is also
# the skeleton of Isotroin, an Isotretinoin brand, so a torn muscle was
# prescribed an acne drug. The skeleton index ran BEFORE the clinical-term
# check: a vowel-dropped guess was outranking positive identification.
check("strain is not a drug",
      judge_medication("স্ট্রেন").tier, "rejected")
check("strain is the diagnosis",
      scan_conditions("দৌড়তে গিয়ে পেশিতে স্ট্রেন হয়েছে"), ["muscle strain"])
# ... but a sub-span match must NOT outrank the skeleton. The lab table
# matches ডেক্সা inside ডেক্সা মিথোসেন and reported a bone-density scan for
# a patient given a steroid injection. Longest span still wins.
check("whole-name drug beats sub-span lab",
      judge_medication("ডেক্সা মিথোসেন").canonical, "Dexamethasone")
check("genuine skeleton recovery survives",
      judge_medication("Rasu Basta Tin").canonical, "Rosuvastatin")

print()
print("=" * 70)
print("23. ONE UTTERANCE IS NOT TWO DRUGS")
print("=" * 70)
# The ASR wrote আসিক্লোফেন্ক and the model romanised the SAME word as
# "Asiklofenken". The first resolved to Aceclofenac on consonants (0.90),
# the second to Diclofenac by edit distance (0.73) - one spoken word, two
# prescription lines, one of them a drug nobody said.
_dup = validate(ExtractedRx(
    source_transcript="আসিক্লোফেন্ক দিলাম",
    medications=[Medication(drug="আসিক্লোফেন্ক"), Medication(drug="Asiklofenken")]))
check("the guess loses to the stronger match",
      [m.canonical for m in _dup.medications], ["Aceclofenac"])
check("and is recorded, not deleted", len(_dup.rejected_terms) >= 1, True)
# Two drugs that merely sound alike must both survive: only a FUZZY
# resolution is ever demoted, and both of these match exactly.
_pair = validate(ExtractedRx(
    source_transcript="Ecosprin আর Ecosprin AV",
    medications=[Medication(drug="Ecosprin"), Medication(drug="Ecosprin AV")]))
check("look-alike real drugs both survive", len(_pair.medications), 2)

print()
print("=" * 70)
print("21. GAZETTEER INTEGRITY")
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
