import sys, io, json, glob
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, r"D:\Claude\voice-to-rx-repo")
from voicerx.glossary import fold, lookup_drug, is_lab_test, is_clinical_term

print("="*66)
print("RESOLUTION TEST - does the term reach the right gazetteer entry?")
print("="*66)
cases = [
    ("space / non-space",        "সি বি সি",     "lab", "CBC"),
    ("+ agglutinative suffix",   "সি বি সি টা",  "lab", "CBC"),
    ("in a sentence",            "আপনি এই সি বি সি টা করবেন", "lab", "CBC"),
    ("joined form",              "সিবিসি",       "lab", "CBC"),
    ("half-letter conjunct",     "মন্টিকুলাষ্ট",  "drug", "Montelukast"),
    ("half-letter dropped",      "মনটিকুলাসট",   "drug", "Montelukast"),
    ("sibilant swap স/ষ",        "মন্টিকুলাস্ট",  "drug", "Montelukast"),
    ("bengali paracetamol",      "প্যারাসিটামল",  "drug", "Paracetamol"),
    ("+ suffix",                 "প্যারাসিটামলটা","drug", "Paracetamol"),
    ("aspiration ভ/ব",           "ভ্যালিয়াম",     "drug", "Diazepam"),
    ("latin brand",              "Calpol",       "drug", "Paracetamol"),
    ("dosage prefix",            "Tab. Amlodipine","drug","Amlodipine"),
    ("spelled ECG",              "ই সি জি",      "lab", "ECG"),
    ("spelled TMT",              "টি এম টি",     "lab", "TMT"),
    ("clinical NOT drug",        "প্রেশারটা",     "term","blood pressure"),
    ("clinical NOT drug",        "সুগার",         "term","blood sugar"),
]
ok = bad = 0
for label, text, kind, want in cases:
    if kind == "drug":
        r = lookup_drug(text); got = r.generic if r else None
    elif kind == "lab":
        got = is_lab_test(text)
    else:
        got = is_clinical_term(text)
    good = (got == want)
    ok += good; bad += not good
    print(f"  {'PASS' if good else 'FAIL'}  {label:24} {text[:26]:28} -> {got}")
print(f"\n  {ok} passed, {bad} failed")
