"""Import clinical TERMS (not drugs) from the MedER Bengali/English dataset.

WHY ONLY TERMS
--------------
The dataset's drug column is generic/chemical names. Measured against what
is actually spoken in these consultations it covers 0/6 of the clinician's
named brands (Nitrocontin, Rybelsus, Ecospirin, Ryzodeg, Dytor, Thyronorm)
and 2/7 of the drugs in the real audio. Brands are what get said in an OPD,
so this dataset cannot carry the allowlist and is not used for it.

Its Disease / Organ / Common Medical Terms columns are a different story.
Those feed CLINICAL_TERMS, whose job is to POSITIVELY rule things out - it
is what stops "sugar", "pressure" and "cholesterol" being classified as
medications. Going from 36 curated terms to a few thousand attacks that
directly.

WHY THE CLEANING IS AGGRESSIVE
------------------------------
Raw ingestion would undo the false-positive work. Real junk found in the
source: "NSAIDs (naproxen" and "celecoxib)" (the file comma-splits inside
parentheses), "অক্সিটোসিনেজ" (oxytocinASE - an enzyme), "β অ্যাড্রেনারজিক
রিসেপ্টর" (a receptor), "20 kDa", "2-4 সপ্তাহ", and the literal string
"Common Medical Terms:" - a header leaked into a data row.

THE SAFETY RULE THAT MATTERS
----------------------------
is_clinical_term() is used by the gate to REJECT medications. So an
imported term that collides with a real drug would cause that drug to be
silently thrown out of a prescription. Every candidate is therefore checked
against the drug gazetteer and dropped on collision - see reject_collisions
below. This is the one rule here that is not about tidiness.

Usage:
    python tools/import_meder.py voicerx/terms_imported.py
"""
from __future__ import annotations

import csv
import re
import sys
from collections import defaultdict

sys.path.insert(0, ".")
from voicerx.glossary import fold, lookup_drug, is_lab_test  # noqa: E402

BN_CSV = "D:/MedER_Dataset_Bengali_Entity_V2.csv"
EN_CSV = "D:/MedER_Dataset_English_Translated.csv"

COLUMN_PAIRS = [
    ("Disease", "Disease_eng"),
    ("Organ", "Organ_eng"),
    ("Common Medical Terms", "Common Medical Terms_eng"),
]

BENGALI = re.compile(r"[\u0980-\u09FF]")
LATIN = re.compile(r"[A-Za-z]")
DIGIT = re.compile(r"\d")

# Substrings that mark a token as biochemistry/pharmacology jargon rather
# than something a patient or doctor says out loud in a consultation.
JARGON = (
    "রিসেপ্টর", "এনজাইম", "ইনহিবিটর", "receptor", "enzyme", "inhibitor",
    "kda", "mg", "ml", "µ", "α", "β", "γ", "অ্যাসিড ডেরিভেটিভ",
    "agonist", "antagonist", "অ্যাগোনিস্ট", "অ্যান্টাগোনিস্ট",
)
# Header text that leaked into data rows.
HEADER_LEAK = ("common medical terms", "medical text", "disease", "organ",
                "pharmacological class", "hormone")


def clean_bn(t: str) -> str | None:
    t = t.strip().strip('"\u201c\u201d').strip()
    t = t.strip("()[]{}.,;:-\u2013 ")
    if not t or not (2 < len(t) <= 30):
        return None
    if not BENGALI.search(t):
        return None                       # must be Bengali to match speech
    if DIGIT.search(t) or LATIN.search(t):
        return None                       # units, codes, script leakage
    if t.count("(") != t.count(")"):
        return None                       # comma-split inside parentheses
    low = t.lower()
    if any(j in low for j in JARGON):
        return None
    return t


def clean_en(t: str) -> str | None:
    t = t.strip().strip('"\u201c\u201d').strip()
    t = t.strip("()[]{}.,;:-\u2013 ").lower()
    if not t or not (2 < len(t) <= 40):
        return None
    if DIGIT.search(t) or not LATIN.search(t):
        return None
    if t in HEADER_LEAK or any(j in t for j in JARGON):
        return None
    if t.count("(") != t.count(")"):
        return None
    return t


def main() -> None:
    bn_rows = list(csv.DictReader(open(BN_CSV, encoding="utf-8-sig")))
    en_rows = list(csv.DictReader(open(EN_CSV, encoding="utf-8-sig")))
    if len(bn_rows) != len(en_rows):
        sys.exit(f"row mismatch: {len(bn_rows)} vs {len(en_rows)}")

    pairs: dict[str, set[str]] = defaultdict(set)
    seen_raw = 0

    for bn_row, en_row in zip(bn_rows, en_rows):
        for bn_col, en_col in COLUMN_PAIRS:
            bn_cell = (bn_row.get(bn_col) or "").replace(";", ",").split(",")
            en_cell = (en_row.get(en_col) or "").replace(";", ",").split(",")
            # Positional pairing only works when both sides split the same
            # way. When they don't, the alignment is unknowable and a wrong
            # pairing would mislabel a term - so the row is skipped.
            if len(bn_cell) != len(en_cell):
                continue
            for bn_t, en_t in zip(bn_cell, en_cell):
                seen_raw += 1
                b, e = clean_bn(bn_t), clean_en(en_t)
                if b and e:
                    pairs[e].add(b)

    # --- the safety rule: never let an imported term shadow a drug -------
    rejected_collisions: list[tuple[str, str, str]] = []
    clean: dict[str, set[str]] = {}
    for en_term, bn_set in pairs.items():
        if lookup_drug(en_term):
            rejected_collisions.append((en_term, "-", "english side is a drug"))
            continue
        keep = set()
        for b in bn_set:
            hit = lookup_drug(b)
            if hit:
                rejected_collisions.append((en_term, b, f"folds onto drug {hit.generic}"))
                continue
            lab = is_lab_test(b)
            if lab:
                rejected_collisions.append((en_term, b, f"folds onto lab {lab}"))
                continue
            keep.add(b)
        if keep:
            clean[en_term] = keep

    # Written as UTF-8 explicitly rather than through stdout: on Windows the
    # default console codec is cp1252 and every Bengali character raises
    # UnicodeEncodeError.
    dest = sys.argv[1] if len(sys.argv) > 1 else "voicerx/terms_imported.py"
    out = open(dest, "w", encoding="utf-8")
    out.write('"""AUTO-GENERATED - do not edit by hand.\n\n')
    out.write("Regenerate with:  python tools/import_meder.py voicerx/terms_imported.py\n\n")
    out.write("Clinical terms imported from the MedER Bengali/English dataset.\n")
    out.write("These are MACHINE-IMPORTED and have NOT been clinically reviewed -\n")
    out.write("a lower trust tier than the curated CLINICAL_TERMS in glossary.py.\n")
    out.write("They are used only to RULE THINGS OUT (a term here can never be a\n")
    out.write("medication), which is the safe direction for unreviewed data.\n\n")
    out.write(f"raw candidate pairs seen : {seen_raw}\n")
    out.write(f"survived cleaning        : {sum(len(v) for v in clean.values())}\n")
    out.write(f"english canonical terms  : {len(clean)}\n")
    out.write(f"dropped for colliding with a drug/lab : {len(rejected_collisions)}\n")
    if rejected_collisions:
        out.write("\ncollisions dropped (these would have shadowed a real drug):\n")
        for en_term, b, why in sorted(rejected_collisions)[:40]:
            out.write(f"    {en_term} / {b} - {why}\n")
    out.write('"""\n\n')
    out.write("IMPORTED_TERMS: dict[str, tuple[str, ...]] = {\n")
    for en_term in sorted(clean):
        variants = ", ".join(f'"{b}"' for b in sorted(clean[en_term]))
        out.write(f'    "{en_term}": ({variants},),\n')
    out.write("}\n")


if __name__ == "__main__":
    main()
