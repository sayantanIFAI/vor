"""Import Indian brand -> generic drug names from the A-Z India dataset.

WHY THIS ONE AND NOT THE LAST ONE
---------------------------------
The MedER dataset covered 0/6 of the brands this clinician actually names.
This one covers 15/15 - Nitrocontin, Rybelsus, Ecosprin, Ryzodeg, Dytor,
Thyronorm, Norflox, Clavam, Ascoril, Valium, Montair, Calpol, Asthalin,
Thyrox, Pantocid. An OPD runs on brands, so this is the file that can
actually carry the allowlist.

WHY NOT JUST LOAD ALL 249,345 OF THEM
-------------------------------------
Because that would recreate, at scale, the bug this project already hit
once: the Lactulose brand "Looz" folds to "los", the English word "loss"
folds to "los", and "hair loss" was resolving to a VERIFIED medication.

With a quarter of a million brand names, many of which are ordinary words
("Total", "Rest", "Cold", "Best"), loading them unfiltered would poison the
gate rather than improve it. So:

  1. Only non-discontinued allopathy entries.
  2. The dosage/form tail is stripped: "Ecosprin 75 Tablet" -> "Ecosprin",
     so 15 pack variants collapse to one gazetteer entry.
  3. Anything that collides - with a curated drug, a lab test, a clinical
     term, or the imported Bengali symptom vocabulary - is DROPPED and
     reported. A brand that shadows a symptom is worse than a missing
     brand, because the missing brand shows up in rejected_terms where a
     human sees it, while the shadowing one silently turns a symptom into a
     drug.
  4. Short names are dropped outright, for the same reason _MIN_DRUG_NGRAM
     exists.

The output is intentionally a SEPARATE table from the curated one in
glossary.py. Curated entries carry Bengali transliterations and a
department; these carry neither, and are a lower trust tier.

Usage:
    python tools/import_india_brands.py voicerx/brands_india.py
"""
from __future__ import annotations

import csv
import re
import sys

sys.path.insert(0, ".")
from voicerx.glossary import (fold, _DRUG_LOOKUP, _LAB_LOOKUP,  # noqa: E402
                               _TERM_LOOKUP)

CSV_PATH = "D:/Extensive_A_Z_medicines_dataset_of_India.csv"

# Dosage/form tail. "Ecosprin 75 Tablet" and "Ecosprin AV 150 Capsule" both
# reduce to a base brand, which is what a clinician actually says out loud.
FORMS = (
    "tablet", "tablets", "capsule", "capsules", "syrup", "suspension",
    "injection", "cream", "ointment", "gel", "drop", "drops", "solution",
    "lotion", "powder", "sachet", "inhaler", "respules", "rotacaps",
    "penfill", "cartridge", "vial", "infusion", "spray", "soap", "shampoo",
    "kit", "patch", "granules", "mouthwash", "eye", "ear", "nasal", "oral",
    "sr", "cr", "xr", "dsr", "md", "mr", "er", "la", "od", "forte", "plus",
)
DOSE = re.compile(r"\b\d+(\.\d+)?\s*(mg|mcg|g|ml|iu|%|units?)\b", re.I)
NUMTAIL = re.compile(r"[\s\-]\d+(\.\d+)?$")

# Brand names below this many folded characters are never imported. Same
# reasoning as _MIN_DRUG_NGRAM in glossary.py - short keys collide with
# ordinary words once folded.
MIN_LEN = 5


def base_name(name: str) -> str | None:
    n = name.strip().lower()
    n = DOSE.sub(" ", n)
    parts = [p for p in re.split(r"\s+", n) if p]
    while parts and parts[-1].strip("().,-") in FORMS:
        parts.pop()
    n = " ".join(parts).strip(" -.,()/")
    n = NUMTAIL.sub("", n).strip()
    if not n or len(n) < 3:
        return None
    if not re.search(r"[a-z]", n):
        return None
    return n


def main() -> None:
    brands: dict[str, str] = {}          # base brand -> generic
    rows = 0
    with open(CSV_PATH, encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            rows += 1
            if (row.get("Is_discontinued") or "").strip().lower() == "true":
                continue
            if (row.get("type") or "").strip().lower() != "allopathy":
                continue
            b = base_name(row.get("name") or "")
            if not b:
                continue
            comp = (row.get("short_composition1") or "").strip()
            # "Amoxycillin  (500mg)" -> "Amoxycillin": the strength belongs
            # to the pack, not to the drug identity.
            generic = re.sub(r"\s*\(.*?\)", "", comp).strip()
            if not generic:
                continue
            brands.setdefault(b, generic)

    # ---- collision filtering: the part that matters -----------------------
    # Uses EXACT folded-key equality against the lookup tables, not the
    # is_clinical_term()/is_lab_test() helpers.
    #
    # Two reasons, and the second is the important one:
    #   - speed: those helpers run n-gram matching over ~2,400 entries at
    #     0.78ms a call, which is ~10 minutes across a quarter-million rows.
    #   - correctness: they match SUBSTRINGS, so any brand merely containing
    #     a symptom word would be dropped. A collision is when the brand IS
    #     the other thing, not when it contains it.
    kept: dict[str, str] = {}
    dropped: list[tuple[str, str]] = []
    for b, generic in brands.items():
        fb = fold(b)
        if len(fb) < MIN_LEN:
            dropped.append((b, "too short - would collide with ordinary words"))
            continue
        if fb in _TERM_LOOKUP:
            dropped.append((b, f"collides with clinical term '{_TERM_LOOKUP[fb]}'"))
            continue
        if fb in _LAB_LOOKUP:
            dropped.append((b, f"collides with lab test '{_LAB_LOOKUP[fb]}'"))
            continue
        if fb in _DRUG_LOOKUP:
            # already curated (with Bengali + department) - curated wins
            continue
        kept[b] = generic

    with open(sys.argv[1] if len(sys.argv) > 1 else "voicerx/brands_india.py",
              "w", encoding="utf-8") as out:
        out.write('"""AUTO-GENERATED - do not edit by hand.\n\n')
        out.write("Regenerate with:\n")
        out.write("    python tools/import_india_brands.py voicerx/brands_india.py\n\n")
        out.write("Indian brand -> generic names from the A-Z India medicines dataset.\n")
        out.write("MACHINE-IMPORTED, NOT CLINICALLY REVIEWED. Lower trust tier than the\n")
        out.write("curated DRUGS table in glossary.py, which always wins on conflict.\n")
        out.write("These carry no Bengali transliteration and no department.\n\n")
        out.write(f"csv rows scanned        : {rows}\n")
        out.write(f"distinct base brands    : {len(brands)}\n")
        out.write(f"kept                    : {len(kept)}\n")
        out.write(f"dropped (collisions)    : {len(dropped)}\n\n")
        out.write("sample of dropped names - each of these would have turned a symptom,\n")
        out.write("a lab test or a common word into a 'verified medication':\n")
        for b, why in sorted(dropped)[:60]:
            out.write(f"    {b} - {why}\n")
        out.write('"""\n\n')
        out.write("INDIA_BRANDS: dict[str, str] = {\n")
        for b in sorted(kept):
            g = kept[b].replace('"', "'")
            out.write(f'    "{b}": "{g}",\n')
        out.write("}\n")

    print(f"rows={rows} base_brands={len(brands)} kept={len(kept)} dropped={len(dropped)}")


if __name__ == "__main__":
    main()
