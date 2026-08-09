"""Learn Bengali drug/lab spellings from human-labelled transcripts.

WHY THIS BEATS TTS DISCOVERY
----------------------------
The TTS route guesses what the ASR *might* emit by speaking a name and
transcribing it. These transcripts carry something better: a human wrote
the Bengali form and the English name side by side -

    "মন্টিলোকাস্ট (Montelukast)"
    "ই সি জি (ECG)"

That is a verified pair, not an inference.

WHAT IT FIXES
-------------
Coverage on 188 real department transcripts was 62.2% for drugs. The
misses were mostly NOT unknown drugs - they were known drugs under a
different Bengali spelling:

    transcript  মন্টিলোকাস্ট      gazetteer  মন্টিকুলাষ্ট     Montelukast
    transcript  লিভোসেট্রিজিন     gazetteer  লেভোসেটিরিজিন    Levocetirizine
    transcript  কিটোরোল্যাক       gazetteer  কিটোরোলাক        Ketorolac

The phonetic fold does not bridge those - the vowels differ too much - and
it should not be loosened until it does, because that was tried and broke
other matching. Adding the observed spelling as an alias is the correct
fix, and the transcripts supply it directly.

WHAT IT REFUSES TO LEARN
------------------------
Not everything in a prescription is a drug. These appear in the ground
truth and must never enter the drug table:

    "ব্রেস্ট পাম্প (Breast Pump)"          a device
    "রিস্ট স্প্লিন্ট (Wrist Splint)"        a device
    "নরম টুথব্রাশ (Soft bristles toothbrush)"  not a medicine
    "নো অ্যাক্টিভ মেডিকেশন রিকোয়ার্ড"        a statement that there is none

Anything whose English side is not resolvable as a drug is emitted for
REVIEW rather than silently added.

Usage:
    python tools/learn_from_transcripts.py D:/v2r/transcript.json
"""
from __future__ import annotations

import collections
import io
import json
import re
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, ".")

from voicerx.gate import judge_medication          # noqa: E402
from voicerx.glossary import fold, is_lab_test, lookup_drug   # noqa: E402

# English sides that are devices, advice or absence-of-treatment.
NOT_A_DRUG = re.compile(
    r"pump|splint|brush|bandage|dressing|no active|not required|none|"
    r"diet|exercise|rest|advice|counsel|physiotherapy|surgery|referral|"
    r"spectacle|glasses|hearing aid|crutch|walker|belt|collar|mask",
    re.I)


def load(path: str) -> list[dict]:
    """These files are concatenated JSON objects with citation markers."""
    s = re.sub(r"\[cite:[^\]]*\]", "", open(path, encoding="utf-8").read())
    dec, i, rows = json.JSONDecoder(), 0, []
    while i < len(s):
        while i < len(s) and s[i] in " \r\n\t,[]":
            i += 1
        if i >= len(s):
            break
        try:
            obj, j = dec.raw_decode(s, i)
            rows.append(obj)
            i = j
        except json.JSONDecodeError:
            i += 1
    return rows


def split_pair(entry: str) -> tuple[str, str]:
    """'মন্টিলোকাস্ট ১০ এমজি (Montelukast 10mg)' -> (bengali, english)"""
    m = re.match(r"^(.*?)\s*\(([^)]*)\)\s*$", entry.strip())
    if not m:
        return entry.strip(), ""
    return m.group(1).strip(), m.group(2).strip()


def strip_strength(text: str) -> str:
    """Drop a trailing dose: '10mg', '৫ এমজি', '2% soap'."""
    t = re.split(r"[\u09e6-\u09ef0-9]", text)[0].strip()
    return t or text.strip()


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else r"D:/v2r/transcript.json"
    rows = load(path)
    print(f"transcripts: {len(rows)}\n")

    new_drug_aliases: dict[str, set[str]] = collections.defaultdict(set)
    new_lab_aliases: dict[str, set[str]] = collections.defaultdict(set)
    unknown_drugs: collections.Counter = collections.Counter()
    refused: collections.Counter = collections.Counter()

    for r in rows:
        pr = r.get("prescription") or {}

        for entry in pr.get("drugs", []):
            bn, en = split_pair(entry)
            if not bn or not en:
                continue
            if NOT_A_DRUG.search(en):
                refused[en] += 1
                continue
            bn_base = strip_strength(bn)
            en_base = strip_strength(en)
            hit = lookup_drug(en_base) or lookup_drug(en)
            if hit:
                # Known drug, possibly new Bengali spelling.
                if not lookup_drug(bn_base):
                    new_drug_aliases[hit.generic].add(bn_base)
            else:
                unknown_drugs[f"{en_base} :: {bn_base}"] += 1

        for entry in pr.get("lab_tests", []):
            bn, en = split_pair(entry)
            if not bn or not en:
                continue
            canon = is_lab_test(en) or is_lab_test(bn)
            if canon and not is_lab_test(bn):
                new_lab_aliases[canon].add(bn.strip())
            elif not canon:
                unknown_drugs[f"[LAB] {en} :: {bn}"] += 1

    print(f"NEW BENGALI SPELLINGS for drugs already known : "
          f"{sum(len(v) for v in new_drug_aliases.values())} "
          f"across {len(new_drug_aliases)} drugs")
    for generic in sorted(new_drug_aliases)[:15]:
        print(f"    {generic:24} += {sorted(new_drug_aliases[generic])}")

    print(f"\nNEW BENGALI SPELLINGS for labs already known  : "
          f"{sum(len(v) for v in new_lab_aliases.values())}")
    for canon in sorted(new_lab_aliases)[:10]:
        print(f"    {canon:24} += {sorted(new_lab_aliases[canon])}")

    print(f"\nNOT IN THE GAZETTEER AT ALL - need curation   : {len(unknown_drugs)}")
    for name, n in unknown_drugs.most_common(25):
        print(f"    x{n:2}  {name}")

    print(f"\nREFUSED (devices / advice, correctly not drugs): {sum(refused.values())}")
    for name, n in refused.most_common(8):
        print(f"    x{n:2}  {name}")

    # Entries the gazetteer lacks entirely. Emitted for REVIEW rather than
    # silently added: the Bengali form is trustworthy (a human wrote it),
    # but whether each is really a drug still needs a person.
    lines = ["PROPOSED_DRUGS = ["]
    for key, n in unknown_drugs.most_common():
        if key.startswith("[LAB]"):
            continue
        parts = (key.split(" :: ") + [""])[:2]
        lines.append('    ("%s", "%s"),   # seen %dx' % (parts[0], parts[1], n))
    lines.append("]")
    lines.append("")
    lines.append("PROPOSED_LABS = [")
    for key, n in unknown_drugs.most_common():
        if not key.startswith("[LAB]"):
            continue
        parts = (key.replace("[LAB] ", "").split(" :: ") + [""])[:2]
        lines.append('    ("%s", "%s"),   # seen %dx' % (parts[0], parts[1], n))
    lines.append("]")
    with open("voicerx/proposed_entries.py", "w", encoding="utf-8") as f:
        f.write(chr(10).join(lines) + chr(10))

    out = "voicerx/learned_forms.py"
    with open(out, "w", encoding="utf-8") as f:
        f.write('"""AUTO-GENERATED from human-labelled transcripts.\n\n')
        f.write("Regenerate:\n")
        f.write("    python tools/learn_from_transcripts.py <transcript.json>\n\n")
        f.write("Bengali spellings a HUMAN paired with an English drug/lab name.\n")
        f.write("Higher trust than TTS discovery, which only infers what the ASR\n")
        f.write("might emit - these were actually written down next to the name.\n\n")
        f.write(f"drug aliases : {sum(len(v) for v in new_drug_aliases.values())}\n")
        f.write(f"lab aliases  : {sum(len(v) for v in new_lab_aliases.values())}\n")
        f.write('"""\n\n')
        f.write("LEARNED_DRUG_FORMS: dict[str, tuple[str, ...]] = {\n")
        for g in sorted(new_drug_aliases):
            v = ", ".join(f'"{x}"' for x in sorted(new_drug_aliases[g]))
            f.write(f'    "{g}": ({v},),\n')
        f.write("}\n\nLEARNED_LAB_FORMS: dict[str, tuple[str, ...]] = {\n")
        for c in sorted(new_lab_aliases):
            v = ", ".join(f'"{x}"' for x in sorted(new_lab_aliases[c]))
            f.write(f'    "{c}": ({v},),\n')
        f.write("}\n")
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
