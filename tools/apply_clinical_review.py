"""Apply the clinician's review of the 102 proposed entries.

The review sorted them into four buckets, and each needs different
treatment:

  1 already present   Tobramycin/Timolol eye drops, Lignocaine gel,
                      Chlorhexidine - no action.
  2 NOT drugs/labs    ORS, "Admit in ICU", "IV Fluids if hypotensive",
                      Foley's catheterisation, compression stockings,
                      monofilament foot exam, proctoscopy, otoscopy, slit
                      lamp. These are advice, devices or procedures the
                      DOCTOR performs - never an orderable item.
  3 real drugs        ~35 absent from ALL_DRUGS.
  4 real labs         ~30 absent from LAB_TESTS.

Bucket 2 is added to CLINICAL_TERMS rather than simply left out. A term
that is positively identified as "a procedure, not a drug" is rejected by
the gate with a reason; a term that is merely absent falls through to
fuzzy matching, which is how "Antibiotic" once became an ear drop.

Bengali surface forms come from the transcripts wherever a human wrote
one - those are what the ASR has to match. Where no observed form exists,
a standard transliteration is used and marked, because a guessed spelling
is weaker evidence and should be replaceable later.

Usage:  python tools/apply_clinical_review.py
"""
from __future__ import annotations

import io
import re
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, ".")

GLOSSARY = "voicerx/glossary.py"

# ---------------------------------------------------------------------------
# Bucket 3 - real drugs. (generic, brands, bengali, indication, department)
# Bengali forms marked * were observed in the transcripts.
# ---------------------------------------------------------------------------
NEW_DRUGS = [
    # supplements
    ("Vitamin C", ("Limcee", "Celin", "Chewcee"), ("ভিটামিন সি",), "vitamin C supplement", "general"),
    ("Evening Primrose Oil", ("Evanova", "EPO"), ("ইভনিং প্রিমরোজ অয়েল",), "PMS / menopause", "gynaecology"),
    ("Biotin", ("Biotin", "Hairbon"), ("বায়োটিন",), "hair and nail supplement", "dermatology"),
    ("Zinc", ("Zincovit", "Z&D"), ("জিঙ্ক", "জিঙ্ক সিরাপ"), "zinc supplement", "general"),
    ("Lycopene", ("Lycostar", "Lycored"), ("লাইকোপিন",), "antioxidant supplement", "general"),
    # systemic
    ("Dexamethasone", ("Decadron", "Dexona"), ("ডেক্সামিথাসোন", "ডেক্সামিথাসোন ইনজেকশন"), "corticosteroid", "general"),
    ("Drotaverine", ("Drotin", "Doverin"), ("ড্রোটাবেরিন", "ড্রোটাভেরিন"), "antispasmodic", "gastro"),
    ("Diltiazem", ("Dilzem", "Angizem"), ("ডিলটিয়াজেম",), "calcium channel blocker", "cardiac"),
    ("Penicillin V", ("Pen-V", "Cilopen"), ("পেনিসিলিন ভি",), "antibiotic", "general"),
    ("Diosmin", ("Daflon", "Venusmin"), ("ডায়োসমিন",), "venotonic / piles", "surgery"),
    ("Methotrexate", ("Folitrax", "Imutrex"), ("মেথোট্রেক্সেট",), "DMARD / psoriasis", "bone"),
    ("Pramipexole", ("Pramipex", "Parkitidin"), ("প্রামিপেক্সোল",), "Parkinson's / RLS", "neurology"),
    ("Piperacillin-Tazobactam", ("Zosyn", "Pipzo"), ("পাইপেরাসিলিন-ট্যাজোব্যাকটাম",), "IV antibiotic", "surgery"),
    ("Iron Sucrose", ("Orofer S", "Encicarb"), ("আয়রন সুক্রোজ",), "IV iron / anaemia", "general"),
    ("Naproxen", ("Naprosyn", "Xenobid"), ("ন্যাপ্রোক্সেন",), "NSAID", "bone"),
    ("Hydroxyzine", ("Atarax", "Hyzine"), ("হাইড্রোক্সিজিন",), "antihistamine / pruritus", "dermatology"),
    ("Cefadroxil", ("Droxyl", "Cefadrox"), ("সেফাড্রক্সিল",), "antibiotic", "general"),
    ("Promethazine", ("Phenergan", "Avomine"), ("প্রোমিথাজিন",), "antihistamine / antiemetic", "general"),
    ("Colchicine", ("Zycolchin", "Goutnil"), ("কোলচিসিন",), "acute gout", "nephrology"),
    ("Albendazole", ("Zentel", "Bandy"), ("অ্যালবেনডাজল",), "anthelmintic", "general"),
    ("Ispaghula Husk", ("Isabgol", "Naturolax"), ("ইসবগুল হাস্ক", "ইসবগুল"), "bulk laxative", "gastro"),
    ("Pheniramine", ("Avil",), ("অ্যাভিল ইনজেকশন", "ফেনিরামিন"), "antihistamine injection", "general"),
    ("Lidocaine", ("Xylocaine", "Lox"), ("লিডোকেন",), "local anaesthetic", "surgery"),
    # topicals
    ("Choline Salicylate gel", ("Zytee", "Dentogel"), ("কোলিন স্যালিসাইলেট ওরাল জেল",), "mouth ulcer gel", "dental"),
    ("Tobramycin+Dexamethasone", ("Tobastar DM", "Tobradex"), ("টোব্রামাইসিন + ডেক্সামিথাসোন আই অয়েন্টমেন্ট",), "eye antibiotic-steroid", "ophthalmology"),
    ("Tannic acid+Iodine gum paint", ("Tannic acid gum paint",), ("ট্যানিক অ্যাসিড + আয়োডিন গাম পেইন্ট",), "gingivitis", "dental"),
    ("Conjugated Estrogen cream", ("Premarin",), ("কনজুগেটেড ইস্ট্রোজেন ভ্যাজাইনাল ক্রিম",), "atrophic vaginitis", "gynaecology"),
    ("Griseofulvin", ("Grisovin", "Walavin"), ("গ্রাইসিওফুলভিন",), "antifungal", "dermatology"),
    ("Petroleum Jelly", ("Vaseline",), ("পেট্রোলিয়াম জেলি",), "emollient", "dermatology"),
    ("Clindamycin+Nicotinamide gel", ("Clinsol NA", "Faceclin"), ("ক্লিনডামাইসিন + নিকোটিনামাইড জেল",), "acne gel", "dermatology"),
    ("Povidone iodine ointment", ("Isodine", "Betadine ointment"), ("আইসোডিন অয়েন্টমেন্ট",), "antiseptic ointment", "surgery"),
    ("Potassium Permanganate", ("KMnO4",), ("পটাশিয়াম পারম্যাঙ্গানেট",), "antiseptic soak", "dermatology"),
    ("Lignocaine+Hydrocortisone cream", ("Anobliss", "Proctosedyl"), ("লিগনোকেইন + হাইড্রোকর্টিসোন ক্রিম",), "piles cream", "surgery"),
    ("Fluticasone cream", ("Flutivate",), ("ফ্লুটিকাসোন ক্রিম",), "topical steroid", "dermatology"),
    ("Liquid Paraffin moisturizer", ("Moisturex", "Venusia"), ("লিকুইড প্যারাফিন ময়েশ্চারাইজার",), "emollient", "dermatology"),
    ("Sodium Hyaluronate eye drops", ("Hyalur", "I-Kul"), ("সোডিয়াম হাইলুরোনেট",), "dry eye", "ophthalmology"),
    ("Carbomer eye gel", ("Lubrigel", "Viscotears"), ("কার্বোমার আই জেল",), "dry eye gel", "ophthalmology"),
    # syrups
    ("Potassium Citrate", ("Alkasol", "Citralka"), ("পটাশিয়াম সাইট্রেট + ম্যাগনেশিয়াম সাইট্রেট সিরাপ", "সিরাপ আলকাসল", "আলকালাইজার সিরাপ"), "urinary alkaliniser", "urology"),
    ("Disodium Hydrogen Citrate", ("Citralka", "Alkacitral"), ("ডাইসোডিয়াম হাইড্রোজেন সাইট্রেট সিরাপ",), "urinary alkaliniser", "urology"),
]

# ---------------------------------------------------------------------------
# Bucket 4 - real lab tests / investigations.
# ---------------------------------------------------------------------------
NEW_LABS = {
    "Serum IgE": ("serum total ige", "total ige", "সিরাম টোটাল আইজিই", "আইজিই"),
    "Haemoglobin": ("hb", "haemoglobin", "hemoglobin", "হিমোগ্লোবিন", "এইচবি"),
    "ASO titre": ("aso titer", "aso titre", "এএসও টাইটার"),
    "C3 level": ("c3 level", "complement c3", "সি৩ লেভেল"),
    "Absolute Eosinophil Count": ("aec", "absolute eosinophil count", "অ্যাবসোলিউট ইওসিনোফিল কাউন্ট"),
    "CA-125": ("ca-125", "ca 125", "সিএ ১২৫"),
    "Serum albumin": ("serum albumin", "সিরাম অ্যালবুমিন"),
    "ABG": ("abg", "arterial blood gas", "এবিজি"),
    "Random blood sugar": ("random blood sugar", "rbs", "র‍্যান্ডম ব্লাড সুগার"),
    "Perimetry": ("perimetry", "vft", "visual field test", "পেরিমেট্রি", "ভিজুয়াল ফিল্ড টেস্ট"),
    "Dix-Hallpike test": ("dix-hallpike", "dix hallpike", "ডিক্স হলপাইক"),
    "Video laryngoscopy": ("video laryngoscopy", "ভিডিও ল্যারিঙ্গোস্কোপি"),
    "Indirect laryngoscopy": ("indirect laryngoscopy", "ইনডাইরেক্ট ল্যারিঙ্গোস্কোপি"),
    "HRCT temporal bone": ("hrct temporal bone", "hrct temporal", "এইচআরসিটি টেম্পোরাল বোন"),
    "Upper GI endoscopy": ("upper gi endoscopy", "ugi endoscopy", "আপার জিআই এন্ডোস্কোপি"),
    "Spirometry": ("spirometry", "pft", "pulmonary function test", "স্পাইরোমেট্রি", "পিএফটি"),
    "Renal biopsy": ("renal biopsy", "kidney biopsy", "রেনাল বায়োপসি"),
    "Incisional biopsy": ("incisional biopsy", "ইনসিশনাল বায়োপসি"),
    "High vaginal swab": ("high vaginal swab", "hvs", "hvs culture", "হাই ভ্যাজাইনাল সোয়াব",
                           "high vaginal swab for wet mount"),
    "Ulcer swab culture": ("ulcer swab", "ulcer swab for culture", "আলসার সোয়াব"),
    "FNAC": ("fnac", "fnac of breast lump", "এফএনএসি"),
    "Tear film breakup time": ("tbut", "tbut test", "tear film breakup time", "টিয়ার ফিল্ম ব্রেকআপ টাইম"),
    "Schirmer's test": ("schirmer", "schirmers test", "schirmer's test", "শির্মের্স টেস্ট"),
    "Wood's lamp examination": ("wood's lamp", "woods lamp examination", "উডস ল্যাম্প এক্সামিনেশন"),
    "Lacrimal sac syringing": ("lacrimal sac syringing", "syringing", "ল্যাক্রিমাল স্যাক সিরিঞ্জিং"),
}

# ---------------------------------------------------------------------------
# Bucket 2 - procedures, devices and instructions. NOT drugs, NOT labs.
#
# Added positively rather than left absent. An absent term falls through to
# fuzzy matching - which is how "Antibiotic" once became an ear drop at
# 0.86 similarity. A named term is rejected with a reason.
# ---------------------------------------------------------------------------
NEW_TERMS = {
    "admit to ICU": ("admit in icu", "icu admission", "আইসিইউ তে ভর্তি", "আইসিইউ"),
    "IV fluids": ("iv fluids", "iv fluids if hypotensive", "আইভি ফ্লুইডস", "স্যালাইন"),
    "catheterisation": ("foley's catheterization", "foleys catheterisation", "catheterisation",
                         "ফলিস ক্যাথেটারাইজেশন", "ক্যাথেটার"),
    "compression stockings": ("compression stockings", "class ii compression stockings",
                               "কম্প্রেশন স্টকিংস"),
    "monofilament foot examination": ("foot examination with monofilament", "monofilament test",
                                       "মনোফিলামেন্ট"),
    "proctoscopy": ("proctoscopy", "প্রক্টোস্কোপি"),
    "otoscopy": ("otoscopy", "otoscopic examination", "ওটোস্কোপি"),
    "slit lamp examination": ("slit lamp", "slit lamp examination", "স্লিট ল্যাম্প এক্সামিনেশন"),
}


def main() -> None:
    src = open(GLOSSARY, encoding="utf-8").read()

    drug_lines = ["\n# ---------------------------------------------------------------------------",
                  "# ADDED FROM CLINICAL REVIEW of 102 entries seen in real transcripts.",
                  "# Bengali forms are the ones a human wrote beside the English name",
                  "# wherever the transcripts supplied one - those are what the ASR must",
                  "# match. Reviewed by the clinician; still not pharmacist-verified.",
                  "# ---------------------------------------------------------------------------",
                  "REVIEWED = ["]
    for generic, brands, bn, indication, dept in NEW_DRUGS:
        b = ", ".join(f'"{x}"' for x in brands) + ("," if len(brands) == 1 else "")
        g = ", ".join(f'"{x}"' for x in bn) + ("," if len(bn) == 1 else "")
        drug_lines.append(f'    Drug("{generic}", ({b}),')
        drug_lines.append(f'         ({g}),')
        drug_lines.append(f'         "{indication}", "{dept}"),')
    drug_lines.append("]\n")

    anchor = "ALL_DRUGS: list[Drug] = (CARDIAC"
    src = src.replace(anchor, "\n".join(drug_lines) + "\n" + anchor, 1)
    src = re.sub(r"(ALL_DRUGS: list\[Drug\] = \([^)]*)\)",
                 r"\1 + REVIEWED)", src, count=1)

    lab_lines = ["    # --- added from clinical review of real transcripts ---"]
    for canon, aliases in NEW_LABS.items():
        a = ", ".join(f'"{x}"' for x in aliases)
        lab_lines.append(f'    "{canon}": ({a}),')
    src = src.replace('    # Generic orders - a test WAS ordered even if unnamed.',
                      "\n".join(lab_lines) + "\n    # Generic orders - a test WAS ordered even if unnamed.", 1)

    term_lines = ["    # --- procedures, devices and instructions: NOT drugs, NOT labs ---",
                  "    # Named positively so the gate rejects them with a reason. An absent",
                  "    # term falls through to fuzzy matching instead."]
    for canon, aliases in NEW_TERMS.items():
        a = ", ".join(f'"{x}"' for x in aliases)
        term_lines.append(f'    "{canon}": ({a}),')
    src = src.replace("    # advice - explicitly NOT medications",
                      "\n".join(term_lines) + "\n    # advice - explicitly NOT medications", 1)

    open(GLOSSARY, "w", encoding="utf-8").write(src)
    print(f"added {len(NEW_DRUGS)} drugs, {len(NEW_LABS)} labs, {len(NEW_TERMS)} procedure terms")


if __name__ == "__main__":
    main()
