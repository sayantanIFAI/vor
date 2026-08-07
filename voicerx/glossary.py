"""Curated clinical gazetteer — the authority on what is and isn't a drug.

WHY THIS REPLACES PROMPT-TUNING
-------------------------------
"Is ORS a medication? Is 'light food' a medication?" is a closed-set
question. Drug names are a finite, enumerable list. Asking a 7B model to
judge set membership is the wrong tool when the answer can simply be looked
up — and two rounds of prompt-tuning in testing produced under-extraction
rather than fewer false positives.

So: the SLM PROPOSES, this glossary DECIDES. A term that is not in
DRUGS cannot enter the medications array, no matter how confident the model
is. It is demoted to unrecognized_terms for a human to judge.

STATUS: DRAFTED BY AI, PENDING CLINICAL REVIEW.
This encodes standard Indian OPD formulary knowledge plus the specific
drugs named by the clinician on this project. It has NOT been reviewed by a
pharmacist. Before any real clinical deployment, a qualified clinician must
verify every entry — particularly the Bengali transliterations, which
determine whether a spoken drug name is recognised at all.

Adding entries is the intended way to grow this. If a real drug is missing,
it gets demoted to unrecognized_terms — visibly, not silently — which is
the safe direction to fail.
"""
from __future__ import annotations

import dataclasses
import unicodedata


@dataclasses.dataclass(frozen=True)
class Drug:
    generic: str                  # canonical generic name
    brands: tuple[str, ...] = ()  # common Indian brand names
    bengali: tuple[str, ...] = () # transliterations the ASR actually emits
    indication: str = ""          # what it treats - shown to the reviewer
    department: str = "general"


# ---------------------------------------------------------------------------
# CARDIAC
# ---------------------------------------------------------------------------
CARDIAC = [
    Drug("Nitroglycerin", ("Nitrocontin", "Sorbitrate", "Angispan"),
         ("নাইট্রোকন্টিন", "নাইট্রোগ্লিসারিন", "সরবিট্রেট"),
         "angina / severe chest pain", "cardiac"),
    Drug("Aspirin", ("Ecosprin", "Ecospirin", "Disprin"),
         ("ইকোস্পিরিন", "একোস্পিরিন", "অ্যাসপিরিন"),
         "antiplatelet / blood thinner", "cardiac"),
    Drug("Clopidogrel", ("Plavix", "Clopilet", "Deplatt"),
         ("ক্লোপিডোগ্রেল", "ক্লোপিলেট"),
         "antiplatelet", "cardiac"),
    Drug("Atorvastatin", ("Atorva", "Lipitor", "Storvas"),
         ("অ্যাটোরভাস্ট্যাটিন", "এটোরভা"),
         "cholesterol / statin", "cardiac"),
    Drug("Rosuvastatin", ("Rosuvas", "Crestor"),
         ("রোসুভাস্ট্যাটিন", "রোসুভাস"),
         "cholesterol / statin", "cardiac"),
    Drug("Amlodipine", ("Amlopres", "Amlokind", "Stamlo"),
         ("অ্যামলোডিপিন", "অ্যামলোপ্রেস"),
         "hypertension", "cardiac"),
    Drug("Telmisartan", ("Telma", "Telsartan"),
         ("টেলমিসারটান", "টেলমা"),
         "hypertension", "cardiac"),
    Drug("Metoprolol", ("Metolar", "Betaloc"),
         ("মেটোপ্রোলল", "মেটোলার"),
         "beta blocker", "cardiac"),
    Drug("Bisoprolol", ("Concor", "Bisolol"),
         ("বিসোপ্রোলল", "কনকর"),
         "beta blocker", "cardiac"),
    Drug("Ramipril", ("Cardace", "Ramistar"),
         ("রামিপ্রিল", "কার্ডেস"),
         "ACE inhibitor / hypertension", "cardiac"),
    Drug("Torsemide", ("Dytor", "Dytor Plus"),
         ("ডাইটর", "টরসেমাইড", "ডায়টর"),
         "diuretic / fluid retention", "cardiac"),
    Drug("Furosemide", ("Lasix", "Frusenex"),
         ("ফুরোসেমাইড", "ল্যাসিক্স"),
         "diuretic", "cardiac"),
    Drug("Spironolactone", ("Aldactone",),
         ("স্পাইরোনোল্যাকটোন", "অ্যালড্যাকটোন"),
         "diuretic", "cardiac"),
    Drug("Isosorbide mononitrate", ("Monotrate", "Ismo"),
         ("আইসোসরবাইড", "মোনোট্রেট"),
         "angina prophylaxis", "cardiac"),
]

# ---------------------------------------------------------------------------
# DIABETES / ENDOCRINE
# ---------------------------------------------------------------------------
ENDOCRINE = [
    Drug("Semaglutide", ("Rybelsus", "Ozempic"),
         ("রাইবেলসাস", "সেমাগ্লুটাইড", "রাইবেলসাস"),
         "oral GLP-1 / type 2 diabetes", "endocrine"),
    Drug("Metformin", ("Glycomet", "Glucophage", "Okamet"),
         ("মেটফরমিন", "গ্লাইকোমেট"),
         "type 2 diabetes, first line", "endocrine"),
    Drug("Glimepiride", ("Amaryl", "Glimestar"),
         ("গ্লিমিপিরাইড", "অ্যামারিল"),
         "sulfonylurea / diabetes", "endocrine"),
    Drug("Gliclazide", ("Diamicron", "Glizid"),
         ("গ্লিক্লাজাইড", "ডায়ামিক্রন"),
         "sulfonylurea / diabetes", "endocrine"),
    Drug("Sitagliptin", ("Januvia", "Istavel"),
         ("সিটাগ্লিপটিন", "জানুভিয়া"),
         "DPP-4 inhibitor / diabetes", "endocrine"),
    Drug("Dapagliflozin", ("Forxiga", "Dapa"),
         ("ড্যাপাগ্লিফ্লোজিন", "ফরজিগা"),
         "SGLT2 inhibitor / diabetes", "endocrine"),
    Drug("Insulin degludec/aspart", ("Ryzodeg",),
         ("রাইজোডেগ", "রাইজোডেগ ইনসুলিন"),
         "combination insulin", "endocrine"),
    Drug("Insulin glargine", ("Lantus", "Basalog", "Glaritus"),
         ("ল্যান্টাস", "ইনসুলিন গ্লার্জিন"),
         "long-acting insulin", "endocrine"),
    Drug("Human insulin", ("Huminsulin", "Actrapid", "Mixtard"),
         ("ইনসুলিন", "হিউমিনসুলিন", "মিক্সটার্ড"),
         "insulin", "endocrine"),
    Drug("Levothyroxine", ("Thyronorm", "Eltroxin", "Thyrox"),
         ("থাইরোনর্ম", "থাইরক্স", "লেভোথাইরক্সিন", "এলট্রক্সিন"),
         "hypothyroidism", "endocrine"),
    Drug("Voglibose", ("Volix", "Vogs"),
         ("ভোগলিবোস",), "diabetes", "endocrine"),
]

# ---------------------------------------------------------------------------
# RESPIRATORY
# ---------------------------------------------------------------------------
RESPIRATORY = [
    Drug("Salbutamol", ("Asthalin", "Ventolin", "Levolin"),
         ("সালবুটামল", "অ্যাসথালিন", "ভেন্টোলিন"),
         "bronchodilator / nebulisation", "respiratory"),
    Drug("Budesonide", ("Budecort", "Pulmicort"),
         ("বুডেসোনাইড", "বুডেকর্ট"),
         "inhaled steroid", "respiratory"),
    Drug("Ipratropium", ("Duolin", "Ipravent"),
         ("আইপ্রাট্রোপিয়াম", "ডুওলিন"),
         "bronchodilator", "respiratory"),
    Drug("Montelukast", ("Montair", "Montek", "Monticope"),
         ("মন্টিকুলাষ্ট", "মন্টিকুলাস", "মন্টেয়ার", "মন্টেক"),
         "asthma / allergic rhinitis", "respiratory"),
    Drug("Ambroxol", ("Ambrodil", "Mucolite"),
         ("অ্যামব্রক্সল", "অ্যামব্রোডিল"),
         "mucolytic / cough", "respiratory"),
    Drug("Ascoril", ("Ascoril LS", "Ascoril D"),
         ("এস্কোরিল", "আস্কোরিল", "অ্যাসকোরিল"),
         "cough syrup (combination)", "respiratory"),
    Drug("Levocetirizine", ("Levocet", "Xyzal"),
         ("লেভোসেটিরিজিন", "লেভোসেট"),
         "antihistamine", "respiratory"),
    Drug("Cetirizine", ("Cetzine", "Alerid"),
         ("সেটিরিজিন", "সেটজিন"),
         "antihistamine", "respiratory"),
    Drug("Fexofenadine", ("Allegra", "Fexova"),
         ("ফেক্সোফেনাডিন", "অ্যালেগ্রা"),
         "antihistamine", "respiratory"),
    Drug("Deriphyllin", ("Deriphyllin",),
         ("ডেরিফাইলিন",), "bronchodilator", "respiratory"),
]

# ---------------------------------------------------------------------------
# GASTROINTESTINAL
# ---------------------------------------------------------------------------
GI = [
    Drug("Norfloxacin+Tinidazole", ("Norflox-TZ", "Norflox TZ", "Normet"),
         ("নরফ্লক্স টি জেড", "নরফ্লক্স", "নলাক্স", "নরফ্লক্স-টিজেড"),
         "infective diarrhoea", "gastro"),
    Drug("Pantoprazole", ("Pantocid", "Pan-D", "Pantop"),
         ("প্যান্টোপ্রাজল", "প্যানটোসিড", "প্যান ডি"),
         "acidity / PPI", "gastro"),
    Drug("Omeprazole", ("Omez", "Ocid"),
         ("ওমিপ্রাজল", "ওমেজ"),
         "acidity / PPI", "gastro"),
    Drug("Rabeprazole", ("Razo", "Rabekind"),
         ("র‍্যাবিপ্রাজল", "রাজো"),
         "acidity / PPI", "gastro"),
    Drug("Domperidone", ("Domstal", "Vomistop"),
         ("ডমপেরিডন", "ডমস্টাল"),
         "nausea / vomiting", "gastro"),
    Drug("Ondansetron", ("Emeset", "Vomikind", "Zofer"),
         ("অনডানসেট্রন", "ইমিসেট"),
         "antiemetic", "gastro"),
    Drug("Metronidazole", ("Flagyl", "Metrogyl"),
         ("মেট্রোনিডাজল", "মেট্রোজিল", "ফ্ল্যাজিল"),
         "anaerobic / amoebic infection", "gastro"),
    Drug("Ofloxacin+Ornidazole", ("O2", "Oflomac-OZ"),
         ("অফ্লক্সাসিন", "ওফ্লোম্যাক"),
         "diarrhoea / infection", "gastro"),
    Drug("Racecadotril", ("Redotil", "Zedott"),
         ("রেসিকাডোট্রিল",), "acute diarrhoea", "gastro"),
    Drug("Sucralfate", ("Sucral", "Sucrafil"),
         ("সুক্রালফেট",), "gastric ulcer", "gastro"),
    Drug("Dicyclomine", ("Cyclopam", "Meftal-Spas"),
         ("ডাইসাইক্লোমিন", "সাইক্লোপাম"),
         "abdominal cramps", "gastro"),
    Drug("Lactulose", ("Duphalac", "Looz"),
         ("ল্যাকটুলোজ", "ডুফালাক"),
         "constipation", "gastro"),
]

# ---------------------------------------------------------------------------
# ANALGESIC / ANTIPYRETIC / ANTIBIOTIC (general OPD)
# ---------------------------------------------------------------------------
GENERAL = [
    Drug("Paracetamol", ("Crocin", "Dolo", "Calpol", "Pyrigesic"),
         ("প্যারাসিটামল", "প্যারাসিটাম", "ক্রোসিন", "ডোলো", "কালপল"),
         "fever / pain", "general"),
    Drug("Ibuprofen", ("Brufen", "Combiflam"),
         ("আইবুপ্রোফেন", "ব্রুফেন", "কম্বিফ্লাম"),
         "pain / inflammation", "general"),
    Drug("Diclofenac", ("Voveran", "Volini"),
         ("ডাইক্লোফেনাক", "ভোভেরান"),
         "pain / inflammation", "general"),
    Drug("Aceclofenac", ("Zerodol", "Hifenac"),
         ("এসিক্লোফেনাক", "জেরোডল"),
         "pain / inflammation", "general"),
    Drug("Amoxicillin+Clavulanate", ("Augmentin", "Clavam", "Moxikind-CV"),
         ("ক্ল্যাভাম", "ক্লাব", "অগমেন্টিন", "অ্যামোক্সিক্লাভ"),
         "broad-spectrum antibiotic", "general"),
    Drug("Amoxicillin", ("Mox", "Novamox"),
         ("অ্যামোক্সিসিলিন", "মক্স"),
         "antibiotic", "general"),
    Drug("Azithromycin", ("Azithral", "Azee", "Zithromax"),
         ("অ্যাজিথ্রোমাইসিন", "অ্যাজিথ্রাল"),
         "antibiotic", "general"),
    Drug("Cefixime", ("Taxim-O", "Zifi", "Mahacef"),
         ("সেফিক্সিম", "ট্যাক্সিম ও", "জিফি"),
         "antibiotic", "general"),
    Drug("Ciprofloxacin", ("Ciplox", "Cifran"),
         ("সিপ্রোফ্লক্সাসিন", "সিপ্লক্স"),
         "antibiotic", "general"),
    Drug("Levofloxacin", ("Levoflox", "Levotas"),
         ("লেভোফ্লক্সাসিন", "লেভোফ্লক্স"),
         "antibiotic", "general"),
    Drug("Doxycycline", ("Doxt", "Doxy-1"),
         ("ডক্সিসাইক্লিন",), "antibiotic", "general"),
    Drug("Diazepam", ("Valium", "Calmpose"),
         ("ভ্যালিয়াম", "ভ্যালুম", "ডায়াজেপাম"),
         "anxiolytic / sedative", "general"),
    Drug("Alprazolam", ("Alprax", "Restyl"),
         ("অ্যালপ্রাজোলাম", "অ্যালপ্র্যাক্স"),
         "anxiolytic", "general"),
    Drug("Prednisolone", ("Omnacortil", "Wysolone"),
         ("প্রেডনিসোলন", "ওমনাকর্টিল"),
         "steroid", "general"),
    Drug("Vitamin B complex", ("Becosules", "Neurobion"),
         ("বিকোসুলস", "নিউরোবিন", "ভিটামিন বি"),
         "supplement", "general"),
    Drug("Vitamin D3", ("Calcirol", "Uprise-D3"),
         ("ভিটামিন ডি", "ক্যালসিরল"),
         "supplement", "general"),
    Drug("Calcium carbonate", ("Shelcal", "Calcimax"),
         ("শেলক্যাল", "ক্যালসিয়াম"),
         "supplement", "general"),
    Drug("Iron/Folic acid", ("Autrin", "Fefol", "Orofer"),
         ("আয়রন", "ফলিক অ্যাসিড"),
         "supplement / anaemia", "general"),
    Drug("Multivitamin", ("Zincovit", "A to Z"),
         ("জিঙ্কোভিট", "মাল্টিভিটামিন"),
         "supplement", "general"),
]

# ---------------------------------------------------------------------------
# UROLOGY / PROSTATE
# ---------------------------------------------------------------------------
UROLOGY = [
    Drug("Tamsulosin", ("Urimax", "Veltam"),
         ("ট্যামসুলোসিন", "ইউরিম্যাক্স"),
         "prostate / BPH", "urology"),
    Drug("Finasteride", ("Finast", "Fincar"),
         ("ফিনাস্টেরাইড",), "prostate / BPH", "urology"),
    Drug("Nitrofurantoin", ("Niftran", "Martifur"),
         ("নাইট্রোফুরানটোইন",), "urinary tract infection", "urology"),
]

ALL_DRUGS: list[Drug] = (CARDIAC + ENDOCRINE + RESPIRATORY + GI
                          + GENERAL + UROLOGY)

# ---------------------------------------------------------------------------
# LAB TESTS / INVESTIGATIONS - these are ORDERS, never medications
# ---------------------------------------------------------------------------
# NOTE ON SPACED ACRONYMS - learned from real audio, not assumed.
# Clinicians SPELL OUT acronyms, and the ASR transcribes each letter as a
# separate Bengali token. Real example from 8.wav: "আপনি এই সি বি সি টা
# করবেন" - that is C-B-C spoken letter by letter. An earlier version of
# this table only had the unspaced "সিবিসি" and therefore matched ZERO lab
# tests across 255 real segments. Always include the spaced form.
LAB_TESTS: dict[str, tuple[str, ...]] = {
    # cardiac
    "ECG": ("ইসিজি", "ই সি জি", "ইকেজি", "electrocardiogram", "e c g"),
    "Echo": ("ইকো", "echocardiogram", "2d echo", "টু ডি ইকো"),
    "TMT": ("টিএমটি", "টি এম টি", "treadmill test", "stress test", "t m t"),
    "Angiography": ("অ্যাঞ্জিওগ্রাফি", "angiogram", "অ্যাঞ্জিওগ্রাম",
                     "এনজিওগ্রাম", "এঞ্জিওগ্রাফি"),
    "Lipid profile": ("লিপিড প্রোফাইল", "cholesterol test", "লিপিড"),
    "Troponin": ("ট্রপোনিন", "ট্রপ"),
    # neuro
    "EEG": ("ইইজি", "ই ই জি", "electroencephalogram", "e e g"),
    "MRI": ("এমআরআই", "এম আর আই", "m r i"),
    "CT scan": ("সিটি স্ক্যান", "সি টি স্ক্যান", "সিটি"),
    # metabolic / blood
    "Creatinine": ("ক্রিয়েটিনিন", "creatine", "ক্রিয়েটিন"),
    "Urea": ("ইউরিয়া",),
    "PP sugar": ("পিপি সুগার", "পি পি সুগার", "পোস্ট প্রান্ডিয়াল",
                  "post prandial sugar", "খাওয়ার পরের সুগার"),
    "Fasting sugar": ("ফাস্টিং সুগার", "FBS", "খালি পেটে সুগার",
                       "এফ বি এস"),
    "HbA1c": ("এইচবিএ১সি", "এইচ বি এ ওয়ান সি", "গ্লাইকোসাইলেটেড হিমোগ্লোবিন"),
    "TSH": ("টিএসএইচ", "টি এস এইচ", "থাইরয়েড টেস্ট", "thyroid profile"),
    # NOTE: "রক্ত পরীক্ষা" / "ব্লাড টেস্ট" are deliberately NOT CBC aliases.
    # "Blood test" is not necessarily a complete blood count, and turning a
    # generic phrase into a specific named order would be the pipeline
    # inventing a clinical decision. They fall through to "blood test
    # (unspecified)" so the reviewer names it.
    "CBC": ("সিবিসি", "সি বি সি", "complete blood count", "c b c",
             "কমপ্লিট ব্লাড কাউন্ট", "কমপ্লিট ব্লাড"),
    "LFT": ("এলএফটি", "এল এফ টি", "liver function test"),
    "KFT": ("কেএফটি", "কে এফ টি", "kidney function test", "RFT",
             "আর এফ টি"),
    "Urine routine": ("ইউরিন", "প্রস্রাব পরীক্ষা", "urine test",
                       "ইউরিন টেস্ট"),
    "Uric acid": ("ইউরিক অ্যাসিড", "ইউরিক"),
    "Vitamin D": ("ভিটামিন ডি টেস্ট",),
    "X-ray": ("এক্স রে", "এক্সরে", "চেস্ট এক্স রে", "এক্স-রে"),
    "USG": ("ইউএসজি", "ইউ এস জি", "আল্ট্রাসাউন্ড", "ultrasound",
             "sonography", "আলট্রাসনোগ্রাফি"),
    "PSA": ("পিএসএ", "পি এস এ"),
    # Generic orders - a test WAS ordered even if unnamed. Surfaced so the
    # reviewer names it, rather than dropped silently or guessed at.
    "blood test (unspecified)": ("রক্ত পরীক্ষা", "ব্লাড টেস্ট", "রক্ত টেস্ট"),
    "test (unspecified)": ("পরীক্ষা করাতে", "টেস্ট দিচ্ছি", "টেস্ট করবেন",
                            "পরীক্ষা করতে", "পরীক্ষা করতে হবে"),
}

# ---------------------------------------------------------------------------
# CLINICAL TERMS - symptoms, findings, advice.
# Present specifically so these are NEVER mistaken for drugs. "Cholesterol",
# "sugar" and "pressure" were being classified as medications.
# ---------------------------------------------------------------------------
CLINICAL_TERMS: dict[str, tuple[str, ...]] = {
    # cardiac
    "angina": ("অ্যাঞ্জাইনা",),
    "chest pain": ("বুকে ব্যথা", "বুক ব্যথা", "চেস্ট পেইন"),
    "blockage": ("ব্লকেজ", "ব্লক"),
    "palpitations": ("ধড়ফড়", "হার্ট বিট"),
    "breathlessness": ("শ্বাসকষ্ট", "দম বন্ধ"),
    "cholesterol": ("কোলেস্টেরল", "কলেস্টেরল"),
    "HDL": ("এইচডিএল",),
    "LDL": ("এলডিএল",),
    "blood pressure": ("প্রেশার", "ব্লাড প্রেশার", "রক্তচাপ", "প্রেসার"),
    # metabolic
    "blood sugar": ("সুগার", "রক্তে চিনি", "ব্লাড সুগার"),
    "thyroid": ("থাইরয়েড",),
    # general symptoms
    "fever": ("জ্বর",),
    "cough": ("কাশি",),
    "headache": ("মাথা ব্যথা", "মাথাব্যথা"),
    "abdominal pain": ("পেট ব্যথা", "পেটে ব্যথা"),
    "loose stools": ("পাতলা পায়খানা", "ডায়রিয়া", "পায়খানা"),
    "vomiting": ("বমি",),
    "nausea": ("গা গোলানো",),
    "body ache": ("শরীর ব্যথা", "গা ব্যথা"),
    "sore throat": ("গলা ব্যথা",),
    "swelling": ("ফোলা", "ফুলে যাওয়া"),
    "weakness": ("দুর্বলতা", "দুর্বল"),
    "difficulty swallowing": ("গিলতে কষ্ট",),
    "phlegm": ("কফ", "শ্লেষ্মা"),
    "infection": ("ইনফেকশন", "সংক্রমণ"),
    # advice - explicitly NOT medications
    "exercise": ("ব্যায়াম", "এক্সারসাইজ", "হাঁটা", "walking"),
    "lean diet": ("লিন ডায়েট", "হালকা খাবার", "light food"),
    "avoid oily food": ("তেল মশলা এড়িয়ে", "তেলমশলা"),
    "drink water": ("বেশি পানি", "জল খাবেন"),
    "ORS": ("ওআরএস", "ওরস"),   # rehydration, NOT a pharmaceutical
    "rest": ("বিশ্রাম",),
    "follow up": ("ফলো আপ", "আবার দেখাবেন"),
    "bandage": ("ব্যাণ্ডেজ", "ব্যান্ডেজ"),
    "dressing": ("ড্রেসিং",),
    "nebulization": ("নেবুলাইজেশন", "নেবুলাইজার"),
    "prescription": ("প্রেসক্রিপশন",),
}


# ===========================================================================
# PHONETIC FOLDING
# ===========================================================================
# The gazetteer used to match on NFC + lowercase only. That failed on real
# audio in four separate ways, and enumerating spellings by hand cannot fix
# any of them because the variants multiply combinatorially:
#
#   1. SPACING      - clinicians spell acronyms out, and the ASR writes each
#                     letter separately: "সি বি সি" vs the written "সিবিসি".
#                     This alone caused ZERO lab tests to match across 255
#                     real segments.
#   2. HALF-LETTERS - Bengali conjuncts (যুক্তাক্ষর) are written with a hasant
#                     that the ASR drops or inserts inconsistently:
#                     "মন্টিকুলাষ্ট" vs "মনটিকুলাসট" are the same word.
#   3. DIALECT      - the three sibilants শ/ষ/স are ONE sound /ʃ/ in spoken
#                     Bengali, as are ণ/ন. Which glyph the ASR picks is
#                     arbitrary. Same for ড়/র and য/জ.
#   4. ACCENT       - aspiration (ক/খ, ত/থ, ব/ভ, প/ফ …) is the least stable
#                     feature across speakers and the one the ASR most often
#                     gets wrong on borrowed drug names.
#
# So instead of listing variants, both the gazetteer and the ASR text are
# pushed through the SAME lossy fold, and matching happens in folded space.
# One entry then covers its whole spelling family.
#
# This is deliberately lossy, which means it can over-merge. That risk is
# not hand-waved: _collisions() below checks every gazetteer entry against
# every other at import time, and the test suite asserts no drug folds onto
# a different drug or onto a clinical term.

_BN_DROP = str.maketrans("", "", (
    "্"   # ্  hasant - collapses ALL conjuncts / half-letters
    "ঁ"   # ঁ  chandrabindu - nasalisation, not phonemic here
    "ঃ"   # ঃ  visarga
    "়"   # ়  nukta
))

# Everything on the left is heard as the thing on the right by a Bengali
# speaker, or is a glyph the ASR swaps freely.
_BN_FOLD = str.maketrans({
    # sibilants - all /ʃ/
    "শ": "স",  # শ -> স
    "ষ": "স",  # ষ -> স
    # nasals
    "ণ": "ন",  # ণ -> ন
    "ঙ": "ং",  # ঙ -> ং
    # rhotics / flaps
    "ড়": "র",  # ড় -> র
    "ঢ়": "র",  # ঢ় -> র
    "ৰ": "র",  # ৰ -> র
    # y / j merge
    "য": "জ",  # য -> জ
    "য়": "য়",  # য় kept - it is a glide, not /dʒ/
    # vowel LENGTH is not contrastive in Bengali
    "ী": "ি",  # ী -> ি
    "ূ": "ু",  # ূ -> ু
    "ঈ": "ই",  # ঈ -> ই
    "ঊ": "উ",  # ঊ -> উ
    "ৃ": "ি",  # ৃ -> ি
    "ঋ": "র",  # ঋ -> র
    "ৎ": "ত",  # ৎ -> ত
    # ASPIRATION - the least stable feature across speakers
    "খ": "ক",  # খ -> ক
    "ঘ": "গ",  # ঘ -> গ
    "ছ": "চ",  # ছ -> চ
    "ঝ": "জ",  # ঝ -> জ
    "ঠ": "ট",  # ঠ -> ট
    "ঢ": "ড",  # ঢ -> ড
    "থ": "ত",  # থ -> ত
    "ধ": "দ",  # ধ -> দ
    "ফ": "প",  # ফ -> প
    "ভ": "ব",  # ভ -> ব
})

# Latin side: drug names arrive romanised too, with the same instability
# ("Montelukast" / "Montuculast", "Ecosprin" / "Ecospirin").
_LAT_FOLD = (
    ("ph", "f"), ("ck", "k"), ("qu", "k"), ("x", "ks"),
    ("y", "i"), ("z", "s"), ("c", "k"), ("w", "v"), ("j", "z"),
)


def fold(s: str) -> str:
    """Collapse a term to its phonetic skeleton for matching.

    Lossy by design - see the block comment above. Never use the output for
    display; it is a lookup key only.
    """
    s = unicodedata.normalize("NFC", s).strip().lower()
    s = s.translate(_BN_DROP).translate(_BN_FOLD)
    for a, b in _LAT_FOLD:
        s = s.replace(a, b)
    # strip spacing and punctuation LAST, so "সি বি সি" == "সিবিসি"
    out = [c for c in s if not (c.isspace() or unicodedata.category(c).startswith("P"))]
    s = "".join(out)
    # Collapse doubled letters ("Ecosprinn", "pantopp") - ASCII ONLY.
    # Applying this to Bengali corrupts spelled-out acronyms: EEG is
    # "ই ই জি", and deduping it to "ইজি" made a 3-character key that
    # false-matched "এই জিভটা" ("stick your tongue out").
    deduped: list[str] = []
    for c in s:
        if deduped and deduped[-1] == c and c.isascii():
            continue
        deduped.append(c)
    return "".join(deduped)


def _norm(s: str) -> str:
    """Display normalisation only. Matching goes through fold()."""
    return unicodedata.normalize("NFC", s.strip().lower())


# Build lookup tables once at import, keyed on the FOLD.
_DRUG_LOOKUP: dict[str, Drug] = {}
for _d in ALL_DRUGS:
    for _key in (_d.generic, *_d.brands, *_d.bengali):
        _DRUG_LOOKUP[fold(_key)] = _d

_LAB_LOOKUP: dict[str, str] = {}
for _canon, _alts in LAB_TESTS.items():
    _LAB_LOOKUP[fold(_canon)] = _canon
    for _a in _alts:
        _LAB_LOOKUP[fold(_a)] = _canon

_TERM_LOOKUP: dict[str, str] = {}
for _canon, _alts in CLINICAL_TERMS.items():
    _TERM_LOOKUP[fold(_canon)] = _canon
    for _a in _alts:
        _TERM_LOOKUP[fold(_a)] = _canon


def collisions() -> list[tuple[str, str, str]]:
    """Entries that fold onto the same key but mean different things.

    The fold is lossy, so this is the guard-rail that keeps it honest. A
    drug colliding with another drug, or with a clinical term, is a real
    bug - the fold has over-merged and must be made less aggressive.
    Exercised by the test suite; returns [] when the gazetteer is clean.
    """
    found: list[tuple[str, str, str]] = []
    seen: dict[str, tuple[str, str]] = {}
    for kind, table in (("drug", {k: v.generic for k, v in _DRUG_LOOKUP.items()}),
                        ("lab", _LAB_LOOKUP),
                        ("term", _TERM_LOOKUP)):
        for key, canon in table.items():
            if key in seen:
                prev_kind, prev_canon = seen[key]
                if prev_canon != canon:
                    found.append((key, f"{prev_kind}:{prev_canon}", f"{kind}:{canon}"))
            else:
                seen[key] = (kind, canon)
    return found


# Dosage-form noise the SLM prepends. Folded, because that is the space
# the comparison happens in ("cap." -> "kap", "syp." -> "sip").
_DOSAGE_PREFIXES = tuple(fold(p) for p in (
    "tab.", "tab", "cap.", "cap", "syp.", "syp", "inj.", "inj",
    "tablet", "capsule", "syrup", "injection", "ট্যাব", "ক্যাপ",
))


def lookup_drug(text: str) -> Drug | None:
    """Exact gazetteer hit in folded space, or None.

    Folding means one entry covers its whole spelling family, so
    "মন্টিকুলাষ্ট", "মনটিকুলাসট" and "Montelukast" all land on the same Drug.
    """
    t = fold(text)
    if not t:
        return None
    if t in _DRUG_LOOKUP:
        return _DRUG_LOOKUP[t]
    for prefix in _DOSAGE_PREFIXES:
        if prefix and t.startswith(prefix):
            stripped = t[len(prefix):]
            if stripped in _DRUG_LOOKUP:
                return _DRUG_LOOKUP[stripped]
    return _ngram_match(text, _DRUG_LOOKUP)


# How many consecutive words a gazetteer entry may span. Covers the longest
# real cases: "সি বি সি" (3), "খাওয়ার পরের সুগার" (3), "তেল মশলা এড়িয়ে" (3),
# plus headroom for a spelled-out four-letter acronym.
_MAX_NGRAM = 5


def _ngram_match(text: str, table: dict):
    """Find a gazetteer entry inside free text, matching only on WHOLE
    word groups.

    Plain substring matching over the space-stripped fold was wrong: it let
    keys straddle word boundaries, so the EEG key matched inside
    "এই জিভটা বার কর". Folding each n-gram of consecutive tokens instead
    means a key must line up with real word starts.

    A trailing-suffix allowance stays, because Bengali is agglutinative -
    "সি বি সি টা" and "প্যারাসিটামলটা" carry a bound suffix that is part of
    the same word. Prefix matching is capped at keys of >= 4 folded
    characters so short keys cannot run away.
    """
    tokens = text.split()
    if not tokens:
        return None
    best = None
    best_len = 0
    for i in range(len(tokens)):
        for n in range(1, min(_MAX_NGRAM, len(tokens) - i) + 1):
            gram = fold("".join(tokens[i:i + n]))
            if not gram:
                continue
            hit = table.get(gram)
            if hit is None and len(gram) >= 4:
                # agglutinative suffix: word starts with the entry
                for key, val in table.items():
                    if len(key) >= 4 and gram.startswith(key):
                        hit = val
                        gram = key
                        break
            # prefer the longest match, so "PP sugar" beats "blood sugar"
            if hit is not None and len(gram) > best_len:
                best, best_len = hit, len(gram)
    return best


# Bengali negation and interrogative particles. Both follow the verb, so a
# short forward window from the end of a match is the right place to look.
#
# This exists because the scan functions became GENERATIVE - pipeline.py
# merges scan_labs() straight into labs_ordered. As a veto over SLM output
# a spurious hit was harmless; as a source of orders it is a fabricated
# prescription line. Measured on real audio, the generic entries scored 1/3
# precision without this: "নতুন কোনো টেস্ট দিচ্ছি না" ("I am NOT giving a new
# test") was reported as a test ordered, and "কোনো পরীক্ষা করাতে হবে কি"
# ("do I need any tests?") is the patient ASKING, not the doctor ordering.
_NEGATORS = {"না", "নি", "নেই", "নয়", "নাই", "কখনো"}
_INTERROGATIVES = {"কি", "কী", "কিনা", "কিনা?"}
_SCOPE_WINDOW = 3


def _span_is_negated(tokens: list[str], start: int, end: int) -> bool:
    """True if a negation or question particle governs the matched span.

    Scans the span ITSELF as well as a short window after it. Checking only
    after the span was wrong: matching is prefix-based, so a longer n-gram
    swallows the negator and then looks clean. Real case -
    "ওই এনজিওগ্রাম করতে চাই না এখন" ("I don't want the angiogram now") was
    suppressed at n<=3 but reported at n=4, because the 4-token span
    absorbed the "না" and the window past it saw only "এখন".
    """
    for tok in tokens[start:end + _SCOPE_WINDOW]:
        clean = tok.strip("।,?!.")
        if clean in _NEGATORS or clean in _INTERROGATIVES:
            return True
    return False


def _ngram_scan_all(text: str, table: dict) -> list:
    """Every distinct gazetteer entry appearing in the text, in order.

    is_lab_test() returns a single best hit, which silently loses data:
    "আপনাকে একটা টি এম টি আর ই সি জি করতে হবে" orders BOTH a TMT and an
    ECG. Anything populating a prescription must use this instead.

    LIMITATION - NO NEGATION HANDLING. This is a gazetteer, not a parser:
    "নতুন কোনো টেস্ট দিচ্ছি না" ("I am NOT giving a new test") still
    reports a test. That is acceptable only because these functions VETO
    what the SLM proposed - they never propose on their own. Do not use
    them as a standalone extractor without adding negation scope.
    """
    tokens = text.split()
    found: list = []
    for i in range(len(tokens)):
        for n in range(1, min(_MAX_NGRAM, len(tokens) - i) + 1):
            gram = fold("".join(tokens[i:i + n]))
            if not gram:
                continue
            hit = table.get(gram)
            if hit is None and len(gram) >= 4:
                for key, val in table.items():
                    if len(key) >= 4 and gram.startswith(key):
                        hit = val
                        break
            if hit is not None and hit not in found:
                if _span_is_negated(tokens, i, i + n):
                    continue
                found.append(hit)
    return found


def scan_labs(text: str) -> list[str]:
    """All lab tests ordered in this segment."""
    return _ngram_scan_all(text, _LAB_LOOKUP)


def scan_drugs(text: str) -> list[Drug]:
    """All gazetteer drugs named in this segment."""
    return _ngram_scan_all(text, _DRUG_LOOKUP)


def scan_terms(text: str) -> list[str]:
    """All clinical terms (symptoms / findings / advice) in this segment."""
    return _ngram_scan_all(text, _TERM_LOOKUP)


def is_lab_test(text: str) -> str | None:
    """Folded n-gram match - this is what makes spelled-out acronyms work:
    the tokens "সি বি সি" join and fold to the CBC key."""
    t = fold(text)
    if not t:
        return None
    if t in _LAB_LOOKUP:
        return _LAB_LOOKUP[t]
    return _ngram_match(text, _LAB_LOOKUP)


def is_clinical_term(text: str) -> str | None:
    """True for symptoms, findings and advice - i.e. things that are
    definitively NOT medications."""
    t = fold(text)
    if not t:
        return None
    if t in _TERM_LOOKUP:
        return _TERM_LOOKUP[t]
    return _ngram_match(text, _TERM_LOOKUP)


def stats() -> dict:
    return {
        "drugs": len(ALL_DRUGS),
        "drug_aliases": len(_DRUG_LOOKUP),
        "lab_tests": len(LAB_TESTS),
        "clinical_terms": len(CLINICAL_TERMS),
        "departments": sorted({d.department for d in ALL_DRUGS}),
    }
