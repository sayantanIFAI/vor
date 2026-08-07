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
LAB_TESTS: dict[str, tuple[str, ...]] = {
    # cardiac
    "ECG": ("ইসিজি", "ই সি জি", "electrocardiogram"),
    "Echo": ("ইকো", "echocardiogram", "2d echo"),
    "TMT": ("টিএমটি", "টি এম টি", "treadmill test", "stress test"),
    "Angiography": ("অ্যাঞ্জিওগ্রাফি", "angiogram", "অ্যাঞ্জিওগ্রাম"),
    "Lipid profile": ("লিপিড প্রোফাইল", "cholesterol test"),
    "Troponin": ("ট্রপোনিন",),
    # neuro
    "EEG": ("ইইজি", "ই ই জি", "electroencephalogram"),
    "MRI": ("এমআরআই", "এম আর আই"),
    "CT scan": ("সিটি স্ক্যান", "সিটি"),
    # metabolic / blood
    "Creatinine": ("ক্রিয়েটিনিন", "creatine"),
    "Urea": ("ইউরিয়া",),
    "PP sugar": ("পিপি সুগার", "পোস্ট প্রান্ডিয়াল", "post prandial sugar"),
    "Fasting sugar": ("ফাস্টিং সুগার", "FBS", "খালি পেটে সুগার"),
    "HbA1c": ("এইচবিএ১সি", "গ্লাইকোসাইলেটেড হিমোগ্লোবিন"),
    "TSH": ("টিএসএইচ", "থাইরয়েড টেস্ট", "thyroid profile"),
    "CBC": ("সিবিসি", "complete blood count", "রক্ত পরীক্ষা"),
    "LFT": ("এলএফটি", "liver function test"),
    "KFT": ("কেএফটি", "kidney function test", "RFT"),
    "Urine routine": ("ইউরিন", "প্রস্রাব পরীক্ষা", "urine test"),
    "Uric acid": ("ইউরিক অ্যাসিড",),
    "Vitamin D": ("ভিটামিন ডি টেস্ট",),
    "X-ray": ("এক্স রে", "এক্সরে", "চেস্ট এক্স রে"),
    "USG": ("ইউএসজি", "আল্ট্রাসাউন্ড", "ultrasound", "sonography"),
    "PSA": ("পিএসএ",),
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


def _norm(s: str) -> str:
    return unicodedata.normalize("NFC", s.strip().lower())


# Build lookup tables once at import.
_DRUG_LOOKUP: dict[str, Drug] = {}
for _d in ALL_DRUGS:
    for _key in (_d.generic, *_d.brands, *_d.bengali):
        _DRUG_LOOKUP[_norm(_key)] = _d

_LAB_LOOKUP: dict[str, str] = {}
for _canon, _alts in LAB_TESTS.items():
    _LAB_LOOKUP[_norm(_canon)] = _canon
    for _a in _alts:
        _LAB_LOOKUP[_norm(_a)] = _canon

_TERM_LOOKUP: dict[str, str] = {}
for _canon, _alts in CLINICAL_TERMS.items():
    _TERM_LOOKUP[_norm(_canon)] = _canon
    for _a in _alts:
        _TERM_LOOKUP[_norm(_a)] = _canon


def lookup_drug(text: str) -> Drug | None:
    """Exact gazetteer hit, or None. Strips common dosage-form noise
    ("Tab.", "Cap", "Syp") the SLM often prepends."""
    t = _norm(text)
    if not t:
        return None
    if t in _DRUG_LOOKUP:
        return _DRUG_LOOKUP[t]
    for prefix in ("tab.", "tab ", "cap.", "cap ", "syp.", "syp ",
                   "inj.", "inj ", "tablet ", "capsule ", "syrup "):
        if t.startswith(prefix):
            stripped = t[len(prefix):].strip()
            if stripped in _DRUG_LOOKUP:
                return _DRUG_LOOKUP[stripped]
    # a drug name embedded in a longer phrase ("Amlodipine 5mg once daily")
    for key, drug in _DRUG_LOOKUP.items():
        if len(key) >= 5 and key in t:
            return drug
    return None


def is_lab_test(text: str) -> str | None:
    t = _norm(text)
    if t in _LAB_LOOKUP:
        return _LAB_LOOKUP[t]
    for key, canon in _LAB_LOOKUP.items():
        if len(key) >= 3 and key in t:
            return canon
    return None


def is_clinical_term(text: str) -> str | None:
    """True for symptoms, findings and advice - i.e. things that are
    definitively NOT medications."""
    t = _norm(text)
    if t in _TERM_LOOKUP:
        return _TERM_LOOKUP[t]
    for key, canon in _TERM_LOOKUP.items():
        if len(key) >= 4 and key in t:
            return canon
    return None


def stats() -> dict:
    return {
        "drugs": len(ALL_DRUGS),
        "drug_aliases": len(_DRUG_LOOKUP),
        "lab_tests": len(LAB_TESTS),
        "clinical_terms": len(CLINICAL_TERMS),
        "departments": sorted({d.department for d in ALL_DRUGS}),
    }
