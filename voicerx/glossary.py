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
         ("রোসুভাস্ট্যাটিন", "রোসুভাস", "রসু ভাস্টাটিন", "রসুভাস্টাটিন",
          "রোসু ভাস্ট্যাটিন", "ভাস্টাটিন"),
         "cholesterol / statin", "cardiac"),
    Drug("Amlodipine", ("Amlopres", "Amlokind", "Stamlo"),
         ("অ্যামলোডিপিন", "অ্যামলোপ্রেস"),
         "hypertension", "cardiac"),
    Drug("Telmisartan", ("Telma", "Telsartan"),
         ("টেলমিসারটান", "টেলমা"),
         "hypertension", "cardiac"),
    Drug("Metoprolol", ("Metolar", "Betaloc", "Metoprolol Succinate"),
         ("মেটোপ্রোলল", "মেটোলার", "মেটো প্রোল", "মেটোপ্রোল",
          "মেটো প্রোলল", "মেটোপ্রোল সাক্সিনেট"),
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
         ("মেটফরমিন", "গ্লাইকোমেট", "মেট ফর্মিন", "মেটফর্মিন", "ফর্মিন"),
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

# ---------------------------------------------------------------------------
# BONE / ORTHOPAEDIC
#
# Added after a live osteoporosis consultation returned no bisphosphonate
# and none of the bone labs. The gazetteer had been built from cardiac,
# diabetic and general-OPD material, so an entire specialty was missing -
# the same way cardiology labs were missing before 200 cardiac
# prescriptions exposed that gap.
# ---------------------------------------------------------------------------
BONE = [
    Drug("Alendronate", ("Fosamax", "Osteofos", "Restofos", "Alendra"),
         ("অ্যালেনড্রোনেট", "অ্যালেন্ড্রোনেট", "ফোসাম্যাক্স"),
         "bisphosphonate / osteoporosis", "bone"),
    Drug("Risedronate", ("Actonel", "Risofos"),
         ("রাইসেড্রোনেট",), "bisphosphonate / osteoporosis", "bone"),
    Drug("Zoledronic acid", ("Zoldria", "Aclasta", "Zybone"),
         ("জোলেড্রনিক অ্যাসিড", "জোলেড্রোনিক"),
         "IV bisphosphonate, yearly", "bone"),
    Drug("Teriparatide", ("Forteo", "Bonista"),
         ("টেরিপ্যারাটাইড",), "anabolic / severe osteoporosis", "bone"),
    Drug("Denosumab", ("Prolia", "Xgeva"),
         ("ডেনোসুম্যাব",), "RANKL inhibitor / osteoporosis", "bone"),
    Drug("Calcitriol", ("Rocaltrol", "Calcirol"),
         ("ক্যালসিট্রায়ল", "ক্যালসিরল"),
         "active vitamin D", "bone"),
    Drug("Cholecalciferol", ("Calcirol", "D-Rise", "Uprise-D3"),
         ("কোলেক্যালসিফেরল", "ভিটামিন ডি থ্রি", "ডি রাইজ"),
         "vitamin D3 supplement", "bone"),
    Drug("Calcium carbonate", ("Shelcal", "Calcimax", "Ostocalcium"),
         ("ক্যালসিয়াম", "শেলক্যাল", "ক্যালশিয়াম", "অস্টোক্যালসিয়াম"),
         "calcium supplement", "bone"),
    Drug("Etoricoxib", ("Etoshine", "Nucoxia"),
         ("ইটোরিকক্সিব", "ইটোশাইন"), "NSAID / joint pain", "bone"),
    Drug("Aceclofenac", ("Zerodol", "Hifenac"),
         ("এসিক্লোফেনাক", "জিরোডল"), "NSAID / joint pain", "bone"),
    Drug("Methylcobalamin", ("Nurokind", "Meconerv"),
         ("মিথাইলকোবালামিন", "নিউরোকাইন্ড"),
         "neuropathy / B12", "bone"),
]

# ---------------------------------------------------------------------------
# DERMATOLOGY (skin)
# ---------------------------------------------------------------------------
DERMATOLOGY = [
    Drug("Clotrimazole", ("Candid", "Canesten", "Clop-G"),
         ("ক্লোট্রিমাজল", "ক্যান্ডিড"), "antifungal cream", "dermatology"),
    Drug("Ketoconazole", ("Nizral", "Ketostar"),
         ("কিটোকোনাজল", "নিজরাল"), "antifungal", "dermatology"),
    Drug("Terbinafine", ("Terbicip", "Lamisil", "Sebifin"),
         ("টারবিনাফাইন", "টারবিসিপ"), "antifungal", "dermatology"),
    Drug("Fluconazole", ("Forcan", "Zocon", "Flucos"),
         ("ফ্লুকোনাজল", "ফোরকান"), "systemic antifungal", "dermatology"),
    Drug("Mupirocin", ("T-Bact", "Bactroban", "Mupinase"),
         ("মিউপিরোসিন", "টি ব্যাক্ট"), "topical antibiotic", "dermatology"),
    Drug("Betamethasone", ("Betnovate", "Betnesol"),
         ("বিটামেথাসোন", "বেটনোভেট"), "topical steroid", "dermatology"),
    Drug("Permethrin", ("Permite", "Scabper"),
         ("পারমেথ্রিন", "পারমাইট"), "scabies", "dermatology"),
    Drug("Ivermectin", ("Ivermectol", "Iverjohn"),
         ("আইভারমেকটিন",), "scabies / parasites", "dermatology"),
    Drug("Isotretinoin", ("Sotret", "Isotroin", "Accufine"),
         ("আইসোট্রেটিনয়েন", "আইসোট্রইন"), "severe acne", "dermatology"),
    Drug("Adapalene", ("Deriva", "Adaferin"),
         ("অ্যাডাপালিন", "ডেরিভা"), "acne", "dermatology"),
    Drug("Tacrolimus ointment", ("Tacroz", "Protopic"),
         ("ট্যাক্রোলিমাস",), "eczema / vitiligo", "dermatology"),
    Drug("Calamine", ("Calamine lotion", "Caladryl"),
         ("ক্যালামিন",), "soothing lotion", "dermatology"),
]

# ---------------------------------------------------------------------------
# OPHTHALMOLOGY (eye)
# ---------------------------------------------------------------------------
OPHTHALMOLOGY = [
    Drug("Moxifloxacin eye drops", ("Vigamox", "5-Moxi", "Milflox", "Moxicip"),
         ("মক্সিফ্লক্সাসিন", "ভিগাম্যাক্স", "মক্সি ফ্লক্সাসিম", "ফ্লক্সাসিম",
          "মক্সিফ্লক্সাসিম", "মক্সি ফ্লক্সাসিন", "মক্সিফ্লক্স"),
         "eye infection", "ophthalmology"),
    Drug("Flurbiprofen eye drops", ("Flur", "Flubiprof", "Ocuflur"),
         ("ফ্লার্বিপ্রোফেন", "ফ্লুরবিপ্রোফেন", "ফ্লারবি"),
         "ocular NSAID", "ophthalmology"),
    Drug("Carboxymethylcellulose", ("Refresh Tears", "Optive", "Lubrex"),
         ("রিফ্রেশ টিয়ার্স", "কৃত্রিম অশ্রু"), "dry eye lubricant", "ophthalmology"),
    Drug("Olopatadine", ("Patanol", "Winolap"),
         ("ওলোপাটাডিন", "প্যাটানল"), "allergic conjunctivitis", "ophthalmology"),
    Drug("Timolol eye drops", ("Glucomol", "Iotim"),
         ("টিমোলল",), "glaucoma", "ophthalmology"),
    Drug("Latanoprost", ("Xalatan", "9PM", "Latoprost"),
         ("ল্যাটানোপ্রস্ট",), "glaucoma", "ophthalmology"),
    Drug("Brimonidine", ("Alphagan", "Brimolol"),
         ("ব্রিমোনিডিন",), "glaucoma", "ophthalmology"),
    Drug("Prednisolone eye drops", ("Predmet", "Omnipred"),
         ("প্রেডনিসোলন ড্রপ",), "ocular inflammation", "ophthalmology"),
    Drug("Tropicamide", ("Tropicacyl", "Mydriacyl"),
         ("ট্রপিকামাইড",), "pupil dilation", "ophthalmology"),
]

# ---------------------------------------------------------------------------
# ENT
# ---------------------------------------------------------------------------
ENT = [
    Drug("Xylometazoline", ("Otrivin", "Nasivion"),
         ("জাইলোমেটাজোলিন", "অট্রিভিন", "নাসিভিয়ন"),
         "nasal decongestant drops", "ent"),
    Drug("Fluticasone nasal spray", ("Flomist", "Nasoflo"),
         ("ফ্লুটিকাসোন", "ফ্লোমিস্ট"), "allergic rhinitis", "ent"),
    Drug("Mometasone nasal spray", ("Nasonex", "Metaspray"),
         ("মোমেটাসোন", "মেটাস্প্রে"), "allergic rhinitis", "ent"),
    Drug("Betahistine", ("Vertin", "Betavert"),
         ("বিটাহিস্টিন", "ভার্টিন"), "vertigo", "ent"),
    Drug("Candibiotic ear drops", ("Candibiotic", "Otek-AC"),
         ("ক্যান্ডিবায়োটিক",), "ear infection", "ent"),
    Drug("Prochlorperazine", ("Stemetil", "Vertigon"),
         ("প্রোক্লোরপেরাজিন", "স্টেমেটিল"), "vertigo / nausea", "ent"),
]

# ---------------------------------------------------------------------------
# DENTAL
# ---------------------------------------------------------------------------
DENTAL = [
    Drug("Chlorhexidine mouthwash", ("Hexidine", "Clohex", "Rexidin"),
         ("ক্লোরহেক্সিডিন", "হেক্সিডিন", "মাউথওয়াশ"),
         "oral antiseptic", "dental"),
    Drug("Ketorolac", ("Ketorol", "Zerodol-K"),
         ("কিটোরোলাক", "কেটোরল"), "severe dental pain", "dental"),
    Drug("Lignocaine gel", ("Mucopain", "Dentogel", "Xylocaine"),
         ("লিগনোকেইন", "মিউকোপেইন"), "topical anaesthetic", "dental"),
    Drug("Triamcinolone oral paste", ("Kenacort", "Tess"),
         ("ট্রায়ামসিনোলোন", "কেনাকর্ট"), "mouth ulcer", "dental"),
]

# ---------------------------------------------------------------------------
# GYNAECOLOGY
# ---------------------------------------------------------------------------
GYNAECOLOGY = [
    Drug("Folic acid", ("Folvite", "Fol-5"),
         ("ফলিক অ্যাসিড", "ফলভাইট"), "pregnancy supplement", "gynaecology"),
    Drug("Tranexamic acid", ("Trapic", "Pause", "Texid"),
         ("ট্রানেক্সামিক অ্যাসিড", "ট্রাপিক"),
         "heavy menstrual bleeding", "gynaecology"),
    Drug("Norethisterone", ("Primolut-N", "Regestrone"),
         ("নরইথিস্টেরন", "প্রিমোলাট"), "menstrual regulation", "gynaecology"),
    Drug("Medroxyprogesterone", ("Deviry", "Meprate"),
         ("মেড্রক্সিপ্রোজেস্টেরন", "ডেভিরি"), "progestin", "gynaecology"),
    Drug("Progesterone", ("Susten", "Duphaston", "Dubagest"),
         ("প্রোজেস্টেরন", "সাসটেন", "ডুফাস্টন"),
         "luteal support", "gynaecology"),
    Drug("Clomiphene", ("Fertyl", "Clomi"),
         ("ক্লোমিফেন", "ফার্টিল"), "ovulation induction", "gynaecology"),
    Drug("Letrozole", ("Letroz", "Fempro"),
         ("লেট্রোজল", "লেট্রোজ"), "ovulation induction", "gynaecology"),
    Drug("Mifepristone+Misoprostol", ("MTP Kit", "Unwanted Kit"),
         ("মিফেপ্রিস্টোন", "মাইসোপ্রোস্টল"),
         "medical termination", "gynaecology"),
]

# ---------------------------------------------------------------------------
# NEPHROLOGY
# ---------------------------------------------------------------------------
NEPHROLOGY = [
    Drug("Sevelamer", ("Renvela", "Sevcar"),
         ("সেভেলামার",), "phosphate binder / CKD", "nephrology"),
    Drug("Calcium acetate", ("Nephrocal", "Royal-CA"),
         ("ক্যালসিয়াম অ্যাসিটেট",), "phosphate binder", "nephrology"),
    Drug("Erythropoietin", ("Eprex", "Epofit", "Relipoietin"),
         ("এরিথ্রোপয়েটিন", "ইপ্রেক্স"), "anaemia of CKD", "nephrology"),
    Drug("Sodium bicarbonate", ("Sodamint", "Nodosis"),
         ("সোডিয়াম বাইকার্বোনেট", "নোডোসিস"),
         "metabolic acidosis", "nephrology"),
    Drug("Febuxostat", ("Febutaz", "Zurig", "Feburic"),
         ("ফেবুক্সোস্ট্যাট", "ফেবুটাজ"), "gout / uric acid", "nephrology"),
    Drug("Allopurinol", ("Zyloric", "Ciploric"),
         ("অ্যালোপিউরিনল", "জাইলোরিক"), "gout / uric acid", "nephrology"),
]

# ---------------------------------------------------------------------------
# NEUROLOGY
# ---------------------------------------------------------------------------
NEUROLOGY = [
    Drug("Levetiracetam", ("Levipil", "Keppra", "Torleva"),
         ("লেভেটিরাসিটাম", "লেভিপিল"), "epilepsy", "neurology"),
    Drug("Sodium valproate", ("Valparin", "Encorate", "Divaa"),
         ("সোডিয়াম ভালপ্রোয়েট", "ভালপারিন", "এনকোরেট"),
         "epilepsy / migraine", "neurology"),
    Drug("Phenytoin", ("Eptoin", "Dilantin"),
         ("ফেনিটয়েন", "এপটোইন"), "epilepsy", "neurology"),
    Drug("Carbamazepine", ("Tegretol", "Mazetol", "Zeptol"),
         ("কার্বামাজেপিন", "টেগ্রেটল"),
         "epilepsy / trigeminal neuralgia", "neurology"),
    Drug("Clobazam", ("Frisium", "Lobazam"),
         ("ক্লোবাজাম", "ফ্রিজিয়াম"), "epilepsy", "neurology"),
    Drug("Pregabalin", ("Pregabid", "Lyrica", "Maxgalin"),
         ("প্রিগাবালিন", "প্রিগাবিড"), "neuropathic pain", "neurology"),
    Drug("Gabapentin", ("Gabapin", "Neurontin"),
         ("গ্যাবাপেন্টিন", "গ্যাবাপিন"), "neuropathic pain", "neurology"),
    Drug("Donepezil", ("Aricept", "Donep", "Dompezil"),
         ("ডোনেপেজিল", "অ্যারিসেপ্ট"), "dementia", "neurology"),
    Drug("Memantine", ("Admenta", "Nemdaa"),
         ("মেমান্টিন",), "dementia", "neurology"),
    Drug("Levodopa+Carbidopa", ("Syndopa", "Tidomet", "Sinemet"),
         ("লেভোডোপা", "সিনডোপা", "টিডোমেট"),
         "Parkinson's disease", "neurology"),
    Drug("Sumatriptan", ("Suminat", "Imitrex"),
         ("সুমাট্রিপটান", "সুমিন্যাট"), "migraine", "neurology"),
    Drug("Flunarizine", ("Sibelium", "Flunarin"),
         ("ফ্লুনারিজিন", "সিবেলিয়াম"), "migraine prophylaxis", "neurology"),
    Drug("Amitriptyline", ("Amitone", "Tryptomer"),
         ("অ্যামিট্রিপটাইলিন", "ট্রিপটোমার"),
         "neuropathic pain / migraine", "neurology"),
    Drug("Clopidogrel+Aspirin", ("Clopitab-A", "Deplatt-A", "Ecosprin-AV"),
         ("ক্লোপিটাব", "ইকোস্পিরিন এভি"), "stroke prevention", "neurology"),
]

# ---------------------------------------------------------------------------
# SURGERY
# ---------------------------------------------------------------------------
SURGERY = [
    Drug("Tramadol", ("Ultracet", "Tramazac", "Domadol"),
         ("ট্রামাডল", "আলট্রাসেট"), "moderate-severe pain", "surgery"),
    Drug("Ceftriaxone", ("Monocef", "Intacef", "Oframax"),
         ("সেফট্রায়াক্সোন", "মনোসেফ"), "injectable antibiotic", "surgery"),
    Drug("Diclofenac", ("Voveran", "Dynapar", "Diclomol"),
         ("ডাইক্লোফেনাক", "ভোভেরান"), "NSAID", "surgery"),
    Drug("Enoxaparin", ("Clexane", "Lomoh"),
         ("এনোক্সাপারিন", "ক্লেক্সেন"), "DVT prophylaxis", "surgery"),
    Drug("Povidone iodine", ("Betadine", "Cipladine"),
         ("পোভিডোন আয়োডিন", "বিটাডিন"), "antiseptic", "surgery"),
    Drug("Lactulose", ("Duphalac", "Looz", "Cremaffin"),
         ("ল্যাকটুলোজ", "ডুফালাক"), "constipation", "surgery"),
]

ALL_DRUGS: list[Drug] = (CARDIAC + ENDOCRINE + RESPIRATORY + GI
                          + GENERAL + UROLOGY + BONE + DERMATOLOGY
                          + OPHTHALMOLOGY + ENT + DENTAL + GYNAECOLOGY
                          + NEPHROLOGY + NEUROLOGY + SURGERY)

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
    "Echo": ("ইকো", "echocardiogram", "2d echo", "টু ডি ইকো",
              "ইকোকার্ডিওগ্রাফি", "ইকোকার্ডিওগ্রাম", "echocardiography",
              "ইকো কার্ডিওগ্রাফি"),
    "TMT": ("টিএমটি", "টি এম টি", "treadmill test", "stress test", "t m t"),
    "Angiography": ("অ্যাঞ্জিওগ্রাফি", "angiogram", "অ্যাঞ্জিওগ্রাম",
                     "এনজিওগ্রাম", "এঞ্জিওগ্রাফি"),
    "Lipid profile": ("লিপিড প্রোফাইল", "cholesterol test", "লিপিড"),
    "Troponin": ("ট্রপোনিন", "ট্রপ", "troponin i", "troponin t",
                  "troponin i/t"),
    # Cardiology-specific tests. Added after 200 real cardiac prescriptions
    # showed only 12/20 labs recognised - the table was built for general
    # OPD and had no Holter, no tilt table, no electrophysiology study.
    "Holter monitor": ("holter", "holter monitor", "হোল্টার",
                        "holter monitor (24-48 hr)", "24 hour holter"),
    "Tilt table test": ("tilt table", "tilt table test", "টিল্ট টেবিল"),
    "Electrophysiology study": ("electrophysiology study", "ep study",
                                 "ইপি স্টাডি"),
    "CK-MB": ("ck-mb", "ck mb", "সিকে এমবি", "creatine kinase"),
    "BNP": ("bnp", "nt-probnp", "bnp or nt-probnp", "pro bnp"),
    "Serum electrolytes": ("serum electrolytes", "electrolytes",
                            "ইলেক্ট্রোলাইট", "na k cl"),
    "2D Echo": ("2d echo", "two d echo", "টু ডি ইকো"),
    # neuro
    "EEG": ("ইইজি", "ই ই জি", "electroencephalogram", "e e g"),
    "MRI": ("এমআরআই", "এম আর আই", "m r i"),
    "CT scan": ("সিটি স্ক্যান", "সি টি স্ক্যান", "সিটি"),
    # metabolic / blood
    "Creatinine": ("ক্রিয়েটিনিন", "creatine", "ক্রিয়েটিন"),
    "Urea": ("ইউরিয়া",),
    # Spoken forms, verbatim from real audio:
    #   "ওসিটি ফাস্টিং ব্লাড সুগার পিপি ইসিজি ব্লাড প্রেসার চেক"
    # Note "ফাস্টিং ব্লাড সুগার" - the interposed "ব্লাড" broke the n-gram
    # against the key "ফাস্টিং সুগার", and bare "পিপি" was never an alias.
    "PP sugar": ("পিপি সুগার", "পি পি সুগার", "পোস্ট প্রান্ডিয়াল",
                  "post prandial sugar", "খাওয়ার পরের সুগার", "পিপি",
                  "পিপি ব্লাড সুগার", "পি পি ব্লাড সুগার", "পোস্ট প্রান্ডিয়াল সুগার"),
    "Fasting sugar": ("ফাস্টিং সুগার", "FBS", "খালি পেটে সুগার",
                       "এফ বি এস", "ফাস্টিং ব্লাড সুগার", "ফাস্টিং",
                       "fasting blood sugar"),
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
                       "ইউরিন টেস্ট", "urine r/e", "urine re",
                       "urine routine examination"),
    "Uric acid": ("ইউরিক অ্যাসিড", "ইউরিক"),
    # NOT the bare "ভিটামিন ডি" - that is ambiguous between the supplement
    # and the blood test, and collisions() flagged it against the
    # Vitamin D3 drug entry. Only unambiguously test-shaped forms here.
    "Vitamin D": ("ভিটামিন ডি টেস্ট", "25 oh vitamin d", "vitamin d3 level",
                   "ভিটামিন ডি লেভেল"),
    # Bone / metabolic. A live osteoporosis consultation ordered every one
    # of these and the table knew none of them.
    "Serum calcium": ("সিরাম ক্যালসিয়াম", "serum calcium", "রক্তে ক্যালসিয়াম",
                       "calcium level", "ক্যালসিয়াম টেস্ট"),
    "Serum phosphorus": ("সিরাম ফসফরাস", "phosphorus", "phosphate",
                          "ফসফরাস", "serum phosphate"),
    "DEXA scan": ("dexa", "dexa scan", "ডেক্সা", "ডেক্সা স্ক্যান",
                   "bone mineral density", "bmd", "বিএমডি", "বোন ডেনসিটি",
                   "bone densitometry"),
    "X-ray LS spine": ("ls spine", "l s spine", "lumbosacral spine",
                        "এল এস স্পাইন", "কোমরের এক্স রে", "ap view ls spine",
                        "ls spine ap view", "এলএস স্পাইন"),
    "PTH": ("pth", "parathyroid hormone", "পিটিএইচ", "প্যারাথাইরয়েড"),
    "Alkaline phosphatase": ("alp", "alkaline phosphatase",
                              "অ্যালকালাইন ফসফেটেজ"),
    "X-ray": ("এক্স রে", "এক্সরে", "চেস্ট এক্স রে", "এক্স-রে"),
    "USG": ("ইউএসজি", "ইউ এস জি", "আল্ট্রাসাউন্ড", "ultrasound",
             "sonography", "আলট্রাসনোগ্রাফি"),
    "PSA": ("পিএসএ", "পি এস এ"),
    # --- department-specific investigations ---------------------------
    # dermatology
    "KOH mount": ("koh mount", "koh", "কেওএইচ", "skin scraping"),
    "Skin biopsy": ("skin biopsy", "ত্বকের বায়োপসি"),
    "Patch test": ("patch test", "প্যাচ টেস্ট", "allergy patch test"),
    # ophthalmology
    "Fundus examination": ("fundus", "ফান্ডাস", "fundoscopy", "retina check"),
    "Intraocular pressure": ("iop", "tonometry", "চোখের প্রেশার",
                              "eye pressure"),
    "Visual acuity": ("visual acuity", "দৃষ্টিশক্তি পরীক্ষা", "vision test"),
    "OCT": ("oct", "optical coherence tomography", "ওসিটি", "ও সি টি"),
    "Biometry": ("biometry", "বায়োমেট্রি", "বায়োমিট্রিক", "বায়োমেট্রিক",
                  "a-scan", "iol power", "আইওএল পাওয়ার"),
    "Viral markers": ("viral marker", "viral markers", "ভাইরাল মার্কার",
                       "hiv", "এইচআইভি", "এইচ আই ভি", "hbsag", "এইচবিএসএজি",
                       "anti hcv", "hiv hbsag hcv"),
    "Refraction": ("refraction", "power test", "চশমার পাওয়ার"),
    # ENT
    "Audiometry": ("audiometry", "pta", "pure tone audiometry",
                    "অডিওমেট্রি", "কানের পরীক্ষা"),
    "Tympanometry": ("tympanometry", "টিমপ্যানোমেট্রি"),
    "Nasal endoscopy": ("nasal endoscopy", "dnc", "নাকের এন্ডোস্কোপি"),
    # dental
    "OPG": ("opg", "orthopantomogram", "ওপিজি", "dental x-ray"),
    "IOPA": ("iopa", "intraoral periapical"),
    # gynaecology
    "USG pelvis": ("usg pelvis", "pelvic ultrasound", "তলপেটের আল্ট্রাসাউন্ড",
                    "tvs", "transvaginal scan"),
    "Pap smear": ("pap smear", "pap test", "প্যাপ স্মিয়ার"),
    "Beta hCG": ("beta hcg", "bhcg", "pregnancy test", "প্রেগন্যান্সি টেস্ট"),
    "Mammography": ("mammography", "mammogram", "ম্যামোগ্রাফি"),
    # nephrology
    "eGFR": ("egfr", "gfr", "creatinine clearance", "জিএফআর"),
    "Urine ACR": ("urine acr", "albumin creatinine ratio", "microalbumin",
                   "মাইক্রোঅ্যালবুমিন"),
    "USG KUB": ("usg kub", "kub", "kidney ultrasound", "কিডনি আল্ট্রাসাউন্ড"),
    # neurology
    "CT brain": ("ct brain", "ct head", "সিটি ব্রেন", "brain ct"),
    "MRI brain": ("mri brain", "এমআরআই ব্রেন", "brain mri"),
    "NCV": ("ncv", "nerve conduction", "এনসিভি"),
    "EMG": ("emg", "electromyography", "ইএমজি"),
    "Carotid doppler": ("carotid doppler", "ক্যারোটিড ডপলার"),
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
    # ASR-observed spellings, not textbook ones. A live consultation
    # produced "ইয়ার্ট অযাটাক" and "হয়াট অ্যাটাক" for "heart attack".
    "heart attack": ("হার্ট অ্যাটাক", "ইয়ার্ট অযাটাক", "হয়াট অ্যাটাক",
                      "হার্ট অ্যাটাক", "হার্ট এটাক", "মায়োকার্ডিয়াল ইনফার্কশন"),
    "heart failure": ("হার্ট ফেইলিওর", "হার্ট ফেলুআর", "হৃদযন্ত্রের অক্ষমতা"),
    "ischemia": ("ইস্কিমিয়া", "ইস্কামিয়া", "ইসকিমিয়া"),
    "diabetes": ("ডায়াবেটিস", "ডায়াবেটিজ", "ডায়াবিটিস", "মধুমেহ"),
    "stent": ("স্টেন্ট", "স্ট্যান্ট", "রিং পরানো"),
    "angioplasty": ("অ্যাঞ্জিওপ্লাস্টি", "এনজিওপ্লাস্টি"),
    "angina": ("অ্যাঞ্জাইনা", "অঞ্জিনা", "এনজাইনা"),
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
    # English symptom phrases the SLM proposes verbatim. Present because
    # fuzzy matching scored them against real drugs once the folded forms
    # got close - "hair loss" folds to "hairlos" and scored 0.714 against
    # Levothyroxine, over the 0.65 floor, so it was being offered as a
    # PROBABLE medication. Naming them positively is more robust than
    # raising the floor, which would start losing real garbled drug names.
    "hair loss": ("চুল পড়া", "চুল পড়ে যাওয়া"),
    "weight loss": ("ওজন কমা", "ওজন কমে যাওয়া"),
    "weight gain": ("ওজন বাড়া", "ওজন বেড়ে যাওয়া"),
    "bone loss": ("হাড় ক্ষয়",),
    "memory loss": ("স্মৃতিশক্তি কমে যাওয়া", "ভুলে যাওয়া"),
    "loss of appetite": ("খিদে কমে যাওয়া", "খাওয়ার ইচ্ছে নেই", "অরুচি"),
    "shortness of breath": ("দম ফুরিয়ে যাওয়া", "হাঁপ ধরা"),
    "loss of consciousness": ("জ্ঞান হারানো", "অজ্ঞান"),
    "blurred vision": ("ঝাপসা দেখা",),
    "dizziness": ("মাথা ঘোরা", "মাথা ঘুরছে"),
    "difficulty swallowing": ("গিলতে কষ্ট",),
    "phlegm": ("কফ", "শ্লেষ্মা"),
    "infection": ("ইনফেকশন", "সংক্রমণ"),
    # --- department-specific symptoms and findings --------------------
    # dermatology
    "itching": ("চুলকানি", "চুলকায়", "খুজলি"),
    "rash": ("র‍্যাশ", "ফুসকুড়ি", "চাকা চাকা দাগ"),
    "acne": ("ব্রণ", "একনি"),
    "hives": ("আমবাত", "চাকা"),
    "fungal infection": ("দাদ", "ছত্রাক", "ফাঙ্গাল ইনফেকশন"),
    "hair fall": ("চুল পড়া", "চুল উঠছে"),
    "dry skin": ("শুষ্ক ত্বক", "চামড়া শুকিয়ে"),
    "boil": ("ফোঁড়া", "বিচি"),
    # ophthalmology
    "blurred vision far": ("দূরে ঝাপসা", "দূরের জিনিস দেখতে"),
    "eye pain": ("চোখে ব্যথা", "চোখ ব্যথা"),
    "watering eyes": ("চোখ দিয়ে জল", "চোখে জল পড়া"),
    "red eye": ("চোখ লাল", "লাল চোখ"),
    "cataract": ("ছানি", "ক্যাটারাক্ট"),
    "glaucoma": ("গ্লুকোমা", "চোখের প্রেশার বেশি"),
    # ENT
    "ear pain": ("কানে ব্যথা", "কান ব্যথা"),
    "hearing loss": ("কানে শুনতে অসুবিধা", "কম শুনছি", "শ্রবণশক্তি কমে"),
    "tinnitus": ("কানে শব্দ", "কানে ভোঁ ভোঁ"),
    "vertigo": ("মাথা ঘোরা ভার্টিগো", "ভার্টিগো", "সবকিছু ঘুরছে"),
    "nasal block": ("নাক বন্ধ", "নাক দিয়ে শ্বাস"),
    "runny nose": ("নাক দিয়ে জল", "সর্দি"),
    "tonsillitis": ("টনসিল", "টনসিলাইটিস"),
    # dental
    "toothache": ("দাঁতে ব্যথা", "দাঁত ব্যথা"),
    "bleeding gums": ("মাড়ি থেকে রক্ত", "মাড়িতে রক্ত"),
    "mouth ulcer": ("মুখে ঘা", "মুখের ঘা"),
    "swollen gums": ("মাড়ি ফোলা", "মাড়ি ফুলে"),
    # gynaecology
    "irregular periods": ("অনিয়মিত পিরিয়ড", "মাসিক অনিয়মিত"),
    "heavy bleeding": ("বেশি রক্তপাত", "অতিরিক্ত রক্তক্ষরণ"),
    "white discharge": ("সাদা স্রাব", "লিউকোরিয়া"),
    "menopause": ("মেনোপজ", "মাসিক বন্ধ"),
    "pregnancy": ("গর্ভাবস্থা", "প্রেগন্যান্ট", "অন্তঃসত্ত্বা"),
    "lower abdominal pain": ("তলপেটে ব্যথা", "তলপেট ব্যথা"),
    # nephrology
    "reduced urine": ("প্রস্রাব কম", "কম প্রস্রাব"),
    "burning urination": ("প্রস্রাবে জ্বালা", "জ্বালাপোড়া"),
    "facial puffiness": ("মুখ ফোলা", "চোখ মুখ ফোলা"),
    "kidney failure": ("কিডনি ফেইলিওর", "কিডনি খারাপ"),
    "dialysis": ("ডায়ালিসিস",),
    # neurology
    "seizure": ("খিঁচুনি", "ফিট", "মৃগী"),
    "stroke": ("স্ট্রোক", "প্যারালাইসিস", "পক্ষাঘাত"),
    "numbness": ("অবশ", "ঝিনঝিন", "অসাড়"),
    "tremor": ("কাঁপুনি", "হাত কাঁপে"),
    "migraine": ("মাইগ্রেন",),
    "memory problem": ("ভুলে যাচ্ছি", "স্মৃতি সমস্যা"),
    "weakness one side": ("একদিক অবশ", "এক পাশ দুর্বল"),
    # surgery
    "lump": ("চাকা", "গোটা", "টিউমার"),
    "hernia": ("হার্নিয়া",),
    "piles": ("পাইলস", "অর্শ"),
    "gallstone": ("পিত্তথলির পাথর", "গলব্লাডার স্টোন"),
    "appendicitis": ("অ্যাপেন্ডিক্স", "অ্যাপেন্ডিসাইটিস"),
    "wound": ("ক্ষত", "ঘা", "কাটা"),
    # --- DRUG CLASSES, not drugs -------------------------------------
    # A doctor says "I'll give you an antibiotic" without naming one. These
    # must never resolve to a specific product.
    #
    # Not hypothetical: adding "Candibiotic" (an ear drop) to the gazetteer
    # made the bare word "Antibiotic" fuzzy-match it at 0.86, so a generic
    # statement became a specific ear medication. Naming the classes
    # positively stops that for every future entry too, because the class
    # check runs before the brand table and before fuzzy.
    "antibiotic": ("অ্যান্টিবায়োটিক", "antibiotics", "এন্টিবায়োটিক"),
    "painkiller": ("পেইনকিলার", "ব্যথার ওষুধ", "analgesic", "pain killer"),
    "antacid": ("অ্যান্টাসিড", "গ্যাসের ওষুধ"),
    "steroid": ("স্টেরয়েড", "steroids"),
    "vitamin supplement": ("ভিটামিন", "vitamins", "supplement", "সাপ্লিমেন্ট"),
    "antihistamine": ("অ্যান্টিহিস্টামিন",),
    "eye drops": ("চোখের ড্রপ", "আই ড্রপ"),
    "ear drops": ("কানের ড্রপ",),
    "nasal spray": ("নাকের স্প্রে",),
    "ointment": ("মলম", "অয়েন্টমেন্ট"),
    "syrup": ("সিরাপ",),
    "tablet": ("ট্যাবলেট", "বড়ি"),
    "injection": ("ইনজেকশন", "ইঞ্জেকশন"),
    "medicine": ("ওষুধ", "মেডিসিন", "ঔষধ"),
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


# ---------------------------------------------------------------------------
# DOSING - frequency and timing, as actually spoken.
#
# A prescription without a frequency is not a prescription. The SLM was
# returning medications with empty dosage/frequency/duration because the
# instruction is spoken in ordinary Bengali - "দুপুরে খাওয়ার পর" - and never
# looks like "BD" or "TDS" to a model expecting clinical shorthand.
#
# Mapped to standard notation so the printed prescription is unambiguous,
# with the plain-English gloss kept alongside for the reviewer.
# ---------------------------------------------------------------------------
DOSING_TERMS: dict[str, tuple[str, ...]] = {
    "after lunch": ("দুপুরে খাওয়ার পর", "দুপুরে খাবার পরে", "দুপুরের খাবারের পর",
                     "দুপুরে খেয়ে"),
    "after dinner": ("রাতে খাবার পরে", "রাতে খাওয়ার পর", "রাতের খাবারের পর",
                      "রাতে খেয়ে"),
    "after breakfast": ("সকালে খাওয়ার পর", "সকালে খাবার পরে", "ব্রেকফাস্টের পর"),
    "before food": ("খাওয়ার আগে", "খাবার আগে", "খালি পেটে", "খাওয়ার পূর্বে"),
    "after food": ("খাওয়ার পরে", "খাবার পরে", "ভরা পেটে", "খাওয়ার পর"),
    "in the morning": ("সকালে", "সকাল বেলা", "রোজ সকালে"),
    "at night": ("রাতে", "রাত্রে", "শোয়ার আগে", "ঘুমানোর আগে"),
    "twice daily": ("দিনে দুবার", "দুবেলা", "সকাল বিকেল", "সকালে আর রাতে"),
    "three times daily": ("দিনে তিনবার", "তিনবেলা", "তিন বেলা"),
    "once daily": ("দিনে একবার", "রোজ একটা", "একবেলা", "প্রতিদিন একবার"),
    "when required": ("দরকার হলে", "প্রয়োজন হলে", "যখন লাগবে", "কষ্ট হলে"),
}

# Frequency shorthand a printed prescription expects.
DOSING_CODES: dict[str, str] = {
    "once daily": "OD",
    "twice daily": "BD",
    "three times daily": "TDS",
    "when required": "SOS",
    "in the morning": "OD (morning)",
    "at night": "OD (night)",
}

DURATION_TERMS: dict[str, tuple[str, ...]] = {
    "3 days": ("তিন দিন", "তিনদিন"),
    "5 days": ("পাঁচ দিন", "পাঁচদিন"),
    "7 days": ("সাত দিন", "সাতদিন", "এক সপ্তাহ"),
    "10 days": ("দশ দিন", "দশদিন"),
    "15 days": ("পনেরো দিন", "পনেরদিন", "দুই সপ্তাহ"),
    "1 month": ("এক মাস", "একমাস", "৩০ দিন"),
    "2 months": ("দুই মাস", "দুমাস"),
    "3 months": ("তিন মাস", "তিনমাস"),
    "6 months": ("ছয় মাস", "ছমাস"),
    "continue": ("চালিয়ে যান", "চলতে থাকবে", "একটানা", "নিয়মিত"),
}


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

# Machine-imported terms from the MedER dataset - a LOWER trust tier than
# the curated table above, and loaded second so a curated entry always wins.
#
# These are only ever used to RULE THINGS OUT: a term here can never become
# a medication, it can only stop something being called one. That is the
# safe direction for unreviewed data - the worst case is a real drug being
# demoted to rejected_terms where a human sees it, not a wrong drug being
# added to a prescription.
#
# Optional by design: the file is generated, so glossary.py must still
# import cleanly in a fresh checkout before anyone has run the importer.
try:
    from .terms_imported import IMPORTED_TERMS
except ImportError:                              # pragma: no cover
    IMPORTED_TERMS = {}

_IMPORTED_COUNT = 0
for _canon, _alts in IMPORTED_TERMS.items():
    for _a in _alts:
        _k = fold(_a)
        # never let an import shadow a curated term, a drug or a lab test
        if _k in _TERM_LOOKUP or _k in _DRUG_LOOKUP or _k in _LAB_LOOKUP:
            continue
        _TERM_LOOKUP[_k] = _canon
        _IMPORTED_COUNT += 1


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


# Minimum folded length for a DRUG key to be matched inside free text.
#
# Short drug keys collide with ordinary words once folded. Found by the
# MedER import: the Lactulose brand "Looz" folds to "los" (z->s, then the
# doubled-o dedup), and the English word "loss" folds to "los" too - so
# "hair loss", "bone loss", "loss of appetite" and "weight loss" all
# resolved to Lactulose and would have entered medications[] as VERIFIED.
#
# collisions() could not catch this: it compares gazetteer entries against
# each other, never against ordinary vocabulary.
#
# Applied to drugs ONLY. Lab acronyms are legitimately short - "tmt", "mri",
# "psa", "usg" are 3 folded characters and must still match inside a
# sentence - and they are far less likely to collide because they are
# consonant clusters, not word-shaped. A short drug name still resolves via
# the whole-string exact lookup in lookup_drug(); it just cannot be fished
# out of the middle of a sentence.
_MIN_DRUG_NGRAM = 4


def _too_short_for_text(gram: str, table: dict) -> bool:
    """Guard against short DRUG keys being fished out of free text."""
    return table is _DRUG_LOOKUP and len(gram) < _MIN_DRUG_NGRAM


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
            if hit is not None and _too_short_for_text(gram, table):
                hit = None
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
            if hit is not None and _too_short_for_text(gram, table):
                hit = None
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


def scan_dosing(text: str) -> tuple[str, str]:
    """(frequency, duration) spoken in ordinary Bengali.

    Prescriptions were coming back with empty frequency and duration
    because a doctor says "দুপুরে খাওয়ার পর" (after lunch), never "BD".
    The SLM, prompted for clinical shorthand, returned nothing rather than
    guessing - correct behaviour, but it left the field blank.

    Returns standard notation where one exists ("twice daily" -> "BD"),
    otherwise the plain-English gloss, which is still unambiguous to a
    pharmacist.
    """
    freqs = _ngram_scan_all(text, _DOSING_LOOKUP)
    durs = _ngram_scan_all(text, _DURATION_LOOKUP)

    # Drop terms implied by a more specific one already present. "after
    # dinner" also matches "at night" and "after food", and printing all
    # three reads like three separate instructions.
    for specific, implied in _SUBSUMES.items():
        if specific in freqs:
            freqs = [f for f in freqs if f == specific or f not in implied]

    freq_parts = [DOSING_CODES.get(f, f) for f in freqs]
    return ", ".join(freq_parts), ", ".join(durs)


_SUBSUMES: dict[str, tuple[str, ...]] = {
    "after dinner": ("at night", "after food"),
    "after lunch": ("after food",),
    "after breakfast": ("in the morning", "after food"),
}


_DOSING_LOOKUP: dict[str, str] = {}
for _canon, _alts in DOSING_TERMS.items():
    _DOSING_LOOKUP[fold(_canon)] = _canon
    for _a in _alts:
        _DOSING_LOOKUP[fold(_a)] = _canon

_DURATION_LOOKUP: dict[str, str] = {}
for _canon, _alts in DURATION_TERMS.items():
    _DURATION_LOOKUP[fold(_canon)] = _canon
    for _a in _alts:
        _DURATION_LOOKUP[fold(_a)] = _canon

# ---------------------------------------------------------------------------
# WHICH CLINICAL TERMS ARE WHAT
#
# CLINICAL_TERMS is one flat table because its original job was only to say
# "this is not a drug". But the entries are three different kinds of thing,
# and treating them alike loses information:
#
#   conditions  cataract, diabetes, heart attack   -> a DIAGNOSIS
#   symptoms    eye pain, itching, breathlessness  -> a SYMPTOM
#   advice/     rest, exercise, antibiotic, syrup  -> neither; must never
#   classes                                           appear as either
#
# A live cataract consultation transcribed "ক্যাটারাক্ট" and "ছানি"
# perfectly, the gazetteer recognised both, and the prescription still came
# back with a blank diagnosis and no mention of cataract - because nothing
# ever carried the term into an output field.
# ---------------------------------------------------------------------------

CONDITIONS: frozenset[str] = frozenset({
    "cataract", "glaucoma", "diabetes", "heart attack", "heart failure",
    "ischemia", "angina", "blockage", "hypertension", "thyroid",
    "kidney failure", "stroke", "migraine", "seizure", "hernia", "piles",
    "gallstone", "appendicitis", "tonsillitis", "fungal infection",
    "infection", "acne", "menopause", "pregnancy", "dementia",
    "osteoporosis", "arthritis",
})

# Neither a symptom nor a diagnosis: advice, dosage forms and drug classes.
NON_CLINICAL_TERMS: frozenset[str] = frozenset({
    "exercise", "lean diet", "avoid oily food", "drink water", "ORS",
    "rest", "follow up", "bandage", "dressing", "nebulization",
    "prescription", "dialysis", "antibiotic", "painkiller", "antacid",
    "steroid", "vitamin supplement", "antihistamine", "eye drops",
    "ear drops", "nasal spray", "ointment", "syrup", "tablet",
    "injection", "medicine",
})


def scan_conditions(text: str) -> list[str]:
    """Diagnosable conditions named in the text."""
    return [t for t in _ngram_scan_all(text, _TERM_LOOKUP) if t in CONDITIONS]


def scan_symptoms(text: str) -> list[str]:
    """Symptoms named in the text - excludes conditions, advice and classes."""
    return [t for t in _ngram_scan_all(text, _TERM_LOOKUP)
            if t not in CONDITIONS and t not in NON_CLINICAL_TERMS]
