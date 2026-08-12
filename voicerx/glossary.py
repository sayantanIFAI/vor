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
import typing
import unicodedata
from difflib import SequenceMatcher


@dataclasses.dataclass(frozen=True)
class Drug:
    generic: str                  # canonical generic name
    brands: tuple[str, ...] = ()  # common Indian brand names
    bengali: tuple[str, ...] = () # transliterations the ASR actually emits
    indication: str = ""          # what it treats - shown to the reviewer
    department: str = "general"   #


# ---------------------------------------------------------------------------
# CARDIAC
# ---------------------------------------------------------------------------
CARDIAC = [
    Drug("Nitroglycerin", ("Nitrocontin", "Sorbitrate", "Angispan"),
         ("নাইট্রোকন্টিন", "নাইট্রোগ্লিসারিন", "সরবিট্রেট"),
         "angina / severe chest pain", "cardiac"), #
    Drug("Aspirin", ("Ecosprin", "Ecospirin", "Disprin"),
         ("ইকোস্পিরিন", "একোস্পিরিন", "অ্যাসপিরিন"),
         "antiplatelet / blood thinner", "cardiac"), #
    Drug("Clopidogrel", ("Plavix", "Clopilet", "Deplatt"),
         ("ক্লোপিডোগ্রেল", "ক্লোপিলেট"),
         "antiplatelet", "cardiac"), #
    Drug("Ticagrelor", ("Brilinta", "Axcer"),
         ("টিকাগ্রেলর", "ব্রিলিন্টা"),
         "antiplatelet", "cardiac"),
    Drug("Atorvastatin", ("Atorva", "Lipitor", "Storvas"),
         ("অ্যাটোরভাস্ট্যাটিন", "এটোরভা"),
         "cholesterol / statin", "cardiac"), #
    Drug("Rosuvastatin", ("Rosuvas", "Crestor", "Rozavel"),
         ("রোসুভাস্ট্যাটিন", "রোসুভাস", "রসু ভাস্টাটিন", "রসুভাস্টাটিন",
          "রোসু ভাস্ট্যাটিন", "ভাস্টাটিন", "রোজ়াভেল"),
         "cholesterol / statin", "cardiac"), #
    Drug("Amlodipine", ("Amlopres", "Amlokind", "Stamlo"),
         ("অ্যামলোডিপিন", "অ্যামলোপ্রেস"),
         "hypertension", "cardiac"), #
    Drug("Cilnidipine", ("Cilacar", "Cilnid"),
         ("সিলনিডিপিন", "সিলাকার"),
         "hypertension / calcium channel blocker", "cardiac"),
    Drug("Telmisartan", ("Telma", "Telsartan", "Tsart"),
         ("টেলমিসারটান", "টেলমা"),
         "hypertension", "cardiac"), #
    # FDC ENHANCEMENT: Explicit combinations to prevent ASR split drop-offs
    Drug("Telmisartan+Amlodipine", ("Telma-AM", "Telsartan-AM", "Creser-AM"),
         ("টেলমা এ এম", "টেলমিসারটান অ্যামলোডিপিন", "টেলমিসারটান আর অ্যামলোডিপিন"),
         "hypertension / FDC", "cardiac"),
    Drug("Sacubitril/Valsartan", ("Cidmus", "Vymada"),
         ("ভায়মাডা", "সিডমাস", "স্যাকুবিট্রিল", "স্যাকুবিট্রিল ভালসারটান"),
         "heart failure / ARNI", "cardiac"),
    Drug("Metoprolol", ("Metolar", "Betaloc", "Metoprolol Succinate"),
         ("মেটোপ্রোলল", "মেটোলার", "মেটো প্রোল", "মেটোপ্রোল",
          "মেটো প্রোলল", "মেটোপ্রোল সাক্সিনেট"),
         "beta blocker", "cardiac"), #
    Drug("Bisoprolol", ("Concor", "Bisolol"),
         ("বিসোপ্রোলল", "কনকর"),
         "beta blocker", "cardiac"), #
    Drug("Ramipril", ("Cardace", "Ramistar"),
         ("রামিপ্রিল", "কার্ডেস"),
         "ACE inhibitor / hypertension", "cardiac"), #
    Drug("Torsemide", ("Dytor", "Dytor Plus", "Tide"),
         ("ডাইটর", "টরসেমাইড", "ডায়টর"),
         "diuretic / fluid retention", "cardiac"), #
    Drug("Furosemide", ("Lasix", "Frusenex"),
         ("ফুরোসেমাইড", "ল্যাসিক্স"),
         "diuretic", "cardiac"), #
    Drug("Spironolactone", ("Aldactone",),
         ("স্পাইরোনোল্যাকটোন", "অ্যালড্যাকটোন"),
         "diuretic / heart failure", "cardiac"), #
    Drug("Isosorbide mononitrate", ("Monotrate", "Ismo"),
         ("আইসোসরবাইড", "মোনোট্রেট"),
         "angina prophylaxis", "cardiac"), #
    Drug("Ivabradine", ("Ivabrad", "Bradia"),
         ("ইভাগ্রাডিন", "ইভাব্রাডিন"),
         "heart failure / angina", "cardiac"),
]

# ---------------------------------------------------------------------------
# DIABETES / ENDOCRINE
# ---------------------------------------------------------------------------
ENDOCRINE = [
    Drug("Semaglutide", ("Rybelsus", "Ozempic"),
         ("রাইবেলসাস", "সেমাগ্লুটাইড", "রাইবেলসাস"),
         "oral GLP-1 / type 2 diabetes", "endocrine"), #
    Drug("Metformin", ("Glycomet", "Glucophage", "Okamet"),
         ("মেটফরমিন", "গ্লাইকোমেট", "মেট ফর্মিন", "মেটফর্মিন", "ফর্মিন"),
         "type 2 diabetes, first line", "endocrine"), #
    Drug("Glimepiride", ("Amaryl", "Glimestar", "Zoryl"),
         ("গ্লিমিপিরাইড", "অ্যামারিল", "গ্লিমেপিরাইড"),
         "sulfonylurea / diabetes", "endocrine"), #
    # FDC ENHANCEMENT
    Drug("Glimepiride+Metformin", ("Amaryl M", "Glycomet GP", "Glimestar M"),
         ("গ্লিমিপিরাইড মেটফরমিন", "অ্যামারিল এম", "গ্লাইকোমেট জিপি"),
         "diabetes FDC", "endocrine"),
    Drug("Gliclazide", ("Diamicron", "Glizid", "Reclimet"),
         ("গ্লিক্লাজাইড", "ডায়ামিক্রন"),
         "sulfonylurea / diabetes", "endocrine"), #
    Drug("Sitagliptin", ("Januvia", "Istavel"),
         ("সিটাগ্লিপটিন", "জানুভিয়া"),
         "DPP-4 inhibitor / diabetes", "endocrine"), #
    Drug("Teneligliptin", ("Zita Plus", "Tenepride", "Tenglyn"),
         ("টেনেগ্লিপটিন", "জিতা প্লাস"),
         "DPP-4 inhibitor / diabetes", "endocrine"),
    Drug("Linagliptin", ("Trajenta", "Linox"),
         ("লিনাগ্লিপটিন", "ট্রাজেন্টা"),
         "DPP-4 inhibitor / diabetes", "endocrine"),
    Drug("Dapagliflozin", ("Forxiga", "Dapa", "Oxra"),
         ("ড্যাপাগ্লিফ্লোজিন", "ফরজিগা", "ডাপা"),
         "SGLT2 inhibitor / diabetes", "endocrine"), #
    Drug("Empagliflozin", ("Jardiance", "Gibtulio"),
         ("এম্পাগ্লিফ্লোজিন", "জার্ডিয়ান্স"),
         "SGLT2 inhibitor / diabetes", "endocrine"),
    Drug("Insulin degludec/aspart", ("Ryzodeg",),
         ("রাইজোডেগ", "রাইজোডেগ ইনসুলিন", "ইনসুলিন ডিগ্লুডেক অ্যাসপার্ট"),
         "combination insulin", "endocrine"), #
    Drug("Insulin glargine", ("Lantus", "Basalog", "Glaritus"),
         ("ল্যান্টাস", "ইনসুলিন গ্লার্জিন", "ব্যাসালগ"),
         "long-acting insulin", "endocrine"), #
    Drug("Human insulin", ("Huminsulin", "Actrapid", "Mixtard"),
         ("ইনসুলিন", "হিউমিনসুলিন", "মিক্সটার্ড"),
         "insulin", "endocrine"), #
    Drug("Levothyroxine", ("Thyronorm", "Eltroxin", "Thyrox"),
         ("থাইরোনর্ম", "থাইরক্স", "লেভোথাইরক্সিন", "এলট্রক্সিন"),
         "hypothyroidism", "endocrine"), #
    Drug("Carbimazole", ("Neo-Mercazole", "Thyrocab"),
         ("কার্বিমাজোল", "নিও মার্কাজোল"),
         "hyperthyroidism", "endocrine"),
    Drug("Voglibose", ("Volix", "Vogs"),
         ("ভোগলিবোস", "ভোলিক্স"), "diabetes", "endocrine"), #
]

# ---------------------------------------------------------------------------
# RESPIRATORY
# ---------------------------------------------------------------------------
RESPIRATORY = [
    Drug("Salbutamol", ("Asthalin", "Ventolin", "Levolin"),
         ("সালবুটামল", "অ্যাসথালিন", "ভেন্টোলিন", "লেভোলিন"),
         "bronchodilator / nebulisation", "respiratory"), #
    Drug("Budesonide", ("Budecort", "Pulmicort"),
         ("বুডেসোনাইড", "বুডেকর্ট"),
         "inhaled steroid", "respiratory"), #
    # FDC ENHANCEMENT
    Drug("Formoterol+Budesonide", ("Foracort", "Budamate"),
         ("ফোরকোর্ট", "বুডামেট", "ফরমোটেরল বুডেসোনাইড", "ফরমোটেরল এবং বুডেসোনাইড"),
         "ICS+LABA / asthma / COPD", "respiratory"),
    Drug("Ipratropium", ("Duolin", "Ipravent"),
         ("আইপ্রাট্রোপিয়াম", "ডুওলিন"),
         "bronchodilator", "respiratory"), #
    Drug("Tiotropium", ("Tiova", "Tiotrop"),
         ("টিওট্রোপিয়াম", "টিওভা"),
         "LAMA / COPD", "respiratory"),
    Drug("Montelukast", ("Montair", "Montek"),
         ("মন্টিকুলাষ্ট", "মন্টিকুলাস", "মন্টেয়ার", "মন্টেক",
          "মাল্টিকুলার", "মাল্টিকুলার স্ট্যাবলেট", "মন্টিকুলার"),
         "asthma / allergic rhinitis", "respiratory"), #
    # FDC ENHANCEMENT
    Drug("Montelukast+Levocetirizine", ("Montair LC", "Montek LC", "Monticope"),
         ("মন্টেয়ার এল সি", "মন্টেক এল সি", "মন্টিকোপ", "মন্টিকুলাষ্ট লেভোসেট্রিজিন", "মন্টিকুলাষ্ট আর লেভোসেট্রিজিন"),
         "asthma / severe allergic rhinitis", "respiratory"),
    # "অ্যাম্ব্রুডিল" is how it came back on a real chest consultation -
    # ব্রু rather than ব্রো. Held only the ব্রো spelling, so the syrup was
    # missed entirely.
    Drug("Ambroxol", ("Ambrodil", "Mucolite"),
         ("অ্যামব্রক্সল", "অ্যামব্রোডিল", "অ্যাম্ব্রুডিল", "অ্যামব্রুডিল",
          "অ্যাম্ব্রোডিল"),
         "mucolytic / cough", "respiratory"), #
    Drug("Ascoril", ("Ascoril LS", "Ascoril D"),
         ("এস্কোরিল", "আস্কোরিল", "অ্যাসকোরিল"),
         "cough syrup (combination)", "respiratory"), #
    Drug("Levocetirizine", ("Levocet", "Xyzal"),
         ("লেভোসেটিরিজিন", "লেভোসেট"),
         "antihistamine", "respiratory"), #
    Drug("Cetirizine", ("Cetzine", "Alerid"),
         ("সেটিরিজিন", "সেটজিন"),
         "antihistamine", "respiratory"), #
    Drug("Fexofenadine", ("Allegra", "Fexova", "Altiva"),
         ("ফেক্সোফেনাডিন", "অ্যালেগ্রা"),
         "antihistamine", "respiratory"), #
    Drug("Deriphyllin", ("Deriphyllin",),
         ("ডেরিফাইলিন",), "bronchodilator", "respiratory"), #
    Drug("Doxofylline", ("Doxovent", "Doxiflo"),
         ("ডক্সোফাইলিন", "ডক্সোভেন্ট"), "bronchodilator", "respiratory"),
]

# ---------------------------------------------------------------------------
# GASTROINTESTINAL
# ---------------------------------------------------------------------------
GI = [
    # FDC ENHANCEMENT
    Drug("Norfloxacin+Tinidazole", ("Norflox-TZ", "Normet"),
         ("নরফ্লক্স টি জেড", "নরফ্লক্স-টিজেড", "নরফ্লক্সাসিন টিনিডাজোল", "নরফ্লক্সাসিন এবং টিনিডাজোল"),
         "infective diarrhoea", "gastro"), #
    Drug("Pantoprazole", ("Pantocid", "Pantop"),
         ("প্যান্টোপ্রাজল", "প্যানটোসিড", "প্যান্টপ"),
         "acidity / PPI", "gastro"), #
    # FDC ENHANCEMENT
    Drug("Pantoprazole+Domperidone", ("Pan-D", "Pantocid DSR"),
         ("প্যান ডি", "প্যানটোসিড ডিএসআর", "প্যান্টোপ্রাজল ডমপেরিডন"),
         "GERD / acidity", "gastro"),
    Drug("Omeprazole", ("Omez", "Ocid"),
         ("ওমিপ্রাজল", "ওমেজ"),
         "acidity / PPI", "gastro"), #
    Drug("Rabeprazole", ("Razo", "Rabekind", "Veloz"),
         ("র‍্যাবিপ্রাজল", "রাজো", "রাবেপ্রাজল"),
         "acidity / PPI", "gastro"), #
    Drug("Esomeprazole", ("Nexpro", "Sompraz", "Esomac"),
         ("ইসোমিপ্রাজোল", "নেক্সপ্রো", "সমপ্রাজ"),
         "acidity / PPI", "gastro"),
    Drug("Domperidone", ("Domstal", "Vomistop"),
         ("ডমপেরিডন", "ডমস্টাল"),
         "nausea / vomiting", "gastro"), #
    Drug("Itopride", ("Itomac", "Ganaton"),
         ("ইটোপ্রাইড", "ইটম্যাক"),
         "prokinetic / dyspepsia", "gastro"),
    Drug("Levosulpiride", ("Nexpro L", "Lesuride"),
         ("লেভোসালপিরাইড", "লেসুরাইড"),
         "prokinetic / GERD", "gastro"),
    Drug("Ondansetron", ("Emeset", "Vomikind", "Zofer"),
         ("অনডানসেট্রন", "ইমিসেট"),
         "antiemetic", "gastro"), #
    Drug("Metronidazole", ("Flagyl", "Metrogyl"),
         ("মেট্রোনিডাজল", "মেট্রোজিল", "ফ্ল্যাজিল"),
         "anaerobic / amoebic infection", "gastro"), #
    # Prescribed alongside Ciprofloxacin on gastroenteritis. "টিনিটা জল" is
    # the ASR's rendering - it splits the name and turns the ending into
    # "জল" (water), so neither the fold nor the skeleton reaches
    # Tinidazole. Without a Bengali form the grounding check could not find
    # the drug in the transcript either: the model named it correctly and
    # it was rejected at 0.60 as a name nobody said.
    Drug("Tinidazole", ("Tiniba", "Fasigyn", "Tinidral"),
         ("টিনিডাজোল", "টিনিডাজল", "টিনিবা", "টিনিটা জল", "টিনিডা জল"),
         "anaerobic / amoebic infection", "gastro"), #
    # ORS was filed as a clinical term - "rehydration, NOT a pharmaceutical".
    # Defensible chemically, wrong operationally: it is dispensed, it carries
    # a dose and a frequency, and on a diarrhoea consultation it is the most
    # important thing prescribed. Filed as a non-drug, the gate rejected it
    # from medications every time, exactly as designed.
    Drug("ORS", ("Electral", "Enerzal", "Walyte"),
         ("ওআরএস", "ওরস", "ওয়ারেস্ট", "ও আর এস", "ওয়ার এস",
          "ওরস্যালাইন", "খাবার স্যালাইন"),
         "oral rehydration", "gastro"), #
    # FDC ENHANCEMENT
    Drug("Ofloxacin+Ornidazole", ("O2", "Oflomac-OZ", "Zenflox-OZ"),
         ("অফ্লক্সাসিন অর্নিডাজোল", "ওফ্লোম্যাক ওজেড", "ও টু", "অফ্লক্সাসিন এবং অর্নিডাজোল"),
         "diarrhoea / infection", "gastro"), #
    Drug("Racecadotril", ("Redotil", "Zedott"),
         ("রেসিকাডোট্রিল", "রেডোটিল", "রোসকাডো ট্রিল", "রোসকাডোট্রিল",
          "রেসিকাডো ট্রিল", "রোসাকাডোট্রিল"),
         "acute diarrhoea", "gastro"), #
    # Missing entirely, and one of the commonest oral antibiotics there is.
    # Prescribed by brand on a gastroenteritis consultation ("ওফ্লোম্যাক")
    # and reported only as an unrecognised term.
    Drug("Ofloxacin", ("Oflomac", "Zanocin", "Zenflox"),
         ("ওফ্লক্সাসিন", "ওফ্লোম্যাক", "জ্যানোসিন", "ওফ্লোমাক"),
         "antibiotic", "gastro"), #
    Drug("Sucralfate", ("Sucral", "Sucrafil"),
         ("সুক্রালফেট", "সুক্রাফিল"), "gastric ulcer", "gastro"), #
    Drug("Dicyclomine", ("Cyclopam", "Meftal-Spas"),
         ("ডাইসাইক্লোমিন", "সাইক্লোপাম", "মেফটাল স্পাস"),
         "abdominal cramps", "gastro"), #
    Drug("Mebeverine", ("Colospa", "Morease"),
         ("মেবেভেরিন", "কোলোস্পা"),
         "IBS / antispasmodic", "gastro"),
    Drug("Lactulose", ("Duphalac", "Looz"),
         ("ল্যাকটুলোজ", "ডুফালাক"),
         "constipation", "gastro"), #
    Drug("Ursodeoxycholic acid", ("Udapa", "Udiliv", "Ursocol"),
         ("ইউডিসিএ", "আরসোডিঅক্সিকোলিক", "উডিলিভ"),
         "gallstones / liver disease", "gastro"),
]

# ---------------------------------------------------------------------------
# ANALGESIC / ANTIPYRETIC / ANTIBIOTIC (general OPD)
# ---------------------------------------------------------------------------
GENERAL = [
    Drug("Paracetamol", ("Crocin", "Dolo", "Calpol", "Pyrigesic"),
         ("প্যারাসিটামল", "প্যারাসিটাম", "ক্রোসিন", "ডোলো", "কালপল"),
         "fever / pain", "general"), #
    Drug("Ibuprofen", ("Brufen", "Combiflam"),
         ("আইবুপ্রোফেন", "ব্রুফেন", "কম্বিফ্লাম"),
         "pain / inflammation", "general"), #
    Drug("Diclofenac", ("Voveran", "Volini"),
         ("ডাইক্লোফেনাক", "ভোভেরান"),
         "pain / inflammation", "general"), #
    Drug("Aceclofenac", ("Zerodol", "Hifenac"),
         ("এসিক্লোফেনাক", "জেরোডল", "অ্যাসেক্লোফেনাক"),
         "pain / inflammation", "general"), #
    # FDC ENHANCEMENT
    Drug("Aceclofenac+Paracetamol", ("Zerodol-P", "Hifenac-P", "Aldigesic-P"), 
         ("জেরোডল পি", "অ্যাসিক্লোফেনাক প্যারাসিটামল", "অ্যাসিক্লোফেনাক এবং প্যারাসিটামল"), 
         "pain / fever", "general"),
    Drug("Nimesulide", ("Nise", "Nimulid"),
         ("নিমেসুলাইড", "নাইস"),
         "pain / inflammation", "general"),
    Drug("Etoricoxib", ("Nucoxia", "Etoshine"),
         ("ইটোরিকক্সিব", "নুকক্সিয়া", "ইটোশাইন"),
         "NSAID / pain", "general"),
    # FDC ENHANCEMENT
    Drug("Amoxicillin+Clavulanate", ("Augmentin", "Clavam", "Moxikind-CV"),
         ("ক্ল্যাভাম", "ক্লাব", "অগমেন্টিন", "অ্যামোক্সিক্লাভ", "মক্সিকাইন্ড সি ভি", "অ্যামোক্সিসিলিন ক্ল্যাভুলানেট", "অ্যামোক্সিসিলিন পটাশিয়াম ক্ল্যাভুলানেট"),
         "broad-spectrum antibiotic", "general"), #
    Drug("Amoxicillin", ("Mox", "Novamox"),
         ("অ্যামোক্সিসিলিন", "মক্স"),
         "antibiotic", "general"), #
    Drug("Azithromycin", ("Azithral", "Azee", "Zithromax"),
         ("অ্যাজিথ্রোমাইসিন", "অ্যাজিথ্রাল", "অ্যাজি"),
         "antibiotic", "general"), #
    Drug("Cefixime", ("Taxim-O", "Zifi", "Mahacef"),
         ("সেফিক্সিম", "ট্যাক্সিম ও", "জিফি"),
         "antibiotic", "general"), #
    Drug("Cefpodoxime", ("Monocef-O", "Gudcef", "Macpod"),
         ("সেফপোডক্সিম", "মনোসেফ ও", "গুডসেফ"),
         "antibiotic", "general"),
    Drug("Ciprofloxacin", ("Ciplox", "Cifran"),
         ("সিপ্রোফ্লক্সাসিন", "সিপ্লক্স"),
         "antibiotic", "general"), #
    Drug("Levofloxacin", ("Levoflox", "Levotas", "Loxof"),
         ("লেভোফ্লক্সাসিন", "লেভোফ্লক্স"),
         "antibiotic", "general"), #
    Drug("Linezolid", ("Lizolid", "Linid"),
         ("লিনেজোলিড", "লিঞ্জোলিড"),
         "antibiotic", "general"),
    Drug("Doxycycline", ("Doxt", "Doxy-1"),
         ("ডক্সিসাইক্লিন", "ডক্সি"), "antibiotic", "general"), #
    Drug("Diazepam", ("Valium", "Calmpose"),
         ("ভ্যালিয়াম", "ভ্যালুম", "ডায়াজেপাম"),
         "anxiolytic / sedative", "general"), #
    Drug("Alprazolam", ("Alprax", "Restyl"),
         ("অ্যালপ্রাজোলাম", "অ্যালপ্র্যাক্স"),
         "anxiolytic", "general"), #
    Drug("Prednisolone", ("Omnacortil", "Wysolone"),
         ("প্রেডনিসোলন", "ওমনাকর্টিল", "ওয়াইসোলন"),
         "steroid", "general"), #
    Drug("Methylprednisolone", ("Medrol", "Macpred"),
         ("মিথাইলপ্রেডনিসোলন", "মেড্রোল"),
         "steroid", "general"),
    Drug("Vitamin B complex", ("Becosules", "Neurobion"),
         ("বিকোসুলস", "নিউরোবিন", "ভিটামিন বি"),
         "supplement", "general"), #
    Drug("Vitamin D3", ("Calcirol", "Uprise-D3"),
         ("ভিটামিন ডি", "ক্যালসিরল", "আপরাইজ ডি থ্রি"),
         "supplement", "general"), #
    Drug("Calcium carbonate", ("Shelcal", "Calcimax"),
         ("শেলক্যাল", "ক্যালসিয়াম"),
         "supplement", "general"), #
    # The bare "ফলিক অ্যাসিড" belongs to Folic acid alone. Sharing it here
    # meant one spoken phrase resolved to two different products, which the
    # strengthened collisions() check caught.
    Drug("Iron/Folic acid", ("Autrin", "Fefol", "Orofer"),
         ("আয়রন", "আয়রন ফলিক অ্যাসিড", "আয়রন ট্যাবলেট", "ওরোফার", "আয়রন এবং ফলিক অ্যাসিড"),
         "supplement / anaemia", "general"), #
    Drug("Multivitamin", ("Zincovit", "A to Z", "Supradyn"),
         ("জিঙ্কোভিট", "মাল্টিভিটামিন", "সুপ্রাডিন"),
         "supplement", "general"), #
]

# ---------------------------------------------------------------------------
# UROLOGY / PROSTATE
# ---------------------------------------------------------------------------
UROLOGY = [
    Drug("Tamsulosin", ("Urimax", "Veltam", "Contiflo"),
         ("ট্যামসুলোসিন", "ইউরিম্যাক্স", "কন্টিফ্লো"),
         "prostate / BPH", "urology"), #
    Drug("Finasteride", ("Finast", "Fincar"),
         ("ফিনাস্টেরাইড", "ফিনাস্ট"), "prostate / BPH", "urology"), #
    Drug("Dutasteride", ("Dutas", "Veltride"),
         ("ডুটাস্টেরাইড", "দুতাস"), "prostate / BPH", "urology"),
    Drug("Nitrofurantoin", ("Niftran", "Martifur"),
         ("নাইট্রোফুরানটোইন", "নিফট্রান"), "urinary tract infection", "urology"), #
    Drug("Silodosin", ("Silodal", "Rapilif"),
         ("সিলোডোসিন", "সিলোডাল"), "BPH / enlarged prostate", "urology"),
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
         ("অ্যালেনড্রোনেট", "অ্যালেন্ড্রোনেট", "ফোসাম্যাক্স", "অস্টিওফস"),
         "bisphosphonate / osteoporosis", "bone"), #
    Drug("Risedronate", ("Actonel", "Risofos"),
         ("রাইসেড্রোনেট",), "bisphosphonate / osteoporosis", "bone"), #
    Drug("Zoledronic acid", ("Zoldria", "Aclasta", "Zybone"),
         ("জোলেড্রনিক অ্যাসিড", "জোলেড্রোনিক", "জোলড্রিয়া"),
         "IV bisphosphonate, yearly", "bone"), #
    Drug("Teriparatide", ("Forteo", "Bonista"),
         ("টেরিপ্যারাটাইড",), "anabolic / severe osteoporosis", "bone"), #
    Drug("Denosumab", ("Prolia", "Xgeva"),
         ("ডেনোসুম্যাব",), "RANKL inhibitor / osteoporosis", "bone"), #
    # Calcirol is CHOLECALCIFEROL, not calcitriol - three entries claimed
    # that brand and the strengthened collision check caught it.
    # "ক্যালসিড ট্রায়াল" is the ASR splitting ক্যালসিট্রায়ল across two
    # tokens on a real osteoporosis consultation, where it was prescribed
    # alongside the alendronate and missed entirely.
    Drug("Calcitriol", ("Rocaltrol", "Calcijunc"),
         ("ক্যালসিট্রায়ল", "ক্যালসিড ট্রায়াল", "ক্যালসি ট্রায়াল",
          "ক্যালসিট্রায়েল"),
         "active vitamin D", "bone"), #
    # "ভিটামিন ব টেলভে" was resolving to Vitamin B COMPLEX - a different
    # product. B12 is what is given for nerve damage, and dropping the
    # "12" changes the drug. Longest match wins, so the qualifier has to
    # exist as its own entry.
    Drug("Vitamin B12", ("Methycobal", "Bcozym"),
         ("ভিটামিন ব টেলভে", "ভিটামিন বি টুয়েলভ", "ভিটামিন বি ১২",
          "ভিটামিন বি১২", "ভিটামিন বি টেলভ", "কোবালামিন"),
         "B12 / nerve repair", "neurology"), #
    Drug("Methylcobalamin", ("Nurokind", "Meconerv"),
         ("মিথাইলকোবালামিন", "নিউরোকাইন্ড"),
         "neuropathy / B12", "bone"), #
    Drug("Pregabalin", ("Pregabid", "Maxgalin"),
         ("প্রিগাবালিন", "প্রেগাবালিন"), "nerve pain", "bone"),
]

# ---------------------------------------------------------------------------
# DERMATOLOGY (skin)
# ---------------------------------------------------------------------------
DERMATOLOGY = [
    Drug("Clotrimazole", ("Candid", "Canesten", "Clop-G"),
         ("ক্লোট্রিমাজল", "ক্যান্ডিড", "ক্লোট্রিমাজোল"), "antifungal cream", "dermatology"), #
    Drug("Ketoconazole", ("Nizral", "Ketostar"),
         ("কিটোকোনাজল", "নিজরাল", "কেটোকোনাজোল"), "antifungal", "dermatology"), #
    Drug("Itraconazole", ("Itramac", "Canditral", "Sporanox"),
         ("ইট্রাকোনাজোল", "ইট্রাম্যাক"), "systemic antifungal", "dermatology"),
    Drug("Luliconazole", ("Lulican", "Luz"),
         ("লুলিকোনাজোল", "লুলিক্যান"), "topical antifungal", "dermatology"),
    Drug("Terbinafine", ("Terbicip", "Lamisil", "Sebifin"),
         ("টারবিনাফাইন", "টারবিসিপ", "টারবিনাফিন"), "antifungal", "dermatology"), #
    Drug("Fluconazole", ("Forcan", "Zocon", "Flucos"),
         ("ফ্লুকোনাজল", "ফোরকান", "ফ্লুকোনাজোল"), "systemic antifungal", "dermatology"), #
    Drug("Mupirocin", ("T-Bact", "Bactroban", "Mupinase"),
         ("মিউপিরোসিন", "টি ব্যাক্ট"), "topical antibiotic", "dermatology"), #
    Drug("Betamethasone", ("Betnovate", "Betnesol"),
         ("বিটামেথাসোন", "বেটনোভেট"), "topical steroid", "dermatology"), #
    Drug("Clobetasol", ("Tenovate", "Lobate", "Clobetamil"),
         ("ক্লোবেটাসোল", "টেনোভেট", "লোবেট"), "topical steroid", "dermatology"),
    Drug("Salicylic acid", ("Salicylix", "Clear"),
         ("স্যালিসাইলিক অ্যাসিড", "স্যালিসাইলিক"), "keratolytic", "dermatology"),
    Drug("Permethrin", ("Permite", "Scabper"),
         ("পারমেথ্রিন", "পারমাইট"), "scabies", "dermatology"), #
    Drug("Ivermectin", ("Ivermectol", "Iverjohn"),
         ("আইভারমেকটিন", "আইভারমেকটোল"), "scabies / parasites", "dermatology"), #
    Drug("Isotretinoin", ("Sotret", "Isotroin", "Accufine"),
         ("আইসোট্রেটিনয়েন", "আইসোট্রইন"), "severe acne", "dermatology"), #
    Drug("Adapalene", ("Deriva", "Adaferin"),
         ("অ্যাডাপালিন", "ডেরিভা"), "acne", "dermatology"), #
    Drug("Tacrolimus ointment", ("Tacroz", "Protopic"),
         ("ট্যাক্রোলিমাস", "ট্যাকরোজ"), "eczema / vitiligo", "dermatology"), #
    Drug("Calamine", ("Calamine lotion", "Caladryl"),
         ("ক্যালামিন", "ক্যালাড্রিল"), "soothing lotion", "dermatology"), #
    Drug("Minoxidil", ("Mintop", "Morr"),
         ("মিনক্সিডিল", "মিনটপ"), "hair loss", "dermatology"),
]

# ---------------------------------------------------------------------------
# OPHTHALMOLOGY (eye)
# ---------------------------------------------------------------------------
OPHTHALMOLOGY = [
    Drug("Moxifloxacin eye drops", ("Vigamox", "5-Moxi", "Milflox", "Moxicip"),
         ("মক্সিফ্লক্সাসিন", "ভিগাম্যাক্স", "মক্সি ফ্লক্সাসিম", "ফ্লক্সাসিম",
          "মক্সিফ্লক্সাসিম", "মক্সি ফ্লক্সাসিন", "মক্সিফ্লক্স"),
         "eye infection", "ophthalmology"), #
    Drug("Tobramycin eye drops", ("Toba", "Eyebrex"),
         ("টোব্রামাইসিন", "টোবা"), "antibiotic eye drops", "ophthalmology"),
    Drug("Flurbiprofen eye drops", ("Flur", "Flubiprof", "Ocuflur"),
         ("ফ্লার্বিপ্রোফেন", "ফ্লুরবিপ্রোফেন", "ফ্লারবি"),
         "ocular NSAID", "ophthalmology"), #
    Drug("Nepafenac eye drops", ("Nevanac", "Nepastar"),
         ("নেপাফেনাক", "নেভান্যাক"), "ocular NSAID", "ophthalmology"),
    Drug("Carboxymethylcellulose", ("Refresh Tears", "Optive", "Lubrex"),
         ("রিফ্রেশ টিয়ার্স", "কৃত্রিম অশ্রু", "কার্বক্সিমিথাইলসেলুলোজ"), "dry eye lubricant", "ophthalmology"), #
    Drug("Olopatadine", ("Patanol", "Winolap"),
         ("ওলোপাটাডিন", "প্যাটানল", "উইনোলাপ"), "allergic conjunctivitis", "ophthalmology"), #
    Drug("Timolol eye drops", ("Glucomol", "Iotim"),
         ("টিমোলল", "গ্লুকোমল"), "glaucoma", "ophthalmology"), #
    Drug("Latanoprost", ("Xalatan", "9PM", "Latoprost"),
         ("ল্যাটানোপ্রস্ট", "ল্যাটোপোস্ট"), "glaucoma", "ophthalmology"), #
    Drug("Bimatoprost", ("Lumigan", "Bimat"),
         ("বিমাটোপ্রস্ট", "লুমিগান"), "glaucoma", "ophthalmology"),
    Drug("Dorzolamide", ("Dorzox", "Trusopt"),
         ("ডোরজোলামাইড", "ডোরজক্স"), "glaucoma", "ophthalmology"),
    Drug("Brimonidine", ("Alphagan", "Brimolol"),
         ("ব্রিমোনিডিন", "অ্যালফাগান"), "glaucoma", "ophthalmology"), #
    Drug("Prednisolone eye drops", ("Predmet", "Omnipred"),
         ("প্রেডনিসোলন ড্রপ", "প্রেডনিসোলন আই ড্রপ"), "ocular inflammation", "ophthalmology"), #
    Drug("Fluorometholone", ("FML", "Fluro"),
         ("ফ্লুরোমেথোলোন", "এফএমএল"), "ocular steroid", "ophthalmology"),
    Drug("Tropicamide", ("Tropicacyl", "Mydriacyl"),
         ("ট্রপিকামাইড", "ট্রপিকাসিল"), "pupil dilation", "ophthalmology"), #
]

# ---------------------------------------------------------------------------
# ENT
# ---------------------------------------------------------------------------
ENT = [
    Drug("Xylometazoline", ("Otrivin",),
         ("জাইলোমেটাজোলিন", "অট্রিভিন"),
         "nasal decongestant drops", "ent"), #
    Drug("Oxymetazoline", ("Nasivion",),
         ("অক্সিমেটাজোলিন", "নাসিভিয়ন ড্রপ"), "nasal decongestant", "ent"),
    Drug("Fluticasone nasal spray", ("Flomist", "Nasoflo"),
         ("ফ্লুটিকাসোন", "ফ্লোমিস্ট", "ফ্লুটিকাসোন নেজাল স্প্রে",
          "ফাল্টিকাসন", "ফাল্টিকাসন নাসাল স্প্রা", "ফ্লুটিকেসোন"),
         "allergic rhinitis", "ent"), #
    Drug("Mometasone nasal spray", ("Nasonex", "Metaspray"),
         ("মোমেটাসোন", "মেটাস্প্রে"), "allergic rhinitis", "ent"), #
    Drug("Betahistine", ("Vertin", "Betavert"),
         ("বিটাহিস্টিন", "ভার্টিন"), "vertigo", "ent"), #
    Drug("Candibiotic ear drops", ("Candibiotic", "Otek-AC"),
         ("ক্যান্ডিবায়োটিক", "ওটেক"), "ear infection", "ent"), #
    Drug("Prochlorperazine", ("Stemetil", "Vertigon"),
         ("প্রোক্লোরপেরাজিন", "স্টেমেটিল"), "vertigo / nausea", "ent"), #
    Drug("Paradichlorobenzene / Chlorbutol", ("Clearwax", "Waxolve"),
         ("ক্লিয়ারওয়্যাক্স", "ওয়্যাক্স সলভেন্ট ড্রপ", "প্যারাডিক্লোরোবেনজিন"), "ear wax solvent", "ent"),
]

# ---------------------------------------------------------------------------
# DENTAL
# ---------------------------------------------------------------------------
DENTAL = [
    Drug("Chlorhexidine mouthwash", ("Hexidine", "Clohex", "Rexidin"),
         ("ক্লোরহেক্সিডিন", "হেক্সিডিন", "মাউথওয়াশ"),
         "oral antiseptic", "dental"), #
    Drug("Ketorolac", ("Ketorol", "Zerodol-K"),
         ("কিটোরোলাক", "কেটোরল"), "severe dental pain", "dental"), #
    Drug("Lignocaine gel", ("Mucopain", "Dentogel", "Xylocaine"),
         ("লিগনোকেইন", "মিউকোপেইন", "ডেন্টোজেল"), "topical anaesthetic", "dental"), #
    Drug("Triamcinolone oral paste", ("Kenacort", "Tess"),
         ("ট্রায়ামসিনোলোন", "কেনাকর্ট"), "mouth ulcer", "dental"), #
]

# ---------------------------------------------------------------------------
# GYNAECOLOGY
# ---------------------------------------------------------------------------
GYNAECOLOGY = [
    Drug("Folic acid", ("Folvite", "Fol-5"),
         ("ফলিক অ্যাসিড", "ফলভাইট"), "pregnancy supplement", "gynaecology"), #
    # NOTE: the iron+folic combination is a DIFFERENT product and must not
    # share the bare "ফলিক অ্যাসিড" alias - collisions() flagged the clash.
    Drug("Tranexamic acid", ("Trapic", "Pause", "Texid"),
         ("ট্রানেক্সামিক অ্যাসিড", "ট্রাপিক"),
         "heavy menstrual bleeding", "gynaecology"), #
    Drug("Norethisterone", ("Primolut-N", "Regestrone"),
         ("নরইথিস্টেরন", "প্রিমোলাট", "রেজেস্ট্রোন"), "menstrual regulation", "gynaecology"), #
    Drug("Medroxyprogesterone", ("Deviry", "Meprate"),
         ("মেড্রক্সিপ্রোজেস্টেরন", "ডেভিরি", "মেপ্রেট"), "progestin", "gynaecology"), #
    Drug("Progesterone", ("Susten", "Duphaston", "Dubagest"),
         ("প্রোজেস্টেরন", "সাসটেন", "ডুফাস্টন"),
         "luteal support", "gynaecology"), #
    Drug("Dienogest", ("Visanne", "Dinogest"),
         ("ডাইনোজেস্ট", "ভিসান"), "endometriosis", "gynaecology"),
    Drug("Drospirenone", ("Crisanta", "Yaz"),
         ("ড্রোসপিরেনোন", "ক্রিসান্টা"), "PCOS / OCP", "gynaecology"),
    Drug("Myo-inositol", ("Myo-inositol", "Oosure"),
         ("মায়ো ইনোসিটল", "ইনোসিটল"), "PCOS", "gynaecology"),
    Drug("Clomiphene", ("Fertyl", "Clomi"),
         ("ক্লোমিফেন", "ফার্টিল"), "ovulation induction", "gynaecology"), #
    Drug("Letrozole", ("Letroz", "Fempro"),
         ("লেট্রোজল", "লেট্রোজ"), "ovulation induction", "gynaecology"), #
    Drug("Cabergoline", ("Cabgolin",),
         ("ক্যাবারগোলিন", "ক্যাবগোলিন"), "hyperprolactinemia", "gynaecology"),
    # FDC ENHANCEMENT
    Drug("Mifepristone+Misoprostol", ("MTP Kit", "Unwanted Kit"),
         ("মিফেপ্রিস্টোন", "মাইসোপ্রোস্টল", "এমটিপি কিট", "মিফেপ্রিস্টোন মাইসোপ্রোস্টল"),
         "medical termination", "gynaecology"), #
]

# ---------------------------------------------------------------------------
# NEPHROLOGY
# ---------------------------------------------------------------------------
NEPHROLOGY = [
    Drug("Sevelamer", ("Renvela", "Sevcar"),
         ("সেভেলামার", "রেণভেলা"), "phosphate binder / CKD", "nephrology"), #
    Drug("Calcium acetate", ("Nephrocal", "Royal-CA"),
         ("ক্যালসিয়াম অ্যাসিটেট", "নেফ্রোকেল"), "phosphate binder", "nephrology"), #
    Drug("Erythropoietin", ("Eprex", "Epofit", "Relipoietin"),
         ("এরিথ্রোপয়েটিন", "ইপ্রেক্স", "ইপোফিট"), "anaemia of CKD", "nephrology"), #
    Drug("Sodium bicarbonate", ("Sodamint", "Nodosis"),
         ("সোডিয়াম বাইকার্বোনেট", "নোডোসিস", "সোডামিন্ট"),
         "metabolic acidosis", "nephrology"), #
    Drug("Alpha ketoanalogue", ("Ketosteril", "Renolog"),
         ("কিটোস্টেরিল", "আলফা কিটোঅ্যানালগ"), "CKD supplement", "nephrology"),
    Drug("Febuxostat", ("Febutaz", "Zurig", "Feburic"),
         ("ফেবুক্সোস্ট্যাট", "ফেবুটাজ", "ফেবুরিক"), "gout / uric acid", "nephrology"), #
    Drug("Allopurinol", ("Zyloric", "Ciploric"),
         ("অ্যালোপিউরিনল", "জাইলোরিক"), "gout / uric acid", "nephrology"), #
    Drug("Mycophenolate", ("Cellcept", "Mycept"),
         ("মাইকোফেনোলেট", "সেলসেপ্ট"), "immunosuppressant", "nephrology"),
]

# ---------------------------------------------------------------------------
# NEUROLOGY & PSYCHIATRY
# ---------------------------------------------------------------------------
NEUROLOGY = [
    Drug("Levetiracetam", ("Levipil", "Keppra", "Torleva"),
         ("লেভেটিরাসিটাম", "লেভিপিল", "কেপরা", "লোভা পল", "লেভা পিল",
          "লোভাপল"), "epilepsy", "neurology"), #
    Drug("Sodium valproate", ("Valparin", "Encorate", "Divaa"),
         ("সোডিয়াম ভালপ্রোয়েট", "ভালপারিন", "এনকোরেট", "ভাল পারিং",
          "ভালপারিং", "ভাল পারিন"),
         "epilepsy / migraine", "neurology"), #
    Drug("Phenytoin", ("Eptoin", "Dilantin"),
         ("ফেনিটয়েন", "এপটোইন"), "epilepsy", "neurology"), #
    Drug("Carbamazepine", ("Tegretol", "Mazetol", "Zeptol"),
         ("কার্বামাজেপিন", "টেগ্রেটল"),
         "epilepsy / trigeminal neuralgia", "neurology"), #
    Drug("Clobazam", ("Frisium", "Lobazam"),
         ("ক্লোবাজাম", "ফ্রিজিয়াম"), "epilepsy", "neurology"), #
    Drug("Gabapentin", ("Gabapin", "Neurontin"),
         ("গ্যাবাপেন্টিন", "গ্যাবাপিন"), "neuropathic pain", "neurology"), #
    Drug("Donepezil", ("Aricept", "Donep", "Dompezil"),
         ("ডোনেপেজিল", "অ্যারিসেপ্ট", "ডনেপ"), "dementia", "neurology"), #
    Drug("Memantine", ("Admenta", "Nemdaa"),
         ("মেমান্টিন", "অ্যাডমেন্টা"), "dementia", "neurology"), #
    # FDC ENHANCEMENT
    Drug("Levodopa+Carbidopa", ("Syndopa", "Tidomet", "Sinemet"),
         ("লেভোডোপা", "সিনডোপা", "টিডোমেট", "কার্বিডোপা", "লেভোডোপা কার্বিডোপা"),
         "Parkinson's disease", "neurology"), #
    Drug("Sumatriptan", ("Suminat", "Imitrex"),
         ("সুমাট্রিপটান", "সুমিন্যাট"), "migraine", "neurology"), #
    Drug("Flunarizine", ("Sibelium", "Flunarin"),
         ("ফ্লুনারিজিন", "সিবেলিয়াম"), "migraine prophylaxis", "neurology"), #
    Drug("Amitriptyline", ("Amitone", "Tryptomer"),
         ("অ্যামিট্রিপটাইলিন", "ট্রিপটোমার", "অ্যামিটোন"),
         "neuropathic pain / migraine", "neurology"), #
    # FDC ENHANCEMENT
    Drug("Clopidogrel+Aspirin", ("Clopitab-A", "Deplatt-A", "Ecosprin-AV"),
         ("ক্লোপিটাব", "ইকোস্পিরিন এভি", "ডেপ্ল্যাট এ", "ক্লোপিডোগ্রেল অ্যাসপিরিন", "ক্লোপিডোগ্রেল এবং অ্যাসপিরিন"), "stroke prevention", "neurology"), #
    Drug("Escitalopram", ("Nexito", "Cilentra"),
         ("এস্কিটালোপ্রাম", "নেক্সিটো"), "depression / anxiety", "psychiatry"),
    Drug("Sertraline", ("Daxid", "Zosert"),
         ("সারট্রালিন", "ড্যাক্সিড"), "depression", "psychiatry"),
    Drug("Clonazepam", ("Clonotril", "Lonazep"),
         ("ক্লোনাজিপাম", "ক্লোনোট্রিল"), "anxiety / panic", "psychiatry"),
    Drug("Zolpidem", ("Zolfresh", "Nitrest"),
         ("জোলপিডেম", "জোলফ্রেশ"), "insomnia", "psychiatry"),
    Drug("Olanzapine", ("Oleanz", "Olanex"),
         ("ওলানজাপিন", "ওলিয়েঞ্জ"), "antipsychotic", "psychiatry"),
    Drug("Quetiapine", ("Qutan", "Seroquel"),
         ("কোয়েটিয়াপিন", "কিউটান"), "antipsychotic", "psychiatry"),
]

# ---------------------------------------------------------------------------
# SURGERY
# ---------------------------------------------------------------------------
SURGERY = [
    Drug("Tramadol", ("Tramazac", "Domadol"),
         ("ট্রামাডল", "ট্রামাজ্যাক"), "moderate-severe pain", "surgery"), #
    # FDC ENHANCEMENT
    Drug("Tramadol+Paracetamol", ("Ultracet", "Tramazac Plus"),
         ("ট্রামাডল প্যারাসিটামল", "আলট্রাসেট", "ট্রামাডল এবং প্যারাসিটামল"), "moderate-severe pain", "surgery"),
    Drug("Ceftriaxone", ("Monocef", "Intacef", "Oframax"),
         ("সেফট্রায়াক্সোন", "মনোসেফ"), "injectable antibiotic", "surgery"), #
    Drug("Enoxaparin", ("Clexane", "Lomoh"),
         ("এনোক্সাপারিন", "ক্লেক্সেন"), "DVT prophylaxis", "surgery"), #
    Drug("Povidone iodine", ("Betadine", "Cipladine"),
         ("পোভিডোন আয়োডিন", "বিটাডিন", "সিপ্লাডিন"), "antiseptic", "surgery"), #
]


# ---------------------------------------------------------------------------
# ADDED FROM CLINICAL REVIEW of 102 entries seen in real transcripts.
# Bengali forms are the ones a human wrote beside the English name
# wherever the transcripts supplied one - those are what the ASR must
# match. Reviewed by the clinician; still not pharmacist-verified.
# ---------------------------------------------------------------------------
REVIEWED = [
    Drug("Vitamin C", ("Limcee", "Celin", "Chewcee"),
         ("ভিটামিন সি",),
         "vitamin C supplement", "general"),
    Drug("Evening Primrose Oil", ("Evanova", "EPO"),
         ("ইভনিং প্রিমরোজ অয়েল",),
         "PMS / menopause", "gynaecology"),
    Drug("Biotin", ("Biotin", "Hairbon"),
         ("বায়োটিন",),
         "hair and nail supplement", "dermatology"),
    Drug("Zinc", ("Z&D", "Zinconia"),
         ("জিঙ্ক", "জিঙ্ক সিরাপ"),
         "zinc supplement", "general"),
    Drug("Lycopene", ("Lycostar", "Lycored"),
         ("লাইকোপিন",),
         "antioxidant supplement", "general"),
    Drug("Dexamethasone", ("Decadron", "Dexona"),
         ("ডেক্সামিথাসোন", "ডেক্সামিথাসোন ইনজেকশন"),
         "corticosteroid", "general"),
    Drug("Drotaverine", ("Drotin", "Doverin"),
         ("ড্রোটাবেরিন", "ড্রোটাভেরিন"),
         "antispasmodic", "gastro"),
    Drug("Diltiazem", ("Dilzem", "Angizem"),
         ("ডিলটিয়াজেম",),
         "calcium channel blocker", "cardiac"),
    Drug("Penicillin V", ("Pen-V", "Cilopen"),
         ("পেনিসিলিন ভি",),
         "antibiotic", "general"),
    Drug("Diosmin", ("Daflon", "Venusmin"),
         ("ডায়োসমিন",),
         "venotonic / piles", "surgery"),
    Drug("Methotrexate", ("Folitrax", "Imutrex"),
         ("মেথোট্রেক্সেট",),
         "DMARD / psoriasis", "bone"),
    Drug("Pramipexole", ("Pramipex", "Parkitidin"),
         ("প্রামিপেক্সোল",),
         "Parkinson's / RLS", "neurology"),
    Drug("Piperacillin-Tazobactam", ("Zosyn", "Pipzo"),
         ("পাইপেরাসিলিন-ট্যাজোব্যাকটাম",),
         "IV antibiotic", "surgery"),
    Drug("Iron Sucrose", ("Orofer S", "Encicarb"),
         ("আয়রন সুক্রোজ",),
         "IV iron / anaemia", "general"),
    Drug("Naproxen", ("Naprosyn", "Xenobid"),
         ("ন্যাপ্রোক্সেন",),
         "NSAID", "bone"),
    Drug("Hydroxyzine", ("Atarax", "Hyzine"),
         ("হাইড্রোক্সিজিন",),
         "antihistamine / pruritus", "dermatology"),
    Drug("Cefadroxil", ("Droxyl", "Cefadrox"),
         ("সেফাড্রক্সিল",),
         "antibiotic", "general"),
    Drug("Promethazine", ("Phenergan", "Avomine"),
         ("প্রোমিথাজিন",),
         "antihistamine / antiemetic", "general"),
    Drug("Colchicine", ("Zycolchin", "Goutnil"),
         ("কোলচিসিন",),
         "acute gout", "nephrology"),
    Drug("Albendazole", ("Zentel", "Bandy"),
         ("অ্যালবেনডাজল",),
         "anthelmintic", "general"),
    Drug("Ispaghula Husk", ("Isabgol", "Naturolax"),
         ("ইসবগুল হাস্ক", "ইসবগুল"),
         "bulk laxative", "gastro"),
    Drug("Pheniramine", ("Avil",),
         ("অ্যাভিল ইনজেকশন", "ফেনিরামিন", "অ্যাভিল", "অ্যাভল", "এভিল"),
         "antihistamine injection", "general"),
    Drug("Choline Salicylate gel", ("Zytee", "Orasore"),
         ("কোলিন স্যালিসাইলেট ওরাল জেল",),
         "mouth ulcer gel", "dental"),
    Drug("Tobramycin+Dexamethasone", ("Tobastar DM", "Tobradex"),
         ("টোব্রামাইসিন + ডেক্সামিথাসোন আই অয়েন্টমেন্ট",),
         "eye antibiotic-steroid", "ophthalmology"),
    Drug("Tannic acid+Iodine gum paint", ("Tannic acid gum paint",),
         ("ট্যানিক অ্যাসিড + আয়োডিন গাম পেইন্ট",),
         "gingivitis", "dental"),
    Drug("Conjugated Estrogen cream", ("Premarin",),
         ("কনজুগেটেড ইস্ট্রোজেন ভ্যাজাইনাল ক্রিম",),
         "atrophic vaginitis", "gynaecology"),
    Drug("Griseofulvin", ("Grisovin", "Walavin"),
         ("গ্রাইসিওফুলভিন",),
         "antifungal", "dermatology"),
    Drug("Petroleum Jelly", ("Vaseline",),
         ("পেট্রোলিয়াম জেলি",),
         "emollient", "dermatology"),
    Drug("Clindamycin+Nicotinamide gel", ("Clinsol NA", "Faceclin"),
         ("ক্লিনডামাইসিন + নিকোটিনামাইড জেল",),
         "acne gel", "dermatology"),
    Drug("Povidone iodine ointment", ("Isodine", "Betadine ointment"),
         ("আইসোডিন অয়েন্টমেন্ট",),
         "antiseptic ointment", "surgery"),
    Drug("Potassium Permanganate", ("KMnO4",),
         ("পটাশিয়াম পারম্যাঙ্গানেট",),
         "antiseptic soak", "dermatology"),
    Drug("Lignocaine+Hydrocortisone cream", ("Anobliss", "Proctosedyl"),
         ("লিগনোকেইন + হাইড্রোকর্টিসোন ক্রিম",),
         "piles cream", "surgery"),
    Drug("Fluticasone cream", ("Flutivate",),
         ("ফ্লুটিকাসোন ক্রিম",),
         "topical steroid", "dermatology"),
    Drug("Liquid Paraffin moisturizer", ("Moisturex", "Venusia"),
         ("লিকুইড প্যারাফিন ময়েশ্চারাইজার",),
         "emollient", "dermatology"),
    Drug("Sodium Hyaluronate eye drops", ("Hyalur", "I-Kul"),
         ("সোডিয়াম হাইলুরোনেট",),
         "dry eye", "ophthalmology"),
    Drug("Carbomer eye gel", ("Lubrigel", "Viscotears"),
         ("কার্বোমার আই জেল",),
         "dry eye gel", "ophthalmology"),
    Drug("Potassium Citrate", ("Alkasol", "K-Cit"),
         ("পটাশিয়াম সাইট্রেট + ম্যাগনেশিয়াম সাইট্রেট সিরাপ", "সিরাপ আলকাসল", "আলকালাইজার সিরাপ"),
         "urinary alkaliniser", "urology"),
    Drug("Disodium Hydrogen Citrate", ("Citralka", "Alkacitral"),
         ("ডাইসোডিয়াম হাইড্রোজেন সাইট্রেট সিরাপ",),
         "urinary alkaliniser", "urology"),
]

ALL_DRUGS: list[Drug] = (CARDIAC + ENDOCRINE + RESPIRATORY + GI
                          + GENERAL + UROLOGY + BONE + DERMATOLOGY
                          + OPHTHALMOLOGY + ENT + DENTAL + GYNAECOLOGY
                          + NEPHROLOGY + NEUROLOGY + SURGERY + REVIEWED)

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
    # "ইকুড ইকো" is the ASR on "ECG, Echo" - the two tests are ordered
    # together and it runs them together. Only the Echo half resolved, so
    # the ECG silently vanished from a cardiac work-up.
    "ECG": ("ইসিজি", "ই সি জি", "ইকেজি", "ইকুড", "ইকজি", "ইসিজিটা",
             "electrocardiogram", "e c g"), #
    # ONE entry. There used to be a separate "2D Echo" as well, and since
    # "2d echo" was an alias of both, a single spoken test resolved to TWO
    # entries - 297 of 311 apparent lab false positives in a 500-transcript
    # eval were this one duplication.
    "2D Echo": ("ইকো", "echo", "echocardiogram", "2d echo", "টু ডি ইকো",
                 "two d echo", "ইকোকার্ডিওগ্রাফি", "ইকোকার্ডিওগ্রাম",
                 "echocardiography", "ইকো কার্ডিওগ্রাফি"), #
    "TMT": ("টিএমটি", "টি এম টি", "treadmill test", "stress test", "t m t"), #
    "Angiography": ("অ্যাঞ্জিওগ্রাফি", "angiogram", "অ্যাঞ্জিওগ্রাম",
                     "এনজিওগ্রাম", "এঞ্জিওগ্রাফি"), #
    "Lipid profile": ("লিপিড প্রোফাইল", "cholesterol test", "লিপিড", "লিপিড টেস্ট"), #
    "Troponin": ("ট্রপোনিন", "ট্রপ", "troponin i", "troponin t",
                  "troponin i/t"), #
    # Cardiology-specific tests. Added after 200 real cardiac prescriptions
    # showed only 12/20 labs recognised - the table was built for general
    # OPD and had no Holter, no tilt table, no electrophysiology study.
    "Holter monitor": ("holter", "holter monitor", "হোল্টার",
                        "holter monitor (24-48 hr)", "24 hour holter"), #
    "Tilt table test": ("tilt table", "tilt table test", "টিল্ট টেবিল"), #
    "Electrophysiology study": ("electrophysiology study", "ep study",
                                 "ইপি স্টাডি"), #
    "CK-MB": ("ck-mb", "ck mb", "সিকে এমবি", "creatine kinase"), #
    "BNP": ("bnp", "nt-probnp", "bnp or nt-probnp", "pro bnp"), #
    "Serum electrolytes": ("serum electrolytes", "electrolytes",
                            "ইলেক্ট্রোলাইট", "na k cl", "সিরাম ইলেকট্রোলাইটস"), #
    # neuro
    # "ইজি" is what came back on a seizure consultation - one token, the
    # doubled vowel lost. It was rejected as an unknown MEDICATION.
    "EEG": ("ইইজি", "ই ই জি", "electroencephalogram", "e e g", "ইজি"), #
    "MRI": ("এমআরআই", "এম আর আই", "m r i"), #
    "CT scan": ("সিটি স্ক্যান", "সি টি স্ক্যান", "সিটি"), #
    # metabolic / blood
    "Creatinine": ("ক্রিয়েটিনিন", "creatine", "ক্রিয়েটিন", "সিরাম ক্রিয়েটিনিন"), #
    "Urea": ("ইউরিয়া", "ব্লাড ইউরিয়া"), #
    # Spoken forms, verbatim from real audio:
    #   "ওসিটি ফাস্টিং ব্লাড সুগার পিপি ইসিজি ব্লাড প্রেসার চেক"
    # Note "ফাস্টিং ব্লাড সুগার" - the interposed "ব্লাড" broke the n-gram
    # against the key "ফাস্টিং সুগার", and bare "পিপি" was never an alias.
    # "খাওয়ার পরের" was held but a doctor says "খাওয়ার পরে" / "খাওয়ার পর".
    # Missed on a diabetes consultation where the fasting sugar ordered in
    # the SAME breath was caught - the pair is ordered together, so half a
    # pair is the conspicuous kind of miss.
    "PP sugar": ("পিপি সুগার", "পি পি সুগার", "পোস্ট প্রান্ডিয়াল",
                  "post prandial sugar", "খাওয়ার পরের সুগার", "পিপি",
                  "খাওয়ার পরে সুগার", "খাওয়ার পর সুগার",
                  "খাবার পরে সুগার", "খাওয়ার পরে ব্লাড সুগার"), #
    "Fasting sugar": ("ফাস্টিং সুগার", "FBS", "খালি পেটে সুগার",
                       "এফ বি এস", "ফাস্টিং", "ফাস্টিং ব্লাড সুগার"), #
    # "এইচ বি এ ওয়ান" without the trailing সি is how it comes back when the
    # ASR clips the last syllable. Four tokens is specific enough to be
    # safe - unlike the bare "এইচবি" alias that was REMOVED for matching
    # inside HbA1c itself.
    "HbA1c": ("এইচবিএ১সি", "এইচ বি এ ওয়ান সি", "গ্লাইকোসাইলেটেড হিমোগ্লোবিন",
               "hba1c", "এইচ বি এ ওয়ান", "এইচবিএওয়ানসি"), #
    "TSH": ("টিএসএইচ", "টি এস এইচ", "থাইরয়েড টেস্ট", "thyroid profile", "থাইরয়েড প্রোফাইল", "t3 t4 tsh"), #
    # NOTE: "রক্ত পরীক্ষা" / "ব্লাড টেস্ট" are deliberately NOT CBC aliases.
    # "Blood test" is not necessarily a complete blood count, and turning a
    # generic phrase into a specific named order would be the pipeline
    # inventing a clinical decision. They fall through to "blood test
    # (unspecified)" so the reviewer names it.
    "CBC": ("সিবিসি", "সি বি সি", "complete blood count", "c b c",
             "কমপ্লিট ব্লাড কাউন্ট", "কমপ্লিট ব্লাড"), #
    "LFT": ("এলএফটি", "এল এফ টি", "liver function test", "লিভার ফাংশন টেস্ট"), #
    "KFT": ("কেএফটি", "কে এফ টি", "kidney function test", "RFT",
             "আর এফ টি"), #
    "Urine routine": ("ইউরিন", "প্রস্রাব পরীক্ষা", "urine test",
                       "ইউরিন টেস্ট", "urine r/e", "urine re",
                       "urine routine examination", "ইউরিন রুটিন"), #
    "Urine culture": ("ইউরিন কালচার", "urine culture"),
    "Stool routine": ("স্টুল টেস্ট", "পায়খানা পরীক্ষা", "stool routine"),
    "Blood culture": ("ব্লাড কালচার", "blood culture"),
    "Uric acid": ("ইউরিক অ্যাসিড", "ইউরিক", "সিরাম ইউরিক অ্যাসিড"), #
    # NOT the bare "ভিটামিন ডি" - that is ambiguous between the supplement
    # and the blood test, and collisions() flagged it against the
    # Vitamin D3 drug entry. Only unambiguously test-shaped forms here.
    "Vitamin D": ("ভিটামিন ডি টেস্ট", "25 oh vitamin d", "vitamin d3 level",
                   "ভিটামিন ডি লেভেল"), #
    "Vitamin B12": ("ভিটামিন বি টুয়েলভ টেস্ট", "ভিটামিন বি ১২ লেভেল", "vitamin b12"),
    # Bone / metabolic. A live osteoporosis consultation ordered every one
    # of these and the table knew none of them.
    "Serum calcium": ("সিরাম ক্যালসিয়াম", "serum calcium", "রক্তে ক্যালসিয়াম",
                       "calcium level", "ক্যালসিয়াম টেস্ট"), #
    "Serum phosphorus": ("সিরাম ফসফরাস", "phosphorus", "phosphate",
                          "ফসফরাস", "serum phosphate"), #
    "DEXA scan": ("dexa", "dexa scan", "ডেক্সা", "ডেক্সা স্ক্যান", "কোমরের ডেক্সা", "কমরে ডেক্সে",
                   "কোমরে ডেক্সা",
                   "bone mineral density", "bmd", "বিএমডি", "বোন ডেনসিটি",
                   "bone densitometry"), #
    "X-ray LS spine": ("ls spine", "l s spine", "lumbosacral spine",
                        "এল এস স্পাইন", "কোমরের এক্স রে", "ap view ls spine",
                        "ls spine ap view", "এলএস স্পাইন", "এমআরআই লাম্বোস্যাক্রাল স্পাইন"), #
    "X-ray Knee": ("এক্স রে হাঁটু", "x-ray knee", "xray knee", "হাঁটুর এক্স রে"),
    "PTH": ("pth", "parathyroid hormone", "পিটিএইচ", "প্যারাথাইরয়েড"), #
    "Alkaline phosphatase": ("alp", "alkaline phosphatase",
                              "অ্যালকালাইন ফসফেটেজ"), #
    "X-ray": ("এক্স রে", "এক্সরে", "এক্স-রে"), #
    # Split out from the generic X-ray: it is the commonest film ordered in
    # a chest clinic, and naming the region is the whole point of the
    # order. "চেস টেক্সটে" is not a typo - it is what the ASR returned for
    # "চেস্ট এক্স-রে" on a real consultation, where the doctor was asking
    # after the lungs ("ফুসফুসের কি অবস্থা").
    "Allergy test": ("অ্যালার্জি টেস্ট", "আলার্জি টেস্ট", "allergy test",
                      "অ্যালার্জি প্রোফাইল"), #
    "Chest X-ray": ("চেস্ট এক্স রে", "চেস্ট এক্সরে", "চেস্ট এক্স-রে",
                     "বুকের এক্স রে", "বুকের এক্সরে", "chest x-ray",
                     "chest xray", "cxr", "সি এক্স আর",
                     "চেস টেক্সটে", "চেস্ট টেক্সটে", "চেস এক্স রে"), #
    # "আল্টা সাউন্ড" is "আল্ট্রাসাউন্ড" with the র dropped AND split in
    # two by the ASR. It used to be recovered by the lab near-skeleton
    # relaxation, which is now single-token only: that relaxation bought
    # exactly this one real case and admitted ordinary two-word speech
    # alongside it - "বলো ডাক্তার" became Blood culture, "অবস্থা সেটা"
    # became Visual acuity. A known garble belongs in the table, where it
    # matches this test and nothing else.
    "USG": ("ইউএসজি", "ইউ এস জি", "আল্ট্রাসাউন্ড", "ultrasound",
             "sonography", "আলট্রাসনোগ্রাফি",
             "আল্টা সাউন্ড", "আলটা সাউন্ড", "আল্ট্রা সাউন্ড"), #
    "USG Whole Abdomen": ("ইউএসজি হোল অ্যাবডোমেন", "হোল অ্যাবডোমেন", "usg whole abdomen"),
    "PSA": ("পিএসএ", "পি এস এ", "প্রস্টেট স্পেসিফিক অ্যান্টিজেন"), #
    "CRP": ("সিআরপি", "সি আর পি", "c reactive protein", "crp"),
    "ESR": ("ইএসআর", "ই এস আর", "esr"),
    "Dengue NS1": ("ডেঙ্গু এনএসওয়ান", "dengue ns1", "ns1 antigen", "ডেঙ্গু টেস্ট"),
    "Widal Test": ("উইডাল টেস্ট", "widal", "টাইফয়েড টেস্ট"),
    "Iron Profile": ("আয়রন প্রোফাইল", "iron profile", "ferritin", "ফেরিটিন"),
    "Lipase": ("লাইপেজ", "lipase", "amylase", "অ্যামাইলেজ"),
    "Hb Electrophoresis": ("হিমোগ্লোবিন ইলেক্ট্রোফোরেসিস", "hb electrophoresis"),
    # --- department-specific investigations ---------------------------
    # dermatology
    "KOH mount": ("koh mount", "koh", "কেওএইচ", "skin scraping"), #
    "Skin biopsy": ("skin biopsy", "ত্বকের বায়োপসি"), #
    "Patch test": ("patch test", "প্যাচ টেস্ট", "allergy patch test"), #
    # ophthalmology
    "Fundus examination": ("fundus", "ফান্ডাস", "fundoscopy", "retina check"), #
    "Intraocular pressure": ("iop", "tonometry", "চোখের প্রেশার",
                              "eye pressure"), #
    "Visual acuity": ("visual acuity", "দৃষ্টিশক্তি পরীক্ষা", "vision test"), #
    "OCT": ("oct", "optical coherence tomography", "ওসিটি", "ও সি টি"), #
    "Biometry": ("biometry", "বায়োমেট্রি", "বায়োমিট্রিক", "বায়োমেট্রিক",
                  "a-scan", "iol power", "আইওএল পাওয়ার"), #
    "Viral markers": ("viral marker", "viral markers", "ভাইরাল মার্কার",
                       "hiv", "এইচআইভি", "এইচ আই ভি", "hbsag", "এইচবিএসএজি",
                       "anti hcv", "hiv hbsag hcv"), #
    "Refraction": ("refraction", "power test", "চশমার পাওয়ার", "রিফ্র্যাকশন টেস্ট"), #
    # ENT
    "Audiometry": ("audiometry", "pta", "pure tone audiometry",
                    "অডিওমেট্রি", "কানের পরীক্ষা", "পিওর টোন অডিওমেট্রি"), #
    "Tympanometry": ("tympanometry", "টিমপ্যানোমেট্রি"), #
    "Nasal endoscopy": ("nasal endoscopy", "dnc", "নাকের এন্ডোস্কোপি"), #
    # dental
    "OPG": ("opg", "orthopantomogram", "ওপিজি", "dental x-ray"), #
    "IOPA": ("iopa", "intraoral periapical", "আইওপিএ"), #
    # gynaecology
    "USG pelvis": ("usg pelvis", "pelvic ultrasound", "তলপেটের আল্ট্রাসাউন্ড",
                    "tvs", "transvaginal scan", "টিভিএস"), #
    "Pap smear": ("pap smear", "pap test", "প্যাপ স্মিয়ার"), #
    "Beta hCG": ("beta hcg", "bhcg", "pregnancy test", "প্রেগন্যান্সি টেস্ট"), #
    "Mammography": ("mammography", "mammogram", "ম্যামোগ্রাফি"), #
    "Prolactin": ("প্রোল্যাকটিন", "prolactin", "সিরাম প্রোল্যাকটিন"),
    # nephrology
    "eGFR": ("egfr", "gfr", "creatinine clearance", "জিএফআর"), #
    "Urine ACR": ("urine acr", "albumin creatinine ratio", "microalbumin",
                   "মাইক্রোঅ্যালবুমিন"), #
    "USG KUB": ("কিডনিতে আল্ট্রাসাউন্ড", "কিডনির আল্ট্রাসাউন্ড",
                 "কিডনি আল্ট্রাসাউন্ড", "কিডনির ইউএসজি", "kidney ultrasound",
                 "কিডনিতে ইউএসজি", "usg kub", "kub", "kidney ultrasound", "কিডনি আল্ট্রাসাউন্ড", "ইউএসজি কেইউবি"), #
    # neurology
    "CT brain": ("ct brain", "ct head", "সিটি ব্রেন", "brain ct"), #
    # "এম আর আই করতে হবে ব্রেনের" - the region IS the order, and a bare
    # "MRI" does not tell radiology what to scan.
    "MRI brain": ("mri brain", "এমআরআই ব্রেন", "brain mri",
                   "এম আর আই ব্রেন", "ব্রেনের এমআরআই", "ব্রেনের এম আর আই",
                   "মাথার এমআরআই", "এম আর আই মাথার", "ব্রেনের এম আর আই টা"), #
    "NCV": ("ncv", "nerve conduction", "এনসিভি"), #
    "EMG": ("emg", "electromyography", "ইএমজি"), #
    "Carotid doppler": ("carotid doppler", "ক্যারোটিড ডপলার"), #
    # --- added from clinical review of real transcripts ---
    "Serum IgE": ("serum total ige", "total ige", "সিরাম টোটাল আইজিই", "আইজিই"),
    # NOT the bare "এইচবি". It is only 4 folded characters, and the gapped
    # matcher then joins "এইচ [ওয়ান] বি" inside a spoken HbA1c and resolves
    # it to Haemoglobin - caught by the regression suite immediately.
    "Haemoglobin": ("haemoglobin", "hemoglobin", "হিমোগ্লোবিন",
                     "হিমোগ্লোবিন লেভেল"),
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
    "High vaginal swab": ("high vaginal swab", "hvs", "hvs culture", "হাই ভ্যাজাইনাল সোয়াব", "high vaginal swab for wet mount"),
    "Ulcer swab culture": ("ulcer swab", "ulcer swab for culture", "আলসার সোয়াব"),
    "FNAC": ("fnac", "fnac of breast lump", "এফএনএসি"),
    "Tear film breakup time": ("tbut", "tbut test", "tear film breakup time", "টিয়ার ফিল্ম ব্রেকআপ টাইম"),
    "Schirmer's test": ("schirmer", "schirmers test", "schirmer's test", "শির্মের্স টেস্ট"),
    "Wood's lamp examination": ("wood's lamp", "woods lamp examination", "উডস ল্যাম্প এক্সামিনেশন"),
    "Lacrimal sac syringing": ("lacrimal sac syringing", "syringing", "ল্যাক্রিমাল স্যাক সিরিঞ্জিং"),
    # Generic orders - a test WAS ordered even if unnamed. Surfaced so the
    # reviewer names it, rather than dropped silently or guessed at.
    "Malaria test": ("malaria test", "ম্যালেরিয়া", "ম্যালেরিয়া পরীক্ষা", "malaria"),
    "blood test (unspecified)": ("রক্ত পরীক্ষা", "ব্লাড টেস্ট", "রক্ত টেস্ট"), #
    "test (unspecified)": ("পরীক্ষা করাতে", "টেস্ট দিচ্ছি", "টেস্ট করবেন",
                            "পরীক্ষা করতে", "পরীক্ষা করতে হবে"), #
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
                      "হার্ট অ্যাটাক", "হার্ট এটাক", "মায়োকার্ডিয়াল ইনফার্কশন"), #
    "heart failure": ("হার্ট ফেইলিওর", "হার্ট ফেলুআর", "হৃদযন্ত্রের অক্ষমতা"), #
    "ischemia": ("ইস্কিমিয়া", "ইস্কামিয়া", "ইসকিমিয়া"), #
    "diabetes": ("ডায়াবেটিস", "ডায়াবেটিজ", "ডায়াবিটিস", "মধুমেহ"), #
    "stent": ("স্টেন্ট", "স্ট্যান্ট", "রিং পরানো"), #
    "angioplasty": ("অ্যাঞ্জিওপ্লাস্টি", "এনজিওপ্লাস্টি"), #
    "angina": ("অ্যাঞ্জাইনা", "অঞ্জিনা", "অন্জিনা", "এনজাইনা",
                "অ্যানজাইনা পেক্টোরিস"), #
    "chest pain": ("বুকে ব্যথা", "বুক ব্যথা", "চেস্ট পেইন", "বুক চেপে ধরা"), #
    # A bare "blockage" names no organ. The doctor said "হার্টের সমস্যার
    # দিকে নির্দেশ করছে ... ব্লকেজ থাকতে পারে" - the site is right there in
    # the sentence, and the same rule that turned "infection" into "stomach
    # infection" applies: the specific entry wins the longer span.
    "heart blockage": ("হার্টের ব্লকেজ", "হার্টে ব্লকেজ", "হৃদযন্ত্রে ব্লকেজ",
                        "হার্টের সমস্যা", "হার্টের সমস্যার",
                        "করোনারি ব্লকেজ", "heart blockage"), #
    "blockage": ("ব্লকেজ", "ব্লক"), #
    # Spellings from a real consultation: "ধরফড়" (r, not ড়) appeared and
    # did not match. Chest pain with sweating, palpitations and
    # breathlessness is the textbook angina presentation - missing three of
    # the four cardinal signs is not a cosmetic gap.
    "palpitations": ("ধড়ফড়", "ধরফড়", "হার্ট বিট", "বুক ধড়ফড়", "বুক ধুকপুক"),
    "sweating": ("ঘাম", "ঘাম হয়", "দরদর করে ঘাম", "ঘেমে যাওয়া", "প্রচুর ঘাম"),
    "chest tightness": ("বুকে চাপ", "বুকটা চেপে", "বুকে ভার", "বুক ভার"),
    "pain radiating to arm": ("হাত দিয়ে নামে", "বাঁ হাতে ব্যথা", "বাঁদিক দিয়ে নামে"),
    "neck pain": ("ঘাড়ে ব্যথা", "ঘাড়ের দিকে ব্যথা", "ঘাড় ব্যথা"), #
    "breathlessness": ("শ্বাসকষ্ট", "দম বন্ধ", "হাঁপ ধরা", "হাফ ধরে",
                        "হাঁপ ধরে", "দম ফুরিয়ে", "শ্বাস নিতে কষ্ট"), #
    "cholesterol": ("কোলেস্টেরল", "কলেস্টেরল"), #
    "HDL": ("এইচডিএল",), #
    "LDL": ("এলডিএল",), #
    "blood pressure": ("প্রেশার", "ব্লাড প্রেশার", "রক্তচাপ", "প্রেসার"), #
    # metabolic
    "blood sugar": ("সুগার", "রক্তে চিনি", "ব্লাড সুগার"), #
    "thyroid": ("থাইরয়েড",), #
    # general symptoms
    "fever": ("জ্বর",), #
    "cough": ("কাশি",), #
    "headache": ("মাথা ব্যথা", "মাথাব্যথা", "চাপ ধরা মাথা ব্যথা"), #
    "abdominal pain": ("পেট ব্যথা", "পেটে ব্যথা"),
    # তলপেট is specifically the LOWER abdomen, and on a gynaecological
    # consultation that is the presenting complaint, not a location detail.
    # MOVED off "abdominal pain" rather than duplicated - two entries
    # claiming one key collide and neither matches.
    #
    # The spoken line was "তলপেটটা খুব ব্যাথা করছে": the টা suffix and an
    # interposed খুব defeated the তলপেটে-anchored key, so the variants are
    # listed rather than relying on the fold to bridge them.
    "lower abdominal pain": ("তলপেটে ব্যথা", "তলপেটে ব্যাথা",
                              "তলপেট ব্যথা", "তলপেটটা ব্যাথা",
                              "তলপেটটা ব্যথা", "তলপেটের ব্যথা",
                              "তলপেটে যন্ত্রণা"),
    # The symptom the doctor REASONED FROM - "এটা তো মেনোপস ... এর লক্ষণ" -
    # and it was not in the vocabulary at all. Anchored to multi-token
    # phrases: গরম alone is ordinary Bengali for hot weather or hot water.
    "hot flushes": ("গরম লাগে হঠাৎ", "হঠাৎ গরম লাগা", "হঠাৎ হঠাৎ গরম",
                     "খুব গরম লাগে", "শরীর গরম হয়ে যাওয়া", "হট ফ্লাশ",
                     "হট ফ্লাশেস"), #
    "loose stools": ("পাতলা পায়খানা", "ডায়রিয়া", "পায়খানা", "diarrhea"), #
    "constipation": ("কোষ্ঠকাঠিন্য", "পায়খানা শক্ত", "constipation"),
    "vomiting": ("বমি",), #
    "nausea": ("গা গোলানো", "বমি ভাব"), #
    "body ache": ("শরীর ব্যথা", "গা ব্যথা", "গা হাত-পা ম্যাজম্যাজ", "হাড়ে ব্যথা"), #
    # An orthopaedic consultation about back and waist pain returned NO
    # symptoms at all - neither phrase existed. "টনটন" (a dull throb) is
    # anchored to a region, not left bare, which would be too generic.
    # "গায়ে কাটা দিয়ে জ্বর" - shivering, which the কাটা key was reading as
    # a wound. The phrase is the symptom.
    "chills": ("গায়ে কাটা দিয়ে", "শীত শীত", "কাঁপুনি দিয়ে জ্বর",
                "গা কাঁপছে"), #
    "gum sore": ("মাড়িতে ঘা", "মাড়ির কাছে ঘা", "মাড়ি কেটে",
                  "মাড়িতে কেটে ঘা"), #
    "breast swelling": ("বুক ফুলে", "বুকটা ফুলে", "স্তন ফুলে",
                         "বুকটা ফুলে পাথর", "বুক শক্ত"), #
    "throat swelling": ("গলা ফুলে", "গলা ফুলেছে", "গলা টোট ফুলে"), #
    "back pain": ("পিঠে ব্যথা", "পিঠের ব্যথা", "পিঠ ব্যথা", "পিঠে ব্যাথা",
                   "পিঠের দিকটা টনটন", "পিঠ টনটন", "পিঠে টনটন"), #
    "low back pain": ("কোমরে ব্যথা", "কোমরের ব্যথা", "কোমর ব্যথা",
                       "কোমরে ব্যাথা"), #
    # A child's seizure described exactly as a parent describes it.
    # খিঁচুনি is NOT listed here - it already belongs to "seizure", and
    # the fold drops the chandrabindu so খিচুনি/খিঁচুনি are one key. Claiming
    # it here made both entries collide and neither matched.
    "convulsions": ("হাত পা ছোঁড়া", "হাত পা ছুঁড়ে", "হাত পা ছোঁড়াছুঁড়ি"), #
    "frothing at the mouth": ("মুখ দিয়ে গেঁজা", "মুখে ফেনা", "গেঁজা বেরোনো"), #
    # These three sat in CONDITIONS with NO alias entry, so they could
    # never be matched and never became a diagnosis. Hypertension is the
    # commonest chronic diagnosis there is.
    # NOT "হাড় ক্ষয়" - that is already "bone loss". A finding and a
    # diagnosis can share the words, and whoever holds the key wins; the
    # diagnosis is carried by the phrases the doctor actually used.
    "osteoporosis": ("হার ক্ষয়", "হাড়ের ক্ষয়", "হাড় ভঙ্গুর",
                      "হারগুলো ভঙ্গুর", "হাড়গুলো ভঙ্গুর", "হাড় দুর্বল",
                      "অস্টিওপোরোসিস", "হাড় ক্ষয়ে যাওয়া"), #
    "hypertension": ("উচ্চ রক্তচাপ", "হাই ব্লাড প্রেশার", "হাইপারটেনশন",
                      "প্রেশার বেশি", "রক্তচাপ বেশি", "হাই প্রেশার",
                      "ব্লাড প্রেশার বেশি"), #
    "dementia": ("ডিমেনশিয়া", "স্মৃতিভ্রংশ"), #
    # NOT "মৃগী" alone - that is already "seizure", which is itself a
    # diagnosable condition, so the consultation still gets a diagnosis.
    "epilepsy": ("মৃগী রোগ", "এপিলেপসি", "মৃগি রোগ"), #
    "sore throat": ("গলা ব্যথা", "গলা খুসখুস", "ঢোঁক গিলতে ব্যথা"), #
    "swelling": ("ফোলা", "ফুলে যাওয়া", "সোয়েলিং"), #
    "weakness": ("দুর্বলতা", "দুর্বল"), #
    "anxiety": ("দুশ্চিন্তা", "উদ্বেগ", "anxiety"),
    "insomnia": ("অনিদ্রা", "ঘুম না হওয়া", "insomnia"),
    # English symptom phrases the SLM proposes verbatim. Present because
    # fuzzy matching scored them against real drugs once the folded forms
    # got close - "hair loss" folds to "hairlos" and scored 0.714 against
    # Levothyroxine, over the 0.65 floor, so it was being offered as a
    # PROBABLE medication. Naming them positively is more robust than
    # raising the floor, which would start losing real garbled drug names.
    "hair loss": ("চুল পড়া", "চুল পড়ে যাওয়া"), #
    # Verb INFLECTION, not a new word: the entry held the noun form
    # ("ওজন কমে যাওয়া") but a patient says it in the continuous
    # ("ওজন কমে যাচ্ছে" - it IS going down), and the fold does not bridge
    # যাওয়া/যাচ্ছে. Real miss on a diabetes consultation.
    "weight loss": ("ওজন কমা", "ওজন কমে যাওয়া", "ওজন কমে যাচ্ছে",
                    "ওজন কমছে", "ওজন কমে গেছে"), #
    "weight gain": ("ওজন বাড়া", "ওজন বেড়ে যাওয়া"), #
    "bone loss": ("হাড় ক্ষয়", "বোন লস"), #
    "memory loss": ("স্মৃতিশক্তি কমে যাওয়া", "ভুলে যাওয়া"), #
    "loss of appetite": ("খিদে কমে যাওয়া", "খাওয়ার ইচ্ছে নেই", "অরুচি", "খাবারে অরুচি"), # #
    "loss of consciousness": ("জ্ঞান হারানো", "অজ্ঞান"), #
    "blurred vision": ("ঝাপসা দেখা", "ঘোলাটে দৃষ্টি"), #
    "dizziness": ("মাথা ঘোরা", "মাথা ঘুরছে"), #
    "difficulty swallowing": ("গিলতে কষ্ট",), #
    "phlegm": ("কফ", "শ্লেষ্মা", "সাদা কফ"), #
    "infection": ("ইনফেকশন", "সংক্রমণ"), #
    # "আপনার পেটে হয়তো ইনফেকশন হয়েছে" - the site is the diagnosis. A bare
    # "infection" tells a reader nothing about what was found.
    # "পোতসাবের আর সাথে ইনফেকশন" plus burning and frequency - the doctor
    # names the site, and "infection" alone loses it.
    "inner ear balance disorder": ("কানের ব্যালেন্স নষ্ট",
                                    "কানের ভারসাম্য নষ্ট",
                                    "ব্যালেন্স নষ্ট হয়ে গেছে",
                                    "ব্যালেন্স নষ্ট"), #
    # An anaphylactic food reaction. The doctor names it outright -
    # "আপনার এই সিভিয়ার আলার্জি ট্রিকার করেছে" - and আমবাত is the Bengali
    # for the hives it presents with.
    "severe allergy": ("সিভিয়ার আলার্জি", "সিভিয়ার অ্যালার্জি",
                        "মারাত্মক অ্যালার্জি", "severe allergy"), #
    # "বুকে ইনফেকশন হয়েছে একটা ইনফেকশন হয়েছে মাস্টিটাইটিস" - stated by name.
    # Named outright - "এটাকে আমরা লাইপোমা বলি" - and absent from the
    # gazetteer, so the diagnosis rested on the model alone.
    # "ডেভিয়েটেড নেজাল সেপ্টেম্বলি আমরা ডাক্তারি ভাষায় ডি এন এস" - the
    # doctor gives it in full AND as the abbreviation, and neither existed.
    "deviated nasal septum": ("ডেভিয়েটেড নেজাল সেপ্টাম", "ডি এন এস",
                               "ডিএনএস", "নেজাল সেপ্টাম বেঁকে",
                               "নাকের পর্দা বেঁকে", "ডেভিয়েটেড নেজাল সেপ্টেম",
                               "deviated nasal septum"), #
    # "ট্রমাটিক একটা আলসার হয়ে গেছে" - a denture edge rubbing the gum.
    # Distinct from an aphthous mouth ulcer: the cause is the appliance,
    # which is what the treatment addresses.
    "traumatic ulcer": ("ট্রমাটিক আলসার", "ট্রম্যাটিক আলসার",
                         "ট্রমাটিক ঘা", "traumatic ulcer"), #
    # "আপনার কার্পেল টানেল সিন্ড্রোম হয়েছে" - stated outright, and absent
    # from the gazetteer, so the consultation came back with NO diagnosis
    # at all. That is not cosmetic: department_for() returns "" without a
    # diagnosis, and the specialty guard added in server._merge_segments
    # is disabled the moment it has no department to compare against. This
    # consultation therefore kept Clobetasol - a DERMATOLOGY steroid -
    # matched from "টেলভে", the word "twelve" out of "ভিটামিন ব টেলভে".
    # A missing diagnosis silently switches off the drug protection.
    #
    # "টানেল" alone is deliberately NOT an alias - it is an ordinary word.
    "carpal tunnel syndrome": ("কার্পেল টানেল সিন্ড্রোম",
                                "কার্পাল টানেল সিন্ড্রোম",
                                "কার্পেল টানেল সিনড্রোম",
                                "কার্পেল টানেল", "কার্পাল টানেল",
                                "carpal tunnel syndrome",
                                "carpal tunnel"), #
    "lipoma": ("লাইপোমা", "লিপোমা", "চর্বির টিউমার", "lipoma"), #
    "mastitis": ("মাস্টিটাইটিস", "স্তনপ্রদাহ", "স্তনে ইনফেকশন",
                  "বুকে ইনফেকশন", "mastitis"), #
    "breast abscess": ("স্তনে অ্যাবসেস", "অ্যাপসেস জমে", "breast abscess"), #
    "urine infection": ("প্রস্রাবে ইনফেকশন", "প্রস্রাবের ইনফেকশন",
                         "মূত্রনালীর সংক্রমণ", "ইউরিন ইনফেকশন",
                         "পেচ্ছাপে ইনফেকশন", "urine infection", "uti"), #
    # "আপনার পোস্টার টা বড় হয়ে থাকতে পারে" - the prostate is enlarged.
    # It is why the PSA was ordered.
    "enlarged prostate": ("প্রস্টেট বড়", "প্রোস্টেট বড়", "প্রস্টেট বড় হয়েছে",
                           "পোস্টার টা বড়", "প্রস্টেট বেড়ে গেছে",
                           "enlarged prostate", "bph"), #
    # "খাদ্য নালীতে ইনফেকশন" - infection in the alimentary canal. The
    # doctor's own words on a food poisoning consultation, and with only
    # the পেট ("stomach") spellings registered, the generic "infection"
    # was all that matched. A named site is what makes it a diagnosis.
    "stomach infection": ("পেটে ইনফেকশন", "পেটের ইনফেকশন",
                           "পেটে সংক্রমণ", "গ্যাস্ট্রোএন্টেরাইটিস",
                           "পেটে ইনফেকশন হয়েছে",
                           "খাদ্য নালীতে ইনফেকশন", "খাদ্যনালীতে ইনফেকশন",
                           "খাদ্য নালীর ইনফেকশন", "পাকস্থলীতে ইনফেকশন"), #
    # "ফুড পয়সা নেই" is the ASR on "ফুড পয়জনিং" - it hears the English
    # loanword as পয়সা ("money"). Registering the garble matters as much as
    # the correct spelling: the doctor named the diagnosis out loud and
    # nothing in the tables could recognise what came back.
    #
    # "ফুড পয়সা নেই" is registered WITH the নেই, and that needs saying.
    # Negation suppression would otherwise read it as "no food poisoning"
    # and drop the diagnosis the doctor had just made. The নেই is not a
    # negation here - it is the tail of the ASR's attempt at "পয়জনিং".
    # Registering the whole garble makes the longest span win, so the token
    # is consumed as part of the term instead of modifying it.
    #
    # A REAL denial is unaffected: "ফুড পয়জনিং নেই" spells the word
    # correctly, matches the shorter alias, and stays suppressed. Only the
    # garble - where পয়সা means "money" and belongs to no denial anyone
    # would utter - is treated this way.
    "food poisoning": ("ফুড পয়জনিং", "ফুড পয়জন", "ফুড পয়সা",
                        "ফুড পয়সা নেই",
                        "খাদ্যে বিষক্রিয়া", "বিষাক্ত খাবার",
                        "food poisoning"), #
    # --- department-specific symptoms and findings --------------------
    # dermatology
    "itching": ("চুলকানি", "চুলকায়", "খুজলি", "ইচিং"), #
    "rash": ("র‍্যাশ", "ফুসকুড়ি", "চাকা চাকা দাগ"), #
    # The bare ব্রণ folds onto বরন, which ভ্রণ ("embryo") also lands on -
    # that key stays blocked, see _AMBIGUOUS_WITH_COMMON_WORD. But the
    # INFLECTED forms fold to distinct keys (বরনো, বরোনো, বরনের) and carry
    # no such ambiguity, so they are added and the acne diagnosis becomes
    # reachable again. Counted over the 16 consultations: three acne
    # tokens to one embryo.
    "acne": ("ব্রণ", "একনি", "ব্রণো", "ব্রোনো", "ব্রণের", "ব্রণো কমানো",
              "একমি", "একমির", "ব্রনো", "ব্রণগুলো"), #
    # আমবাদ is the ASR's spelling; চাকা is how a patient describes the
    # wheals ("সারা গায়ে লাল লাল চাকা হয়ে ফুলে গেছে").
    "hives": ("আমবাত", "আর্টিকেরিয়া", "আমবাদ", "লাল লাল চাকা",
               "চাকা হয়ে ফুলে", "চাকা চাকা"), #
    # "ইনফেকশান" is how the ASR writes the loanword about as often as
    # "ইনফেকশন", and only the second spelling was registered - so the
    # doctor's own words, "এক প্রকার ফাঙ্গাল ইনফেকশান", matched nothing.
    "fungal infection": ("দাদ", "ছত্রাক", "ফাঙ্গাল ইনফেকশন",
                          "ফাঙ্গাল ইনফেকশান", "ফাংগাল ইনফেকশন",
                          "ছত্রাক সংক্রমণ", "fungal infection"), #
    # ছুলি is what patients and doctors call it; পিটাইরিয়াসিস ভার্সিকালার
    # is the name written on the prescription. The ASR renders the latter
    # as "পিটাইরিয়াস শিশ ওয়াশ কালার ভ্যাসিকালার" - it loses the word
    # boundaries entirely - so the spoken form is the one that has to
    # carry it, and the garble is registered alongside.
    "pityriasis versicolor": ("ছুলি", "সাদা ছোপ", "পিটাইরিয়াসিস ভার্সিকালার",
                               "পিটাইরিয়াসিস ভার্সিকলার",
                               "পিটাইরিয়াস ভ্যাসিকালার",
                               "কালার ভ্যাসিকালার",
                               "pityriasis versicolor"), #
    "hair fall": ("চুল উঠছে",), #
    "dry skin": ("শুষ্ক ত্বক", "চামড়া শুকিয়ে"), #
    "boil": ("ফোঁড়া", "বিচি"), #
    "corn": ("কড়া পড়া", "কর্ন"),
    # ophthalmology
    "blurred vision far": ("দূরে ঝাপসা", "দূরের জিনিস দেখতে"), #
    "eye pain": ("চোখে ব্যথা", "চোখ ব্যথা"), #
    "watering eyes": ("চোখ দিয়ে জল", "চোখে জল পড়া"), #
    "red eye": ("চোখ লাল", "লাল চোখ"), #
    "cataract": ("ছানি", "ক্যাটারাক্ট"), #
    "glaucoma": ("গ্লুকোমা", "চোখের প্রেশার বেশি"), #
    # ENT
    "ear pain": ("কানে ব্যথা", "কান ব্যথা"), #
    "hearing loss": ("কানে শুনতে অসুবিধা", "কম শুনছি", "শ্রবণশক্তি কমে", "হিয়ারিং লস"), #
    "tinnitus": ("কানে শব্দ", "কানে ভোঁ ভোঁ"), #
    # Vertigo is the DIAGNOSIS in an ENT clinic, not merely a complaint -
    # "আপনার ভার্টিকও হয়েছে" states it. ভার্টিকও is the ASR's rendering.
    "vertigo": ("মাথা ঘোরা ভার্টিগো", "ভার্টিগো", "সবকিছু ঘুরছে",
                 "ভার্টিকও", "ভারটিগো"), #
    "nasal block": ("নাক বন্ধ", "নাক দিয়ে শ্বাস"), #
    "runny nose": ("নাক দিয়ে জল", "সর্দি", "হলদে সর্দি"), #
    "tonsillitis": ("টনসিল", "টনসিলাইটিস"), #
    # dental
    "toothache": ("দাঁতে ব্যথা", "দাঁত ব্যথা", "টুথএক"), #
    "bleeding gums": ("মাড়ি থেকে রক্ত", "মাড়িতে রক্ত", "গাম ব্লিডিং"), #
    "mouth ulcer": ("মুখে ঘা", "মুখের ঘা", "অ্যাপথাস আলসার"), #
    "swollen gums": ("মাড়ি ফোলা", "মাড়ি ফুলে"), #
    # gynaecology
    "irregular periods": ("অনিয়মিত পিরিয়ড", "মাসিক অনিয়মিত", "ইরেগুলার পিরিয়ড"), #
    "heavy bleeding": ("বেশি রক্তপাত", "অতিরিক্ত রক্তক্ষরণ", "হেভি ব্লিডিং"), #
    "white discharge": ("সাদা স্রাব", "লিউকোরিয়া"), #
    "menopause": ("মেনোপজ", "মাসিক বন্ধ"), #
    "pregnancy": ("গর্ভাবস্থা", "প্রেগন্যান্ট", "অন্তঃসত্ত্বা"), # #
    # nephrology
    "reduced urine": ("প্রস্রাব কম", "কম প্রস্রাব", "ইউরিন আউটপুট কমে যাওয়া"), #
    "burning urination": ("প্রস্রাবে জ্বালা", "জ্বালাপোড়া"), #
    # The other two thirds of the classic diabetic triad. Only "weakness"
    # was being reported from a consultation that stated all three, and
    # polyuria and polydipsia are what make the picture diabetic.
    # Anchored on the short verb phrase, not the whole sentence. The line
    # spoken was "রাতে দু তিনবার কেন চারবার বাথরুম ছুটতে হয়" - four tokens
    # between রাতে and বাথরুম, where gapped matching allows only one.
    "frequent urination": ("বারবার প্রস্রাব", "ঘন ঘন প্রস্রাব",
                            "বারবার বাথরুম", "রাতে বারবার বাথরুম",
                            "বাথরুম ছুটতে", "বাথরুম যেতে হয়",
                            "বারবার প্রস্রাব পায়", "ঘন ঘন বাথরুম"), #
    # "শুকিয়ে কাঠ" (dried to wood) is the idiom that carries the meaning
    # and survives the words around it - the spoken line was
    # "গলাটাও কেমন শুকিয়ে কাঠ হয়ে থাকে", where an interposed কেমন and the
    # ও suffix defeat a গলা-anchored key.
    "excessive thirst": ("খুব তেষ্টা", "বেশি তেষ্টা", "তেষ্টা পায়",
                          "গলা শুকিয়ে যাওয়া", "গলা শুকিয়ে কাঠ",
                          "গলাটা শুকিয়ে কাঠ", "জল তেষ্টা", "মুখ শুকিয়ে যাওয়া",
                          "শুকিয়ে কাঠ", "গলা শুকিয়ে"), #
    "facial puffiness": ("মুখ ফোলা", "চোখ মুখ ফোলা", "পেরিঅরবিটাল এডিমা"), #
    "kidney failure": ("কিডনি ফেইলিওর", "কিডনি খারাপ"), #
    "dialysis": ("ডায়ালিসিস",), #
    # neurology
    # "ফিট" folds to পিট, and so does "পিঠ" - the BACK. fold() maps ফ->প
    # and ঠ->ট, so an ordinary body part became a neurological diagnosis:
    # a dermatology consultation about spots on the back and chest came
    # back diagnosed as seizure, from the word পিঠ alone. The bare key is
    # blocked in _AMBIGUOUS_WITH_COMMON_WORD; the inflected forms below
    # fold distinctly and carry the verb that makes them a finding.
    "seizure": ("খিঁচুনি", "ফিট", "মৃগী", "ফিট হয়ে", "ফিট হয়েছে",
                 "ফিট হয়ে গেছে", "ফিট খেয়েছে", "খিঁচুনি হয়েছে"), #
    "stroke": ("স্ট্রোক", "প্যারালাইসিস", "পক্ষাঘাত"), #
    "numbness": ("অবশ", "ঝিনঝিন", "অসাড়"), #
    "tremor": ("কাঁপুনি", "হাত কাঁপে", "রেস্টিং ট্রেমর"), #
    "migraine": ("মাইগ্রেন",), #
    "memory problem": ("ভুলে যাচ্ছি", "স্মৃতি সমস্যা"), #
    "weakness one side": ("একদিক অবশ", "এক পাশ দুর্বল"), #
    # surgery
    "lump": ("চাকা", "গোটা", "টিউমার"), #
    "hernia": ("হার্নিয়া",), #
    "piles": ("পাইলস", "অর্শ", "হেমোরয়েডস"), #
    "gallstone": ("পিত্তথলির পাথর", "গলব্লাডার স্টোন", "কোলিথিওসিস"), #
    "appendicitis": ("অ্যাপেন্ডিক্স", "অ্যাপেন্ডিসাইটিস"), #
    "wound": ("ক্ষত", "ঘা", "কাটা"), #
    # respiratory
    "asthma": ("হাঁপানি", "অ্যাজমা", "ব্রঙ্কিয়াল অ্যাজমা"),
    "wheezing": ("শোঁ শোঁ আওয়াজ", "wheezing"),
    # The doctor STATED this one - "আপনার ডাস্ট আলার্জি তাহলে বুঝতে পারবো" -
    # and the consultation still came back with no diagnosis at all,
    # because only "allergies" existed and that is a symptom entry, not a
    # diagnosable condition.
    "dust allergy": ("ডাস্ট অ্যালার্জি", "ডাস্ট আলার্জি", "ধুলোর অ্যালার্জি",
                      "ধুলো অ্যালার্জি", "ধুলোয় অ্যালার্জি", "dust allergy"),
    "allergic rhinitis": ("অ্যালার্জিক রাইনাইটিস", "নাকের অ্যালার্জি",
                           "allergic rhinitis"),
    # orthopedics
    "arthritis": ("বাতের ব্যথা", "গাঁটে ব্যথা", "অস্টিওআর্থারাইটিস"),
    "joint pain": ("জয়েন্টে ব্যথা", "হাঁটুতে ব্যথা"),
    "muscle cramp": ("শিরায় টান", "পেশিতে টান"),
    # Doctors say the English word, in Bengali script, on sports injuries.
    # Registering it is not only about naming the diagnosis: until it was
    # here, স্ট্রেন was not recognised as ANYTHING, and its consonant
    # skeleton "strn" is shared with Isotroin, an Isotretinoin brand - so
    # a torn muscle was prescribed an acne drug. A term the tables can
    # name positively can no longer be guessed at. See gate.py step 2b.
    "muscle strain": ("স্ট্রেন", "মাসল স্ট্রেন", "পেশিতে টান লেগেছে",
                       "মাংসপেশিতে টান", "muscle strain", "strain"),
    "sprain": ("মচকে গেছে", "মচকানো", "sprain"),
    # --- DRUG CLASSES, not drugs -------------------------------------
    # A doctor says "I'll give you an antibiotic" without naming one. These
    # must never resolve to a specific product.
    #
    # Not hypothetical: adding "Candibiotic" (an ear drop) to the gazetteer
    # made the bare word "Antibiotic" fuzzy-match it at 0.86, so a generic
    # statement became a specific ear medication. Naming the classes
    # positively stops that for every future entry too, because the class
    # check runs before the brand table and before fuzzy.
    "antibiotic": ("অ্যান্টিবায়োটিক", "antibiotics", "এন্টিবায়োটিক"), #
    "painkiller": ("পেইনকিলার", "ব্যথার ওষুধ", "analgesic", "pain killer"), #
    "antacid": ("অ্যান্টাসিড", "গ্যাসের ওষুধ"), #
    "steroid": ("স্টেরয়েড", "steroids"), #
    "vitamin supplement": ("ভিটামিন", "vitamins", "supplement", "সাপ্লিমেন্ট"), #
    "antihistamine": ("অ্যান্টিহিস্টামিন",), #
    "eye drops": ("চোখের ড্রপ", "আই ড্রপ"), #
    "ear drops": ("কানের ড্রপ",), #
    "nasal spray": ("নাকের স্প্রে", "নেজাল স্প্রে"), #
    "ointment": ("মলম", "অয়েন্টমেন্ট"), #
    "syrup": ("সিরাপ",), #
    "tablet": ("ট্যাবলেট", "বড়ি"), #
    "injection": ("ইনজেকশন", "ইঞ্জেকশন"), #
    "medicine": ("ওষুধ", "মেডিসিন", "ঔষধ"), #
    # --- procedures, devices and instructions: NOT drugs, NOT labs ---
    # Named positively so the gate rejects them with a reason. An absent
    # term falls through to fuzzy matching instead.
    "admit to ICU": ("admit in icu", "icu admission", "আইসিইউ তে ভর্তি", "আইসিইউ"),
    "IV fluids": ("iv fluids", "iv fluids if hypotensive", "আইভি ফ্লুইডস", "স্যালাইন"),
    "catheterisation": ("foley's catheterization", "foleys catheterisation", "catheterisation", "ফলিস ক্যাথেটারাইজেশন", "ক্যাথেটার"),
    "compression stockings": ("compression stockings", "class ii compression stockings", "কম্প্রেশন স্টকিংস"),
    "monofilament foot examination": ("foot examination with monofilament", "monofilament test", "মনোফিলামেন্ট"),
    "proctoscopy": ("proctoscopy", "প্রক্টোস্কোপি"),
    "otoscopy": ("otoscopy", "otoscopic examination", "ওটোস্কোপি"),
    "slit lamp examination": ("slit lamp", "slit lamp examination", "স্লিট ল্যাম্প এক্সামিনেশন"),
    # advice - explicitly NOT medications
    "exercise": ("ব্যায়াম", "এক্সারসাইজ", "হাঁটা", "walking"), #
    "lean diet": ("লিন ডায়েট", "হালকা খাবার", "light food"), #
    "avoid oily food": ("তেল মশলা এড়িয়ে", "তেলমশলা"), #
    "drink water": ("প্রচুর জল খাবেন", "বেশি করে জল খাবেন", "অনেক জল খাবেন", "বেশি পানি", "জল খাবেন"), #
    # "ওয়ারেস্ট" is the ASR's rendering on a diarrhoea consultation, where
    # ORS is the single most important instruction given.
    # ORS moved to the DRUG table - it is dispensed with a dose, and while
    # it sat here the gate rejected it out of every prescription. Its
    # spellings live with the Drug entry now; keeping them here as well
    # would fold to the same key and trip the collision check.
    "rest": ("বিশ্রাম",), #
    # Advice the consultations actually give, in the words they use. It
    # was recognised as non-clinical and then dropped, because nothing
    # carried it anywhere - see scan_advice.
    "use less soap": ("সাবান মাখবেন না", "বেশি সাবান মাখবেন না",
                       "কম সাবান", "সাবান কম"), #
    "do not scratch": ("চুলকাবেন না", "খুঁটবেন না", "নখ দিয়ে খুঁটবেন না",
                        "খোঁটাখুঁটি করবেন না"), #
    # Advice that is the whole treatment plan on these two consultations:
    # a six-month food exclusion, and the handling instructions that stop
    # a mastitis becoming an abscess.
    "avoid allergic foods": ("চিংড়ি কাঁকড়া বেগুন বন্ধ",
                              "যেগুলোতে অ্যালার্জি হয় বন্ধ",
                              "অ্যালার্জির খাবার বন্ধ", "চিংড়ি বন্ধ"), #
    "do not press the breast": ("চাপাচাপি করবেন না", "জোর করে চাপবেন না",
                                 "চাপবেন না", "চাপাচাপি করবেন না"), #
    "use a breast pump": ("ব্রেস্ট পাম্প", "বেস্ট প্রাম্প", "ব্রেস্ট পাম্প ব্যবহার",
                           "breast pump"), #
    "keep wearing the denture": ("পরা বন্ধ করবেন না", "পরাবন্ধ করবেন না",
                                  "পরা ছাড়বেন না", "বন্ধ করবেন না"), #
    "wear a wrist splint": ("রিস্ট স্প্লিন্ট", "রিস্ট প্লিন্ট",
                             "কবজির বেল্ট", "কব্জির বেল্ট",
                             "wrist splint"), #
    "warm compress": ("গরম সেক", "গরম সেঁক", "গরম জলে সেঁক", "গরম শেক"), #
    "avoid dust": ("ধুলো এড়িয়ে চলুন", "ধুলো থেকে দূরে", "ধুলোবালি এড়ান"), #
    "follow up": ("ফলো আপ", "আবার দেখাবেন"), #
    "bandage": ("ব্যাণ্ডেজ", "ব্যান্ডেজ"), #
    "dressing": ("ড্রেসিং",), #
    "nebulization": ("নেবুলাইজেশন", "নেবুলাইজার"), #
    "prescription": ("প্রেসক্রিপশন",), #
}


# Combined list of all drugs for easy access
DRUGS = (CARDIAC + ENDOCRINE + RESPIRATORY + GI + GENERAL + UROLOGY + BONE +
         DERMATOLOGY + OPHTHALMOLOGY + ENT + DENTAL + GYNAECOLOGY +
         NEPHROLOGY + NEUROLOGY + SURGERY + REVIEWED)


# ===========================================================================
# Combined list of all drugs for easy access
DRUGS = (CARDIAC + ENDOCRINE + RESPIRATORY + GI + GENERAL + UROLOGY + BONE +
         DERMATOLOGY + OPHTHALMOLOGY + ENT + DENTAL + GYNAECOLOGY +
         NEPHROLOGY + NEUROLOGY + SURGERY + REVIEWED)


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
    "ৎ": "ট",  # ৎ -> ট
    # DENTAL vs RETROFLEX. Contrastive in native Bengali words, but not in
    # English loanwords - which is all the drug vocabulary is. Bengali
    # writes English /t/ and /d/ as retroflex ট/ড ("মেটফরমিন"), while
    # transliterators emit dental ত/দ ("মেত্ফোরমিন"), and speakers vary.
    # Merging them lets one entry cover both conventions.
    #
    # This is the most aggressive rule here, so it is justified by
    # measurement, not intuition: collisions() and tests_regression.py are
    # run after adding it, and it stays only while both are clean.
    "ত": "ট",  # ত -> ট
    "দ": "ড",  # দ -> ড
    # ASPIRATION - the least stable feature across speakers
    "খ": "ক",  # খ -> ক
    "ঘ": "গ",  # ঘ -> গ
    "ছ": "চ",  # ছ -> চ
    "ঝ": "জ",  # ঝ -> জ
    "ঠ": "ট",  # ঠ -> ট
    "ঢ": "ড",  # ঢ -> ড
    "থ": "ট",  # থ -> ট (dental/retroflex merged)
    "ধ": "ড",  # ধ -> ড (dental/retroflex merged)
    "ফ": "প",  # ফ -> প
    "ভ": "ব",  # ভ -> ব
})

# Latin side: drug names arrive romanised too, with the same instability
# ("Montelukast" / "Montuculast", "Ecosprin" / "Ecospirin").
_LAT_FOLD = (
    ("ph", "f"), ("ck", "k"), ("qu", "k"), ("x", "ks"),
    ("y", "i"), ("z", "s"), ("c", "k"), ("w", "v"), ("j", "z"),
) #


# TRIED AND REJECTED: stripping vowel diacritics (matras).
#
# It would let transliterator output match real spellings - "মেত্ফ়োর্মিন্"
# and "মেটফরমিন" both reduce to the same consonant skeleton. Tempting,
# because transliterators and speakers disagree constantly about which
# vowel sign an English loanword takes.
#
# But it collapses too much: tests_regression.py failed immediately,
# reinstating the HbA1c-matches-CT-scan over-reach and breaking a CBC case.
# Consonant skeletons alone are not distinctive enough for a vocabulary
# this phonetically dense. Do not re-add it without a much larger reviewed
# set to measure against.
def fold(s: str) -> str:
    """Collapse a term to its phonetic skeleton for matching.

    Lossy by design - see the block comment above. Never use the output for
    display; it is a lookup key only.
    """
    s = unicodedata.normalize("NFC", s).strip().lower() #
    s = s.translate(_BN_DROP).translate(_BN_FOLD) #
    for a, b in _LAT_FOLD: #
        s = s.replace(a, b) #
    # strip spacing and punctuation LAST, so "সি বি সি" == "সিবিসি"
    out = [c for c in s if not (c.isspace() or unicodedata.category(c).startswith("P"))] #
    s = "".join(out) #
    # Collapse doubled letters ("Ecosprinn", "pantopp") - ASCII ONLY.
    # Applying this to Bengali corrupts spelled-out acronyms: EEG is
    # "ই ই জি", and deduping it to "ইজি" made a 3-character key that
    # false-matched "এই জিভটা" ("stick your tongue out").
    deduped: list[str] = [] #
    for c in s: #
        if deduped and deduped[-1] == c and c.isascii(): #
            continue #
        deduped.append(c) #
    return "".join(deduped) #


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
                     "দুপুরে খেয়ে"), #
    "after dinner": ("রাতে খাবার পরে", "রাতে খাওয়ার পর", "রাতের খাবারের পর",
                      "রাতে খেয়ে"), #
    "after breakfast": ("সকালে খাওয়ার পর", "সকালে খাবার পরে", "ব্রেকফাস্টের পর"), #
    "before food": ("খাওয়ার আগে", "খাবার আগে", "খালি পেটে", "খাওয়ার পূর্বে"), #
    "after food": ("খাওয়ার পরে", "খাবার পরে", "ভরা পেটে", "খাওয়ার পর"), #
    "in the morning": ("সকালে", "সকাল বেলা", "রোজ সকালে"), #
    "at night": ("রাতে", "রাত্রে", "শোয়ার আগে", "ঘুমানোর আগে",
                  "রোজ রাতে", "রোজরাতে", "প্রতি রাতে", "প্রতিদিন রাতে"), #
    "twice daily": ("দিনে দুবার", "দুবেলা", "সকাল বিকেল", "সকালে আর রাতে"), #
    "three times daily": ("দিনে তিনবার", "তিনবেলা", "তিন বেলা"), #
    "once daily": ("দিনে একবার", "রোজ একটা", "একবেলা", "প্রতিদিন একবার"), #
    "when required": ("দরকার হলে", "প্রয়োজন হলে", "যখন লাগবে", "কষ্ট হলে"), #
}

# Frequency shorthand a printed prescription expects.
DOSING_CODES: dict[str, str] = {
    "once daily": "OD", #
    "twice daily": "BD", #
    "three times daily": "TDS", #
    "when required": "SOS", #
    "in the morning": "OD (morning)", #
    "at night": "OD (night)", #
}

DURATION_TERMS: dict[str, tuple[str, ...]] = {
    "3 days": ("তিন দিন", "তিনদিন"), #
    "5 days": ("পাঁচ দিন", "পাঁচদিন"), #
    "7 days": ("সাত দিন", "সাতদিন", "এক সপ্তাহ"), #
    "10 days": ("দশ দিন", "দশদিন"), #
    "15 days": ("পনেরো দিন", "পনেরদিন", "দুই সপ্তাহ"), #
    "1 month": ("এক মাস", "একমাস", "৩০ দিন"), #
    "2 months": ("দুই মাস", "দুমাস"), #
    "3 months": ("তিন মাস", "তিনমাস"), #
    "6 months": ("ছয় মাস", "ছমাস"), #
    "continue": ("চালিয়ে যান", "চলতে থাকবে", "একটানা", "নিয়মিত"), #
}


def _norm(s: str) -> str:
    """Display normalisation only. Matching goes through fold()."""
    return unicodedata.normalize("NFC", s.strip().lower()) #


# Build lookup tables once at import, keyed on the FOLD.
_DRUG_LOOKUP: dict[str, Drug] = {} #
for _d in ALL_DRUGS: #
    for _key in (_d.generic, *_d.brands, *_d.bengali): #
        _DRUG_LOOKUP[fold(_key)] = _d #

_LAB_LOOKUP: dict[str, str] = {} #
for _canon, _alts in LAB_TESTS.items(): #
    _LAB_LOOKUP[fold(_canon)] = _canon #
    for _a in _alts: #
        _LAB_LOOKUP[fold(_a)] = _canon #

_TERM_LOOKUP: dict[str, str] = {} #
for _canon, _alts in CLINICAL_TERMS.items(): #
    _TERM_LOOKUP[fold(_canon)] = _canon #
    for _a in _alts: #
        _TERM_LOOKUP[fold(_a)] = _canon #

# The CURATED terms alone, captured before the machine-imported vocabulary
# is merged in below.
#
# WHY THE TWO ARE KEPT APART
# A curated table may GENERATE a finding; an imported one may only
# VALIDATE a candidate. This is the same rule gate.py already applies to
# the 174k Indian brand register, and for the same reason - see the note
# there on why fishing names out of raw transcripts is a different and
# much more dangerous operation than checking a name the model proposed.
#
# The imported MedER vocabulary is pharmacology prose, not a symptom list.
# Scanning transcripts with it put these on real prescriptions as findings:
#
#     doctor  report  age  serum  wound  inhaler  milk  dna  capacity
#
# and it holds ~1,900 more of the same kind - "absorption", "adult size",
# "alternative splicing", "adverse effects", "HDL". No blocklist keeps up
# with that; the table is simply not a source of symptoms. It stays fully
# available to is_clinical_term(), which only ever VETOES - rejecting a
# proposed medication that is really a symptom or finding, the safe
# direction, where the worst case is a human reviewing it.
# Folded keys that an ORDINARY Bengali word also lands on. Each alias here
# is a genuine clinical word - the collision is with everyday vocabulary,
# which is the gap collisions() explicitly cannot see: it compares
# gazetteer entries against each other, never against the language.
#
# Measured on the 16 real consultations, matched tokens counted:
#
#   গা    ক্ষত/ঘা "sore"      also গা "body"        3 of 5 hits were wrong
#   কসট   ক্ষত "wound"        also কষ্ট "difficulty" 2 of 2 hits were wrong
#   বরন   ব্রণ "acne"         also ভ্রণ "embryo"     1 of 1 hits was wrong
#
# Blocked from GENERATING a finding, not from the gazetteer. They stay in
# _TERM_LOOKUP, so is_clinical_term() still vetoes them as non-drugs - the
# safe direction. If a patient really does say ঘা, the extraction model
# reports it: the model understands ordinary words, and it is the fold, not
# the model, that cannot tell these pairs apart.
#   কাটা  ক্ষত "cut"          also "গায়ে কাটা দিয়ে" = shivering
#                              0 true positives, 1 false, over the 16
#   পিট   ফিট "fit/seizure"   also পিঠ "back"       1 of 1 hits was wrong
#
# পিট is the worst of them, because the two words are not even related and
# the false positive is a diagnosis rather than a symptom: a dermatology
# consultation about spots on the back and chest was reported as SEIZURE,
# and the summary repeated it. ফ->প and ঠ->ট are both correct fold rules;
# it is their combination on a three-letter word that is unsafe.
_AMBIGUOUS_WITH_COMMON_WORD = frozenset({"গা", "কসট", "বরন", "কাটা", "পিট"}) #

_CURATED_TERM_LOOKUP: dict[str, str] = { #
    _k: _v for _k, _v in _TERM_LOOKUP.items() #
    if _k not in _AMBIGUOUS_WITH_COMMON_WORD #
} #

# Bengali spellings LEARNED from human-labelled transcripts.
#
# Higher trust than anything machine-inferred: a human wrote the Bengali
# form next to the English drug name. Coverage on 188 real department
# transcripts was 62.2% for drugs, and most misses were not unknown drugs -
# they were known drugs under a spelling the gazetteer did not have:
#
#     transcript  মন্টিলোকাস্ট     gazetteer  মন্টিকুলাষ্ট     Montelukast
#     transcript  লিভোসেট্রিজিন    gazetteer  লেভোসেটিরিজিন    Levocetirizine
#
# The fold cannot bridge those - the vowels differ too much - and loosening
# it far enough was tried and broke other matching. Recording the observed
# spelling is the correct fix.
try:
    from .learned_forms import LEARNED_DRUG_FORMS, LEARNED_LAB_FORMS
except ImportError:                              # pragma: no cover
    LEARNED_DRUG_FORMS, LEARNED_LAB_FORMS = {}, {}

_LEARNED_COUNT = 0
for _generic, _forms in LEARNED_DRUG_FORMS.items():
    _drug = _DRUG_LOOKUP.get(fold(_generic))
    if _drug is None:
        continue
    for _f in _forms:
        _k = fold(_f)
        if _k and _k not in _DRUG_LOOKUP and _k not in _LAB_LOOKUP:
            _DRUG_LOOKUP[_k] = _drug
            _LEARNED_COUNT += 1

for _canon, _forms in LEARNED_LAB_FORMS.items():
    for _f in _forms:
        _k = fold(_f)
        if _k and _k not in _LAB_LOOKUP and _k not in _DRUG_LOOKUP:
            _LAB_LOOKUP[_k] = _canon
            _LEARNED_COUNT += 1


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
    from .terms_imported import IMPORTED_TERMS #
except ImportError:                              # pragma: no cover
    IMPORTED_TERMS = {} #

_IMPORTED_COUNT = 0 #
for _canon, _alts in IMPORTED_TERMS.items(): #
    for _a in _alts: #
        _k = fold(_a) #
        # never let an import shadow a curated term, a drug or a lab test
        if _k in _TERM_LOOKUP or _k in _DRUG_LOOKUP or _k in _LAB_LOOKUP: #
            continue #
        _TERM_LOOKUP[_k] = _canon #
        _IMPORTED_COUNT += 1 #


def _build_collisions() -> list[tuple[str, str, str]]:
    """Aliases that two DIFFERENT canonical entries both claim.

    collisions() below inspects the finished lookup tables, which cannot
    show this: during construction `table[fold(alias)] = canon` silently
    overwrites, so only the last writer survives and the clash disappears.

    That blind spot let "Echo" and "2D Echo" coexist as separate lab tests
    both claiming the alias "2d echo". One spoken test then resolved to two
    entries, producing 297 phantom false positives in a 500-transcript
    evaluation and sending me looking for a detection bug that did not
    exist.
    """
    found: list[tuple[str, str, str]] = [] #
    for label, table in (("lab", LAB_TESTS), ("term", CLINICAL_TERMS)): #
        seen: dict[str, str] = {} #
        for canon, aliases in table.items(): #
            for alias in (canon, *aliases): #
                key = fold(alias) #
                if key in seen and seen[key] != canon: #
                    found.append((alias, f"{label}:{seen[key]}", f"{label}:{canon}")) #
                else:
                    seen[key] = canon #
    for i, d1 in enumerate(ALL_DRUGS): #
        for d2 in ALL_DRUGS[i + 1:]: #
            shared = {fold(a) for a in (d1.generic, *d1.brands, *d1.bengali)} & \
                     {fold(a) for a in (d2.generic, *d2.brands, *d2.bengali)} #
            for key in shared: #
                found.append((key, f"drug:{d1.generic}", f"drug:{d2.generic}")) #
    return found #


def collisions() -> list[tuple[str, str, str]]:
    """Entries that fold onto the same key but mean different things.

    The fold is lossy, so this is the guard-rail that keeps it honest. A
    drug colliding with another drug, or with a clinical term, is a real
    bug - the fold has over-merged and must be made less aggressive.
    Exercised by the test suite; returns [] when the gazetteer is clean.
    """
    found: list[tuple[str, str, str]] = _build_collisions() #
    seen: dict[str, tuple[str, str]] = {} #
    for kind, table in (("drug", {k: v.generic for k, v in _DRUG_LOOKUP.items()}), #
                        ("lab", _LAB_LOOKUP), #
                        ("term", _TERM_LOOKUP)): #
        for key, canon in table.items(): #
            if key in seen: #
                prev_kind, prev_canon = seen[key] #
                if prev_canon != canon: #
                    found.append((key, f"{prev_kind}:{prev_canon}", f"{kind}:{canon}")) #
            else:
                seen[key] = (kind, canon) #
    return found #


# Dosage-form noise the SLM prepends. Folded, because that is the space
# the comparison happens in ("cap." -> "kap", "syp." -> "sip").
_DOSAGE_PREFIXES = tuple(fold(p) for p in (
    "tab.", "tab", "cap.", "cap", "syp.", "syp", "inj.", "inj",
    "tablet", "capsule", "syrup", "injection", "ট্যাব", "ক্যাপ",
)) #


def lookup_drug(text: str) -> Drug | None:
    """Exact gazetteer hit in folded space, or None.

    Folding means one entry covers its whole spelling family, so
    "মন্টিকুলাষ্ট", "মনটিকুলাসট" and "Montelukast" all land on the same Drug.
    """
    t = fold(text) #
    if not t: #
        return None #
    if t in _DRUG_LOOKUP: #
        return _DRUG_LOOKUP[t] #
    for prefix in _DOSAGE_PREFIXES: #
        if prefix and t.startswith(prefix): #
            stripped = t[len(prefix):] #
            if stripped in _DRUG_LOOKUP: #
                return _DRUG_LOOKUP[stripped] #
    return _ngram_match(text, _DRUG_LOOKUP) #


# How many consecutive words a gazetteer entry may span. Covers the longest
# real cases: "সি বি সি" (3), "খাওয়ার পরের সুগার" (3), "তেল মশলা এড়িয়ে" (3),
# plus headroom for a spelled-out four-letter acronym.
_MAX_NGRAM = 5 #


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
_MIN_DRUG_NGRAM = 4 #


def _too_short_for_text(gram: str, table: dict) -> bool:
    """Guard against short DRUG keys being fished out of free text."""
    return table is _DRUG_LOOKUP and len(gram) < _MIN_DRUG_NGRAM #


# Below this, a drug key is a brand ABBREVIATION - টোবা (Tobra), টেলমা
# (Telma), ডোলো (Dolo), নাইস (Nise) - short enough to collide with ordinary
# Bengali vocabulary. Such a key may only match a WHOLE TOKEN; it may never
# be assembled by joining separate words.
#
# The bounded suffix tail alone did not close this. Two of the three false
# medications came back by other routes:
#
#     তো মেনোপস বা   dropping the interior token joins "তো"+"বা" -> টোবা,
#                    which is then an EXACT hit, so no tail rule applies
#     তেল মাখলাম     folds to টেলমাকলাম, trailing টেলমা by exactly 4 -
#                    inside any suffix bound loose enough to allow গুলো
#
# Both are two ordinary words being welded into a drug name. Requiring a
# single token refuses that while leaving the real uses intact: "ডোলোটা"
# is one token and still matches, and long keys - Rosuvastatin,
# Metformin - keep the gapped matching that recovers "রসু ভাস্টা টিন".
_SAFE_DRUG_KEY = 6 #


def _needs_whole_token(key: str, table: dict) -> bool:
    """Whether this key is too short to be built from joined words."""
    return table is _DRUG_LOOKUP and len(key) < _SAFE_DRUG_KEY #


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
    tokens = text.split() #
    if not tokens: #
        return None #
    best = None #
    best_len = 0 #
    for i in range(len(tokens)): #
        for n in range(1, min(_MAX_NGRAM, len(tokens) - i) + 1): #
            hit, key = _lookup_span(_span_variants(tokens, i, n), table, n) #
            # prefer the longest match, so "PP sugar" beats "blood sugar"
            if hit is not None and len(key) > best_len: #
                best, best_len = hit, len(key) #
    return best #


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
_NEGATORS = {"না", "নি", "নেই", "নয়", "নাই", "কখনো"} #
_INTERROGATIVES = {"কি", "কী", "কিনা", "কিনা?"} #
_SCOPE_WINDOW = 3 #


def _span_is_negated(tokens: list[str], start: int, end: int) -> bool:
    """True if a negation or question particle governs the matched span.

    Scans the span ITSELF as well as a short window after it. Checking only
    after the span was wrong: matching is prefix-based, so a longer n-gram
    swallows the negator and then looks clean. Real case -
    "ওই এনজিওগ্রাম করতে চাই না এখন" ("I don't want the angiogram now") was
    suppressed at n<=3 but reported at n=4, because the 4-token span
    absorbed the "না" and the window past it saw only "এখন".
    """
    for tok in tokens[start:end + _SCOPE_WINDOW]: #
        clean = tok.strip("।,?!.") #
        if clean in _NEGATORS or clean in _INTERROGATIVES: #
            return True #
    return False #


# Aliases that were AUTHORED with a negator inside them.
#
# _span_is_negated scans the span itself, and must keep doing so - a longer
# n-gram swallowing a real "না" is how "I don't want the angiogram" got
# reported. But an alias that CONTAINS a negator is a different thing: the
# token belongs to the term's own spelling, not to a denial of it.
#
# Two real cases, and the second was already broken:
#   "ফুড পয়সা নেই"      the ASR's rendering of "ফুড পয়জনিং". Its নেই is the
#                        tail of a mangled English word, and reading it as a
#                        denial discarded the diagnosis just given.
#   "খাওয়ার ইচ্ছে নেই"   IS loss of appetite. The নেই is the finding.
#
# A correctly-spelled denial still suppresses, because it matches the
# shorter alias that carries no negator: "ফুড পয়জনিং নেই" stays dropped.
def _build_negator_bearing_keys() -> frozenset[str]:
    out: set[str] = set() #
    for aliases in CLINICAL_TERMS.values(): #
        for alias in aliases: #
            toks = [t.strip("।,?!.") for t in alias.split()] #
            if any(t in _NEGATORS for t in toks): #
                key = fold(alias) #
                if key: #
                    out.add(key) #
    return frozenset(out) #


_NEGATOR_BEARING_KEYS = _build_negator_bearing_keys() #


def _span_variants(tokens: list[str], i: int, n: int) -> list[str]:
    """Folded forms of tokens[i:i+n]: the full span, plus each variant with
    ONE interior token removed.

    Contiguous-only matching was the single biggest source of misses. The
    key is "ফাস্টিং সুগার" but a doctor says "ফাস্টিং ব্লাড সুগার", and the
    interposed word made the n-gram miss - so every such phrasing needed
    its own hand-written alias, and each one was found only after a real
    consultation lost data.

    Dropping one interior token generalises that: the anchor words still
    have to be present, in order, adjacent-but-one. Only interior tokens
    are droppable - removing the first or last would match a different
    phrase entirely.
    """
    span = tokens[i:i + n] #
    out = [fold("".join(span))] #
    if n >= 3: #
        for skip in range(1, n - 1): #
            out.append(fold("".join(span[:skip] + span[skip + 1:]))) #
    return out #


# How much may trail a key and still count as an inflectional suffix.
#
# The prefix allowance exists for Bengali agglutination: the key is
# "সিবিসি" but a doctor says "সি বি সি টা", and "প্রেশার" arrives as
# "প্রেশারটা". Those suffixes - টা, টি, কে, তে, র, গুলো, টাকে - are one to
# four characters. The allowance was UNBOUNDED, which is a different and
# much weaker claim: that any word merely BEGINNING with a key is that key.
#
# On real consultations that produced three false medications, and they
# are the dangerous kind - ordinary Bengali words turning into drugs:
#
#     তো বাচ্চাকে   "to the child"      -> Tobramycin eye drops   (key টোবা)
#     তো মেনোপস বা  "...menopause..."   -> Tobramycin eye drops   (key টোবা)
#     তেল মাখলাম    "I applied oil"     -> Telmisartan            (key টেলমা)
#
# Each trailed its key by 4-5 characters of unrelated word. The short keys
# themselves are legitimate and stay - টোবা is Tobra, টেলমা is Telma, and
# so are ডোলো (Dolo), ওমেজ (Omez), নাইস (Nise), কালপল (Calpol). Deleting
# real brand names to fix a matching rule would trade a precision bug for
# a recall bug.
#
# 4 is the longest genuine suffix (গুলো, টাকে). Verified against the
# regression corpus: every legitimate agglutinated match needs <= 2.
_MAX_SUFFIX_TAIL = 4 #


def _lookup_span(grams: list[str], table: dict, n_tokens: int = 1,
                  allow_near: bool = False):
    """First table hit among a span's folded variants.

    grams[0] is the full contiguous span; the rest have one interior token
    dropped. The agglutinative-suffix allowance (prefix matching) is applied
    ONLY to grams[0].

    n_tokens is how many words grams[0] spans. Short drug keys require it to
    be 1 - see _SAFE_DRUG_KEY.

    Stacking both relaxations was too loose and produced a real false
    positive: "এইচ ওয়ান বি এ সি" (HbA1c) matched CT scan, because dropping
    an interior token yielded a fragment that merely STARTED with the "সিটি"
    key. A gapped match is already a relaxation; a gapped match that also
    only has to match a prefix is barely a match at all.

    The prefix allowance is also BOUNDED - see _MAX_SUFFIX_TAIL.
    """
    for idx, gram in enumerate(grams): #
        if not gram: #
            continue #
        hit = table.get(gram) #
        if (hit is not None and not _too_short_for_text(gram, table) #
                and not (_needs_whole_token(gram, table) and n_tokens > 1)): #
            return hit, gram #
        if idx == 0 and len(gram) >= 4: #
            for key, val in table.items(): #
                if (len(key) >= 4 and gram.startswith(key) #
                        and len(gram) - len(key) <= _MAX_SUFFIX_TAIL #
                        and not (_needs_whole_token(key, table) and n_tokens > 1)): #
                    return val, key #

    # Last resort. DRUGS: vowel-level variation, exact skeleton equality.
    # LABS: a dropped or inserted consonant, NEAR match - and only when
    # SCANNING free text (allow_near). is_lab_test() is used by the gate
    # to VETO a drug candidate, and a near match is not strong enough to
    # do that: it rejected "আইশো ট্রোটন নয়েন ক্যাপসুল" - Isotretinoin - as
    # though it were a lab test.
    if grams and grams[0]: #
        if table is _DRUG_LOOKUP: #
            # A span of SEVERAL words needs more consonant evidence than a
            # single token. Concatenating words manufactures skeletons that
            # were never a name: "মানে ওইটু জোরে" ("meaning, that one,
            # loudly") collapses to mntjr, which is also মন্টেয়ার - Montair,
            # a Montelukast brand. An asthma drug was therefore proposed on
            # an ANGINA consultation, at similarity 0.9, from three ordinary
            # words. Real split names carry more: Dexamethasone heard as
            # "ডেক্সা মিথোসেন" gives dksmtsn (7), Rosuvastatin heard as
            # "Rasu Basta Tin" gives rsbsttn (7). The collision sat at the
            # 5-char floor, so the bar for a multi-word span is above it.
            _sk = _skeleton(grams[0]) #
            if n_tokens == 1 or len(_sk) >= _MIN_SPAN_SKELETON: #
                hit = _DRUG_SKELETON.get(_sk) #
                if hit is not None: #
                    return hit, grams[0] #
        elif (table is _LAB_LOOKUP and allow_near
                and n_tokens <= _MAX_LAB_SPAN_TOKENS): #
            # BOUNDED BY SPAN WIDTH, and for a stronger reason than the
            # drug branch above: this match tolerates a dropped or inserted
            # consonant, so it is looser than the exact skeleton equality
            # drugs get, and gluing a whole clause together hands it a
            # string that was never a test name. Every wrong lab across
            # recordings 32-47 arrived this way, out of ordinary speech:
            #
            #   "দেখুন তো আরে আপনি শান্ত"          -> Troponin, on a LIPOMA
            #   "মিডিয়ান নার্ভটা হাতে ঢোকে সেখানে"  -> DEXA scan
            #   "নেবেন ফুসফুসের কি অবস্থা সেটা"     -> Visual acuity
            #   "মানে কি বলবো বলো ডাক্তার"          -> Blood culture
            #
            # All four are five-word spans carrying 10-15 consonants. The
            # relaxation was written for a garbled NAME, and a name is one
            # or two words.
            #
            # Narrowing this to one token costs nothing real. The only
            # genuine two-token case was "আল্ট্রাসাউন্ড" arriving as "আল্টা
            # সাউন্ড", and that is now an alias on USG. Genuine multi-word
            # labs are otherwise spelled-out acronyms ("ই সি জি", "টি এম
            # টি"), which sit in the table verbatim, match exactly, and
            # never reach this fallback at all.
            hit = _lab_by_near_skeleton(grams[0]) #
            if hit is not None: #
                return hit, grams[0] #
    return None, "" #


def _ngram_scan_all(text: str, table: dict, honour_negation: bool = True) -> list:
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
    return [hit for hit, _span, _i, _j in #
            _ngram_scan_spans(text, table, honour_negation)] #


def _ngram_scan_spans(text: str, table: dict, #
                       honour_negation: bool = True) -> list[tuple]: #
    """As _ngram_scan_all, but keeps the TEXT THAT MATCHED alongside each hit.

    The surface form is not decoration. A drug entry is one generic with
    several brands, so collapsing a hit to entry.generic loses which name
    was actually spoken: "সরবিট্রেট" and "Nitrocontin" both resolve to
    Nitroglycerin, and a prescription that prints the generic for a brand
    the doctor never said reads as a substitution. It was reported as one.
    """
    tokens = text.split() #
    found: list[tuple] = [] #
    seen: list = [] #
    for i in range(len(tokens)): #
        for n in range(1, min(_MAX_NGRAM, len(tokens) - i) + 1): #
            hit, _key = _lookup_span(_span_variants(tokens, i, n), table, n, #
                                      allow_near=True) #
            if hit is not None and hit not in seen: #
                if (honour_negation #
                        and _key not in _NEGATOR_BEARING_KEYS #
                        and _span_is_negated(tokens, i, i + n)): #
                    continue #
                seen.append(hit) #
                found.append((hit, " ".join(tokens[i:i + n]), i, i + n)) #
    return found #


# ---------------------------------------------------------------------------
# CROSS-TABLE ARBITRATION
#
# Each table is scanned independently over the same words, and nothing
# decided what happens when two of them claim the SAME words. That is a
# false-positive generator, and it produced the worst kind - a drug read
# as an investigation:
#
#     "ডেক্সা মিথোসেন ইনজেকশন"   Dexamethasone, a drug
#      ^^^^^^                    the lab table took this as a DEXA scan
#
# so an allergy patient given a steroid injection was recorded as having
# had a bone-density scan ordered. The same shape as "ইজি" (an EEG) being
# rejected as an unknown MEDICATION, and "electrolytes" verifying as a
# drug before the table order was fixed in gate.py.
#
# The rule is the one already used WITHIN a table for subsumption, applied
# ACROSS tables: the longest span wins. A match covering more of what was
# actually said is a better account of it than one covering less.
# ---------------------------------------------------------------------------
def _drop_overlapped(spans: list[tuple], rivals: list[tuple]) -> list[tuple]: #
    """Drop any span a STRICTLY LONGER rival span overlaps.""" #
    out = [] #
    for hit, text_, i, j in spans: #
        if any(ri < j and i < rj and (rj - ri) > (j - i) #
               for _h, _t, ri, rj in rivals): #
            continue #
        out.append((hit, text_, i, j)) #
    return out #


def scan_labs(text: str) -> list[str]:
    """All lab tests ordered in this segment, least specific dropped.

    A REGION-SPECIFIC order also contains the generic one, so both matched
    the same words: "চেস্ট এক্স রে" returned Chest X-ray AND X-ray, and
    "USG pelvis" returned USG alongside it. Two lines for one order reads
    as two investigations, and a patient sent for both pays twice.

    Only exact name containment is dropped, so "Fasting sugar" and
    "PP sugar" - genuinely two orders, neither inside the other - both
    survive.
    """
    spans = _drop_overlapped(_ngram_scan_spans(text, _LAB_LOOKUP), #
                              _ngram_scan_spans(text, _DRUG_LOOKUP)) #
    return _drop_subsumed_labs( #
        _attach_region([hit for hit, _t, _i, _j in spans], text)) #


# An imaging order names a MODALITY and a REGION, and Bengali does not
# require them to sit together: "এম আর আই করতে হবে ব্রেনের" puts two words
# between them, where the n-gram scan tolerates one. The result was a bare
# "MRI", which does not tell radiology what to scan.
#
# So the region is attached afterwards rather than matched inline: if a
# generic modality was ordered and a region is named anywhere in the same
# segment, and a specific entry exists for that pair, that entry is the
# order. Nothing is invented - both halves were spoken.
_REGIONS: dict[str, tuple[str, ...]] = { #
    "brain": ("ব্রেন", "ব্রেনের", "মাথার", "মস্তিষ্ক", "brain"), #
    "chest": ("চেস্ট", "বুকের", "বুক", "ফুসফুস", "ফুসফুসের", "chest"), #
    "pelvis": ("পেলভিস", "তলপেট", "তলপেটের", "pelvis"), #
    "knee": ("হাঁটু", "হাঁটুর", "knee"), #
    "kub": ("কিডনি", "কিডনিতে", "কিডনির", "kidney"), #
    # For CONDITIONS rather than imaging: the site of an infection is
    # named several words away from the word "infection".
    "urine": ("প্রস্রাব", "প্রস্রাবে", "প্রস্রাবের", "পেচ্ছাপ", "পেচ্ছাপের",
               "পোতসাব", "পোতসাবের", "প্রশ্রাব", "প্রশ্রাবে", "urine"), #
    "stomach": ("পেট", "পেটে", "পেটের", "stomach"), #
} #


def _attach_region(found: list[str], text: str, table=None) -> list[str]: #
    """Upgrade a bare entry to its region-specific one when both were said.""" #
    table = LAB_TESTS if table is None else table #
    if not found: #
        return found #
    tokens = {fold(t) for t in text.split()} #
    named = {region for region, words in _REGIONS.items() #
             if any(fold(w) in tokens for w in words)} #
    if not named: #
        return found #
    out = [] #
    for lab in found: #
        upgraded = lab #
        for region in named: #
            for canon in table: #
                if (canon != lab and lab.lower() in canon.lower() #
                        and region in canon.lower()): #
                    upgraded = canon #
                    break #
            if upgraded != lab: #
                break #
        if upgraded not in out: #
            out.append(upgraded) #
    return out #


def _drop_subsumed_labs(found: list[str]) -> list[str]: #
    """Drop a lab whose whole name sits inside another one found here.""" #
    out = [] #
    for lab in found: #
        low = lab.lower() #
        if any(other is not lab and low in other.lower() #
               and len(other) > len(lab) for other in found): #
            continue #
        out.append(lab) #
    return out #


def scan_drugs(text: str) -> list[Drug]:
    """All gazetteer drugs named in this segment."""
    return _ngram_scan_all(text, _DRUG_LOOKUP) #


# Consonant skeletons, for pairing a spoken BENGALI brand with its Latin
# spelling. fold() cannot do this: it normalises within a script, so
# "সরবিট্রেট" and "Sorbitrate" never meet. Dropping vowels and collapsing
# each consonant to one canonical letter puts both scripts in one alphabet:
#
#     সরবিট্রেট  -> srbtrt        Sorbitrate -> srbtrt      identical
#     ইকোস্পিরিন -> ksprn         Ecosprin   -> ksprn       identical
#
# Vowels are dropped rather than mapped because they carry the least
# reliable information across a transliteration - Bengali marks length and
# inherent vowels that English spelling simply does not have.
#
# USED ONLY TO CHOOSE A DISPLAY NAME among one drug entry's OWN brands.
# It never decides whether something is a drug, and it cannot introduce a
# different molecule. Measured over the gazetteer: 454/536 Bengali forms
# pair to a name at >= 0.75.
_BN_SKEL = { #
    'ক': 'k', 'খ': 'k', 'গ': 'g', 'ঘ': 'g', 'ঙ': 'n', #
    'চ': 'c', 'ছ': 'c', 'জ': 'j', 'ঝ': 'j', 'ঞ': 'n', #
    'ট': 't', 'ঠ': 't', 'ড': 'd', 'ঢ': 'd', 'ণ': 'n', #
    'ত': 't', 'থ': 't', 'দ': 'd', 'ধ': 'd', 'ন': 'n', #
    'প': 'p', 'ফ': 'f', 'ব': 'b', 'ভ': 'b', 'ম': 'm', #
    'য': 'j', 'র': 'r', 'ল': 'l', 'ৎ': 't', 'ং': 'n', #
    'শ': 's', 'ষ': 's', 'স': 's', 'হ': 'h', #
    'ড়': 'r', 'ঢ়': 'r', 'য়': '', #
} #

_LAT_DIGRAPH = (('ph', 'f'), ('th', 't'), ('ch', 'c'), ('sh', 's'), #
                ('gh', 'g'), ('kh', 'k'), ('bh', 'b'), ('dh', 'd'), #
                ('ck', 'k')) #


def _skeleton(text: str) -> str: #
    """Consonant skeleton, in one alphabet for both scripts.""" #
    if any('ঀ' <= c <= '৿' for c in text): #
        return "".join(_BN_SKEL.get(c, '') for c in text) #
    t = text.lower() #
    for a, b in _LAT_DIGRAPH: #
        t = t.replace(a, b) #
    out = [] #
    for ch in t: #
        if ch in "aeiouywh '-.": #
            continue #
        if ch == 'x': #
            out.append('k') #
            ch = 's' #
        else: #
            ch = {'c': 'k', 'z': 's', 'v': 'b', 'q': 'k'}.get(ch, ch) #
        if ch.isalpha(): #
            out.append(ch) #
    return "".join(out) #


# Drug keys indexed by CONSONANT SKELETON, for vowel-level ASR variation.
#
# fold() normalises consonants but leaves vowels alone, so every vowel the
# ASR hears differently is a miss that has to be added by hand:
#
#     ডেক্সা মিথোসেন  vs  ডেক্সামিথাসোন   Dexamethasone
#     অ্যাভল          vs  অ্যাভিল         Avil
#     রোসকাডো ট্রিল   vs  রেসিকাডোট্রিল   Racecadotril
#
# Each of those was a real miss, and adding spellings one at a time does
# not converge - the ASR invents a new vowel pattern on the next
# recording. The skeletons are identical in every pair above.
#
# STRIPPING VOWELS INSIDE fold() WAS TRIED AND REJECTED - it broke
# matching elsewhere, which is why this is a separate, LAST-RESORT index
# rather than a change to the fold. It applies only after exact and
# prefix matching have both failed.
#
# Safety, because dropping vowels is lossy:
#   - exact skeleton equality only, never fuzzy
#   - minimum 5 consonants (see below)
#   - AMBIGUOUS skeletons are excluded outright. Measured: 11 of 962
#     collide, and they are exactly the pairs that must never be confused
#     (Linagliptin/Olanzapine, Ofloxacin/Zinc, Hydroxyzine/Levothyroxine).
#
# The minimum was 4 and is now 5, on measurement. At four consonants a
# skeleton is the same size as an ordinary Bengali word, and collisions()
# cannot warn about it - it compares gazetteer entries against each other,
# never against everyday vocabulary. Both known false positives were
# exactly four:
#
#   bjlm   বুঝলাম  "I understand"   == ভ্যালুম   (Valium)     -> Diazepam
#   strn   স্ট্রেন  "strain"         == Isotroin (Isotretinoin)
#
# and every true recovery was five or more:
#
#   tntjl  টিনিটা জল    -> Tinidazole      5
#   sklfnk আসিক্লোফেন্ক -> Aceclofenac     6
#   rsbsttn Rasu Basta Tin -> Rosuvastatin 7
#
# Same caveat as every threshold here: five observations, not fifty. It is
# the gap the data shows, and raising it costs nothing measurable - the
# regression suite is unchanged at 177.
_MIN_SKELETON = 5 #
# How wide a span the LAB near-skeleton fallback may glue together.
#
# ONE. Two was tried and does not hold: at two tokens the false matches
# are indistinguishable from the real one on every measure available -
# "বলো ডাক্তার" -> Blood culture and "অবস্থা সেটা" -> Visual acuity carry
# 5-6 consonants across 2 words, exactly like "আল্টা সাউন্ড" -> USG. The
# single real case is now an alias on USG itself, so the relaxation no
# longer has to cover it. See _lookup_span.
_MAX_LAB_SPAN_TOKENS = 1 #
# Same index, but reached by concatenating MULTIPLE words - see _lookup_span.
_MIN_SPAN_SKELETON = 6 #
_DRUG_SKELETON: dict[str, Drug] = {} #
_skel_owners: dict[str, set] = {} #
for _k, _d in _DRUG_LOOKUP.items(): #
    _s = _skeleton(_k) #
    if len(_s) >= _MIN_SKELETON: #
        _skel_owners.setdefault(_s, set()).add(_d.generic) #
        _DRUG_SKELETON[_s] = _d #
for _s, _owners in _skel_owners.items(): #
    if len(_owners) > 1: #
        _DRUG_SKELETON.pop(_s, None) #
del _skel_owners #


# Lab keys by skeleton, for a DROPPED or INSERTED consonant.
#
# Vowel drift is handled by exact skeleton equality (see _DRUG_SKELETON),
# but the ASR also loses whole consonants: "আল্টা সাউন্ড" is
# "আল্ট্রাসাউন্ড" with the র gone, so the skeletons differ too - ltsnd
# against ltrsnd - and no exact index can bridge that.
#
# So this one is NEAR-match, and that is a weaker claim, fenced
# accordingly:
#   - minimum 5 consonants, so short acronyms cannot drift into each other
#   - 0.90 similarity, which is roughly one dropped character in six
#   - any key within 0.90 of a DIFFERENT lab is excluded outright.
#     Measured: 4 such pairs among 298 keys.
#
# Labs only. The same relaxation on DRUGS would be reckless - a dropped
# consonant is exactly how one drug name becomes another - and drugs
# already have the exact-skeleton index plus a scored fuzzy tier in the
# gate, both of which propose rather than assert.
_LAB_NEAR_MIN = 5 #
_LAB_NEAR_FLOOR = 0.90 #
_LAB_SKELETONS: list[tuple] = [] #
_lab_sk = [(k, _skeleton(k), v) for k, v in _LAB_LOOKUP.items() #
           if len(_skeleton(k)) >= _LAB_NEAR_MIN #
           and "unspecified" not in v.lower()] #
for _k, _sk, _v in _lab_sk: #
    if any(_v2 != _v and SequenceMatcher(a=_sk, b=_sk2).ratio() >= _LAB_NEAR_FLOOR #
           for _k2, _sk2, _v2 in _lab_sk): #
        continue                    # ambiguous with another lab - excluded #
    _LAB_SKELETONS.append((_sk, _v)) #
del _lab_sk #


def _lab_by_near_skeleton(span: str): #
    """Closest lab whose consonant skeleton nearly matches this span.""" #
    sk = _skeleton(span) #
    if len(sk) < _LAB_NEAR_MIN: #
        return None #
    best, best_score = None, 0.0 #
    for cand_sk, canon in _LAB_SKELETONS: #
        score = SequenceMatcher(a=sk, b=cand_sk).ratio() #
        if score > best_score: #
            best, best_score = canon, score #
    return best if best_score >= _LAB_NEAR_FLOOR else None #


# Below this the skeletons are too far apart to claim they are the same
# name, and the generic is used instead.
_SKEL_FLOOR = 0.75 #
# Within this of the best score, the GENERIC wins. A brand is a stronger
# claim than a molecule - it names a specific product - so it has to be
# clearly better, not merely tied. Without this "মন্টিকুলাস" (plainly
# Montelukast) printed as "Montek", a brand nobody said.
_SKEL_MARGIN = 0.08 #


def display_name(drug: Drug, spoken: str) -> str: #
    """The name to print for a drug the transcript named as `spoken`.""" #
    key = fold(spoken) #
    for brand in drug.brands: #
        if fold(brand) == key: #
            return brand            # said the brand outright, in Latin #
    if fold(drug.generic) == key: #
        return drug.generic #

    # Cross-script: pick whichever of this drug's own names the spoken
    # form most resembles, generic winning ties.
    said = _skeleton(spoken) #
    if not said: #
        return drug.generic #
    scored = [(SequenceMatcher(a=said, b=_skeleton(n)).ratio(), n) #
              for n in (drug.generic,) + tuple(drug.brands)] #
    best, best_name = max(scored, key=lambda s: s[0]) #
    if best < _SKEL_FLOOR: #
        return drug.generic #
    generic_score = scored[0][0] #
    if best - generic_score <= _SKEL_MARGIN: #
        return drug.generic #
    return best_name #


class DrugMention(typing.NamedTuple): #
    """One drug found in a transcript, with both names it needs.""" #
    drug: Drug      # the gazetteer entry #
    printed: str    # the name to put on the prescription #
    spoken: str     # the text that actually matched, verbatim #
    exact: bool = True   # False when only the consonant skeleton matched #


def matched_exactly(span: str) -> bool: #
    """Whether this span hits a drug key outright, vowels and all.

    A skeleton match is weaker evidence - it drops every vowel - so a
    caller must be able to tell the two apart. "Rasu Basta Tin" IS
    Rosuvastatin and should be recovered, but not silently promoted to the
    same confidence as a name that matched outright.
    """ #
    return fold(span) in _DRUG_LOOKUP #


def scan_drugs_spoken(text: str) -> list[DrugMention]: #
    """As scan_drugs, but keeps the name to PRINT and the text that was SAID.

    The printed name is the name the doctor actually used, in either
    script, and the generic when that cannot be established:

        "Sorbitrate"   -> printed "Sorbitrate"     brand, said in Latin
        "সরবিট্রেট"      -> printed "Sorbitrate"     brand, said in Bengali
        "রসু ভাস্টাটিন"   -> printed "Rosuvastatin"

    The Bengali case goes through _display_name's consonant skeleton.
    Drug.brands and Drug.bengali are independent tuples with no shared
    order, so the pairing is not in the data and has to be derived from
    the names themselves. Where it cannot be, the generic is used - always
    clinically correct - and `spoken` keeps the original either way.
    """
    own = _ngram_scan_spans(text, _DRUG_LOOKUP) #
    spans = _drop_overlapped(_drop_overlapped(own, own), #
                              _ngram_scan_spans(text, _LAB_LOOKUP)) #
    return [DrugMention(drug, display_name(drug, span), span, #
                         matched_exactly(span)) #
            for drug, span, _i, _j in spans] #


def scan_terms(text: str) -> list[str]:
    """All clinical terms (symptoms / findings / advice) in this segment."""
    return _ngram_scan_all(text, _TERM_LOOKUP) #


def is_lab_test(text: str) -> str | None:
    """Folded n-gram match - this is what makes spelled-out acronyms work:
    the tokens "সি বি সি" join and fold to the CBC key."""
    t = fold(text) #
    if not t: #
        return None #
    if t in _LAB_LOOKUP: #
        return _LAB_LOOKUP[t] #
    return _ngram_match(text, _LAB_LOOKUP) #


def is_clinical_term(text: str) -> str | None:
    """True for symptoms, findings and advice - i.e. things that are
    definitively NOT medications."""
    t = fold(text) #
    if not t: #
        return None #
    if t in _TERM_LOOKUP: #
        return _TERM_LOOKUP[t] #
    return _ngram_match(text, _TERM_LOOKUP) #


def stats() -> dict:
    return {
        "drugs": len(ALL_DRUGS), #
        "drug_aliases": len(_DRUG_LOOKUP), #
        "lab_tests": len(LAB_TESTS), #
        "clinical_terms": len(CLINICAL_TERMS), #
        "departments": sorted({d.department for d in ALL_DRUGS}), #
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
    freqs = _ngram_scan_all(text, _DOSING_LOOKUP) #
    durs = _ngram_scan_all(text, _DURATION_LOOKUP) #

    # Drop terms implied by a more specific one already present. "after
    # dinner" also matches "at night" and "after food", and printing all
    # three reads like three separate instructions.
    for specific, implied in _SUBSUMES.items(): #
        if specific in freqs: #
            freqs = [f for f in freqs if f == specific or f not in implied] #

    freq_parts = [DOSING_CODES.get(f, f) for f in freqs] #
    return ", ".join(freq_parts), ", ".join(durs) #


_SUBSUMES: dict[str, tuple[str, ...]] = {
    "after dinner": ("at night", "after food"), #
    "after lunch": ("after food",), #
    "after breakfast": ("in the morning", "after food"), #
}


_DOSING_LOOKUP: dict[str, str] = {} #
for _canon, _alts in DOSING_TERMS.items(): #
    _DOSING_LOOKUP[fold(_canon)] = _canon #
    for _a in _alts: #
        _DOSING_LOOKUP[fold(_a)] = _canon #

_DURATION_LOOKUP: dict[str, str] = {} #
for _canon, _alts in DURATION_TERMS.items(): #
    _DURATION_LOOKUP[fold(_canon)] = _canon #
    for _a in _alts: #
        _DURATION_LOOKUP[fold(_a)] = _canon #

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
    "osteoporosis", "arthritis", "asthma", "muscle strain", "sprain",
    "dust allergy", "allergic rhinitis", "epilepsy", "stomach infection",
    "heart blockage",
    "pityriasis versicolor",
    "food poisoning",
    "urine infection", "enlarged prostate",
    "vertigo", "inner ear balance disorder",
    "severe allergy", "hives", "mastitis", "breast abscess", "lipoma", "traumatic ulcer",
    "deviated nasal septum",
    "carpal tunnel syndrome",
})

# Neither a symptom nor a diagnosis: advice, dosage forms and drug classes.
NON_CLINICAL_TERMS: frozenset[str] = frozenset({
    "exercise", "lean diet", "avoid oily food", "drink water",
    "use less soap", "avoid dust", "avoid allergic foods",
    "do not press the breast", "use a breast pump", "warm compress",
    "wear a wrist splint", "keep wearing the denture",
    "do not scratch", "avoid sweets", "walk daily",
    "rest", "follow up", "bandage", "dressing", "nebulization",
    "prescription", "dialysis", "antibiotic", "painkiller", "antacid",
    "steroid", "vitamin supplement", "antihistamine", "eye drops",
    "ear drops", "nasal spray", "ointment", "syrup", "tablet",
    "injection", "medicine",
})


# Which specialty a diagnosis belongs to. Used ONLY to break ties between
# similar-sounding drugs - see gate._CONTEXT_BONUS - never to suggest a
# drug or a test. Clobazam and Clonazepam differ by 0.78 similarity and by
# which clinic you are sitting in.
DEPARTMENT_BY_CONDITION: dict[str, str] = {
    "angina": "cardiac", "heart attack": "cardiac",
    "heart failure": "cardiac", "ischemia": "cardiac",
    "blockage": "cardiac", "heart blockage": "cardiac", "hypertension": "cardiac",
    "diabetes": "endocrine", "thyroid": "endocrine",
    "seizure": "neurology", "epilepsy": "neurology",
    "stroke": "neurology", "migraine": "neurology",
    "dementia": "neurology",
    "asthma": "respiratory", "dust allergy": "respiratory",
    "allergic rhinitis": "respiratory",
    "cataract": "ophthalmology", "glaucoma": "ophthalmology",
    "osteoporosis": "bone", "arthritis": "bone",
    "muscle strain": "bone", "sprain": "bone",
    "kidney failure": "nephrology",
    "stomach infection": "gastro", "gastritis": "gastro",
    "food poisoning": "gastro",
    "urine infection": "urology", "enlarged prostate": "urology",
    "vertigo": "ent", "inner ear balance disorder": "ent",
    "severe allergy": "general", "hives": "dermatology",
    "mastitis": "gynaecology", "breast abscess": "gynaecology",
    "lipoma": "surgery", "deviated nasal septum": "ent",
    "carpal tunnel syndrome": "neurology",
    "traumatic ulcer": "dental",
    "piles": "surgery", "hernia": "surgery",
    "appendicitis": "surgery", "gallstone": "surgery",
    "menopause": "gynaecology", "pregnancy": "gynaecology",
    "acne": "dermatology", "fungal infection": "dermatology",
    "pityriasis versicolor": "dermatology",
    "tonsillitis": "ent",
}


def department_for(conditions: list[str]) -> str:
    """The specialty a consultation belongs to, from its diagnoses."""
    for c in conditions:
        dept = DEPARTMENT_BY_CONDITION.get(c.strip().lower())
        if dept:
            return dept
    return ""


_GENERIC_CONDITIONS = frozenset({"infection"}) #


def scan_conditions(text: str) -> list[str]:
    """Diagnosable conditions named in the text, least specific dropped.

    "পেটে ইনফেকশন" matched both "stomach infection" and "infection", and a
    diagnosis line reading "stomach infection, infection" names one
    finding twice while implying two. Same containment rule as scan_labs.
    """
    found = [t for t in _ngram_scan_all(text, _CURATED_TERM_LOOKUP) #
             if t in CONDITIONS] #
    found = _drop_subsumed_labs(_attach_region(found, text, CONDITIONS)) #
    # "mastitis, infection" names one finding twice. Containment cannot
    # see it - the words do not overlap - but a catch-all diagnosis beside
    # a specific one carries nothing.
    if len(found) > 1: #
        found = [c for c in found if c not in _GENERIC_CONDITIONS] or found #
    return found #


# Terms too generic to be a symptom on their own. They arrive from the
# imported MedER vocabulary, where "লক্ষণ" (symptoms) and organ names like
# "হার্ট" are legitimate entries - but a prescription listing "heart" and
# "symptoms" as findings is noise, and it crowds out the real ones.
_TOO_GENERIC = frozenset({
    "symptoms", "heart", "pain", "infection", "medicine", "problem",
    "eye", "ear", "nose", "throat", "skin", "bone", "blood", "body",
    "stomach", "chest", "head", "leg", "hand", "back",
})


def _drop_subsumed(found: list[str]) -> list[str]:
    """Remove a term when a MORE SPECIFIC one is also present.

    A real consultation produced ["chest pain", "pain", "palpitations",
    "symptoms", "heart"]. Only the specific ones are clinically useful;
    "pain" beside "chest pain" is not a second finding, and "heart" is an
    organ rather than a symptom.
    """
    out = []
    for term in found:
        if term in _TOO_GENERIC and any(
                other != term and term in other for other in found):
            continue                       # "pain" when "chest pain" exists
        if term in _TOO_GENERIC and len(term.split()) == 1:
            continue                       # bare organ / catch-all noun
        out.append(term)
    return out


def scan_advice(text: str) -> list[str]: #
    """Dietary and lifestyle instructions named in this segment.

    These terms were already classified as NON_CLINICAL - correctly, they
    are not symptoms - but that only kept them OUT of the symptom list.
    Nothing carried them anywhere, so "বেশি সাবান মাখবেন না আর প্রচুর জল
    খাবেন" was simply lost. On a skin or diabetes consultation the advice
    is half the prescription.
    """ #
    # honour_negation=False: negation suppression exists so a REFUSED
    # order is not recorded as an order ("টেস্ট দিচ্ছি না"). Advice is the
    # opposite case - "সাবান মাখবেন না", "খুঁটবেন না" are prohibitions, and
    # the না IS the instruction. Suppressing them deleted the advice.
    return [t for t in #
            _ngram_scan_all(text, _CURATED_TERM_LOOKUP, honour_negation=False) #
            if t in NON_CLINICAL_TERMS and t not in _NOT_ADVICE] #


# Dosage forms and drug classes also live in NON_CLINICAL_TERMS. They are
# not advice - "tablet" is not something a doctor tells you to do.
_NOT_ADVICE = frozenset({ #
    "tablet", "syrup", "injection", "ointment", "eye drops", "ear drops", #
    "nasal spray", "medicine", "antibiotic", "painkiller", "antacid", #
    "steroid", "vitamin supplement", "antihistamine", "prescription", #
}) #


def scan_symptoms(text: str) -> list[str]:
    """Symptoms named in the text - excludes conditions, advice and classes.

    Scans the CURATED table only. See _CURATED_TERM_LOOKUP for why the
    imported vocabulary cannot be a source of findings.
    """
    found = [t for t in _ngram_scan_all(text, _CURATED_TERM_LOOKUP)
             if t not in CONDITIONS and t not in NON_CLINICAL_TERMS]
    return _drop_subsumed(found) #