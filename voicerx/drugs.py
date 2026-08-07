"""Node 3 support: a known-drug list for cross-checking LLM output.

This is NOT a claim of complete drug coverage - it's a safety net. Anything
the extraction LLM names as a specific drug that does not match this list
gets auto-flagged for human review, on top of whatever the LLM already
flagged itself. Given the proven failure mode this session (a garbled ASR
fragment resolved into "Naloxone" - a real, wrong, dangerous drug), the
purpose here is to catch exactly that pattern: a confident-sounding but
unrecognized drug name, even if the LLM didn't flag it itself.

Generic names + common Indian brand names, covering the drug classes most
likely to appear in a general-OPD Bengali consultation. Extend this list
before deploying against a wider patient population - it is deliberately
conservative, not exhaustive.
"""

KNOWN_DRUGS = {
    # analgesics / antipyretics
    "paracetamol", "acetaminophen", "crocin", "dolo", "calpol", "ibuprofen",
    "combiflam", "diclofenac", "aspirin", "aceclofenac", "naproxen",
    "tramadol", "mefenamic acid",

    # antibiotics
    "amoxicillin", "amoxiclav", "azithromycin", "azithral", "ciprofloxacin",
    "cifran", "cefixime", "taxim", "metronidazole", "flagyl", "doxycycline",
    "levofloxacin", "ofloxacin", "clindamycin", "erythromycin",
    "co-amoxiclav", "ampicillin", "cephalexin", "norfloxacin",

    # antacids / PPI / GI
    "omeprazole", "pantoprazole", "pantocid", "ranitidine", "domperidone",
    "domstal", "ondansetron", "emeset", "esomeprazole", "rabeprazole",
    "sucralfate", "digene", "gelusil", "eno", "loperamide", "ors",
    "electral", "zinc sulphate",

    # antihistamines / allergy / cold
    "cetirizine", "levocetirizine", "chlorpheniramine", "cheston",
    "montelukast", "fexofenadine", "loratadine", "ambroxol", "bromhexine",
    "levosalbutamol", "salbutamol", "budesonide", "phenylephrine",

    # antidiabetic
    "metformin", "glimepiride", "glibenclamide", "insulin", "sitagliptin",
    "voglibose", "gliclazide",

    # antihypertensive / cardiac
    "amlodipine", "telmisartan", "losartan", "atenolol", "metoprolol",
    "ramipril", "enalapril", "hydrochlorothiazide", "atorvastatin",
    "rosuvastatin", "clopidogrel", "aspirin cardio",

    # vitamins / supplements
    "vitamin b complex", "vitamin d", "vitamin d3", "calcium", "iron",
    "folic acid", "zincovit", "becosules", "shelcal", "vitamin c",

    # topical / other common
    "betadine", "povidone iodine", "silver sulfadiazine", "clotrimazole",
    "hydrocortisone", "mupirocin",

    # Found in the user's own corrections of these 10 consultations - real
    # drugs actually prescribed that this list was missing. Norflox-TZ in
    # particular is the drug behind the fragment that an earlier, looser
    # prompt hallucinated into "Naloxone".
    "norflox", "norflox-tz", "norflox tz", "norfloxacin", "tinidazole",
    "clavam", "ascoril", "valium", "diazepam", "montelukast",
}


def normalize(name: str) -> str:
    return name.strip().lower()


def is_known_drug(name: str) -> bool:
    n = normalize(name)
    if not n:
        return False
    if n in KNOWN_DRUGS:
        return True
    # loose substring match: catches "Tab. Paracetamol 500mg" etc.
    return any(known in n or n in known for known in KNOWN_DRUGS)
