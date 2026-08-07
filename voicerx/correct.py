"""Post-ASR correction layer, built from real human corrections.

Why this exists: analysis of 67 human-corrected segments from 10 real
consultations showed the ASR's errors are NOT random - they cluster hard on
English medical loanwords transliterated into Bengali. The model mangles
exactly the vocabulary that matters most for a prescription:

    ক্লাব          -> ক্ল্যাভাম      (Clavam, an antibiotic)
    আস্কোরিল       -> এস্কোরিল       (Ascoril, a cough syrup)
    ভ্যালুম        -> ভ্যালিয়াম     (Valium)
    মল্টিকুলার     -> মন্টিকুলাষ্ট   (Montelukast)
    প্যারাসিটাম    -> প্যারাসিটামল   (Paracetamol)
    ইনফেকটআন       -> ইনফেকশন        (infection, seen 3x)
    প্রেস্কৃপ্তিন  -> প্রেসক্রিপশন   (prescription, seen 2x)

Two confidence tiers, deliberately kept separate:

  HIGH   - substitutions observed more than once across different
           consultations. Applied silently.
  MEDIUM - single observations with high character similarity (>=0.6),
           i.e. plausible phonetic confusions rather than coincidence.
           Applied, but every hit is recorded so Node 3 can flag the
           segment and a human can confirm the swap was right.

Anything not in these tables is left alone. A correction layer that guesses
is worse than none at all - the whole point of this pipeline is to not
invent clinical content.
"""
from __future__ import annotations

import dataclasses

# Observed >1x across different consultations - safe to apply silently.
HIGH_CONFIDENCE = {
    "ইনফেকটআন": "ইনফেকশন",
    "লেট": "ওয়েট",
    "প্রেস্কৃপ্তিন": "প্রেসক্রিপশন",
    "বন্দগে": "ব্যাণ্ডেজ",
    "ক্লাব": "ক্ল্যাভাম",
    "নেবুলাইজাতিওন": "নেবুলাইজেশন",
}

# Single observations, high character similarity. Applied but recorded.
# Drug and clinical terms are the bulk of these, which is the point.
MEDIUM_CONFIDENCE = {
    "প্যারাসিটাম": "প্যারাসিটামল",
    "আস্কোরিল": "এস্কোরিল",
    "ভ্যালুম": "ভ্যালিয়াম",
    "মল্টিকুলার": "মন্টিকুলাষ্ট",
    "অযান্টিবায়োটিক": "এন্টিবায়োটিক",
    "হমিপ্যাতি": "হোমিপ্যাথি",
    "প্রেস্ক্রিপ্তিন": "প্রেসক্রিপশন",
    "ফার্মাচী": "ফার্মেসী",
    "দোস": "ডোস",
    "গিল্ডে": "গিলতে",
    "কন্টিনু": "কন্টিনিউ",
    "চাঞ্জ": "চেঞ্জ",
    "ঠাংক": "থ্যাংক",
    "থাকবে": "থাকবেন",
    "অ্যাস": "অ্যাজ",
    "সমে": "সেম",
}

# Deliberately EXCLUDED despite appearing in the mined patterns, because
# applying them would corrupt correct text far more often than it helps:
#   'বুকে' -> 'কে', 'বাবা' -> 'বা', 'ইস' -> 'ই', 'বুঝতে' -> 'ঝতে',
#   'নোন' -> 'নন', 'উস' -> 'ইউস', 'অ্যালার্জির' -> 'যালার্জির'
# These are truncations/artifacts of one specific utterance, not general
# phonetic confusions - 'বুকে' (chest) is a common, correct Bengali word.


@dataclasses.dataclass
class CorrectionResult:
    text: str
    high_conf_applied: list[tuple[str, str]]
    medium_conf_applied: list[tuple[str, str]]

    @property
    def any_applied(self) -> bool:
        return bool(self.high_conf_applied or self.medium_conf_applied)

    @property
    def needs_confirmation(self) -> bool:
        """Medium-confidence swaps should be eyeballed by a human."""
        return bool(self.medium_conf_applied)


def correct_transcript(text: str) -> CorrectionResult:
    words = text.split()
    high_applied: list[tuple[str, str]] = []
    medium_applied: list[tuple[str, str]] = []
    out: list[str] = []

    for w in words:
        if w in HIGH_CONFIDENCE:
            out.append(HIGH_CONFIDENCE[w])
            high_applied.append((w, HIGH_CONFIDENCE[w]))
        elif w in MEDIUM_CONFIDENCE:
            out.append(MEDIUM_CONFIDENCE[w])
            medium_applied.append((w, MEDIUM_CONFIDENCE[w]))
        else:
            out.append(w)

    return CorrectionResult(
        text=" ".join(out),
        high_conf_applied=high_applied,
        medium_conf_applied=medium_applied,
    )
