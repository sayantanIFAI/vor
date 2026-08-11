#!/usr/bin/env python3
"""
TASK 7: Audit skeleton collisions against Bengali vocabulary.
Finds drug skeletons that collide with everyday Bengali words.
"""

import json
from pathlib import Path
from collections import defaultdict

# Seed list of common Bengali words (known to have caused issues)
_SEED_BENGALI_WORDS = {
    "পিঠ",      # back, body part
    "বুঝলাম",   # I understood
    "খাবার",    # food
    "খাওয়া",    # eat
    "জল",       # water
    "ডাক্তার",   # doctor
    "হাসপাতাল", # hospital
    "বেদনা",    # pain
    "ব্যথা",    # ache
    "জ্বর",     # fever
    "কাশি",     # cough
    "ঠান্ডা",    # cold
    "ঘুম",      # sleep
    "দাঁত",     # tooth
    "চোখ",      # eye
    "কান",      # ear
    "নাক",      # nose
    "পায়",      # leg/foot
    "হাত",      # hand
    "মাথা",     # head
    "পেট",      # stomach
    "হৃদয়",     # heart
    "রক্ত",     # blood
    "শ্বাস",     # breath
    "স্ট্রেন",   # strain (body symptom, not drug)
    "খেয়েছি",   # (I) ate
    "করছেন",    # (you) are doing
    "হয়েছে",    # (has) happened
    "খাবেন",    # (will) eat
    "দিচ্ছি",    # (I) am giving
    "দেখেছেন",   # (have) seen
    "করবেন",    # (will) do
    "দিলেম",    # (I) gave
    "দেখলাম",   # (I) saw
    "বলেছি",    # (I) said
    "এসেছি",    # (I) came
    "গেছি",     # (I) went
    "খেয়ে",     # (eating)
    "করে",      # (doing)
    "হয়ে",      # (become/happening)
    "দিয়ে",     # (with/giving)
}


def load_bengali_words(wordlist_path: str | None = None) -> set[str]:
    """Load common Bengali words from file or use seed."""
    words = set()

    if wordlist_path and Path(wordlist_path).exists():
        try:
            with open(wordlist_path, 'r', encoding='utf-8') as f:
                for line in f:
                    word = line.strip().split()[0] if line.strip() else ""
                    if word:
                        words.add(word)
            print(f"✓ Loaded {len(words)} words from {wordlist_path}")
        except Exception as e:
            print(f"⚠️ Failed to load {wordlist_path}: {e}. Using seed list.")
            return _SEED_BENGALI_WORDS.copy()
    else:
        if wordlist_path:
            print(f"⚠️ {wordlist_path} not found. Using seed list of {len(_SEED_BENGALI_WORDS)} words.")
        return _SEED_BENGALI_WORDS.copy()

    return words


def fold(text: str) -> str:
    """
    Bengali phonetic folding: collapse vowels and diacritics to consonant skeleton.
    Match the same logic as voicerx/glossary.py:fold()
    """
    # Vowels and diacritics to remove
    vowels = "আইঈউঊঋএঐওৌ়্ািীুূেৈোৌ"

    result = ""
    for char in text:
        if char not in vowels:
            result += char

    return result.lower()


def audit_skeleton_collisions(
    wordlist_path: str | None = None,
    min_skeleton_length: int = 5
) -> dict:
    """
    Find all drug skeleton keys that collide with everyday Bengali words.

    Returns structure:
    {
        "total_drugs": int,
        "collisions": {
            "bjlm": {
                "skeleton": "bjlm",
                "drug_name": "Valium",
                "drug_bengali": ["ভ্যালুম"],
                "collides_with": ["বুঝলাম"],
                "action": "BLOCK"
            },
            ...
        },
        "summary": {
            "total_collisions": int,
            "to_block": int,
            "to_monitor": int
        }
    }
    """

    # Load Bengali word list
    bengali_words = load_bengali_words(wordlist_path)

    # Import drug data from glossary
    try:
        from voicerx.glossary import DRUGS
    except ImportError:
        print("ERROR: Cannot import DRUGS from voicerx.glossary")
        print("Make sure you're in the repo directory and voicerx is in PYTHONPATH")
        return {}

    # Build skeleton→drug mapping
    skeleton_to_drug = {}
    for drug_list in DRUGS.values():
        for drug in drug_list:
            # Fold the generic name
            skeleton = fold(drug.generic)
            if len(skeleton) >= min_skeleton_length:
                if skeleton not in skeleton_to_drug:
                    skeleton_to_drug[skeleton] = drug

    # Find collisions
    collisions = {}

    for word in bengali_words:
        word_skeleton = fold(word)

        # Check if this skeleton is already assigned to a drug
        if word_skeleton in skeleton_to_drug:
            drug = skeleton_to_drug[word_skeleton]

            if word_skeleton not in collisions:
                collisions[word_skeleton] = {
                    "skeleton": word_skeleton,
                    "drug_name": drug.generic,
                    "drug_bengali": list(drug.bengali) if drug.bengali else [],
                    "collides_with": [],
                    "collision_count": 0
                }

            collisions[word_skeleton]["collides_with"].append(word)
            collisions[word_skeleton]["collision_count"] += 1

    # Assign actions
    for skeleton, info in collisions.items():
        if info["collision_count"] >= 2:
            info["action"] = "BLOCK"
        else:
            info["action"] = "MONITOR"

    # Summarize
    summary = {
        "total_drugs": sum(len(v) for v in DRUGS.values()),
        "total_bengali_words": len(bengali_words),
        "total_collisions": len(collisions),
        "to_block": sum(1 for v in collisions.values() if v["action"] == "BLOCK"),
        "to_monitor": sum(1 for v in collisions.values() if v["action"] == "MONITOR"),
    }

    return {
        "collisions": collisions,
        "summary": summary
    }


def print_report(audit_result: dict):
    """Print human-readable collision report."""

    if not audit_result or "collisions" not in audit_result:
        print("No collisions found or audit failed.")
        return

    collisions = audit_result["collisions"]
    summary = audit_result["summary"]

    print("\n" + "=" * 80)
    print("SKELETON COLLISION AUDIT")
    print("=" * 80)

    print(f"\nStatistics:")
    print(f"  Total drugs: {summary['total_drugs']}")
    print(f"  Bengali words checked: {summary['total_bengali_words']}")
    print(f"  Skeletons with collisions: {summary['total_collisions']}")
    print(f"    → To BLOCK: {summary['to_block']}")
    print(f"    → To MONITOR: {summary['to_monitor']}")

    if collisions:
        print(f"\n{'BLOCKED':-^80}")
        for skeleton in sorted(collisions.keys()):
            info = collisions[skeleton]
            if info["action"] == "BLOCK":
                print(f"\n  Skeleton: {skeleton}")
                print(f"  Drug: {info['drug_name']}")
                if info["drug_bengali"]:
                    print(f"  Bengali forms: {', '.join(info['drug_bengali'])}")
                print(f"  Collides with: {', '.join(info['collides_with'])}")

        print(f"\n{'MONITORED':-^80}")
        for skeleton in sorted(collisions.keys()):
            info = collisions[skeleton]
            if info["action"] == "MONITOR":
                print(f"\n  Skeleton: {skeleton}")
                print(f"  Drug: {info['drug_name']}")
                if info["drug_bengali"]:
                    print(f"  Bengali forms: {', '.join(info['drug_bengali'])}")
                print(f"  Collides with: {', '.join(info['collides_with'])}")

    print("\n" + "=" * 80)
    print("RECOMMENDED ACTIONS")
    print("=" * 80)

    print(f"""
1. Add to glossary.py _BLOCKED_SKELETONS:
   _BLOCKED_SKELETONS = frozenset([
""")

    for skeleton in sorted(collisions.keys()):
        info = collisions[skeleton]
        if info["action"] == "BLOCK":
            print(f"""       "{skeleton}",  # {info['drug_name']} collides with {info['collides_with'][0]}""")

    print("""   ])

2. Run regression tests to verify drugs still resolve via other paths

3. Before production deployment, verify this script runs without errors
""")


if __name__ == "__main__":
    import sys

    wordlist_path = None
    if len(sys.argv) > 1:
        wordlist_path = sys.argv[1]

    print("Running skeleton collision audit...")
    audit_result = audit_skeleton_collisions(wordlist_path)
    print_report(audit_result)

    # Save results
    output_file = "/workspace/collision_audit.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(audit_result, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Results saved to {output_file}")
