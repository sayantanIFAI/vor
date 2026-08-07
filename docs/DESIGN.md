# Voice-to-Rx — Design

How the system is built and **why**. For setup and running, see
[LOCAL_SETUP.md](LOCAL_SETUP.md).

Every design choice below is followed by the evidence that produced it.
Where something is unproven, it says so.

---

## 1. The problem

A Bengali doctor–patient consultation must become a structured English
prescription — symptoms, diagnosis, medications, lab tests, follow-up —
**fully offline**, on hardware a clinic can own.

Two properties make this different from ordinary transcription:

**It is safety-critical.** A wrong drug name on a prescription can hurt
someone. This constrains the architecture more than accuracy targets do.

**The input is bad and will stay bad.** Bengali medical ASR runs at ~25.7%
WER on this audio. Drug names are the worst case, because they are English
words spoken in Bengali and transliterated inconsistently every time. The
design assumes garbled input as the normal case, not the error case.

---

## 2. Architecture

```
  audio
    │
    ▼
 ┌─────────────────┐
 │ 1  VAD          │  Silero, max 25 s segments
 └────────┬────────┘
          │  segments
          ▼
 ┌─────────────────┐
 │ 2  ASR          │  IndicConformer, CTC + RNNT run independently
 └────────┬────────┘  agreement between them = a confidence signal
          │  Bengali text
          ├──────────────────────────────┐
          ▼                              ▼
 ┌─────────────────┐            ┌─────────────────┐
 │ 2a Correction   │            │ 4  GAZETTEER    │
 │    learned      │            │    reads the    │
 │    ASR fixes    │            │    BENGALI      │
 └────────┬────────┘            │    directly     │
          ▼                     └────────┬────────┘
 ┌─────────────────┐                     │
 │ 2b Translate    │  optional           │  labs, terms
 │    bn -> en     │  (--translate)      │
 └────────┬────────┘                     │
          ▼                              │
 ┌─────────────────┐                     │
 │ 3  EXTRACT      │  Qwen2.5 proposes   │
 │    (the SLM)    │  a prescription     │
 └────────┬────────┘                     │
          │  proposal                    │
          ▼                              ▼
 ┌──────────────────────────────────────────┐
 │ 5  GATE + VALIDATE                       │
 │    the gazetteer DECIDES what is a drug  │
 └────────┬─────────────────────────────────┘
          ▼
    prescription + review flags
```

### The organising principle

> **The model proposes. The gazetteer decides.**

"Is this string a drug?" is a **closed-set** question — drug names are
finite and enumerable. Asking a 7B model to judge set membership is using
the wrong tool for a question that can simply be looked up.

This was tested, not assumed. Two rounds of prompt-tuning aimed at reducing
false-positive medications produced **under-extraction** instead: the model
started hiding real symptoms in `raw_uncertain_terms` rather than becoming
more accurate. Moving the decision out of the model and into a lookup table
fixed it in one step.

---

## 3. Design decisions

### 3.1 Two ASR decoders, not one

CTC and RNNT share an encoder but decode independently. Where they disagree,
something is wrong — and that is a signal **neither decoder's own confidence
score can provide**. Word-level Jaccard agreement below 0.5 on a segment
with real content raises a review flag.

Measured: RNNT used on 183/188 segments, CTC fallback on 5.

> RNNT must use `strategy="greedy"`. The default `greedy_batch` returns
> empty text on longer segments. This cost hours to find.

### 3.2 Per-segment extraction, never whole-file

Unsegmented audio was the root cause of hallucination in an earlier version:
a long run-on transcript caused the model to invent a symptom with no
textual basis, and to resolve a garbled fragment into **"Naloxone"** — a
real, wrong, dangerous drug.

Per-segment extraction keeps each model call short and grounded. It is also
what makes the 3-second interaction budget reachable (§6).

### 3.3 Phonetic folding, not string equality

The gazetteer originally matched on NFC + lowercase. That failed on real
audio four ways at once, and the variants multiply combinatorially:

| | example | why |
|---|---|---|
| spacing | `সি বি সি` vs `সিবিসি` | clinicians spell acronyms out; the ASR writes each letter separately |
| half-letters | `মন্টিকুলাষ্ট` vs `মনটিকুলাসট` | conjuncts carry a hasant the ASR drops or adds |
| dialect | `শ` `ষ` `স` | one sound in speech; also `ণ/ন`, `ড়/র`, `য/জ` |
| accent | `ব` vs `ভ` | aspiration is the least stable feature across speakers |

The spacing case alone caused **zero** lab tests to match across 255 real
segments.

Both the gazetteer and the ASR text now pass through the same lossy fold, so
one entry covers a whole spelling family. It recovers `CBC` from the ASR's
own mangled `সি ভিসিটা`.

**The fold is lossy, so it is policed.** `collisions()` checks every entry
against every other at import. A drug folding onto another drug, or onto a
symptom, is a bug — not a tuning knob.

### 3.4 Token n-grams, not substrings

Substring matching over space-stripped folded text let keys straddle word
boundaries. The `EEG` key matched inside `এই জিভটা বার কর` — "stick your
tongue out". Matching now aligns to whole word groups, with a trailing-suffix
allowance because Bengali is agglutinative (`সি বি সি টা`).

### 3.5 Three tiers, not an allowlist

A strict allowlist was tried first and was too harsh: it correctly rejected
13 of 18 false positives but also discarded `Montuculast` and
`মন ডেগুলাস্ট` — both real Montelukast prescriptions.

| tier | meaning | behaviour |
|---|---|---|
| **verified** | exact gazetteer hit after folding | enters `medications[]` |
| **probable** | close to a real drug (`Montuculast` → Montelukast, 0.82) | **kept**, canonical name attached as a *proposal* in a separate field |
| **rejected** | not a drug | moved to `rejected_terms` — recorded, never deleted |

**`probable` is never auto-applied.** The canonical name goes in its own
field; `drug` always holds what was actually said. Auto-resolving a drug
name is the one thing this pipeline must not do — that is what produced
"Naloxone".

**`rejected` is never silently dropped.** A silent deletion is
indistinguishable from a term that was never extracted, and if the gate is
wrong, `rejected_terms` is the only place the evidence survives. The UI
shows it in its own panel.

### 3.6 The fuzzy threshold is derived, not chosen

`SIMILARITY_FLOOR = 0.65`. Scored against every medication the SLM proposed
across 10 real consultations, the classes separate cleanly:

```
0.818  Montuculast      -> Montelukast      REAL DRUG
0.700  মন ডেগুলাস্ট      -> Montelukast      REAL DRUG
──────────────────────────────────────────  empty band
0.588  মেডিসিন           ~ Prednisolone      false positive
0.571  জিন টাকে          ~ Insulin glargine  false positive
0.500  Antibiotic       ~ Pantoprazole      false positive
```

0.65 sits in the gap. **Caveat:** derived from 13 non-exact samples. That is
the best evidence available, not a lot of it. Re-derive from the
distribution when a larger reviewed set exists — do not nudge it to fix one
case.

### 3.7 Trust hierarchy

Order of checks is a safety property, not an implementation detail:

```
1. curated drug table       highest trust - Bengali + department, hand-written
2. curated labs / terms     reject outright
3. imported brand register  174k entries, public dataset, unreviewed
4. combination products     "A/B" resolves when every component resolves
5. fuzzy                    proposes only, never decides
```

This order was wrong once. The imported register sat above the curated lab
table, so `electrolytes` — a lab test — matched a commercial product and was
returned as a **verified medication 227 times** in a 500-transcript run.
**Curated beats imported** is now explicit.

### 3.8 Translation is a bridge, not a rewrite

Qwen2.5-7B is weak at Bengali clinical text, and fine-tuning does not fix
that: teaching a model a language is a continued-pretraining problem
measured in billions of tokens, not a LoRA over a few thousand
consultations. Training it on 25.7%-WER transcripts would mostly teach
confident guessing.

So IndicTrans2 (offline, 200M) carries the **narrative** — symptoms,
diagnosis, summary. **Drug and lab names never pass through translation.**
The gazetteer reads the original Bengali, and MT is least reliable exactly
where stakes are highest: transliterated brand names are not really Bengali
words. A translation failure degrades readability; it cannot invent a drug.

Off by default. The Bengali-only path is the one actually verified; the
bridge is opt-in so the two can be compared rather than assumed.

### 3.9 Output is forced into English — two failures, two treatments

A live consultation returned this as the **summary**:

```
医生提到患者有多年的糖尿病，并进行了冠状动脉支架植入手术。
```

and `糖尿病`, `右脚疼痛`, `শ্বাসকষ্ট`, `লক্ষণগুলো` among the symptoms. Two
different failures, so `english.py` treats them differently:

| script | why it appeared | treatment |
|---|---|---|
| **Chinese** | Qwen2.5 is Chinese-origin and falls back to it on Bengali input | **dropped** — no legitimate use here, and no reviewer can read it |
| **Bengali** | real clinical content the model just didn't translate | **translated via the gazetteer** — `শ্বাসকষ্ট` → *breathlessness* |

Bengali the gazetteer can't translate is **not** deleted — it moves to
`raw_uncertain_terms`. Losing a symptom to a formatting rule would be
losing patient data.

> A CJK drop deliberately does **not** quote the offending text. The final
> CJK sweep would then delete the audit note itself — the summary vanished
> with no record at all in testing. The note says how many characters were
> removed instead.

Applied to output fields only. `source_transcript` stays Bengali: it is
evidence, and a reviewer must be able to check the original.

### 3.10 Dosing is spoken, not written

Prescriptions came back with empty frequency and duration. The cause isn't
extraction failure — a doctor says **"দুপুরে খাওয়ার পর"** (after lunch),
never "BD". The model, prompted for clinical shorthand, correctly returned
nothing rather than guessing.

`DOSING_TERMS` and `DURATION_TERMS` map spoken Bengali to standard
notation, with subsumption so "after dinner" doesn't also print "at night,
after food". These fill **only** empty fields — the model is never
overwritten.

### 3.11 What the gazetteer may and may not do

The brand register is used **only to validate a name the SLM already
proposed** — exact folded lookup. It is deliberately **not** used to scan
raw transcripts.

Many of its 174k entries are ordinary words. Scanning free text with it
would reproduce, at scale, a bug this project already hit: the Lactulose
brand *Looz* folds to `los`, so does the English word "loss", and
`hair loss` was resolving to a **verified medication**.

Cost of that restriction, measured on 500 transcripts: gazetteer-only drug
recall is 44% rather than ~54%. Accepted deliberately — the real path is
SLM-proposes → gate-validates, which scores 48/48.

---

## 4. Data model

`ExtractedRx` (see `schema.py`). Strict on the way out, lenient on the way
in — LLMs drift, so `from_llm_json` coerces common drift patterns rather
than failing.

> A single rejected field drops the **whole** extraction. When
> `confidence_note: null` was rejected, 4 of 16 segments were lost on one
> file — including one carrying a real symptom. Coercion is a safety
> feature, not convenience.

Fields that exist purely for review:

| field | why |
|---|---|
| `rejected_terms` | what the gate refused, so a human can catch the gate being wrong |
| `drug_candidates` | proposed names for garbled tokens, never applied |
| `review_reasons` | re-derived from hard rules, **not** the model's self-assessment |
| `transcript_en` | kept alongside the Bengali so a reviewer can check the translation |
| `tier`, `canonical` | separate from `drug`, so a proposal is never mistaken for what was said |

**The model's own confidence is not trusted.** It was observed correctly
flagging one garbled term while confidently mis-resolving another *in the
same response*.

---

## 5. Failure modes

| failure | mitigation |
|---|---|
| ASR garbles a drug name | fold absorbs spelling/dialect/accent; fuzzy proposes; human confirms |
| SLM invents a drug | gate rejects anything not in the gazetteer |
| SLM proposes advice as medication | clinical-terms table rejects positively |
| Gate wrongly rejects a real drug | surfaced in `rejected_terms` panel, never deleted |
| Extraction fails on a segment | placeholder emitted — never silently dropped, keeps arrays aligned |
| Decoders disagree | agreement score raises a review flag |
| Negated order read as an order | negation/interrogative scope on gazetteer scans |

**Known gap, unmitigated:** negation scope is a heuristic, not a parser. It
handles `টেস্ট দিচ্ছি না` but will not handle arbitrary constructions.

---

## 6. Performance model

A 32-segment consultation measured **61.8 s** end to end on an RTX 4090 —
almost entirely LLM (~2 s × 32).

| step | cost |
|---|---|
| `judge_medication` | 0.007 ms |
| `scan_labs` per segment | 0.76 ms |
| gazetteer import | 1.3 s (one-time) |
| ASR per segment | ~0.3 s |
| **LLM per segment** | **~2 s** |
| merge + validate + render | ~50 ms |

The gate and glossary are free at these timescales. **Optimise the LLM call
count, nothing else.**

The 3-second interaction target is met architecturally, not by tuning:
extract each segment *during* the consultation, so the click only merges,
validates and renders. The per-segment design already supports this;
`server.py` has not yet been switched to streaming capture.

---

## 7. What is not proven

Stated plainly so nobody mistakes these for settled.

- **`glossary.py` has not been clinically reviewed.** Drafted by an AI. The
  Bengali transliterations decide what is recognised at all.
- **The 174k brand register is unreviewed** — a public dataset, not a
  validated formulary.
- **Accuracy figures come from 255 segments.** Far too few to support a
  percentage claim. The stated 98% target is not currently measurable.
- **The translation bridge is unmeasured.** Built, not yet A/B'd.
- **The 5,000-record cardiology set is template-generated** — 113k turns,
  only 9.4k unique. Useful for evaluation; misleading as training data.
