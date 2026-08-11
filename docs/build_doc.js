const fs = require('fs');
const d = require('docx');
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
  Table, TableRow, TableCell, WidthType, ShadingType, BorderStyle,
  PageBreak, TableOfContents, LevelFormat, PositionalTab,
  PositionalTabAlignment, PositionalTabLeader, convertInchesToTwip
} = d;

// A4 portrait, 1" margins -> content width
const W = 9020;

const NAVY   = '1F3864';
const SLATE  = '44546A';
const RED    = 'C00000';
const GREEN  = '375623';
const AMBER  = '7F6000';
const LGREY  = 'F2F2F2';
const HDRBG  = '1F3864';

// ---------- helpers ----------------------------------------------------
const p = (text, opts = {}) => new Paragraph({
  spacing: { after: opts.after ?? 120, before: opts.before ?? 0, line: 276 },
  alignment: opts.align,
  indent: opts.indent,
  border: opts.border,
  shading: opts.shading,
  children: [new TextRun({
    text, bold: opts.bold, italics: opts.italics, size: opts.size ?? 21,
    color: opts.color, font: opts.font
  })]
});

// rich paragraph from [{t, b, i, c, f}] runs
const rp = (runs, opts = {}) => new Paragraph({
  spacing: { after: opts.after ?? 120, before: opts.before ?? 0, line: 276 },
  alignment: opts.align,
  indent: opts.indent,
  shading: opts.shading,
  border: opts.border,
  children: runs.map(r => new TextRun({
    text: r.t, bold: r.b, italics: r.i, color: r.c,
    font: r.f, size: r.s ?? opts.size ?? 21
  }))
});

const h1 = (text) => new Paragraph({
  heading: HeadingLevel.HEADING_1, spacing: { before: 360, after: 160 },
  children: [new TextRun({ text, bold: true, size: 32, color: NAVY })]
});
const h2 = (text) => new Paragraph({
  heading: HeadingLevel.HEADING_2, spacing: { before: 280, after: 120 },
  children: [new TextRun({ text, bold: true, size: 26, color: NAVY })]
});
const h3 = (text) => new Paragraph({
  heading: HeadingLevel.HEADING_3, spacing: { before: 220, after: 100 },
  children: [new TextRun({ text, bold: true, size: 23, color: SLATE })]
});

// monospace block (diagrams / code)
const mono = (lines, opts = {}) => lines.map((l, i) => new Paragraph({
  spacing: { after: i === lines.length - 1 ? 140 : 0, before: i === 0 ? 60 : 0, line: 240 },
  shading: { type: ShadingType.CLEAR, fill: opts.fill ?? LGREY },
  indent: { left: 170, right: 170 },
  children: [new TextRun({ text: l || ' ', font: 'Consolas', size: opts.size ?? 17,
                           color: opts.color ?? '1A1A1A' })]
}));

const bullet = (text, opts = {}) => new Paragraph({
  numbering: { reference: 'bullets', level: opts.level ?? 0 },
  spacing: { after: 70, line: 276 },
  children: [new TextRun({ text, size: 21, bold: opts.bold })]
});
const bulletR = (runs, opts = {}) => new Paragraph({
  numbering: { reference: 'bullets', level: opts.level ?? 0 },
  spacing: { after: 70, line: 276 },
  children: runs.map(r => new TextRun({ text: r.t, bold: r.b, italics: r.i,
                                        color: r.c, font: r.f, size: 21 }))
});
const step = (text) => new Paragraph({
  numbering: { reference: 'steps', level: 0 },
  spacing: { after: 80, line: 276 },
  children: [new TextRun({ text, size: 21 })]
});

const cell = (content, o = {}) => new TableCell({
  width: { size: o.w, type: WidthType.DXA },
  shading: o.fill ? { type: ShadingType.CLEAR, fill: o.fill } : undefined,
  margins: { top: 60, bottom: 60, left: 110, right: 110 },
  verticalAlign: 'center',
  children: Array.isArray(content) ? content : [
    new Paragraph({
      spacing: { after: 0, line: 252 },
      alignment: o.align,
      children: [new TextRun({
        text: String(content), bold: o.b, size: o.s ?? 19,
        color: o.c, font: o.f
      })]
    })
  ]
});

// table: headers[], rows[][], widths[]
const table = (headers, rows, widths, opts = {}) => new Table({
  columnWidths: widths,
  width: { size: widths.reduce((a, b) => a + b, 0), type: WidthType.DXA },
  borders: {
    top:    { style: BorderStyle.SINGLE, size: 2, color: 'BFBFBF' },
    bottom: { style: BorderStyle.SINGLE, size: 2, color: 'BFBFBF' },
    left:   { style: BorderStyle.SINGLE, size: 2, color: 'BFBFBF' },
    right:  { style: BorderStyle.SINGLE, size: 2, color: 'BFBFBF' },
    insideHorizontal: { style: BorderStyle.SINGLE, size: 2, color: 'D9D9D9' },
    insideVertical:   { style: BorderStyle.SINGLE, size: 2, color: 'D9D9D9' },
  },
  rows: [
    new TableRow({
      tableHeader: true,
      children: headers.map((hh, i) =>
        cell(hh, { w: widths[i], fill: HDRBG, b: true, c: 'FFFFFF', s: 19 }))
    }),
    ...rows.map((r, ri) => new TableRow({
      children: r.map((c, i) => {
        const isObj = c && typeof c === 'object' && !Array.isArray(c);
        const val = isObj ? c.t : c;
        return cell(val, {
          w: widths[i],
          fill: ri % 2 ? 'FAFAFA' : undefined,
          b: isObj ? c.b : (opts.firstBold && i === 0),
          c: isObj ? c.c : undefined,
          f: isObj ? c.f : undefined,
          s: opts.s
        });
      })
    }))
  ]
});

// callout box
const callout = (label, text, color) => new Table({
  columnWidths: [W],
  width: { size: W, type: WidthType.DXA },
  borders: {
    top:    { style: BorderStyle.SINGLE, size: 2, color: color },
    bottom: { style: BorderStyle.SINGLE, size: 2, color: color },
    left:   { style: BorderStyle.SINGLE, size: 18, color: color },
    right:  { style: BorderStyle.SINGLE, size: 2, color: color },
    insideHorizontal: { style: BorderStyle.NONE },
    insideVertical: { style: BorderStyle.NONE },
  },
  rows: [new TableRow({
    children: [new TableCell({
      width: { size: W, type: WidthType.DXA },
      shading: { type: ShadingType.CLEAR, fill: 'FBFBFB' },
      margins: { top: 110, bottom: 110, left: 160, right: 140 },
      children: [
        new Paragraph({
          spacing: { after: 50 },
          children: [new TextRun({ text: label, bold: true, size: 19, color: color })]
        }),
        new Paragraph({
          spacing: { after: 0, line: 264 },
          children: [new TextRun({ text, size: 20 })]
        })
      ]
    })]
  })]
});

const spacer = (h = 120) => new Paragraph({ spacing: { after: h }, children: [] });
const pbreak = () => new Paragraph({ children: [new PageBreak()] });
const rule = () => new Paragraph({
  spacing: { before: 60, after: 160 },
  border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: NAVY } },
  children: []
});

// ======================================================================
const children = [];

// ---------- COVER ------------------------------------------------------
children.push(
  spacer(1700),
  p('VOICE-TO-RX', { size: 56, bold: true, color: NAVY, align: AlignmentType.CENTER, after: 60 }),
  p('Bengali Consultation to Structured Prescription', {
    size: 28, color: SLATE, align: AlignmentType.CENTER, after: 40 }),
  p('System Design & Implementation Guide', {
    size: 24, italics: true, color: SLATE, align: AlignmentType.CENTER, after: 500 }),
  new Paragraph({
    spacing: { after: 400 },
    alignment: AlignmentType.CENTER,
    border: { bottom: { style: BorderStyle.SINGLE, size: 8, color: NAVY } },
    children: []
  }),
);

children.push(
  table(
    ['', ''],
    [
      ['Document', { t: 'Design review (Part I) + implementation runbook (Part II)' }],
      ['Audience', { t: 'Chief AI Architect · AI Engineers' }],
      ['System', { t: 'IndicConformer ASR → Qwen2.5-7B extraction → clinical gazetteer' }],
      ['Repository', { t: 'github.com/sayantanIFAI/vor', f: 'Consolas' }],
      ['Commit', { t: '137d8d8', f: 'Consolas' }],
      ['Status', { t: 'Deployed and verified end-to-end on GPU', c: GREEN, b: true }],
      ['Date', { t: '11 August 2026' }],
    ],
    [2200, 6820], { firstBold: true }
  ),
  spacer(600)
);

children.push(
  callout('SAFETY POSTURE',
    'This system produces prescriptions. Its design assumes that a missing drug is recoverable ' +
    'and a wrong drug is not. Every uncertainty therefore surfaces as a visible blank or a review ' +
    'flag, never as a confident-looking guess. Section 5 records where that principle cost ' +
    'convenience, and Section 6 records where the supporting evidence is thin.',
    RED),
  pbreak()
);

// ---------- HOW TO READ ------------------------------------------------
children.push(
  h1('How to read this document'),
  rule(),
  p('This document serves two readers. Both parts are self-contained.'),
  table(
    ['Reader', 'Read', 'Purpose'],
    [
      [{ t: 'Chief AI Architect', b: true }, 'Part I (§1–§7)',
       'Architecture, the decision log with alternatives costed, threshold evidence and its limits, residual risk.'],
      [{ t: 'AI Engineer (new to the project)', b: true }, 'Part II (§8–§14)',
       'Prerequisites, local install, per-stage verification, repo map, common change recipes, troubleshooting.'],
      [{ t: 'Both', b: true }, '§2, §5, §6',
       'The governing principle, why each decision was made, and which constants must not be casually tuned.'],
    ],
    [2400, 1700, 4920]
  ),
  spacer(200),
  p('Conventions', { bold: true, size: 22, color: NAVY, before: 120 }),
  bullet('Monospaced grey blocks are commands, code, or diagrams — copy them verbatim.'),
  bullet('Red callouts are safety-critical. Ignoring one can put a wrong drug on a prescription.'),
  bullet('Amber callouts are traps that cost real debugging time on this project.'),
  bullet('Every threshold cites the measurement behind it. Where evidence is thin, it says so.'),
  spacer(150),
  callout('A NOTE ON THE EVIDENCE',
    'Numbers in this document come from real consultations, not synthetic tests. Where a sample ' +
    'is small, the sample size is stated next to the number. Two safety thresholds rest on about ' +
    'twelve observations each — see §6 before changing either.',
    AMBER),
  pbreak()
);

// ---------- TOC --------------------------------------------------------
children.push(
  h1('Contents'),
  rule(),
  new TableOfContents('Contents', { hyperlink: true, headingStyleRange: '1-3' }),
  pbreak()
);

// ======================================================================
// PART I
// ======================================================================
children.push(
  spacer(2200),
  p('PART I', { size: 44, bold: true, color: NAVY, align: AlignmentType.CENTER, after: 80 }),
  p('SYSTEM DESIGN', { size: 32, color: SLATE, align: AlignmentType.CENTER, after: 120 }),
  p('Architecture, decisions, and the evidence behind them', {
    size: 22, italics: true, color: SLATE, align: AlignmentType.CENTER }),
  pbreak()
);

// --- 1 problem
children.push(
  h1('1. The problem'),
  rule(),
  p('A doctor speaks Bengali. The output must be a prescription a pharmacist can fill. ' +
    'Three properties separate this from ordinary speech-to-text.'),
  h3('1.1  It is safety-critical in one direction'),
  p('A missing drug prompts a question. A wrong drug does not — it reads as a decision. ' +
    'This asymmetry drives every design choice that follows: the system is built to fail ' +
    'towards blanks and flags, never towards plausible-looking specifics.'),
  h3('1.2  Drug names are the highest-risk and worst-recognised field'),
  p('Bengali ASR garbles transliterated brand names badly. These are not edge cases; they are ' +
    'the normal condition of the input:'),
  ...mono([
    '  রসু ভাস্টাটিন      Rosuvastatin, split across two words',
    '  মন ডেগুলাস্ট        Montelukast, 0.70 similarity',
    '  মেট ফর্মিন          Metformin, split',
    '  Rasu Basta Tin     not a drug, not a brand, not a word',
  ]),
  h3('1.3  The answer set is closed'),
  p('"Is this a drug?" is enumerable. That single fact is what the entire design leans on — ' +
    'a closed-set question is looked up, not reasoned about.'),
  spacer(120),
  h2('1.4  The two failures that define the system'),
  callout('FAILURE 1 — NALOXONE',
    'An early version resolved a garbled ASR fragment into "Naloxone": a real drug, wrong ' +
    'patient, dangerous. Root cause: the extraction model pattern-matched garble to the closest ' +
    'real drug name it knew.',
    RED),
  spacer(120),
  callout('FAILURE 2 — ERYTHROMYCIN',
    'A urology prescription printed "Erythromycin", a drug the consultation never mentions. ' +
    'Root cause: the model invented the name, and the gate verified it against a 179,002-entry ' +
    'imported brand register nobody had clinically reviewed. A fabrication arrived wearing the ' +
    'same badge as a drug the doctor actually said. The gazetteer answered "is this a real drug"; ' +
    'it was never asked "was this one said".',
    RED),
  spacer(140),
  p('Both are the same class of error: a specific, plausible, wrong name presented with ' +
    'confidence. Everything in Part I exists to make that outcome structurally difficult.'),
  pbreak()
);

// --- 2 principle
children.push(
  h1('2. Governing principle'),
  rule(),
  new Paragraph({
    spacing: { before: 160, after: 200 },
    alignment: AlignmentType.CENTER,
    shading: { type: ShadingType.CLEAR, fill: 'EDF1F8' },
    border: {
      top: { style: BorderStyle.SINGLE, size: 8, color: NAVY },
      bottom: { style: BorderStyle.SINGLE, size: 8, color: NAVY },
    },
    children: [new TextRun({
      text: 'The model proposes.  The gazetteer decides.  A human disposes.',
      bold: true, size: 28, color: NAVY })]
  }),
  p('The LLM is good at reading narrative and bad at closed-set recall under garble. The ' +
    'gazetteer is the opposite. They are split by what each is actually good at, and neither is ' +
    'trusted to check itself.'),
  spacer(100),
  h2('2.1  Three corollaries'),
  table(
    ['Rule', 'What it means in the code'],
    [
      [{ t: 'Nothing is silently discarded', b: true },
       'Rejected drugs → rejected_terms. Unresolved labs → raw_uncertain_terms. Uncorroborated symptoms → symptoms_unconfirmed. A reviewer can always see what was removed and why.'],
      [{ t: 'Nothing uncertain is silently applied', b: true },
       'A below-certainty resolution stays a proposal: verified=False, review_reason set, heard_as preserves the original string.'],
      [{ t: 'Context reorders; it never lowers the bar', b: true },
       'Specialty decides WHICH real drug a garble is. It can never decide WHETHER a garble is a drug. Raw similarity must clear the floor unaided.'],
    ],
    [3000, 6020]
  ),
  spacer(160),
  callout('WHY THE THIRD RULE MATTERS MOST',
    'A neurology consultation must not start turning noise into antiepileptics. Department ' +
    'context adds +0.12 to ranking only — the raw score still has to clear the threshold on its ' +
    'own evidence. This is the core safety argument for context-aware matching.',
    RED),
  pbreak()
);

// --- 3 architecture
children.push(
  h1('3. Architecture'),
  rule(),
  h2('3.1  End-to-end flow'),
  ...mono([
    '   Audio (browser / file)',
    '        |',
    '        v',
    '   +----------------------+',
    '   |  Silero VAD          |   utterances, max 25s, 0.6s merge gap',
    '   +----------------------+',
    '        |',
    '        v',
    '   +----------------------+',
    '   |  IndicConformer      |',
    '   |   +-- CTC decoder    |----+',
    '   |   +-- RNNT decoder   |----+---> agreement score (Jaccard)',
    '   +----------------------+    |',
    '        |  RNNT if non-empty,  |',
    '        |  else CTC fallback   |',
    '        v                      |',
    '   +----------------------+    |',
    '   |  correct.py          |    |    learned ASR fixes',
    '   +----------------------+    |',
    '        |                      |',
    '        +----------+-----------+',
    '        |          |',
    '        v          v',
    '  +-----------+  +--------------------+',
    '  | Qwen2.5   |  | Gazetteer scanners |   drugs, labs, symptoms,',
    '  | per seg   |  | (deterministic)    |   conditions, advice, dosing',
    '  +-----------+  +--------------------+',
    '        |          |',
    '        +----+-----+     MERGE (union, not substitution)',
    '             |',
    '             v',
    '   +----------------------+',
    '   |  validate.py         |   gate | grounding | corroboration',
    '   |                      |   lab gate | recovery',
    '   +----------------------+',
    '             |',
    '             v',
    '     ExtractedRx + review flags',
  ]),
  spacer(120),
  h2('3.2  Why two independent paths'),
  p('The LLM path reads narrative. The gazetteer path reads the same Bengali text ' +
    'deterministically. Results are merged as a union, never substituted — so anything either ' +
    'path catches survives, and validation then judges the combined set.'),
  spacer(100),
  table(
    ['Observation', 'Consequence'],
    [
      ['The LLM missed EVERY lab order across 10 consultations',
       'The gazetteer found CBC inside the ASR\'s mangled "সি ভিসিটা". Without the second path, all lab orders are lost.'],
      ['The gazetteer cannot read narrative dosing',
       '"দুপুরে খাওয়ার পর" (after lunch) needs the LLM. Without the first path, dosing instructions are lost.'],
    ],
    [3400, 5620]
  ),
  spacer(160),
  p('Neither path alone is sufficient. That is the justification for the redundancy and its ' +
    'compute cost.'),
  spacer(140),
  h2('3.3  Live path vs. optional bridge'),
  ...mono([
    '   Bengali text ------------------> Qwen extraction     [LIVE]',
    '                \\                  ^',
    '                 \\                /',
    '                  -> IndicTrans2 -                      [OFF]',
    '                     (bn -> en)',
  ]),
  callout('DOCUMENTED, NOT ACTIVE',
    'server.py:56 builds VoiceToRxPipeline() with no translator. pipeline.py:36 states why: ' +
    'translator=None "keeps the Bengali-only path that was actually verified", so the bridge can ' +
    'be compared on the same audio rather than assumed to help. Qwen extracts from Bengali today. ' +
    'The IndicTrans2 model is staged on the pod and ready for that A/B.',
    AMBER),
  pbreak()
);

// --- 4 nodes
children.push(
  h1('4. Node design'),
  rule(),
  h2('4.1  Segmentation — why it happens before ASR'),
  p('Two independent failures both point at long audio, and segmentation fixes both causes at once.'),
  table(
    ['Failure', 'Evidence'],
    [
      ['RNNT silently drops content',
       'Frame-level alignment showed ZERO non-blank tokens for the first 78% of a 49-second file. Slicing the same audio decoded it correctly — proving a sequence-length bug, not a model-quality one.'],
      ['Long context increases hallucination',
       'A run-on 49-second block produced an invented symptom AND an invented drug name. The same content, isolated into utterances, did not.'],
    ],
    [2700, 6320]
  ),
  spacer(140),
  p('Silero VAD. max_segment_s = 25.0, merge_gap_s = 0.6. The merge gap keeps ' +
    '"um... four or five times" as one utterance instead of three fragments.'),
  spacer(160),

  h2('4.2  Dual decoder — CTC and RNNT'),
  p('Both decoders share an encoder but decode independently. All three options were considered:'),
  table(
    ['Option', 'Pros', 'Cons'],
    [
      [{ t: 'RNNT only', b: true },
       'More accurate on real segments. Correctly split "চার্জিনারস" into "চার্জ নার্স".',
       'Returns empty on ~3.5% of segments (2 of 57, both ≤3.1s).'],
      [{ t: 'CTC only', b: true },
       'Never empty. No length ceiling.',
       'Measurably less accurate on the same audio.'],
      [{ t: 'Both — CHOSEN', b: true, c: GREEN },
       'RNNT quality with a CTC safety net, PLUS a free disagreement signal.',
       'Doubles ASR compute (~2s of a ~45s pipeline — not the bottleneck).'],
    ],
    [1700, 3760, 3560]
  ),
  spacer(150),
  p('The third benefit is what justifies the cost. Decoder agreement is a confidence signal ' +
    'neither decoder can produce alone:'),
  ...mono([
    '  agreement = |words(ctc) ∩ words(rnnt)| / |words(ctc) ∪ words(rnnt)|',
    '',
    '  0.29   segment that produced hallucinated eye symptoms',
    '  0.56   median segment',
    '  0.50   LOW_AGREEMENT threshold -> demote symptoms below this',
  ]),
  p('Below the threshold the decoders could not agree what was said, so anything built on top ' +
    'is speculation. validate.py demotes symptoms from such segments to raw_uncertain_terms — ' +
    'demoted, not deleted.'),
  spacer(160),

  h2('4.3  Extraction — per segment, not whole transcript'),
  table(
    ['Option', 'Pros', 'Cons'],
    [
      [{ t: 'Whole transcript', b: true },
       'Model sees full context; can link symptom to diagnosis across the consultation.',
       'Reproduced hallucination. One failure loses the entire consultation.'],
      [{ t: 'Per segment — CHOSEN', b: true, c: GREEN },
       'Small, grounded context per call. A failure costs one segment.',
       'Cross-segment reasoning is lost. N times more LLM calls.'],
    ],
    [2100, 3560, 3360]
  ),
  spacer(150),
  p('The prompt encodes each lesson as a hard rule:'),
  bulletR([{ t: 'Rule 1 — Grounding. ', b: true },
           { t: 'Every symptom and medication must be traceable to words that actually appear in the transcript.' }]),
  bulletR([{ t: 'Rule 2 — Never resolve garble into a drug name. ', b: true },
           { t: 'Carries the Naloxone example verbatim, with the correct behaviour shown alongside the wrong one.' }]),
  bulletR([{ t: 'Rule 3b — Report every symptom actually stated. ', b: true },
           { t: 'Added after the opposite over-correction: on a cardiac consultation, sweating, palpitations and breathlessness were all spoken and all omitted — losing the three findings that make the diagnosis.' }]),
  bulletR([{ t: 'Rule 3c — Route and instructions are part of the prescription. ', b: true },
           { t: '"জিভের তলায়" (under the tongue) is sublingual, not an unknown term.' }]),
  spacer(120),
  callout('EXTRACTION FAILURE NEVER DROPS A SEGMENT',
    'A placeholder ExtractedRx is emitted carrying the transcript and a loud confidence_note. ' +
    'Silent drops previously lost real clinical content AND misaligned segments[] against ' +
    'extractions[] for every downstream caller that zips them together.',
    RED),
  pbreak()
);

// --- gazetteer recovery
children.push(
  h2('4.4  Gazetteer recovery — six scanners'),
  p('Each scanner was added to close a specific, measured loss. None is speculative.'),
  table(
    ['Scanner', 'The failure it closes'],
    [
      [{ t: 'scan_labs', f: 'Consolas', b: true }, 'The LLM missed EVERY lab order in 10 consultations.'],
      [{ t: 'scan_drugs_spoken', f: 'Consolas', b: true }, '"মেট ফর্মিন", "রসু ভাস্টাটিন", "মেটো প্রোল" sat in the transcript but were absent from medications[].'],
      [{ t: 'scan_symptoms', f: 'Consolas', b: true }, 'Colloquial complaints the model classed as chit-chat.'],
      [{ t: 'scan_conditions', f: 'Consolas', b: true }, 'Cataract was transcribed perfectly and recognised, yet the prescription returned a blank diagnosis — nothing carried the term to an output field.'],
      [{ t: 'scan_advice', f: 'Consolas', b: true }, 'Advice was understood, then dropped because the schema had nowhere to put it.'],
      [{ t: 'scan_dosing', f: 'Consolas', b: true }, '"দুপুরে খাওয়ার পর" returned blank — the model expects clinical shorthand.'],
    ],
    [2350, 6670]
  ),
  spacer(150),
  callout('scan_drugs_spoken, NOT scan_drugs',
    'The brand the doctor actually named has to survive. Scanning for the generic is what put ' +
    '"Nitroglycerin" on a prescription where the spoken word was সরবিট্রেট (Sorbitrate). ' +
    'Correct pharmacology, but it reads as a substitution — and it was reported as one.',
    AMBER),
  spacer(160),

  h3('4.4.1  The single-medication dosing rule'),
  ...mono([
    '   scan_dosing finds frequency / duration',
    '              |',
    '              v',
    '     how many medications in this segment?',
    '              |',
    '      +-------+--------+',
    '      |                |',
    '   exactly 1        2 or more',
    '      |                |',
    '      v                v',
    '  fill blanks      FILL NOTHING',
  ]),
  p('A segment\'s timing is a segment fact. Attributing it to a particular drug is a guess, and ' +
    'copying it to every drug made that guess silently, several times over. It produced a ' +
    'clinically wrong instruction on a real cardiology consultation:'),
  ...mono([
    '  "রোজ সকালে খাওয়ার পর ... ইকোস্পিডিন আর রসু ভাস্টা টিন ...',
    '   আর বুকে ব্যাথা উঠলে ... জিভের তলায় একটা সর্বিট্রেট"',
  ], { fill: 'FDF2F2' }),
  callout('WHY THIS IS NOT A COSMETIC BUG',
    'One sentence, two schedules: a daily tablet and an as-needed sublingual. "After breakfast" ' +
    'was copied onto the Sorbitrate — which is taken WHEN THE CHEST PAIN STARTS. A patient ' +
    'following that instruction takes it at breakfast and has none during angina. With several ' +
    'drugs present, the model\'s own attribution is the only one with sentence structure to go ' +
    'on, so nothing is filled in behind it. A blank prompts a question; a wrong schedule does not.',
    RED),
  pbreak()
);

// --- 5 gate
children.push(
  h1('5. The gate and validation layer'),
  rule(),
  h2('5.1  Medication gate — decision order'),
  ...mono([
    '  proposed name',
    '       |',
    '  empty / null? ------------------------------> REJECTED',
    '       |',
    '  fold + exact lookup --- exact --------------> VERIFIED   (1.00)',
    '       |               `- skeleton only ------> PROBABLE   (0.90)  vowels dropped',
    '       |',
    '  strip dosage form, retry --- resolves ------> VERIFIED / PROBABLE',
    '       |',
    '  known clinical term or lab test? -----------> REJECTED   identified as something else',
    '       |',
    '  179k brand / generic register --- exact ----> VERIFIED',
    '       |',
    '  combination A/B, all parts resolve ---------> VERIFIED',
    '       |',
    '  fuzzy vs curated table (+dept bonus)',
    '       |--- raw >= 0.65 ------------------------> PROBABLE   CONFIRM',
    '       `--- below --------------------------------> REJECTED',
  ]),
  spacer(140),
  callout('ORDER MATTERS — AND IT USED TO BE WRONG',
    'The clinical-term and lab checks originally sat AFTER the brand register, so "electrolytes" ' +
    '— a lab test — matched some product in the register and came back VERIFIED as a medication. ' +
    'Found on a 500-transcript dry run, where it fired 227 times. The rule now: CURATED BEATS ' +
    'IMPORTED. The curated tables are clinically reviewed; the 179k register is not.',
    RED),
  spacer(140),
  table(
    ['Design point', 'Rationale'],
    [
      [{ t: 'Dosage forms stripped before judging', b: true },
       '"অ্যাম্ব্রুডিল সিরাপ" matched the non-clinical term "syrup" and the whole drug was demoted, though "অ্যাম্ব্রুডিল" alone resolves. The form must not become a verdict about the name.'],
      [{ t: '179k register validates, never scans', b: true },
       'Fishing 179,000 names out of raw transcript would be far more dangerous — many are ordinary words. Restricted to validating a claim the model already made.'],
      [{ t: 'Skeleton hits are PROBABLE, never VERIFIED', b: true },
       'Recovered on consonants only, with vowels dropped. Real enough to keep, not certain enough to assert.'],
    ],
    [2900, 6120]
  ),
  spacer(160),

  h2('5.2  Why three tiers rather than a filter'),
  p('A strict allowlist was implemented and measured. It correctly rejected 13 of 18 false ' +
    'positives — but also discarded "Montuculast" and "মন ডেগুলাস্ট", both real Montelukast ' +
    'prescriptions. The PROBABLE tier exists for exactly those cases: kept, flagged, never ' +
    'silently rewritten.'),
  spacer(120),
  table(
    ['Tier', 'Meaning', 'Behaviour'],
    [
      [{ t: 'VERIFIED', b: true, c: GREEN }, 'Exact gazetteer hit after phonetic folding',
       'Printed as spoken. verified=True.'],
      [{ t: 'PROBABLE', b: true, c: AMBER }, 'Close to a real drug, not exact',
       'Kept, canonical name attached as a PROPOSAL, review_reason set, original preserved in heard_as.'],
      [{ t: 'REJECTED', b: true, c: RED }, 'Not a drug',
       'Removed from medications[], recorded in rejected_terms with the reason.'],
    ],
    [1500, 3300, 4220]
  ),
  pbreak()
);

// --- grounding
children.push(
  h2('5.3  The grounding check — "was this one said?"'),
  p('The gate answers "is this a real drug?" — a fabrication passes that question easily. ' +
    'The grounding check asks the question the gate structurally cannot.'),
  ...mono([
    '  drug survived the gate',
    '       |',
    '  resolve to gazetteer entry',
    '       |',
    '  collect ALL known forms:  generic + brands + Bengali spellings',
    '       |',
    '  any form literally present in transcript? --- yes --> score 1.0  KEEP',
    '       | no',
    '       v',
    '  consonant skeletons vs 1/2/3-token spans',
    '       |',
    '  best >= 0.78 ? --- yes --> KEEP',
    '       | no',
    '       v',
    '  demote to rejected_terms + review reason',
  ]),
  spacer(140),
  callout('THE OBVIOUS APPROACH DOES NOT WORK — AND WAS MEASURED FAILING',
    'Checking the model\'s raw output string against the transcript does not separate the two ' +
    'classes. "Colonsalicyl" is a TRUE reading of "কোলন স্যালেসাইল" scoring 0.80, while the ' +
    'bogus "Traject" scores 0.89 against an unrelated word. No threshold separates them. What ' +
    'does work is scoring every KNOWN FORM of the RESOLVED drug — which produced a real gap.',
    AMBER),
  spacer(150),
  table(
    ['Score', 'Drug', 'Actually said?'],
    [
      [{ t: '0.62', f: 'Consolas' }, 'Lignocaine + Hydrocortisone', { t: 'NO', c: RED, b: true }],
      [{ t: '0.67', f: 'Consolas' }, 'Erythromycin', { t: 'NO', c: RED, b: true }],
      [{ t: '0.75', f: 'Consolas' }, 'Linagliptin', { t: 'NO', c: RED, b: true }],
      [{ t: '— 0.78 —', f: 'Consolas', b: true }, { t: 'THRESHOLD (gap of 0.05)', b: true }, { t: '', b: true }],
      [{ t: '0.80', f: 'Consolas' }, 'Ecosprin', { t: 'yes — ইকোস্পিডিন', c: GREEN }],
      [{ t: '0.80', f: 'Consolas' }, 'Norethisterone', { t: 'yes — ট্রিমোলাট', c: GREEN }],
      [{ t: '0.82', f: 'Consolas' }, 'Choline Salicylate', { t: 'yes — কোলন স্যালেসাইল', c: GREEN }],
      [{ t: '0.89+', f: 'Consolas' }, 'everything else', { t: 'yes', c: GREEN }],
    ],
    [1300, 4100, 3620]
  ),
  spacer(150),
  p('Cross-script matching is why consonant skeletons are needed: the model romanises what the ' +
    'ASR wrote in Bengali, so the two never meet under phonetic folding alone.'),
  spacer(160),

  h2('5.4  Department clash'),
  p('A fuzzy match that also lands in the wrong specialty is demoted. On a menopause ' +
    'consultation, "Traject" resolved by edit distance alone to Linagliptin — a diabetes drug — ' +
    'and was printed as a medication.'),
  bullet('Grounding could not separate it (0.89, above the floor).'),
  bullet('Specialty could: it cleared Trimolat, ট্রাফিক, অ্যামোরাল, Adapaline, Colonsalicyl and Montuculast, catching only Traject and Roxatodil.'),
  bullet('Both sides must be specific — "general" drugs such as paracetamol and antacids are prescribed in every clinic and never clash.'),
  spacer(160),

  h2('5.5  Uncertain-term recovery'),
  p('The model files anything it doubts into raw_uncertain_terms, and nothing ever looked at ' +
    'that list again. Measured on a diabetes consultation: "অ্যামোরাল (possible medication ' +
    'name, ASR unclear)". The gate resolves it to Glimepiride at 0.75 — and Glimepiride ' +
    'alongside the Metformin in the same sentence is the standard pairing. The drug was ' +
    'recoverable the whole time; it was simply never offered to the gate.'),
  callout('THE MODEL\'S DOUBT IS NOT A VERDICT',
    'Doubt is a reason to check, not a reason to discard. Recoveries land as PROBABLE at best, ' +
    'keep their original text, and are always flagged. This closes the OPPOSITE failure to ' +
    'Naloxone: not a garbled fragment becoming a confident drug, but a real drug staying invisible.',
    AMBER),
  spacer(160),

  h2('5.6  Dosing is normalised, never resolved'),
  ...mono([
    '  heard:     "One Eightti EMI Tab Five Days"',
    '  obvious:   180 mg tab, 5 days',
    '  system:    kept verbatim, row FLAGGED for confirmation',
  ]),
  p('Converting it would invent a specific dose from garble, in the one field where a tenfold ' +
    'error is unrecoverable by a reader who trusts the number. A doctor reads "as heard" and ' +
    'types the dose — a five-second correction. A confidently wrong "18 mg" is not a correction ' +
    'at all, because nothing signals that it is wrong.'),
  pbreak()
);

// --- 6 thresholds
children.push(
  h1('6. Thresholds and their evidence'),
  rule(),
  table(
    ['Constant', 'Value', 'Basis', 'Confidence'],
    [
      [{ t: 'SIMILARITY_FLOOR', f: 'Consolas', b: true }, { t: '0.65', f: 'Consolas' },
       'Sits in a 0.06-wide empty band between 0.700 (real drug) and 0.588 (false positive).',
       { t: 'LOW — 13 samples', c: RED, b: true }],
      [{ t: '_GROUNDING_FLOOR', f: 'Consolas', b: true }, { t: '0.78', f: 'Consolas' },
       '0.05 gap between 0.75 (Linagliptin, not said) and 0.80 (Ecosprin, said).',
       { t: 'LOW — 12 samples', c: RED, b: true }],
      [{ t: '_CONTEXT_BONUS', f: 'Consolas', b: true }, { t: '0.12', f: 'Consolas' },
       'Enough to reorder Clobazam / Clonazepam, which are 0.78 alike.',
       { t: 'Structural', c: GREEN }],
      [{ t: 'LOW_AGREEMENT', f: 'Consolas', b: true }, { t: '0.50', f: 'Consolas' },
       'Hallucinating segment 0.29 vs median 0.56.',
       { t: 'Moderate', c: AMBER }],
      [{ t: 'max_segment_s', f: 'Consolas', b: true }, { t: '25.0', f: 'Consolas' },
       'RNNT begins dropping content well before 49s.',
       { t: 'Moderate', c: AMBER }],
      [{ t: '_MIN_FUZZY_LEN', f: 'Consolas', b: true }, { t: '5', f: 'Consolas' },
       'Shorter strings match almost anything.',
       { t: 'Structural', c: GREEN }],
    ],
    [2150, 900, 4070, 1900]
  ),
  spacer(180),
  callout('READ THIS BEFORE CHANGING EITHER FLOOR',
    'The two safety thresholds rest on roughly twelve observations each, with gaps of about 0.05. ' +
    'They are the best evidence available, not strong evidence. Re-derive them from a scored ' +
    'distribution when a larger reviewed set exists. DO NOT nudge either to fix a single ' +
    'consultation — a nudge that fixes one case silently moves every other case across the same ' +
    'boundary, and nothing in the test suite will tell you which ones moved.',
    RED),
  spacer(160),
  h3('6.1  A rejected refinement, and why'),
  p('A second, higher floor for SUBSTITUTING a name (as opposed to merely keeping the row) was ' +
    'considered. Printing a specific name is a stronger claim than keeping a flagged row, so a ' +
    'stricter threshold would be defensible in principle.'),
  p('The scored data does not support one. Real drugs land at 0.700 and 0.818; false positives ' +
    'at 0.588 and below. Any second threshold would sit inside the same single gap as ' +
    'SIMILARITY_FLOOR and separate nothing. Adding a constant that no measurement distinguishes ' +
    'is false precision, and it would imply a rigour the data does not contain.'),
  pbreak()
);

// --- 7 decision log + risk
children.push(
  h1('7. Decision log'),
  rule(),
  h3('D1 — Gazetteer over pure LLM'),
  table(
    ['Option', 'Pros', 'Cons'],
    [
      ['Pure LLM', 'No curation burden; handles unseen names.', 'Hallucinates specific names — the Naloxone failure.'],
      ['Gazetteer-only, no LLM', 'Fully deterministic and auditable.', 'Cannot read narrative: dosing, instructions, symptom phrasing.'],
      [{ t: 'Hybrid — CHOSEN', b: true, c: GREEN }, 'Closed-set question gets a closed-set answer; narrative still read by the LLM.', 'Needs curation; misses names absent from the tables.'],
    ],
    [2000, 3600, 3420]
  ),
  p('Rejected names are recorded rather than dropped, so coverage gaps surface as data instead ' +
    'of silence.', { before: 100 }),
  spacer(140),

  h3('D2 — Merge, not replace'),
  p('Gazetteer results are appended to LLM output. Replacing would discard narrative fields the ' +
    'scanners cannot produce; appending means either path alone suffices for a given item.'),
  spacer(120),

  h3('D3 — Print the spoken name when VERIFIED'),
  table(
    ['Case', 'Printed', 'Why'],
    [
      ['VERIFIED, Latin script', 'As spoken', '"Ecosprin" → "Aspirin" is correct pharmacology but reads as a substitution, and was reported as one.'],
      ['VERIFIED, Bengali script', 'That drug\'s own spoken name', 'সর্বিট্রেট → "Sorbitrate", not "Nitroglycerin". A doctor should read back what they said.'],
      ['PROBABLE', 'The resolved name', '"Rasu Basta Tin" is not a drug, a brand, or a word — printing it verbatim puts a non-existent medicine on the script.'],
    ],
    [2300, 2200, 4520]
  ),
  spacer(140),

  h3('D4 — Bengali-only extraction (translator staged but off)'),
  table(
    ['Option', 'Pros', 'Cons'],
    [
      [{ t: 'Bengali → Qwen — CURRENT', b: true, c: GREEN }, 'The path actually verified. No MT corruption of drug names.', 'Qwen reads English clinical text better than Bengali.'],
      ['Bengali → IndicTrans2 → Qwen', 'Better narrative comprehension.', 'MT is least reliable exactly on transliterated brand names. Unverified against the current path.'],
    ],
    [2500, 3300, 3220]
  ),
  p('If enabled, extract.py\'s bilingual header pins drug names to the Bengali original and uses ' +
    'the English only for narrative — because translation is worst precisely where the risk is ' +
    'highest.', { before: 100 }),
  spacer(140),

  h3('D5 — Ollama over direct inference'),
  p('Local, offline, no per-token cost, and the model is swappable in one constant. The cost is ' +
    'a service dependency whose absence looks like a content bug — see §13.'),
  spacer(180),

  h2('7.1  Residual risks'),
  table(
    ['Risk', 'Exposure', 'Mitigation / status'],
    [
      [{ t: 'Thresholds under-evidenced', b: true }, 'Two safety floors on ~12 samples each.',
       { t: 'OPEN — needs a larger reviewed set', c: RED }],
      [{ t: '179k register unreviewed', b: true }, 'Machine-imported; contains ordinary words.',
       'Contained: validation-only, plus the grounding check.'],
      [{ t: 'Translator silent fallback', b: true }, 'If enabled and it fails to load, output degrades with no error.',
       { t: 'Currently off. Log line is the only positive signal.', c: AMBER }],
      [{ t: 'Cross-segment reasoning lost', b: true }, 'Per-segment extraction cannot link findings across the consultation.',
       'Accepted trade for grounding.'],
      [{ t: 'Gazetteer coverage', b: true }, 'A drug absent from the tables is rejected.',
       'Visible: lands in rejected_terms, so gaps are measurable.'],
    ],
    [2200, 3400, 3420]
  ),
  pbreak()
);

// ======================================================================
// PART II
// ======================================================================
children.push(
  spacer(2200),
  p('PART II', { size: 44, bold: true, color: NAVY, align: AlignmentType.CENTER, after: 80 }),
  p('IMPLEMENTATION GUIDE', { size: 32, color: SLATE, align: AlignmentType.CENTER, after: 120 }),
  p('Set it up locally, verify each stage, and make your first change', {
    size: 22, italics: true, color: SLATE, align: AlignmentType.CENTER }),
  pbreak()
);

children.push(
  h1('8. Prerequisites'),
  rule(),
  table(
    ['Requirement', 'Version / note'],
    [
      ['GPU', 'NVIDIA with CUDA. Verified on RTX 4090. CPU works but is impractically slow.'],
      ['Python', '3.12'],
      ['Disk', '~18 GB — 8.1 GB packages, 4.4 GB Qwen, 2.2 GB Ollama, 2.3 GB models'],
      ['OS', 'Linux (the NeMo fork and Ollama builds assume it)'],
      ['ffmpeg', 'Required to transcode browser audio'],
      ['HF_TOKEN', 'Both AI4Bharat models are gated — token AND accepted licence'],
    ],
    [2100, 6920], { firstBold: true }
  ),
  spacer(180),
  callout('THE GATING TRAP — 30 MINUTES LOST TO THIS',
    'gated=auto means acceptance is instant, but it is still required. A VALID TOKEN ALONE ' +
    'RETURNS 403. Worse, an unaccepted repo downloads "successfully" — yielding a ~2 MB folder ' +
    'containing only LICENSE and README, with no weights and no error. Always check the folder ' +
    'size. Accept the licence in a browser first, at the model page for each of the two ' +
    'ai4bharat repos.',
    RED),
  pbreak()
);

children.push(
  h1('9. Local setup'),
  rule(),
  p('One script does all of it. Read §10 to verify each stage rather than trusting it silently.'),
  h3('Step 1 — Choose a persistent location'),
  ...mono([
    '  export PREFIX=/workspace          # any persistent path',
    '  mkdir -p $PREFIX',
  ]),
  callout('WHY THIS MATTERS ON CLOUD GPUS',
    'On RunPod (and most container hosts) /, /root and /usr/local are rebuilt from the image on ' +
    'every restart. Only the mounted volume survives. Installing to the wrong tier is what turns ' +
    'a restart into a multi-hour rebuild — this happened, and it is the reason boot.sh exists.',
    AMBER),
  h3('Step 2 — Set the token and clone'),
  ...mono([
    '  export HF_TOKEN=hf_xxxxxxxxxxxx',
    '  git clone https://github.com/sayantanIFAI/vor.git $PREFIX/voice-to-rx-repo',
    '  cd $PREFIX/voice-to-rx-repo',
  ]),
  h3('Step 3 — Run setup'),
  ...mono([
    '  bash setup_offline.sh             # online',
    '  # or, on an air-gapped machine:',
    '  WHEELS=/path/to/bundle/wheels bash setup_offline.sh --offline',
  ]),
  p('This clones the AI4Bharat NeMo fork, applies the numpy patch, installs from ' +
    'requirements-lock.txt under a torch constraint, downloads both gated models, installs ' +
    'Ollama, and pulls qwen2.5:7b.'),
  h3('Step 4 — Start'),
  ...mono([
    '  bash boot.sh',
  ]),
  p('Serves on port 8000. boot.sh downloads nothing — if it ever starts downloading, an asset ' +
    'landed on the wrong storage tier.'),
  spacer(170),
  h2('9.1  Building an offline bundle'),
  p('On a connected machine with the SAME OS, Python version and CPU architecture as the target ' +
    '(wheels are platform-specific — a bundle built on Windows or Python 3.11 will not install ' +
    'on a Python 3.12 Linux box):'),
  ...mono([
    '  export HF_TOKEN=hf_xxxxxxxxxxxx',
    '  bash bundle.sh /path/to/output',
  ]),
  p('Produces wheels, the Ollama tarball, the NeMo fork, both HF models, the Qwen blob store, ' +
    'and a README-OFFLINE.txt with the transfer steps.'),
  pbreak()
);

// --- 10 verify
children.push(
  h1('10. Verifying each stage'),
  rule(),
  p('Verify in this order. Each stage depends on the one above it, so a failure at stage 2 makes ' +
    'stage 4 meaningless.'),
  spacer(120),
  h3('10.1  GPU and torch'),
  ...mono([
    '  python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"',
    '  # expect:  2.8.0+cu128 True',
  ]),
  h3('10.2  Imports'),
  ...mono([
    '  for m in pydantic fastapi uvicorn soundfile silero_vad \\',
    '           pytorch_lightning nemo.collections.asr; do',
    '      python3 -c "import $m" 2>/dev/null && echo "OK   $m" || echo "FAIL $m"',
    '  done',
  ]),
  p('nemo.collections.asr is the slow one (~2 minutes) and the one that fails most often.'),
  h3('10.3  Ollama and Qwen'),
  ...mono([
    '  curl -s localhost:11434/api/tags',
    '  # expect qwen2.5:7b listed, size 4683087332',
  ]),
  h3('10.4  Server health'),
  ...mono([
    '  curl -s localhost:8000/api/health',
    '  # {"status":"ok","cuda":true,"gpu":"...","model_loaded":true}',
  ]),
  h3('10.5  End to end — the only test that counts'),
  ...mono([
    '  curl -s -X POST -F file=@sample.wav \\',
    '       http://localhost:8000/api/transcribe | python3 -m json.tool',
  ]),
  p('Expect a diagnosis, medications with tiers, labs, advice, and review flags. Reference ' +
    'timing on an RTX 4090: ASR ~2 s, extraction ~43 s for a 28-second consultation. Extraction ' +
    'dominates; that is expected.'),
  spacer(160),
  callout('WHAT "WORKING" ACTUALLY LOOKS LIKE',
    'Do not treat a populated JSON response as success. A healthy result has review flags on ' +
    'anything uncertain. A result with NO flags on a garbled consultation is more suspicious ' +
    'than one with many — it suggests the validation layer is not running.',
    AMBER),
  pbreak()
);

// --- 11 repo map
children.push(
  h1('11. Repository map'),
  rule(),
  p('Read pipeline.py first. It is the orchestrator and every other module hangs off it.'),
  table(
    ['File', 'Role', 'Read when'],
    [
      [{ t: 'voicerx/pipeline.py', f: 'Consolas', b: true }, 'Orchestrator — audio to validated output', 'Always. Start here.'],
      [{ t: 'server.py', f: 'Consolas', b: true }, 'FastAPI, builds the pipeline, serves the UI', 'Changing the API or wiring'],
      [{ t: 'voicerx/vad.py', f: 'Consolas' }, 'Silero VAD, utterance merging', 'Segment lengths look wrong'],
      [{ t: 'voicerx/asr.py', f: 'Consolas' }, 'IndicConformer, dual decoder, agreement', 'Transcription quality'],
      [{ t: 'voicerx/correct.py', f: 'Consolas' }, 'Learned post-ASR corrections', 'A known garble recurs'],
      [{ t: 'voicerx/extract.py', f: 'Consolas' }, 'Qwen prompt and Ollama client', 'The model misreads content'],
      [{ t: 'voicerx/glossary.py', f: 'Consolas', b: true }, 'Gazetteer, folding, scanners, arbitration', 'A term is missed or wrong'],
      [{ t: 'voicerx/gate.py', f: 'Consolas', b: true }, 'Three-tier medication verdict', 'A drug is wrongly kept or rejected'],
      [{ t: 'voicerx/validate.py', f: 'Consolas', b: true }, 'Grounding, corroboration, recovery, flags', 'Review flags look wrong'],
      [{ t: 'voicerx/schema.py', f: 'Consolas' }, 'ExtractedRx, Medication', 'Adding a field'],
      [{ t: 'voicerx/brands_india.py', f: 'Consolas' }, '179,002 brands (imported, unreviewed)', 'Rarely — do not hand-edit'],
      [{ t: 'voicerx/translate.py', f: 'Consolas' }, 'IndicTrans2 bridge (optional, off)', 'Running the bn→en A/B'],
    ],
    [2450, 3700, 2870]
  ),
  pbreak()
);

// --- 12 common tasks
children.push(
  h1('12. Common tasks'),
  rule(),
  h2('12.1  Adding a gazetteer entry'),
  step('Find whether a related entry already exists. EXTEND it rather than adding a duplicate — two entries folding to the same key is a hard test failure.'),
  step('If the bare Bengali word is also an ordinary word, add inflected forms instead. They fold to distinct keys.'),
  step('Run python tests_glossary_fold.py — it asserts collisions() == 0.'),
  step('Add a regression case to tests_regression.py covering the transcript that motivated the change.'),
  spacer(120),
  callout('THE AMBIGUITY TRAP',
    'Four keys are deliberately blocked from generating entries because they are ordinary Bengali ' +
    'words: গা, কসট, বরন, কাটা. Blocking বরন once made acne undiagnosable — the fix was adding ' +
    'INFLECTED forms, which fold to distinct keys, while leaving the ambiguous bare key blocked. ' +
    'Check _AMBIGUOUS_WITH_COMMON_WORD before adding a short term.',
    AMBER),
  spacer(160),

  h2('12.2  Adding a scanner'),
  p('Follow scan_advice as the template. Return canonical strings, merge in pipeline.py WITHOUT ' +
    'overwriting model output, then gate the result in validate.py. A scanner that overwrites ' +
    'rather than merges will silently delete correct LLM output.'),
  spacer(140),

  h2('12.3  Extending the schema'),
  callout('A FIELD THE SCHEMA LACKS IS A FIELD THE PIPELINE DISCARDS',
    'This is exactly how clinical advice was lost: the model understood it, and the prompt then ' +
    'instructed it to throw the content away because there was nowhere to put it. Six files must ' +
    'change together — missing any one loses the content downstream, silently.',
    RED),
  table(
    ['#', 'File', 'Change'],
    [
      ['1', { t: 'voicerx/schema.py', f: 'Consolas' }, 'Add the field to ExtractedRx or Medication'],
      ['2', { t: 'voicerx/extract.py', f: 'Consolas' }, 'Add it to the prompt\'s JSON shape and describe it in the rules'],
      ['3', { t: 'voicerx/pipeline.py', f: 'Consolas' }, 'Merge gazetteer findings into it'],
      ['4', { t: 'server.py', f: 'Consolas' }, 'Carry it through the response merge'],
      ['5', { t: 'tools/rerun_opd.py', f: 'Consolas' }, 'Add it to the correction worksheet'],
      ['6', { t: 'ui/dist/index.html', f: 'Consolas' }, 'Display it'],
    ],
    [600, 2600, 5820]
  ),
  spacer(160),

  h2('12.4  Changing a threshold'),
  p('Do not, unless you have a scored distribution. If you must:'),
  step('Compute scores for ALL known positives and negatives, not just the case in front of you.'),
  step('Confirm the two bands remain separated with the new value.'),
  step('Record the new measurement in the constant\'s inline comment. Every threshold in this codebase carries its evidence next to it — that convention is worth preserving.'),
  step('Run the full suite. If nothing fails, you have not proven the change is safe; you have proven the suite does not cover it. Add a case.'),
  pbreak()
);

// --- 13 testing + troubleshooting
children.push(
  h1('13. Testing and troubleshooting'),
  rule(),
  h2('13.1  The suite'),
  ...mono([
    '  python tests_regression.py      # 151 checks - end-to-end behaviour',
    '  python tests_glossary_fold.py   #  16 checks - folding + collisions',
    '  python tests_gate.py            # tier assignment',
    '  python tests_labs.py            # lab scanning + ordering',
  ]),
  p('tests_regression.py is organised by the failure each section prevents. When you fix a ' +
    'defect, add the case that would have caught it — that is how the suite grew from 100 to 151 ' +
    'checks across one review cycle.'),
  spacer(160),
  h2('13.2  Troubleshooting'),
  table(
    ['Symptom', 'Cause', 'Fix'],
    [
      ['Extraction empty; segments_flagged == segments_total',
       'Ollama down or model missing',
       'curl localhost:11434/api/tags, then check ollama.log'],
      ['Server will not start; KeyError: \'dir\'',
       'Mainline NeMo cannot load IndicConformer\'s aggregate tokenizer',
       'Use the AI4Bharat fork, branch nemo-v2'],
      ['Imports fine, dies on first audio',
       'numpy 2.0 removed np.sctypes; the fork still calls it',
       'Apply the segment.py patch in setup_offline.sh'],
      ['RuntimeError: operator torchvision::nms does not exist',
       'A --target install shadowed the container\'s torch',
       'Remove pylibs/torch*, nvidia, cuda, triton'],
      ['Raw Bengali appears in medications',
       'Normal on the Bengali-only path',
       'Not a translation bug — there is no translation. Investigate as extraction behaviour.'],
      ['Everything reinstalls after restart',
       'Assets on the container disk, not the volume',
       'Set HF_HOME, TORCH_HOME, OLLAMA_MODELS before anything downloads'],
      ['Model folder is ~2 MB with only LICENSE and README',
       'Gated repo licence not accepted',
       'Accept it in a browser, then re-download'],
      ['llama-server binary not found; generate returns 500',
       'Only the ollama binary was copied, not its runtime libs',
       'Keep the extracted dist; ensure ../lib/ollama resolves'],
    ],
    [2900, 2700, 3420]
  ),
  spacer(170),
  callout('THE ONE FAILURE THAT DOES NOT ANNOUNCE ITSELF',
    'Every row above fails loudly. The dangerous exception is the translator: if it is enabled ' +
    'and fails to load, translate.py catches the error and passes Bengali through untranslated. ' +
    'Output still arrives, and it is quietly worse. The log line "IndicTrans2 loaded" is the only ' +
    'positive signal — plausible-looking output is not evidence that it ran.',
    RED),
  pbreak()
);

// --- 14 deployment
children.push(
  h1('14. Deployment and persistence'),
  rule(),
  h2('14.1  The two storage tiers'),
  ...mono([
    '  CONTAINER DISK  /  /root  /usr/local        WIPED on every restart',
    '  NETWORK VOLUME  /workspace                  SURVIVES restart & termination',
  ]),
  table(
    ['Asset', 'Size', 'Must live on'],
    [
      ['Python packages', '8.1 GB', 'volume'],
      ['Qwen model', '4.4 GB', 'volume'],
      ['Ollama distribution', '2.2 GB', 'volume'],
      ['IndicConformer', '503 MB', 'volume'],
      ['IndicTrans2', '1.75 GB', 'volume'],
    ],
    [3600, 1900, 3520], { firstBold: true }
  ),
  spacer(150),
  p('Four environment variables decide where downloads land. Set them BEFORE anything downloads, ' +
    'or it goes to the tier that gets destroyed:'),
  ...mono([
    '  export HF_HOME=/workspace/.cache/huggingface     # HF models',
    '  export TORCH_HOME=/workspace/.cache/torch        # silero-vad',
    '  export OLLAMA_MODELS=/workspace/ollama/models    # the 4.4 GB',
    '  export HF_HUB_ENABLE_HF_TRANSFER=0               # set in images, package absent',
  ]),
  spacer(140),
  h2('14.2  Restart procedure'),
  ...mono([
    '  bash /workspace/boot.sh',
  ]),
  p('That is the whole procedure. It re-exports the environment, restores the SSH key, starts ' +
    'Ollama, starts the server, and waits for health.'),
  spacer(150),
  callout('THE INVARIANT WORTH REMEMBERING',
    'boot.sh must never download anything. If it does, an asset was installed to the container ' +
    'disk and will be lost again on the next restart. That is the signal to investigate — not to ' +
    'wait for the download to finish.',
    AMBER),
  spacer(180),
  h2('14.3  Reference documents in the repository'),
  table(
    ['File', 'Contents'],
    [
      [{ t: 'DESIGN.md', f: 'Consolas', b: true }, 'This document\'s Part I, with Mermaid diagrams that render on GitHub'],
      [{ t: 'MODELS.md', f: 'Consolas', b: true }, 'The three model artifacts: IDs, sizes, paths, and how each one fails'],
      [{ t: 'setup_offline.sh', f: 'Consolas' }, 'One-shot install, online or air-gapped'],
      [{ t: 'bundle.sh', f: 'Consolas' }, 'Builds the no-internet bundle'],
      [{ t: 'boot.sh', f: 'Consolas' }, 'Restart recovery; downloads nothing'],
      [{ t: 'requirements-lock.txt', f: 'Consolas' }, 'Freeze of an environment verified end-to-end'],
    ],
    [2600, 6420]
  ),
  spacer(220),
  rule(),
  p('END OF DOCUMENT', { bold: true, color: NAVY, align: AlignmentType.CENTER, size: 22 }),
  p('Voice-to-Rx · Design & Implementation Guide · commit 137d8d8',
    { italics: true, color: SLATE, align: AlignmentType.CENTER, size: 18 })
);

// ======================================================================
const doc = new Document({
  creator: 'Voice-to-Rx',
  title: 'Voice-to-Rx — System Design & Implementation Guide',
  description: 'Design review and implementation runbook',
  numbering: {
    config: [
      {
        reference: 'bullets',
        levels: [
          { level: 0, format: LevelFormat.BULLET, text: '•', alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 420, hanging: 220 } } } },
          { level: 1, format: LevelFormat.BULLET, text: '◦', alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 800, hanging: 220 } } } },
        ]
      },
      {
        reference: 'steps',
        levels: [
          { level: 0, format: LevelFormat.DECIMAL, text: '%1.', alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 420, hanging: 260 } } } },
        ]
      }
    ]
  },
  styles: {
    default: {
      document: { run: { font: 'Calibri', size: 21 } }
    }
  },
  sections: [{
    properties: {
      page: { margin: { top: 1300, bottom: 1200, left: 1440, right: 1440 } }
    },
    children
  }]
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync(process.argv[2] || 'VoiceToRx.docx', buf);
  console.log('written', (buf.length / 1024).toFixed(0) + 'KB');
});
