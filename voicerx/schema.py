"""Strict output schema for the extraction node.

The LLM is asked to produce this shape, but LLMs drift (nested objects where
a string was asked for, missing fields, etc). `ExtractedRx.from_llm_json`
is deliberately lenient on the way IN - coercing common drift patterns back
into the strict shape - while the shape itself stays strict, so everything
downstream can rely on it without re-checking types.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, field_validator


def _flatten_to_text(value) -> str:
    """Coerce a str, or a dict/list the LLM emitted instead of a str, to text."""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        parts = [str(v).strip() for v in value.values() if v not in (None, "")]
        return ", ".join(parts)
    if isinstance(value, list):
        return ", ".join(_flatten_to_text(v) for v in value)
    return str(value).strip()


class Medication(BaseModel):
    drug: str = ""
    dosage: str = ""
    frequency: str = ""
    duration: str = ""

    # set by the validator (Node 3), never by the LLM itself
    verified: bool = False
    review_reason: str = ""

    # set by the medication gate (gate.py), never by the LLM.
    # tier is "verified" or "probable" - "rejected" never reaches here,
    # those are moved to ExtractedRx.rejected_terms instead.
    tier: str = ""
    # The gazetteer's canonical generic name. For a PROBABLE match this is
    # a PROPOSAL for the reviewer, deliberately kept in a separate field so
    # it can never be mistaken for what was actually said - `drug` always
    # holds the original text.
    canonical: str = ""
    department: str = ""
    indication: str = ""
    match_similarity: float = 0.0

    @field_validator("drug", "dosage", "frequency", "duration", mode="before")
    @classmethod
    def _coerce_text(cls, v):
        return _flatten_to_text(v) if v is not None else ""


class ExtractedRx(BaseModel):
    # PII. Only populated when the patient's name is explicitly spoken in the
    # consultation. Never inferred, never guessed - a wrong name on a
    # prescription is its own category of harm.
    patient_name: Optional[str] = None

    symptoms: list[str] = Field(default_factory=list)
    diagnosis: Optional[str] = None
    labs_ordered: list[str] = Field(default_factory=list)
    medications: list[Medication] = Field(default_factory=list)
    follow_up: Optional[str] = None
    summary: Optional[str] = None
    raw_uncertain_terms: list[str] = Field(default_factory=list)
    confidence_note: str = ""

    # set by the pipeline, not the LLM
    needs_human_review: bool = False
    review_reasons: list[str] = Field(default_factory=list)
    source_transcript: str = ""
    audio_file: str = ""
    segment_start_s: Optional[float] = None
    segment_end_s: Optional[float] = None

    # set by the ASR node - which decoder's text was actually used, and how
    # much the two decoders agreed. Low agreement is an independent review
    # signal Node 3 checks regardless of what the LLM itself flagged.
    decoder_used: str = ""
    decoder_agreement: float = 1.0

    # set by the correction layer (correct.py). Medium-confidence swaps are
    # applied but must be confirmed by a human, so Node 3 flags them.
    corrections_applied: list[str] = Field(default_factory=list)
    correction_needs_confirmation: bool = False

    # Proposed - NEVER auto-applied - drug names for mangled tokens.
    # Surfaced for human confirmation. See fuzzy_drugs.py for why these are
    # never substituted automatically.
    drug_candidates: list[str] = Field(default_factory=list)

    # Terms the SLM called medications that the gazetteer refused (gate.py):
    # "drink water and rest", "Antibiotic", "Boot", null. Recorded rather
    # than dropped, because a silent deletion is indistinguishable from a
    # term that was never extracted - and if the gate is ever WRONG, this
    # list is the only place the evidence survives.
    rejected_terms: list[str] = Field(default_factory=list)

    @field_validator("confidence_note", mode="before")
    @classmethod
    def _coerce_note(cls, v):
        """The LLM emits null here freely - especially once the prompt's
        example JSON contains other nulls. Rejecting it cost 4 of 16
        segments on one file, including one carrying a real symptom, since
        a validation failure drops the whole extraction."""
        return "" if v is None else _flatten_to_text(v)

    @field_validator("symptoms", "labs_ordered", "raw_uncertain_terms", mode="before")
    @classmethod
    def _coerce_str_list(cls, v):
        if v is None:
            return []
        if not isinstance(v, list):
            v = [v]
        return [_flatten_to_text(item) for item in v if item not in (None, "")]

    @field_validator("diagnosis", "follow_up", "summary", "patient_name", mode="before")
    @classmethod
    def _coerce_optional_text(cls, v):
        if v is None:
            return None
        text = _flatten_to_text(v)
        return text or None

    @classmethod
    def from_llm_json(cls, raw: dict, audio_file: str = "", transcript: str = "",
                       seg_start: Optional[float] = None,
                       seg_end: Optional[float] = None) -> "ExtractedRx":
        obj = cls.model_validate(raw)
        obj.audio_file = audio_file
        obj.source_transcript = transcript
        obj.segment_start_s = seg_start
        obj.segment_end_s = seg_end
        return obj
