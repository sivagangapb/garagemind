"""LangGraph state schema for the GarageMind diagnostic agent."""
from __future__ import annotations

from typing import Optional, TypedDict


class CandidateCause(TypedDict):
    dtc_code: str
    cause: str
    likelihood: str
    score: float


class DiagnosisState(TypedDict, total=False):
    # --- input ---
    user_input: str

    # --- parse_input output ---
    extracted_codes: list[str]
    extracted_symptoms: list[str]

    # --- retrieve_context output ---
    graph_context: str          # rendered text block for LLM grounding
    matched_dtcs: list[str]
    ranked_causes: list[CandidateCause]
    fixes_by_dtc: dict[str, list[str]]
    related_dtcs: list[str]
    evidence_nodes: list[str]

    # --- routing ---
    needs_clarification: bool
    clarification_question: Optional[str]

    # --- diagnose output ---
    diagnosis_summary: str

    # --- generate_report output (final) ---
    final_report: dict
