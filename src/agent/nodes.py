"""Node functions for the GarageMind LangGraph agent.

Each node is a plain function: `(state: DiagnosisState) -> partial state dict`.
LangGraph merges the returned dict into the running state. Keeping nodes as
pure-ish functions of state (graph + LLM model are injected via closures
from `build_agent`) keeps them independently unit-testable -- see
tests/test_agent.py, which drives them with the offline MockChatModel.
"""
from __future__ import annotations

import re

import networkx as nx
from langchain_core.language_models.chat_models import BaseChatModel

from src.agent.state import DiagnosisState
from src.graph.retrieval import RetrievalResult, result_to_context_block, retrieve

DTC_PATTERN = re.compile(r"\bP0[0-9]{3}\b", re.IGNORECASE)

# A query is treated as too vague to answer confidently when it matched
# purely on symptom overlap (no explicit code given) and pulled in more
# unrelated candidate codes than this.
AMBIGUITY_THRESHOLD = 3


def make_parse_input_node(llm: BaseChatModel):
    def parse_input(state: DiagnosisState) -> dict:
        text = state["user_input"]

        regex_codes = sorted(set(m.upper() for m in DTC_PATTERN.findall(text)))

        # Ask the model too -- on real providers this catches codes written
        # in unusual formats; on the offline mock it just re-derives the
        # same regex match, which is fine (see MockChatModel._respond).
        llm_response = llm.invoke(f"TASK: extract_codes\nUser said: {text}")
        llm_codes = sorted(
            set(m.upper() for m in DTC_PATTERN.findall(llm_response.content or ""))
        )

        all_codes = sorted(set(regex_codes) | set(llm_codes))

        # Split the free text into rough clauses so multi-symptom
        # descriptions ("rough idle, and the check engine light is on")
        # each get matched against the graph independently.
        clauses = [c.strip() for c in re.split(r",| and |;|\.", text) if c.strip()]
        symptom_phrases = clauses or [text]

        return {
            "extracted_codes": all_codes,
            "extracted_symptoms": symptom_phrases,
        }

    return parse_input


def make_retrieve_context_node(graph: nx.DiGraph):
    def retrieve_context(state: DiagnosisState) -> dict:
        result: RetrievalResult = retrieve(
            graph,
            dtc_codes=state.get("extracted_codes", []),
            symptom_phrases=state.get("extracted_symptoms", []),
        )

        needs_clarification = result.is_empty() or (
            not state.get("extracted_codes")
            and len(result.matched_dtcs) > AMBIGUITY_THRESHOLD
        )

        clarification_question = None
        if result.is_empty():
            clarification_question = (
                "I couldn't match that to anything in the knowledge base. "
                "Can you give me the exact OBD-II code from a scan tool "
                "(e.g. 'P0171'), or describe the symptom more specifically "
                "(e.g. 'rough idle' rather than 'it feels off')?"
            )
        elif needs_clarification:
            preview = ", ".join(result.matched_dtcs[:5])
            clarification_question = (
                f"Those symptoms match {len(result.matched_dtcs)} different "
                f"possible codes ({preview}, ...) -- too broad to narrow down "
                "confidently from symptoms alone. Do you have the actual "
                "OBD-II code from a scan tool? That would let me pinpoint it."
            )

        return {
            "graph_context": result_to_context_block(result),
            "matched_dtcs": result.matched_dtcs,
            "ranked_causes": [
                {
                    "dtc_code": c.dtc_code,
                    "cause": c.cause,
                    "likelihood": c.likelihood,
                    "score": c.score,
                }
                for c in result.ranked_causes
            ],
            "fixes_by_dtc": result.fixes_by_dtc,
            "related_dtcs": result.related_dtcs,
            "evidence_nodes": result.evidence_nodes,
            "needs_clarification": needs_clarification,
            "clarification_question": clarification_question,
        }

    return retrieve_context


def route_after_retrieval(state: DiagnosisState) -> str:
    return "ask_clarification" if state.get("needs_clarification") else "diagnose"


def ask_clarification(state: DiagnosisState) -> dict:
    return {
        "final_report": {
            "status": "needs_clarification",
            "question": state.get("clarification_question"),
            "possible_codes": state.get("matched_dtcs", []),
        }
    }


def make_diagnose_node(llm: BaseChatModel):
    def diagnose(state: DiagnosisState) -> dict:
        prompt = (
            "TASK: diagnose\n"
            "You are an automotive diagnostic assistant. Using ONLY the "
            "evidence below (retrieved from a structured OBD-II knowledge "
            "graph -- do not invent causes that aren't listed), write a "
            "short, plain-language summary (3-5 sentences) of the most "
            "likely problem and what to check first. Be direct and "
            "practical, like an experienced mechanic explaining it to a "
            "customer.\n\n"
            f"User report: {state['user_input']}\n\n"
            f"Evidence:\n{state['graph_context']}"
        )
        response = llm.invoke(prompt)
        return {"diagnosis_summary": response.content}

    return diagnose


def make_generate_report_node(graph: nx.DiGraph):
    def generate_report(state: DiagnosisState) -> dict:
        matched = state.get("matched_dtcs", [])
        severities = {
            code: graph.nodes.get(f"DTC::{code}", {}).get("severity", "unknown")
            for code in matched
        }
        ranked = state.get("ranked_causes", [])
        top = ranked[0] if ranked else None
        # crude confidence heuristic: how much the top cause's score
        # separates from the runner-up -- a clear leader reads as
        # higher-confidence than a near-tie.
        if top and len(ranked) > 1:
            gap = top["score"] - ranked[1]["score"]
            confidence = "high" if gap >= 1.0 else "medium"
        elif top:
            confidence = "high"
        else:
            confidence = "low"

        report = {
            "status": "diagnosed",
            "matched_dtcs": matched,
            "severity_by_code": severities,
            "top_cause": top,
            "confidence": confidence,
            "summary": state.get("diagnosis_summary", ""),
            "recommended_fixes": state.get("fixes_by_dtc", {}),
            "related_codes_to_check": state.get("related_dtcs", []),
            "grounded_in_graph_nodes": len(state.get("evidence_nodes", [])),
        }
        return {"final_report": report}

    return generate_report
