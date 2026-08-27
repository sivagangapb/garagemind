"""Tests for the compiled LangGraph agent, run entirely offline against the
MockChatModel so this suite needs zero API keys and costs nothing to run in
CI. It exercises the full graph (parse_input -> retrieve_context -> route ->
diagnose/ask_clarification -> generate_report), not just the retrieval
layer, and separately scores the agent against a small hand-labeled
evaluation set (tests/eval_set.json) the way you'd score any retrieval/
classification system.
"""
import json
from pathlib import Path

import pytest

from src.agent.graph_agent import build_agent
from src.graph.build_graph import build_knowledge_graph
from src.llm.provider import MockChatModel

EVAL_SET_PATH = Path(__file__).parent / "eval_set.json"

# Below this, something in retrieval/routing regressed -- fail loudly rather
# than let accuracy quietly drift down over time.
MIN_EVAL_ACCURACY = 0.9


@pytest.fixture(scope="module")
def agent():
    graph = build_knowledge_graph()
    llm = MockChatModel()
    return build_agent(graph, llm)


def test_exact_code_resolves_to_diagnosed(agent):
    result = agent.invoke({"user_input": "P0171"})
    report = result["final_report"]
    assert report["status"] == "diagnosed"
    assert report["matched_dtcs"] == ["P0171"]
    assert report["top_cause"]["dtc_code"] == "P0171"
    assert report["confidence"] in ("high", "medium", "low")


def test_vague_input_asks_for_clarification(agent):
    result = agent.invoke({"user_input": "my car feels weird sometimes"})
    report = result["final_report"]
    assert report["status"] == "needs_clarification"
    assert report["question"]


def test_unknown_code_asks_for_clarification(agent):
    result = agent.invoke({"user_input": "P9999"})
    assert result["final_report"]["status"] == "needs_clarification"


def test_report_is_grounded_in_graph_evidence(agent):
    result = agent.invoke({"user_input": "P0420"})
    report = result["final_report"]
    assert report["grounded_in_graph_nodes"] > 0
    assert "P0420" in report["recommended_fixes"]


def _load_eval_cases() -> list[dict]:
    with open(EVAL_SET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)["cases"]


def test_eval_set_accuracy(agent):
    cases = _load_eval_cases()
    results = []

    for case in cases:
        state = agent.invoke({"user_input": case["input"]})
        report = state["final_report"]
        ok = report.get("status") == case["expected_status"]

        if ok and case.get("expected_top_dtc"):
            top = report.get("top_cause") or {}
            ok = top.get("dtc_code") == case["expected_top_dtc"]

        if ok and case.get("expected_matched_dtcs_include"):
            matched = set(report.get("matched_dtcs", []))
            ok = set(case["expected_matched_dtcs_include"]) <= matched

        results.append({"input": case["input"], "passed": ok, "got_status": report.get("status")})

    accuracy = sum(r["passed"] for r in results) / len(results)
    failures = [r for r in results if not r["passed"]]

    assert accuracy >= MIN_EVAL_ACCURACY, (
        f"Eval accuracy {accuracy:.0%} below {MIN_EVAL_ACCURACY:.0%} threshold. "
        f"Failing cases: {failures}"
    )
