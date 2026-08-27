"""Unit tests for the knowledge graph builder and GraphRAG retrieval layer."""
import pytest

from src.graph.build_graph import build_knowledge_graph, graph_stats
from src.graph.retrieval import retrieve


@pytest.fixture(scope="module")
def graph():
    return build_knowledge_graph()


def test_graph_has_expected_node_types(graph):
    stats = graph_stats(graph)
    assert stats["node_types"]["DTC"] == 30
    for expected_type in ("Symptom", "Component", "Cause", "Fix", "System"):
        assert stats["node_types"][expected_type] > 0


def test_every_dtc_has_at_least_one_cause(graph):
    dtc_nodes = [n for n, a in graph.nodes(data=True) if a.get("type") == "DTC"]
    for node in dtc_nodes:
        causes = [
            v for _, v, d in graph.out_edges(node, data=True) if d.get("relation") == "CAUSED_BY"
        ]
        assert causes, f"{node} has no CAUSED_BY edges"


def test_retrieve_by_exact_code_returns_only_that_code(graph):
    result = retrieve(graph, dtc_codes=["P0171"])
    assert result.matched_dtcs == ["P0171"]
    assert result.ranked_causes
    assert result.ranked_causes[0].dtc_code == "P0171"


def test_retrieve_by_code_ignores_generic_symptom_noise(graph):
    # Regression test: mentioning "check engine light" alongside an exact
    # code must not pull in unrelated DTCs via symptom fan-out.
    result = retrieve(graph, dtc_codes=["P0171"], symptom_phrases=["check engine light", "rough idle"])
    assert result.matched_dtcs == ["P0171"]


def test_retrieve_ranks_high_likelihood_causes_above_low(graph):
    result = retrieve(graph, dtc_codes=["P0171"])
    scores = [c.score for c in result.ranked_causes]
    assert scores == sorted(scores, reverse=True)
    assert result.ranked_causes[0].likelihood == "high"


def test_retrieve_unknown_code_returns_empty(graph):
    result = retrieve(graph, dtc_codes=["P9999"])
    assert result.is_empty()


def test_multi_symptom_corroboration_narrows_candidates(graph):
    # "shudder at cruising speed" is unique to P0740; "poor fuel economy"
    # alone spans many codes. Together they should narrow to P0740, not
    # return the full union.
    result = retrieve(graph, symptom_phrases=["shudder at cruising speed", "poor fuel economy"])
    assert result.matched_dtcs == ["P0740"]


def test_single_generic_symptom_returns_many_candidates(graph):
    # A single vague symptom with no corroboration should NOT be narrowed
    # (there's no second clue to narrow it with) -- this is what should
    # trigger the agent's clarification path upstream.
    result = retrieve(graph, symptom_phrases=["stalling"])
    assert len(result.matched_dtcs) > 1


def test_fixes_are_returned_per_matched_dtc(graph):
    result = retrieve(graph, dtc_codes=["P0420"])
    assert "P0420" in result.fixes_by_dtc
    assert len(result.fixes_by_dtc["P0420"]) > 0


def test_related_codes_exclude_the_matched_code_itself(graph):
    result = retrieve(graph, dtc_codes=["P0300"])
    assert "P0300" not in result.related_dtcs
    assert set(result.related_dtcs) <= {n for n in _all_codes(graph)}


def _all_codes(graph):
    return {a["code"] for _, a in graph.nodes(data=True) if a.get("type") == "DTC"}
