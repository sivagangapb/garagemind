"""GraphRAG retrieval over the DTC knowledge graph.

Given one or more seed entities (DTC codes and/or free-text symptom phrases),
this module walks outward through the NetworkX graph to assemble a bounded,
ranked "evidence subgraph": the causes, components, fixes, related codes and
symptoms that are actually connected to what the user reported. That
evidence subgraph -- not the raw dataset and not the LLM's imagination -- is
what gets handed to the LLM as grounding context in the `diagnose` node of
the LangGraph agent. This is the "graph" half of GraphRAG: retrieval is a
graph traversal, not a vector similarity search.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import networkx as nx

# How many hops from a seed node we're willing to walk. 2 hops from a DTC
# reaches its causes and each cause's fixes; 1 hop from a symptom reaches
# the DTC(s) that list it.
MAX_HOPS = 2


@dataclass
class RankedCause:
    dtc_code: str
    cause: str
    likelihood: str
    score: float


@dataclass
class RetrievalResult:
    matched_dtcs: list[str]
    matched_symptoms: list[str]
    ranked_causes: list[RankedCause]
    fixes_by_dtc: dict[str, list[str]]
    related_dtcs: list[str]
    evidence_nodes: list[str]  # for citation / "grounded in" transparency

    def is_empty(self) -> bool:
        return not self.matched_dtcs and not self.matched_symptoms


def _dtc_predecessor_count(g: nx.DiGraph, node: str) -> int:
    return sum(1 for pred in g.predecessors(node) if g.nodes[pred].get("type") == "DTC")


def _generic_symptom_threshold(g: nx.DiGraph) -> int:
    """IDF-style cutoff: a symptom reported by more than ~25% of all DTCs
    (e.g. "check engine light", which is on nearly every code) is too
    common to discriminate between codes and shouldn't seed new candidate
    DTCs on its own -- same idea as stopword filtering in classic IR,
    applied to graph fan-out instead of term frequency."""
    total_dtcs = sum(1 for _, a in g.nodes(data=True) if a.get("type") == "DTC")
    return max(3, round(total_dtcs * 0.25))


def _find_symptom_nodes(g: nx.DiGraph, symptom_text: str) -> list[str]:
    """Fuzzy-ish match: substring match against known Symptom node names.

    Kept deliberately simple (no embeddings) so the retrieval step has zero
    external dependencies and zero cost -- the LLM is only ever used for
    phrasing/reasoning on top of what this function already found.
    """
    symptom_text = symptom_text.lower()
    matches = []
    for node, attrs in g.nodes(data=True):
        if attrs.get("type") != "Symptom":
            continue
        name = attrs["name"].lower()
        if name in symptom_text or symptom_text in name:
            matches.append(node)
        else:
            # word-overlap fallback, e.g. "engine shaking" vs "engine shaking/rough running"
            words = set(symptom_text.split())
            name_words = set(name.replace("/", " ").split())
            if words & name_words and len(words & name_words) >= 1 and len(name_words) <= 4:
                overlap_ratio = len(words & name_words) / max(len(name_words), 1)
                if overlap_ratio >= 0.5:
                    matches.append(node)
    return matches


def retrieve(
    g: nx.DiGraph,
    dtc_codes: list[str] | None = None,
    symptom_phrases: list[str] | None = None,
    top_k_causes: int = 6,
) -> RetrievalResult:
    """Core GraphRAG retrieval: seed the graph from known entities, walk
    outward, rank causes by (edge likelihood weight, corroborating evidence
    count), and return a compact, ranked evidence bundle."""
    dtc_codes = dtc_codes or []
    symptom_phrases = symptom_phrases or []

    seed_dtc_nodes: set[str] = set()
    matched_dtcs: list[str] = []
    for code in dtc_codes:
        node = f"DTC::{code.upper()}"
        if g.has_node(node):
            seed_dtc_nodes.add(node)
            matched_dtcs.append(code.upper())

    generic_threshold = _generic_symptom_threshold(g)
    matched_symptom_names: list[str] = []

    # Track which DTCs each *distinct symptom phrase* points to, so a query
    # describing several symptoms ("shudder at cruising speed, poor fuel
    # economy") can disambiguate the way a mechanic would: a code
    # corroborated by more than one reported symptom outranks one that only
    # matched a single, more generic clue.
    phrase_dtc_sets: list[set[str]] = []
    for phrase in symptom_phrases:
        phrase_dtcs: set[str] = set()
        for sym_node in _find_symptom_nodes(g, phrase):
            matched_symptom_names.append(g.nodes[sym_node]["name"])
            if _dtc_predecessor_count(g, sym_node) > generic_threshold:
                continue  # too generic (e.g. "check engine light") to seed a code on its own
            for pred in g.predecessors(sym_node):
                if g.nodes[pred].get("type") == "DTC":
                    phrase_dtcs.add(pred)
        if phrase_dtcs:
            phrase_dtc_sets.append(phrase_dtcs)

    corroboration: Counter[str] = Counter()
    for s in phrase_dtc_sets:
        corroboration.update(s)

    symptom_seed_dtcs: set[str] = set(corroboration)
    if len(phrase_dtc_sets) > 1 and corroboration:
        max_corr = max(corroboration.values())
        if max_corr > 1:
            # multiple independent symptom clauses agree on a subset of
            # codes -- narrow to those instead of the full union
            symptom_seed_dtcs = {n for n, c in corroboration.items() if c == max_corr}

    # An explicit DTC code is authoritative: trust it over symptom fan-out.
    # Symptom text alongside a given code is corroborating narrative, not a
    # search query for *additional* unrelated codes.
    all_seed_dtcs_set = seed_dtc_nodes if seed_dtc_nodes else symptom_seed_dtcs
    # Deterministic order: set iteration order is not guaranteed stable
    # across runs, and downstream ranking ties should break predictably.
    all_seed_dtcs = sorted(all_seed_dtcs_set, key=lambda n: g.nodes[n]["code"])

    if not all_seed_dtcs:
        return RetrievalResult([], [], [], {}, [], [])

    # Score each candidate DTC by how much seed evidence points to it:
    # a directly-named code scores higher than one only reached via a
    # symptom match, and a code corroborated by multiple symptoms scores
    # higher still.
    dtc_evidence_count: dict[str, int] = {}
    for node in seed_dtc_nodes:
        dtc_evidence_count[node] = dtc_evidence_count.get(node, 0) + 3
    for node in symptom_seed_dtcs:
        dtc_evidence_count[node] = dtc_evidence_count.get(node, 0) + corroboration.get(node, 1)

    ranked_causes: list[RankedCause] = []
    fixes_by_dtc: dict[str, list[str]] = {}
    evidence_nodes: set[str] = set(all_seed_dtcs)

    for dtc_node in all_seed_dtcs:
        code = g.nodes[dtc_node]["code"]
        evidence_boost = dtc_evidence_count.get(dtc_node, 1)

        # Fixes are recorded once per DTC (the dataset writes them as an
        # ordered remedy sequence for that code, not 1:1 per cause), so we
        # collect them once here rather than duplicating under every cause.
        dtc_fixes: list[str] = []
        seen_fix_nodes: set[str] = set()

        for _, cause_node, edata in g.out_edges(dtc_node, data=True):
            if edata.get("relation") != "CAUSED_BY":
                continue
            cause_attrs = g.nodes[cause_node]
            score = edata.get("weight", 0.5) * evidence_boost
            ranked_causes.append(
                RankedCause(
                    dtc_code=code,
                    cause=cause_attrs["name"],
                    likelihood=cause_attrs["likelihood"],
                    score=round(score, 3),
                )
            )
            evidence_nodes.add(cause_node)
            for _, fix_node, fedata in g.out_edges(cause_node, data=True):
                if fedata.get("relation") != "FIXED_BY" or fix_node in seen_fix_nodes:
                    continue
                seen_fix_nodes.add(fix_node)
                dtc_fixes.append(g.nodes[fix_node]["name"])
                evidence_nodes.add(fix_node)

        if dtc_fixes:
            fixes_by_dtc[code] = dtc_fixes

    # Sort by score descending; ties break on code/cause text so results are
    # reproducible regardless of dict/set iteration order.
    ranked_causes.sort(key=lambda c: (-c.score, c.dtc_code, c.cause))
    ranked_causes = ranked_causes[:top_k_causes]

    related_dtcs: set[str] = set()
    for dtc_node in all_seed_dtcs:
        for _, related_node, edata in g.out_edges(dtc_node, data=True):
            if edata.get("relation") == "RELATED_TO":
                related_dtcs.add(g.nodes[related_node]["code"])
    related_dtcs -= set(matched_dtcs)

    return RetrievalResult(
        matched_dtcs=sorted({g.nodes[n]["code"] for n in all_seed_dtcs}),
        matched_symptoms=sorted(set(matched_symptom_names)),
        ranked_causes=ranked_causes,
        fixes_by_dtc=fixes_by_dtc,
        related_dtcs=sorted(related_dtcs),
        evidence_nodes=sorted(evidence_nodes),
    )


def result_to_context_block(result: RetrievalResult) -> str:
    """Renders a RetrievalResult as a compact text block suitable for
    inclusion in an LLM prompt -- this is the "grounding context" the
    diagnose node injects so the model reasons over retrieved facts rather
    than prior knowledge."""
    if result.is_empty():
        return "No matching DTC codes or symptoms found in the knowledge graph."

    lines = [f"Matched DTC code(s): {', '.join(result.matched_dtcs) or 'none'}"]
    if result.matched_symptoms:
        lines.append(f"Matched symptom(s): {', '.join(result.matched_symptoms)}")
    lines.append("Ranked candidate causes (highest-scoring first, grounded in the knowledge graph):")
    for i, c in enumerate(result.ranked_causes, 1):
        lines.append(f"  {i}. [{c.dtc_code}] {c.cause} (likelihood: {c.likelihood}, score: {c.score})")
    if result.fixes_by_dtc:
        lines.append("Recommended fix sequence per code:")
        for code, fixes in result.fixes_by_dtc.items():
            lines.append(f"  {code}: " + "; ".join(fixes))
    if result.related_dtcs:
        lines.append(f"Related codes worth checking too: {', '.join(result.related_dtcs)}")
    return "\n".join(lines)
