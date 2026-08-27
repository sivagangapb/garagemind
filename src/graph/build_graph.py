"""Builds a typed NetworkX knowledge graph from the curated OBD-II DTC dataset.

Node types: DTC, Symptom, Component, Cause, Fix, System
Edge types (all directed, DTC-centric):
    DTC  -[HAS_SYMPTOM]->   Symptom
    DTC  -[INVOLVES]->      Component
    DTC  -[CAUSED_BY]->     Cause        (weight = likelihood: high=1.0, medium=0.6, low=0.3)
    Cause-[FIXED_BY]->      Fix
    DTC  -[BELONGS_TO]->    System
    DTC  -[RELATED_TO]->    DTC          (bidirectional, from the curated related_codes list)

This graph is the substrate the GraphRAG retriever (src/graph/retrieval.py)
walks over to ground the LangGraph agent's reasoning in structured facts
instead of letting the LLM free-associate about car problems.
"""
from __future__ import annotations

import json
from pathlib import Path

import networkx as nx

LIKELIHOOD_WEIGHT = {"high": 1.0, "medium": 0.6, "low": 0.3}

DEFAULT_DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "dtc_knowledge_base.json"


def load_dtc_data(path: Path | str = DEFAULT_DATA_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_knowledge_graph(path: Path | str = DEFAULT_DATA_PATH) -> nx.DiGraph:
    """Constructs the full DTC knowledge graph as a directed, weighted NetworkX graph."""
    data = load_dtc_data(path)
    g = nx.DiGraph()

    for entry in data["dtc_codes"]:
        code = entry["code"]
        dtc_node = f"DTC::{code}"
        g.add_node(
            dtc_node,
            type="DTC",
            code=code,
            description=entry["description"],
            severity=entry["severity"],
        )

        system_node = f"System::{entry['system']}"
        g.add_node(system_node, type="System", name=entry["system"])
        g.add_edge(dtc_node, system_node, relation="BELONGS_TO")

        for symptom in entry["symptoms"]:
            sym_node = f"Symptom::{symptom.lower()}"
            g.add_node(sym_node, type="Symptom", name=symptom)
            g.add_edge(dtc_node, sym_node, relation="HAS_SYMPTOM", weight=1.0)

        for component in entry["components"]:
            comp_node = f"Component::{component.lower()}"
            g.add_node(comp_node, type="Component", name=component)
            g.add_edge(dtc_node, comp_node, relation="INVOLVES", weight=1.0)

        for cause in entry["causes"]:
            cause_node = f"Cause::{code}::{cause['cause'].lower()}"
            weight = LIKELIHOOD_WEIGHT.get(cause["likelihood"], 0.5)
            g.add_node(
                cause_node,
                type="Cause",
                name=cause["cause"],
                likelihood=cause["likelihood"],
                dtc=code,
            )
            g.add_edge(dtc_node, cause_node, relation="CAUSED_BY", weight=weight)

        for fix in entry["fixes"]:
            fix_node = f"Fix::{code}::{fix.lower()}"
            g.add_node(fix_node, type="Fix", name=fix, dtc=code)
            # Every fix in the dataset is written to address the DTC's cause set
            # as a whole, so we fan FIXED_BY out from every cause node of this DTC.
            for cause in entry["causes"]:
                cause_node = f"Cause::{code}::{cause['cause'].lower()}"
                g.add_edge(cause_node, fix_node, relation="FIXED_BY", weight=1.0)

    # Related-code edges (bidirectional), added after all DTC nodes exist.
    for entry in data["dtc_codes"]:
        src = f"DTC::{entry['code']}"
        for related in entry.get("related_codes", []):
            dst = f"DTC::{related}"
            if g.has_node(dst):
                g.add_edge(src, dst, relation="RELATED_TO", weight=0.5)
                g.add_edge(dst, src, relation="RELATED_TO", weight=0.5)

    return g


def graph_stats(g: nx.DiGraph) -> dict:
    type_counts: dict[str, int] = {}
    for _, attrs in g.nodes(data=True):
        t = attrs.get("type", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1
    return {
        "num_nodes": g.number_of_nodes(),
        "num_edges": g.number_of_edges(),
        "node_types": type_counts,
    }


if __name__ == "__main__":
    graph = build_knowledge_graph()
    print(graph_stats(graph))
