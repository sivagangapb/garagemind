"""Wires the GarageMind LangGraph agent together.

    START -> parse_input -> retrieve_context --route_after_retrieval-->
                                                    |-- diagnose -> generate_report -> END
                                                    |-- ask_clarification -> END

`retrieve_context` is where GraphRAG happens: it walks the NetworkX
knowledge graph (src/graph) and grounds everything downstream in real
retrieved facts. `diagnose` is the only node that calls the LLM for open-
ended generation, and it's instructed to reason only over that retrieved
evidence. `ask_clarification` is a safety valve -- if the graph can't
confidently place the report, the agent asks a targeted follow-up instead
of guessing.
"""
from __future__ import annotations

import networkx as nx
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, StateGraph

from src.agent.nodes import (
    ask_clarification,
    make_diagnose_node,
    make_generate_report_node,
    make_parse_input_node,
    make_retrieve_context_node,
    route_after_retrieval,
)
from src.agent.state import DiagnosisState


def build_agent(graph: nx.DiGraph, llm: BaseChatModel):
    """Compiles and returns a runnable LangGraph agent bound to the given
    knowledge graph and chat model."""
    workflow = StateGraph(DiagnosisState)

    workflow.add_node("parse_input", make_parse_input_node(llm))
    workflow.add_node("retrieve_context", make_retrieve_context_node(graph))
    workflow.add_node("diagnose", make_diagnose_node(llm))
    workflow.add_node("generate_report", make_generate_report_node(graph))
    workflow.add_node("ask_clarification", ask_clarification)

    workflow.set_entry_point("parse_input")
    workflow.add_edge("parse_input", "retrieve_context")
    workflow.add_conditional_edges(
        "retrieve_context",
        route_after_retrieval,
        {"diagnose": "diagnose", "ask_clarification": "ask_clarification"},
    )
    workflow.add_edge("diagnose", "generate_report")
    workflow.add_edge("generate_report", END)
    workflow.add_edge("ask_clarification", END)

    return workflow.compile()


def run_diagnosis(user_input: str, graph: nx.DiGraph | None = None, llm: BaseChatModel | None = None) -> dict:
    """Convenience entrypoint: builds (or reuses) the graph/model and runs
    one diagnosis turn, returning the final report dict."""
    if graph is None:
        from src.graph.build_graph import build_knowledge_graph

        graph = build_knowledge_graph()
    if llm is None:
        from src.llm.provider import get_chat_model

        llm = get_chat_model()

    agent = build_agent(graph, llm)
    result_state = agent.invoke({"user_input": user_input})
    return result_state["final_report"]
