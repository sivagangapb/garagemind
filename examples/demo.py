#!/usr/bin/env python3
"""Quick CLI demo -- no server, no n8n required.

    python examples/demo.py "P0171 and rough idle"
    python examples/demo.py --interactive

Runs entirely offline (deterministic MockChatModel) unless you've set an
API key per .env.example, in which case it picks that up automatically via
src/llm/provider.get_chat_model().
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv  # noqa: E402

from src.agent.graph_agent import build_agent  # noqa: E402
from src.graph.build_graph import build_knowledge_graph, graph_stats  # noqa: E402
from src.llm.provider import get_chat_model  # noqa: E402


def print_report(report: dict) -> None:
    if report["status"] == "needs_clarification":
        print("\n🔧 GarageMind needs more detail:\n")
        print(f"   {report['question']}\n")
        if report.get("possible_codes"):
            print(f"   Possible codes on file: {', '.join(report['possible_codes'][:10])}")
        return

    print("\n🔧 GarageMind diagnosis:\n")
    print(f"   Matched code(s): {', '.join(report['matched_dtcs'])}")
    print(f"   Confidence: {report['confidence']}")
    if report.get("top_cause"):
        tc = report["top_cause"]
        print(f"   Most likely cause: {tc['cause']} ({tc['likelihood']} likelihood)")
    print(f"\n   Summary:\n   {report['summary']}\n")
    print("   Recommended fix sequence:")
    for code, fixes in report["recommended_fixes"].items():
        print(f"     {code}:")
        for i, fix in enumerate(fixes, 1):
            print(f"       {i}. {fix}")
    if report.get("related_codes_to_check"):
        print(f"\n   Related codes worth checking: {', '.join(report['related_codes_to_check'][:8])}")
    print(f"\n   (grounded in {report['grounded_in_graph_nodes']} knowledge-graph nodes)")


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="GarageMind CLI demo")
    parser.add_argument("report", nargs="?", help="Free-text problem report, e.g. 'P0171 rough idle'")
    parser.add_argument("--interactive", action="store_true", help="Prompt in a loop instead")
    parser.add_argument("--json", action="store_true", help="Print raw JSON report")
    args = parser.parse_args()

    graph = build_knowledge_graph()
    llm = get_chat_model()
    agent = build_agent(graph, llm)

    print(f"GarageMind ready. LLM provider: {type(llm).__name__} | knowledge graph: {graph_stats(graph)}")

    def run_once(text: str) -> None:
        result = agent.invoke({"user_input": text})
        report = result["final_report"]
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            print_report(report)

    if args.interactive:
        print("Type a DTC code or describe symptoms. Ctrl+C to exit.\n")
        try:
            while True:
                text = input("> ").strip()
                if text:
                    run_once(text)
                    print()
        except (KeyboardInterrupt, EOFError):
            print("\nbye")
        return

    if not args.report:
        parser.error("provide a report string, or pass --interactive")

    run_once(args.report)


if __name__ == "__main__":
    main()
