"""FastAPI service exposing the GarageMind LangGraph agent over HTTP.

This is the seam n8n automates against: the /diagnose endpoint is what the
n8n workflow (n8n/garagemind_workflow.json) calls from an HTTP Request node
after its webhook trigger fires, so a diagnosis can be kicked off by
anything that can POST JSON -- a scan-tool app, a form, a chat bot -- with
n8n handling the "what happens after" (routing by severity, notifying,
logging) instead of that logic living inside the agent itself.
"""
from __future__ import annotations

import os
import time

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.agent.graph_agent import build_agent
from src.graph.build_graph import build_knowledge_graph, graph_stats
from src.llm.provider import get_chat_model

load_dotenv()

app = FastAPI(
    title="GarageMind API",
    description="Agentic automotive diagnostic assistant (LangGraph + GraphRAG over OBD-II data).",
    version="0.1.0",
)

# Built once at process startup and reused across requests -- the knowledge
# graph and compiled LangGraph agent are both cheap to hold in memory and
# expensive to rebuild per-request.
_graph = build_knowledge_graph()
_llm = get_chat_model()
_agent = build_agent(_graph, _llm)


class DiagnoseRequest(BaseModel):
    report: str = Field(
        ...,
        min_length=3,
        description="Free-text description of the problem: a DTC code, symptoms, or both.",
        examples=["P0171 and the engine idles rough"],
    )


class DiagnoseResponse(BaseModel):
    status: str
    report: dict
    latency_ms: int


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "llm_provider": type(_llm).__name__, "graph": graph_stats(_graph)}


@app.post("/diagnose", response_model=DiagnoseResponse)
def diagnose(payload: DiagnoseRequest) -> DiagnoseResponse:
    if not payload.report.strip():
        raise HTTPException(status_code=400, detail="`report` must not be empty")

    start = time.perf_counter()
    result_state = _agent.invoke({"user_input": payload.report})
    elapsed_ms = int((time.perf_counter() - start) * 1000)

    final_report = result_state.get("final_report", {})
    return DiagnoseResponse(
        status=final_report.get("status", "unknown"),
        report=final_report,
        latency_ms=elapsed_ms,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("API_PORT", "8000")))
