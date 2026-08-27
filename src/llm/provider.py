"""LLM provider abstraction.

Supports OpenAI, Anthropic, and Groq (Groq is OpenAI-API-compatible, so it's
wired through ChatOpenAI with a custom base_url -- no extra dependency
needed). When no API key is configured for any provider, `get_chat_model()`
returns a deterministic offline `MockChatModel` so the rest of the system
(retrieval, graph construction, LangGraph wiring, tests, CI) works with zero
signup and zero cost. This is what lets `pytest` and the CLI demo both run
green in a fresh clone before anyone has plugged in a real key.
"""
from __future__ import annotations

import os
import re
from typing import Any, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class MockChatModel(BaseChatModel):
    """A zero-cost, deterministic stand-in for a real chat model.

    It doesn't call out to anywhere. Callers in this codebase only ever ask
    the model to do one of two structured jobs (extract DTC codes / symptoms
    from free text, or turn already-retrieved graph evidence into a written
    diagnosis) -- both jobs are simple enough that light heuristics produce
    a genuinely useful offline result rather than a placeholder string, so
    the demo is honest about what's templated vs. LLM-authored.
    """

    @property
    def _llm_type(self) -> str:
        return "mock-offline"

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs: Any) -> ChatResult:
        prompt = "\n".join(m.content for m in messages if isinstance(m.content, str))
        content = self._respond(prompt)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])

    @staticmethod
    def _respond(prompt: str) -> str:
        # The two call sites in this codebase (src/agent/nodes.py) tag their
        # prompts with a marker so the mock can branch without any real NLU.
        if "TASK: extract_codes" in prompt:
            codes = sorted(set(re.findall(r"\bP0[0-9]{3}\b", prompt.upper())))
            return "\n".join(codes)
        if "TASK: diagnose" in prompt:
            return (
                "[offline mode -- templated summary, no LLM call made]\n"
                "Based on the retrieved knowledge-graph evidence above, the highest-"
                "scoring cause(s) are the most likely explanation. Work through the "
                "recommended fix sequence in order, starting with the highest-"
                "likelihood item, and re-scan for the code after each step."
            )
        return "[offline mode] no handler for this prompt"


def _make_openai_compatible(model: str, api_key: str, base_url: Optional[str] = None) -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    kwargs: dict[str, Any] = {"model": model, "api_key": api_key, "temperature": 0.2}
    if base_url:
        kwargs["base_url"] = base_url
    return ChatOpenAI(**kwargs)


def get_chat_model(provider: Optional[str] = None) -> BaseChatModel:
    """Returns a LangChain chat model chosen by (in priority order):

    1. explicit `provider` argument
    2. LLM_PROVIDER env var
    3. auto-detect from whichever *_API_KEY is set (openai > anthropic > groq)
    4. offline MockChatModel
    """
    provider = (provider or os.getenv("LLM_PROVIDER") or "").strip().lower()

    if not provider:
        if os.getenv("OPENAI_API_KEY"):
            provider = "openai"
        elif os.getenv("ANTHROPIC_API_KEY"):
            provider = "anthropic"
        elif os.getenv("GROQ_API_KEY"):
            provider = "groq"
        else:
            provider = "mock"

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return MockChatModel()
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        return _make_openai_compatible(model, api_key)

    if provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            return MockChatModel()
        from langchain_anthropic import ChatAnthropic

        model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022")
        return ChatAnthropic(model=model, api_key=api_key, temperature=0.2)

    if provider == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return MockChatModel()
        model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        return _make_openai_compatible(model, api_key, base_url="https://api.groq.com/openai/v1")

    return MockChatModel()
