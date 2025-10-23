# strategies.py — Strategy pattern for LLM backend and toolkits.

from __future__ import annotations
from typing import Literal, Callable, Dict, Any
from dataclasses import dataclass
import os
from .utils import log

# ---------- LLM Strategy ----------

@dataclass
class LLMResponse:
    text: str

class RuleLLM:
    """A tiny deterministic 'LLM' for demos: templates + heuristics."""
    def generate(self, prompt: str) -> LLMResponse:
        # Extremely simple rules to keep this project offline.
        # You can expand with regex/templates for your tasks.
        if "fib" in prompt.lower() or "fibonacci" in prompt.lower():
            return LLMResponse("Plan: use iterative DP; compute F(n) with loop; explain steps.")
        if "search:" in prompt.lower():
            return LLMResponse("Use SEARCH tool with given keywords, then summarize top findings.")
        if "critic" in prompt.lower():
            return LLMResponse("Check correctness, clarity, reproducibility, and missing steps.")
        if "execute" in prompt.lower():
            return LLMResponse("Deliver concise final answer with a brief explanation.")
        # Fallback generic response:
        return LLMResponse("Proceed with available tools; if unknown, compute or search; explain result.")

class OpenAILLM:
    """Optional 'real' LLM if OPENAI_API_KEY is set."""
    def __init__(self):
        from openai import OpenAI
        self.client = OpenAI()

    def generate(self, prompt: str) -> LLMResponse:
        # NOTE: simple, no streaming; replace with your preferred model.
        res = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return LLMResponse(res.choices[0].message.content)

def make_llm(strategy: Literal["rule","openai"] = "rule"):
    """Factory: choose LLM backend by name."""
    strategy = (strategy or "rule").lower().strip()
    if strategy == "rule":
        return RuleLLM()
    if strategy == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            log("OPENAI_API_KEY not set; falling back to RuleLLM.")
            return RuleLLM()
        return OpenAILLM()
    raise ValueError(f"Unknown LLM strategy: {strategy}")
