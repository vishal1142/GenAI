# agent.py — Agent base class + concrete Planner / Researcher / Critic / Executor

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Tuple
from .utils import log, timed, log_calls
from .strategies import make_llm, LLMResponse
from .tools import LocalSearch, SafePython
from .memory import ChatMemory, Scratchpad
from . import prompts

@dataclass
class AgentContext:
    goal: str
    memory: ChatMemory
    scratch: Scratchpad
    search: LocalSearch
    pytool: SafePython
    llm_name: str = "rule"  # or 'openai'

class BaseAgent:
    def __init__(self, name: str, ctx: AgentContext):
        self.name = name
        self.ctx = ctx
        self.llm = make_llm(ctx.llm_name)

    def say(self, text: str) -> None:
        self.ctx.memory.add(self.name, text)
        log(f"{self.name}: {text}")

class PlannerAgent(BaseAgent):
    @timed
    @log_calls
    def run(self) -> str:
        prompt = prompts.PLANNER.format(goal=self.ctx.goal)
        out: LLMResponse = self.llm.generate(prompt)
        self.say(out.text)
        self.ctx.scratch.add(f"plan: {out.text}")
        return out.text

class ResearcherAgent(BaseAgent):
    @timed
    @log_calls
    def run(self, plan: str) -> str:
        prompt = prompts.RESEARCHER.format(goal=self.ctx.goal, plan=plan)
        draft = self.llm.generate(prompt).text

        # Tool-use protocol: look for "search:" and "python:" commands in the draft.
        findings: List[str] = []
        for line in draft.splitlines():
            low = line.strip().lower()
            if low.startswith("search:"):
                query = line.split(":",1)[1].strip()
                hits = self.ctx.search.search(query, top_k=3)
                summary = "; ".join(f"[{n}] score={s:.2f}" for _,s,n in hits)
                findings.append(f"search({query}) -> {summary}")
            elif low.startswith("python:"):
                code = line.split(":",1)[1].strip()
                ok, res = self.ctx.pytool.run(code)
                findings.append(f"python({code}) -> {res}")

        # If no explicit commands, still add a helpful note:
        if not findings:
            findings.append("no tools invoked; relying on prior knowledge")

        # Build a concise researcher draft.
        text = "findings:\n" + "\n".join(f"- {f}" for f in findings)
        self.say(text)
        self.ctx.scratch.add(text)
        return text

class CriticAgent(BaseAgent):
    @timed
    @log_calls
    def run(self, draft: str) -> str:
        prompt = prompts.CRITIC.replace("{draft}", draft)
        out = self.llm.generate("You are the critic; review the draft for errors. Provide 'verdict: ok|revise' and a reason.")
        # Our minimalist LLM gives generic advice; we add a simple heuristic:
        verdict = "ok" if "error" not in draft.lower() else "revise"
        text = f"verdict: {verdict}; reason: basic sanity checks applied"
        self.say(text)
        self.ctx.scratch.add(text)
        return verdict

class ExecutorAgent(BaseAgent):
    @timed
    @log_calls
    def run(self, plan: str, draft: str) -> str:
        # Try to extract any numeric results from the draft (very naive):
        final_note = draft
        # Compose final message:
        prompt = prompts.EXECUTOR
        out = self.llm.generate(prompt).text
        final = f"{out}\n\nPlan:\n{plan}\n\nDraft notes:\n{draft}"
        self.say(final)
        self.ctx.scratch.add(final)
        return final
