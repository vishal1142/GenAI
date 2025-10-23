# orchestrator.py — a simple multi-agent loop controller

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from .utils import log, timed
from .agent import AgentContext, PlannerAgent, ResearcherAgent, CriticAgent, ExecutorAgent
from .tools import LocalSearch, SafePython
from .memory import ChatMemory, Scratchpad

@dataclass
class Task:
    goal: str
    llm: str = "rule"   # 'rule' or 'openai'
    max_rounds: int = 2

class Orchestrator:
    def __init__(self, corpus_dir: Path):
        self.search = LocalSearch(corpus_dir)
        self.search.load()
        self.pytool = SafePython()

    @timed
    def run(self, task: Task) -> str:
        mem = ChatMemory(max_turns=50)
        scratch = Scratchpad()
        ctx = AgentContext(
            goal = task.goal,
            memory = mem,
            scratch = scratch,
            search = self.search,
            pytool = self.pytool,
            llm_name = task.llm,
        )

        planner    = PlannerAgent("planner", ctx)
        researcher = ResearcherAgent("researcher", ctx)
        critic     = CriticAgent("critic", ctx)
        executor   = ExecutorAgent("executor", ctx)

        plan = planner.run()

        draft = ""
        for i in range(task.max_rounds):
            log(f"--- ROUND {i+1}/{task.max_rounds} ---")
            draft = researcher.run(plan)
            verdict = critic.run(draft)
            if verdict == "ok":
                break

        final = executor.run(plan, draft)
        return final
