# main.py — runs an end-to-end multi-agent session on a single goal.

from __future__ import annotations
import argparse
from pathlib import Path
from src.utils import set_seed, log, ensure_dir
from src.orchestrator import Orchestrator, Task

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--goal", type=str, default="Compute the 10th Fibonacci number and explain the approach")
    p.add_argument("--llm", type=str, default="rule", choices=["rule","openai"])
    p.add_argument("--corpus", type=str, default="corpus")
    p.add_argument("--rounds", type=int, default=2)
    return p.parse_args()

def main():
    args = parse_args()
    set_seed(42)

    # ensure a tiny corpus directory; you can drop your .txt files here
    corpus_dir = Path(args.corpus)
    ensure_dir(corpus_dir)
    seed_file = corpus_dir / "rag.txt"
    if not seed_file.exists():
        seed_file.write_text(
            "Retrieval-Augmented Generation (RAG) retrieves relevant context before generation to reduce hallucinations.",
            encoding="utf-8"
        )

    orch = Orchestrator(corpus_dir)
    final = orch.run(Task(goal=args.goal, llm=args.llm, max_rounds=args.rounds))
    log("\n=== FINAL ANSWER ===\n" + final)

if __name__ == "__main__":
    main()
