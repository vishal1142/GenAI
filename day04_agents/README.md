# Day 04 — Autonomous Agents (from scratch)

Multi-agent loop (Planner → Researcher → Critic → Executor) with:
- Strategy pattern for LLM backend ('rule' or 'openai')
- Tools: local search (over ./corpus/*.txt) + safe Python evaluator
- Memory: chat buffer + TF-IDF scratchpad
- Decorators for timing and call logging
- End-to-end demo task

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
# (optional) export OPENAI_API_KEY=sk-...  # to try strategy=openai

# Run with the default 'rule' LLM (no API needed):
python main.py --goal "Compute the 10th Fibonacci number and explain the approach"
