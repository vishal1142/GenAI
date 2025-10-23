# qa.py — Basic extractive answer composition for RAG pipeline
from __future__ import annotations
from typing import List, Tuple
from src.utils import timed


@timed
def compose_answer(results: List[Tuple[str, float]]) -> str:
    """
    Build a simple extractive-style answer by summarizing top retrieved chunks.

    Args:
        results: List of (chunk_text, normalized_score)

    Returns:
        String containing formatted context and a synthesized "answer"
    """
    if not results:
        return "No relevant chunks found."

    # Build short preview lines for context display
    lines = []
    for i, (txt, s) in enumerate(results):
        clean_txt = txt[:200].replace("\n", " ")  # escape outside f-string
        lines.append(f"[{i+1}] (score={s:.2f}) → {clean_txt}...")

    context = "\n".join(lines)

    # Build extractive-style pseudo-answer (top sentences)
    answer = " ".join([r[0] for r in results if r[1] > 0.1])[:400]

    return f"Top chunks:\n{context}\n\n**Extractive Answer:**\n{answer}"
