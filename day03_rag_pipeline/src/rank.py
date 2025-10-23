# rank.py — Post-ranking step (score normalization)
from __future__ import annotations
from typing import List, Tuple

from src.utils import log, timed


@timed
def rerank(results: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
    """
    Normalize retrieved results' scores for cleaner visualization and comparison.

    Args:
        results: List of tuples (text_chunk, similarity_score).

    Returns:
        List of tuples (text_chunk, normalized_score).
    """
    if not results:
        log("No results to rerank.")
        return []

    max_score = max(score for _, score in results)
    if max_score == 0:
        log("All scores are zero; skipping normalization.")
        return results

    normalized = [(txt, score / max_score) for txt, score in results]
    log(f"Reranked {len(normalized)} results (normalized to 0–1).")
    return normalized
