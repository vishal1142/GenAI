# memory.py — short-term chat buffer + TF-IDF scratchpad for contextual recall.

from __future__ import annotations
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ChatMemory:
    def __init__(self, max_turns: int = 20):
        self.max_turns = max_turns
        self.buf: List[Tuple[str,str]] = []  # (role, text)

    def add(self, role: str, text: str) -> None:
        self.buf.append((role, text))
        if len(self.buf) > self.max_turns:
            self.buf = self.buf[-self.max_turns:]

    def to_prompt(self) -> str:
        lines = [f"{r.upper()}: {t}" for r, t in self.buf]
        return "\n".join(lines)

class Scratchpad:
    """Light TF-IDF memory for quick 'what did we learn so far?' retrieval."""
    def __init__(self):
        self.vec = TfidfVectorizer(stop_words="english")
        self.notes: List[str] = []
        self.matrix = None

    def add(self, note: str) -> None:
        self.notes.append(note)
        self.matrix = self.vec.fit_transform(self.notes)

    def recall(self, query: str, top_k: int = 2) -> List[str]:
        if not self.notes: return []
        q = self.vec.transform([query])
        scores = cosine_similarity(q, self.matrix).ravel()
        idxs = scores.argsort()[-top_k:][::-1]
        return [self.notes[i] for i in idxs]
