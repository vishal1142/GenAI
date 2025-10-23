# tools.py — simple tools agents can call via Python functions.
# We keep tools explicit (no code execution by LLM), and routed by the agent.

from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Dict, Any
import math, builtins
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------- Local Search Tool (over ./corpus/*.txt) --------

class LocalSearch:
    """Index and search a local text corpus using TF-IDF."""
    def __init__(self, corpus_dir: Path):
        self.corpus_dir = corpus_dir
        self.docs: List[str] = []
        self.names: List[str] = []
        self.vec = TfidfVectorizer(stop_words="english")
        self.matrix = None

    def load(self) -> None:
        self.docs, self.names = [], []
        for f in sorted(self.corpus_dir.glob("*.txt")):
            self.docs.append(f.read_text(encoding="utf-8"))
            self.names.append(f.name)
        if not self.docs:
            # seed with a default doc if none provided
            self.docs = ["RAG combines retrieval with generation to ground LLM answers in external context."]
            self.names = ["seed.txt"]
        self.matrix = self.vec.fit_transform(self.docs)

    def search(self, query: str, top_k: int = 3) -> List[Tuple[str, float, str]]:
        q = self.vec.transform([query])
        scores = cosine_similarity(q, self.matrix).ravel()
        idxs = scores.argsort()[-top_k:][::-1]
        return [(self.docs[i], float(scores[i]), self.names[i]) for i in idxs]

# -------- Safe Python Eval Tool --------

class SafePython:
    """
    Very small 'sandbox': evaluate numeric/python snippets with minimal builtins.
    WARNING: still limited; never expose in production as-is.
    """
    def __init__(self):
        self.env: Dict[str, Any] = {"math": math, "__builtins__": {"abs": abs, "min": min, "max": max, "range": range}}

    def run(self, code: str) -> Tuple[bool, str]:
        try:
            # if it's an expression, eval; else exec and return a name if set
            try:
                val = eval(code, self.env, {})
                return True, repr(val)
            except SyntaxError:
                exec(code, self.env, {})
                return True, "OK"
        except Exception as e:
            return False, f"ERROR: {e}"
