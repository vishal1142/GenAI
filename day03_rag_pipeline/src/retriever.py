# retriever.py — Build TF-IDF index and perform semantic retrieval
from __future__ import annotations
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

from src.utils import log, timed


class TfidfRetriever:
    """
    Classic TF-IDF Retriever for text chunks.
    - Fits TF-IDF vectors on provided chunks.
    - Computes cosine similarity for search queries.
    """

    def __init__(self, stop_words: str = "english"):
        self.vectorizer = TfidfVectorizer(stop_words=stop_words)
        self.doc_matrix = None
        self.chunks = None

    # -------------------------------------------------------------
    @timed
    def fit(self, chunks: list[str]) -> None:
        """Build TF-IDF index for provided text chunks."""
        log("Building TF-IDF index…")
        self.chunks = chunks
        self.doc_matrix = self.vectorizer.fit_transform(chunks)
        log(f"TF-IDF index built on {len(chunks)} chunks.")

    # -------------------------------------------------------------
    @timed
    def search(self, query: str, top_k: int = 3) -> list[tuple[str, float]]:
        """Return top_k most similar chunks to the given query."""
        q_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self.doc_matrix).ravel()
        idxs = scores.argsort()[-top_k:][::-1]
        return [(self.chunks[i], float(scores[i])) for i in idxs]

    # -------------------------------------------------------------
    def save(self, path: Path) -> None:
        """Serialize retriever to disk."""
        with open(path, "wb") as f:
            pickle.dump(self, f)
        log(f"Saved retriever to {path}")

    @staticmethod
    def load(path: Path) -> "TfidfRetriever":
        """Load a previously saved retriever."""
        with open(path, "rb") as f:
            retriever = pickle.load(f)
        log(f"Loaded retriever from {path}")
        return retriever
