# config.py — holds global paths and hyperparameters
from __future__ import annotations
from pathlib import Path

class Config:
    """Global configuration for the RAG pipeline."""

    def __init__(self):
        # Always resolve paths relative to the project root (day03_rag_pipeline)
        self.root = Path(__file__).resolve().parents[1]   # -> day03_rag_pipeline/
        self.data_dir = self.root / "data"                # -> day03_rag_pipeline/data/
        self.artifacts = self.root / "artifacts"          # -> day03_rag_pipeline/artifacts/
        self.stop_words = "english"

        # Chunking hyperparameters
        self.chunk_size = 5
        self.overlap = 1

    def __repr__(self) -> str:
        return f"<Config data_dir={self.data_dir} artifacts={self.artifacts}>"
