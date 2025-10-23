# chunk.py — Split text into chunks (Strategy Pattern: “sentence” or “paragraph”)
from __future__ import annotations
from typing import List
from tqdm import tqdm
import nltk
from nltk.tokenize import sent_tokenize

from src.utils import timed

# ------------------------------------------------------------------------------
# Ensure necessary tokenizer models are downloaded
# ------------------------------------------------------------------------------
# "punkt" handles sentence tokenization for multiple languages.
# The "quiet=True" flag suppresses verbose output for cleaner logs.
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

# ------------------------------------------------------------------------------
# Main Function: make_chunks
# ------------------------------------------------------------------------------
@timed
def make_chunks(
    texts: List[str],
    mode: str = "sentence",
    size: int = 5,
    overlap: int = 1
) -> List[str]:
    """
    Split a list of documents into overlapping chunks for retrieval or embedding.

    Args:
        texts (List[str]):
            A list of raw text documents to be chunked.
        mode (str, optional):
            "sentence" → splits by sentences (default),
            "paragraph" → splits by double newlines.
        size (int, optional):
            Number of sentences per chunk (for sentence mode). Defaults to 5.
        overlap (int, optional):
            Number of sentences repeated between consecutive chunks.
            Creates smooth context transitions. Defaults to 1.

    Returns:
        List[str]: Flattened list of all text chunks.
    """
    chunks: List[str] = []

    if not texts:
        print("⚠️ No input documents found — skipping chunking.")
        return chunks

    for txt in tqdm(texts, desc="Chunking", ncols=80):
        if not txt.strip():
            continue

        if mode.lower() == "sentence":
            # Sentence-based chunking
            sents = sent_tokenize(txt)
            if not sents:
                continue

            # Slide a window with overlap for smoother continuity
            step = max(1, size - overlap)
            for i in range(0, len(sents), step):
                chunk = " ".join(sents[i: i + size]).strip()
                if chunk:
                    chunks.append(chunk)

        elif mode.lower() == "paragraph":
            # Paragraph-based chunking
            for p in txt.split("\n\n"):
                p = p.strip()
                if p:
                    chunks.append(p)
        else:
            raise ValueError(f"❌ Unknown mode: {mode}. Use 'sentence' or 'paragraph'.")

    print(f"✅ Generated {len(chunks)} chunks.")
    return chunks
