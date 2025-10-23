# ingest.py — load raw text files (.txt, .md) safely into memory
from __future__ import annotations
from pathlib import Path
from typing import List
from src.utils import log, timed


@timed
def load_documents(folder: str | Path) -> List[str]:
    """
    Load all text documents from the given folder into memory.
    Supports .txt and .md files.
    """
    folder = Path(folder)
    texts = []

    if not folder.exists():
        log(f"❌ Data folder not found: {folder}")
        return []

    # Load .txt and .md files
    for f in folder.glob("*.txt"):
        try:
            log(f"📄 Loading {f.name}")
            text = f.read_text(encoding="utf-8", errors="ignore").strip()
            if text:
                texts.append(text)
        except Exception as e:
            log(f"⚠️ Error reading {f.name}: {e}")

    for f in folder.glob("*.md"):
        try:
            log(f"📄 Loading {f.name}")
            text = f.read_text(encoding="utf-8", errors="ignore").strip()
            if text:
                texts.append(text)
        except Exception as e:
            log(f"⚠️ Error reading {f.name}: {e}")

    log(f"✅ Loaded {len(texts)} document(s) from {folder}")
    return texts
