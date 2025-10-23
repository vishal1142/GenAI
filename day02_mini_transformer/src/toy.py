# toy.py — tiny whitespace tokenizer with a growable vocab for the toy demo.

from __future__ import annotations
from typing import List, Dict

class TinyTokenizer:
    def __init__(self):
        self.tok2id: Dict[str, int] = {"<pad>": 0, "<unk>": 1}
        self.id2tok: Dict[int, str] = {0: "<pad>", 1: "<unk>"}

    def encode(self, text: str) -> List[int]:
        ids: List[int] = []
        for t in text.strip().split():
            if t not in self.tok2id:
                idx = len(self.tok2id)
                self.tok2id[t] = idx
                self.id2tok[idx] = t
            ids.append(self.tok2id.get(t, 1))
        return ids

    def decode(self, ids: List[int]) -> str:
        return " ".join(self.id2tok.get(i, "<unk>") for i in ids)

    @property
    def vocab_size(self) -> int:
        return len(self.tok2id)
