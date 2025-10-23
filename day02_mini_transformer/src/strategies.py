# strategies.py — Strategy Pattern factories to swap attention / FFN variants via if/elif.

from __future__ import annotations
import torch.nn as nn
import torch
from typing import Literal

# --- Feed-Forward (FFN) Strategy -------------------------------------------------

class FFNRelu(nn.Module):
    """Position-wise MLP with ReLU activation."""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class FFNGelu(nn.Module):
    """Position-wise MLP with GELU activation."""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def make_ff(ff_type: Literal["relu","gelu"], d_model: int, d_ff: int, dropout: float = 0.1) -> nn.Module:
    """Factory: choose FFN variant by name."""
    ff_type = (ff_type or "gelu").lower().strip()
    if ff_type == "relu":
        return FFNRelu(d_model, d_ff, dropout)
    elif ff_type == "gelu":
        return FFNGelu(d_model, d_ff, dropout)
    raise ValueError(f"Unknown FF type: {ff_type}")

# --- Attention Strategy -----------------------------------------------------------

class ScaledDotAttention(nn.Module):
    """
    Core attention: softmax((Q K^T) / sqrt(d_k)) V
    Expects inputs shaped for multi-head attention: (B, H, S, D).
    """
    def __init__(self): super().__init__()
    def forward(self, Q, K, V, mask=None):
        d_k = Q.size(-1)
        scores = (Q @ K.transpose(-2, -1)) / (d_k ** 0.5)  # (B, H, S, S)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = torch.softmax(scores, dim=-1)                # (B, H, S, S)
        out = attn @ V                                      # (B, H, S, D)
        return out, attn

def make_attention(attn_type: Literal["scaled_dot"] | None):
    """Factory: choose attention type by name (extensible for linear/flash etc.)."""
    attn_type = (attn_type or "scaled_dot").lower().strip()
    if attn_type == "scaled_dot":
        return ScaledDotAttention()
    raise ValueError(f"Unknown attention type: {attn_type}")
# --- Example Usage ---------------------------------------------------------------
if __name__ == "__main__":
    # Example: create FFN and Attention instances via factories
    ffn = make_ff("gelu", d_model=512, d_ff=2048, dropout=0.1)
    attn = make_attention("scaled_dot")
    print(ffn)
    print(attn)