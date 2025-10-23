# transformer.py — Minimal Transformer Encoder with detailed comments.
# Implements:
#   - PositionalEncoding (sin/cos)
#   - Multi-head self-attention (uses attention strategy)
#   - EncoderLayer (MHSA + FFN with residual + LayerNorm)
#   - Encoder (N stacked layers, token embedding + PE)
# Notes: this is a teaching build prioritizing clarity over trick optimizations.

from __future__ import annotations
import math, torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

from .strategies import make_attention, make_ff
from .utils import log_calls, timed

# --- Positional Encoding ---------------------------------------------------------

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encodings:
      PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
      PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    Added to token embeddings to inject order information.
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)                            # (max_len, d_model)
        pos = torch.arange(0, max_len).float().unsqueeze(1)           # (max_len, 1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(pos * div)                            # even dims
        pe[:, 1::2] = torch.cos(pos * div)                            # odd dims
        self.register_buffer("pe", pe)                                # not a learnable param

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, D)
        S = x.size(1)
        return x + self.pe[:S].unsqueeze(0)                           # broadcast add

# --- Multi-Head Self-Attention ---------------------------------------------------

class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention:
      - project x into Q, K, V
      - split heads
      - attention per head (strategy)
      - concat heads
      - output projection
    Shapes:
      x      : (B, S, D)
      Q/K/V  : (B, S, D)
      heads  : H, depth per head: Dh = D // H
      split  : (B, H, S, Dh)
    """
    def __init__(self, d_model: int, num_heads: int, attn_type: str = "scaled_dot"):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model, self.num_heads = d_model, num_heads
        self.dh = d_model // num_heads

        # linear projections
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)

        # attention core via strategy
        self.core = make_attention(attn_type)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (B, S, D) -> (B, H, S, Dh)
        B, S, D = x.size()
        x = x.view(B, S, self.num_heads, self.dh).permute(0, 2, 1, 3).contiguous()
        return x

    def _combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (B, H, S, Dh) -> (B, S, D)
        B, H, S, Dh = x.size()
        x = x.permute(0, 2, 1, 3).contiguous().view(B, S, H * Dh)
        return x

    @log_calls
    @timed
    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # project to Q/K/V
        Q = self.Wq(x)                                               # (B, S, D)
        K = self.Wk(x)                                               # (B, S, D)
        V = self.Wv(x)                                               # (B, S, D)

        # split heads
        Qh = self._split_heads(Q)                                    # (B, H, S, Dh)
        Kh = self._split_heads(K)                                    # (B, H, S, Dh)
        Vh = self._split_heads(V)                                    # (B, H, S, Dh)

        # attention per head
        out_h, attn = self.core(Qh, Kh, Vh, mask=mask)               # (B, H, S, Dh), (B, H, S, S)

        # combine heads
        out = self._combine_heads(out_h)                              # (B, S, D)
        out = self.Wo(out)                                            # final projection
        return out, attn

# --- Encoder Layer ---------------------------------------------------------------

class EncoderLayer(nn.Module):
    """
    One encoder block: MHSA -> Add+Norm -> FFN -> Add+Norm
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int,
                 ff_type: str = "gelu", attn_type: str = "scaled_dot", dropout: float = 0.1):
        super().__init__()
        self.mhsa = MultiHeadSelfAttention(d_model, num_heads, attn_type=attn_type)
        self.ffn  = make_ff(ff_type, d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    @timed
    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # MHSA with residual + norm
        attn_out, attn = self.mhsa(x, mask=mask)                      # (B,S,D), (B,H,S,S)
        x = self.norm1(x + self.drop(attn_out))
        # FFN with residual + norm
        ff = self.ffn(x)                                              # (B,S,D)
        x = self.norm2(x + self.drop(ff))
        return x, attn

# --- Encoder Stack ---------------------------------------------------------------

class Encoder(nn.Module):
    """
    N-layer Transformer encoder with token embedding and positional encoding.
    """
    def __init__(self, vocab_size: int, d_model: int = 128, num_layers: int = 2,
                 num_heads: int = 4, d_ff: int = 256, ff_type: str = "gelu",
                 attn_type: str = "scaled_dot", max_len: int = 256, dropout: float = 0.1):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)                  # token embeddings
        self.pe  = PositionalEncoding(d_model, max_len)               # sinusoidal PE
        self.drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, ff_type=ff_type, attn_type=attn_type, dropout=dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None, return_attn: bool = False):
        # x: (B, S) token IDs
        h = self.emb(x)                                               # (B, S, D)
        h = self.pe(h)                                                # add PE
        h = self.drop(h)

        att_all: List[torch.Tensor] = []
        for layer in self.layers:
            h, att = layer(h, mask=mask)                              # propagate through blocks
            att_all.append(att)

        return (h, att_all) if return_attn else h
# --- Example Usage ---------------------------------------------------------------
if __name__ == "__main__":
    # Example: create a small Encoder and pass dummy data
    vocab_size = 1000
    seq_len = 20
    batch_size = 4

    model = Encoder(vocab_size, d_model=64, num_layers=2, num_heads=4, d_ff=128)
    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len))  # (B, S)
    output = model(dummy_input)                                         # (B, S, D)
    print(output.shape)  # should be (4, 20, 64)
    print(model)