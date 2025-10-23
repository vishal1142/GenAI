# main.py — Build encoder, pass a toy sentence, plot attention heatmaps.
# Includes CLI flags to demonstrate Strategy Pattern switches.

from __future__ import annotations
from pathlib import Path
import argparse
import torch
import matplotlib.pyplot as plt

from src.transformer import Encoder
from src.toy import TinyTokenizer
from src.utils import ensure_dir, set_seed, log

ART_DIR = Path("artifacts")

def plot_attention_grid(attn: torch.Tensor, tokens: list[str], out_path: Path) -> None:
    """
    Draw a grid of attention heatmaps (one per head) for the top layer.
    attn: (H, S, S) tensor for a single batch item.
    """
    heads, S, _ = attn.shape
    cols = min(4, heads)
    rows = (heads + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3.5*rows))
    # normalize axes to a 2D list
    if rows == 1 and cols == 1: axes = [[axes]]
    elif rows == 1: axes = [axes]

    for h in range(heads):
        r, c = divmod(h, cols)
        ax = axes[r][c]
        ax.imshow(attn[h].detach().cpu().numpy(), aspect="auto")
        ax.set_title(f"Head {h}")
        ax.set_xticks(range(S)); ax.set_xticklabels(tokens, rotation=45, ha="right")
        ax.set_yticks(range(S)); ax.set_yticklabels(tokens)
        ax.set_xlabel("Key positions"); ax.set_ylabel("Query positions")

    # hide any unused subplots
    for h in range(heads, rows*cols):
        r, c = divmod(h, cols); fig.delaxes(axes[r][c])

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--layers", type=int, default=2, help="number of encoder layers")
    p.add_argument("--heads", type=int, default=4, help="number of attention heads")
    p.add_argument("--d_model", type=int, default=128, help="model dimensionality")
    p.add_argument("--ff", type=int, default=256, help="feed-forward hidden size")
    p.add_argument("--ff_type", type=str, default="gelu", choices=["relu", "gelu"], help="FFN variant")
    p.add_argument("--attn_type", type=str, default="scaled_dot", choices=["scaled_dot"], help="attention variant")
    return p.parse_args()

def main() -> None:
    set_seed(42)                       # stable demo
    ensure_dir(ART_DIR)                # make artifacts dir

    # toy sentence
    sent = "the quick brown fox jumps over the lazy dog"
    tok = TinyTokenizer()
    ids = tok.encode(sent)             # int ids
    x = torch.tensor(ids, dtype=torch.long).unsqueeze(0)  # (B=1, S)

    # hyperparams via CLI
    args = parse_args()
    model = Encoder(
        vocab_size=tok.vocab_size,
        d_model=args.d_model,
        num_layers=args.layers,
        num_heads=args.heads,
        d_ff=args.ff,
        ff_type=args.ff_type,
        attn_type=args.attn_type,
    )

    # forward pass and capture attention
    with torch.no_grad():
        _, attn_list = model(x, return_attn=True)
    top_attn = attn_list[-1][0]        # (H, S, S) for batch 0

    out_path = ART_DIR / "attention_heatmap.png"
    plot_attention_grid(top_attn, sent.split(), out_path)
    log(f"Saved attention heatmap to: {out_path}")

if __name__ == "__main__":
    main()
