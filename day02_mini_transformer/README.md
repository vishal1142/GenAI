# Day 02 – Mini Transformer Encoder (from scratch) + Attention Heatmap

This project builds a **Transformer Encoder** from first principles:
- Positional Encoding (sin/cos)
- Scaled Dot-Product Attention
- Multi-Head Self-Attention
- Encoder Layer(s) with residual + LayerNorm
- Strategy Pattern for attention & feed-forward variants
- Decorators for timing / call logging
- Attention heatmap visualization on a toy sentence

## Run

```bash
# optional: virtual env
python -m venv .venv
# mac/linux
source .venv/bin/activate
# windows
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt

# default run (2 layers, 4 heads, GELU feed-forward)
python main.py

# experiment with strategies / sizes:
python main.py --layers 3 --heads 8 --d_model 256 --ff 512 \
  --ff_type gelu --attn_type scaled_dot
