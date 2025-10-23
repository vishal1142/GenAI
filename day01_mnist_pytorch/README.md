# Day 01 – MNIST Classifier (PyTorch) with Strategy + Decorators

A from-scratch, production-style **MNIST** digit classifier demonstrating:

- Clean project layout
- **Strategy pattern** (choose optimizer/augmentations at runtime)
- **Decorators** (timing + logging wrappers)
- Config-first design (Pydantic)
- Reproducible training with plots
- Line-by-line code comments

## Setup

```bash
# macOS/Linux
python -m venv .venv && source .venv/bin/activate
# Windows (PowerShell)
python -m venv .venv; .\.venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install -r requirements.txt
