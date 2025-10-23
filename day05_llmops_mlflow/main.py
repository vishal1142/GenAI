# main.py — orchestrates the full LLMOps + MLflow pipeline

from __future__ import annotations
import argparse
from src.utils import load_yaml, set_seed
from src.strategies import make_model
from src.data import get_dataloader
from src.tracker import init_mlflow
from src.train import run_training


def parse_args():
    parser = argparse.ArgumentParser(description="Run LLMOps training pipeline with MLflow tracking.")
    parser.add_argument("--cfg", default="src/config.yaml", help="Path to YAML configuration file.")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_yaml(args.cfg)

    # 1️⃣ Set random seed
    set_seed(cfg["experiment"]["seed"])

    # 2️⃣ Initialize MLflow tracking
    init_mlflow(cfg["tracking"])

    # 3️⃣ Load model and tokenizer
    model, tokenizer = make_model(cfg["model"]["name"], cfg["model"]["strategy"])

    # 4️⃣ Load dataset and create dataloader (pass both data + model configs)
    loader = get_dataloader(cfg["data"], tokenizer, cfg["model"])

    # 5️⃣ Run training and log metrics
    run_training(model, tokenizer, loader, cfg["model"])


if __name__ == "__main__":
    main()
