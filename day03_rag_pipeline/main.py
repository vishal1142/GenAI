# main.py — End-to-End RAG Pipeline driver with step-wise logging and decorators
from __future__ import annotations
import argparse
from pathlib import Path

from src.utils import ensure_dir, log, set_seed
from src.config import Config
from src.ingest import load_documents
from src.chunk import make_chunks
from src.retriever import TfidfRetriever
from src.rank import rerank
from src.qa import compose_answer


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for query and retrieval configuration."""
    parser = argparse.ArgumentParser(description="Run RAG pipeline")
    parser.add_argument("--query", type=str, default="What is AI?", help="User query")
    parser.add_argument("--top_k", type=int, default=3, help="Number of top documents to retrieve")
    return parser.parse_args()


def get_next_result_filename(artifact_dir: Path) -> Path:
    """
    Generate sequential result filenames like:
    1_results.txt, 2_results.txt, etc.
    """
    ensure_dir(artifact_dir)
    existing = sorted(artifact_dir.glob("*_results.txt"))
    if not existing:
        next_index = 1
    else:
        last = max(int(f.stem.split("_")[0]) for f in existing if f.stem.split("_")[0].isdigit())
        next_index = last + 1
    return artifact_dir / f"{next_index}_results.txt"


def main() -> None:
    """Run full retrieval-augmented generation pipeline."""
    args = parse_args()
    set_seed()

    # 1️⃣ Initialize configuration and folders
    cfg = Config()
    ensure_dir(cfg.artifacts)
    log("Initialized configuration and artifacts directory.")

    # 2️⃣ Load text documents
    docs = load_documents(cfg.data_dir)
    log(f"Loaded {len(docs)} documents from {cfg.data_dir}")

    if not docs:
        log("⚠️ No documents found in data directory — please add .txt files to proceed.")
        return

    # 3️⃣ Split documents into chunks
    chunks = make_chunks(docs, mode="sentence", size=cfg.chunk_size, overlap=cfg.overlap)
    log(f"Created {len(chunks)} chunks from input documents.")

    # 4️⃣ Build and fit the retriever
    retriever = TfidfRetriever(stop_words=cfg.stop_words)
    retriever.fit(chunks)
    log("TF-IDF retriever trained on all chunks.")

    # 5️⃣ Perform retrieval, reranking, and generate answer
    results = rerank(retriever.search(args.query, args.top_k))
    answer = compose_answer(results)

    # 6️⃣ Display formatted answer in console
    print("\n" + "=" * 80)
    print(f"🧠 Query: {args.query}\n")
    print(answer)
    print("=" * 80 + "\n")

    # 7️⃣ Save output to sequential results file
    out_path = get_next_result_filename(cfg.artifacts)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"Query: {args.query}\n\n{answer}")
    log(f"Saved final answer to {out_path}")


if __name__ == "__main__":
    main()
