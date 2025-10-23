# Day 03 – RAG Pipeline Baseline (from scratch)

Learn how Retrieval-Augmented Generation works under the hood.

### Features
- Document ingestion (.txt or .md)
- Text chunking by sentence or paragraph (strategy pattern)
- TF-IDF retrieval + cosine ranking
- Extractive Answer synthesis (Top-k sentences)
- Logging & timing decorators for full traceability

### Run
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py --query "What is RAG?" --top_k 3

python -m nltk.downloader punkt punkt_tab
python -m nltk.downloader all
