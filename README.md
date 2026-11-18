# Literature Review Local AI

Search research papers and chat with uploaded PDFs using local models. Derived from [Automated-Literature-Review-with-Multi-Agent-AI](https://github.com/CosmicAlpaca/Automated-Literature-Review-with-Multi-Agent-AI).

Crossref retrieves paper metadata and a local MiniLM cross-encoder reranks the results. PDF chat uses local semantic and keyword retrieval, cross-encoder reranking, and FLAN-T5-small to generate answers based on retrieved passages. The cited pages are sources consulted, not proof that every claim is supported. Verify answers against the papers. A concept map shows terms that occur together in passages, not verified relationships.

Models run locally without an inference API, but download their weights on first use. Crossref is an external metadata service. The small reader works best for simple English questions and can make mistakes. PDF sessions live in backend memory and are lost when the service restarts. Only text PDFs are supported; scanned PDFs require OCR.

## Run locally

Start PostgreSQL and create a database. Use Python 3.12 and Node.js 22 or later.

```bash
cd backend
python -m venv .venv
# Activate the virtual environment for your platform
pip install -r requirements.txt
cp .env.example .env
# Set DATABASE_URL and a long random SECRET_KEY in .env
python -m app.init_db
uvicorn app.main:app --reload
```

```bash
cd frontend
npm ci
cp .env.local.example .env.local
npm run dev
```

The example backend environment enables Crossref, local reranking, and local abstract analysis. The backend Dockerfile installs CPU PyTorch. Model loading requires network access, memory and CPU. Run a single backend worker while PDF sessions remain in memory. Never use the example secret in production.

## Models and training

Search uses `cross-encoder/ms-marco-MiniLM-L6-v2`; PDF embeddings use `sentence-transformers/all-MiniLM-L6-v2`; the PDF reader uses `google/flan-t5-small`. Consult the model cards for terms and limitations. `ml/scripts/train_scifact_experiment.py` demonstrates a query-separated fine-tuning experiment using BEIR SciFact; install `datasets` and `accelerate` for training. Its small held-out evaluation is not evidence of general performance.

This public snapshot excludes legacy experiments, upstream Scopus spreadsheets, notebook outputs and credentials. No license is granted here for upstream code or datasets; check the original project and applicable model and dataset licenses.
