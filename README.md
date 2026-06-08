# Low-Resource Code Summary (paper artifact)

VS Code thin client + FastAPI backend implementing the paper's pivot-retrieval
4-stage pipeline. Components ship as stubs so the shell runs end-to-end; plug
real implementations into `backend/app/components/*.py`.

## Backend
```bash
cd backend
python3.11 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pytest -v
python -m uvicorn app.main:app --port 8000
```

Heavy ML deps (rank-bm25, sentence-transformers, tree-sitter) are an optional
extra, needed only by the real component implementations:
```bash
pip install -e ".[full]"
```

## CLI (reproducible run)
```bash
python cli.py --code-file snippet.rb --language ruby \
  --base-url http://localhost:8080/v1 --model local-model
```

## Extension
```bash
cd extension && npm install && npm run build
# Press F5 in VS Code to launch the Extension Development Host.
```
Select code → right-click → "Code Summary: Summarize Selection". Toggle
online/offline from the status bar item.

## Online vs offline
Single OpenAI-compatible client; only `base_url` differs.
- Online: `https://api.openai.com/v1` (or DeepSeek, etc.)
- Offline: `http://localhost:8080/v1` (llama.cpp `llama-server`)

## Plugging in real implementations
Each component's constructor + method signatures are final. Replace stub bodies
in place — no API, pipeline, extension, or schema changes required:
- `translator.py` — multi-temperature sampling, AST validate/repair,
  back-translation selection (BLEU + SBERT via `self.embedder`).
- `retriever.py` — build BM25 index from `CS_CORPUS_PATH`; query in `retrieve`.
- `extractor.py` — load classifier from `CS_EXTRACTOR_WEIGHTS`; AST semantic
  split + scoring + threshold.
- `embedder.py` — set `CS_LOAD_SBERT=1` to load the real SBERT model.
- `generator.py` — already real; adjust the prompt template / tags to match
  Appendix A.2 if needed.
