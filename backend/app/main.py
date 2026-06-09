from __future__ import annotations
from fastapi import FastAPI
from app.config import settings
from app.schemas import SummarizeRequest, SummarizeResponse, ModelConfig
from app.models.llm import LLMClient
from app.models.embedder import Embedder
from app.components.translator import Translator
from app.components.retriever import Retriever
from app.components.extractor import Extractor
from app.components.generator import Generator
from app.pipeline import Pipeline

app = FastAPI(title="Code Summary Backend")

# Resident resources loaded once at import (process start).
embedder = Embedder(load=settings.load_sbert)
retriever = Retriever(settings.corpus_path)
extractor = Extractor(settings.extractor_weights,
                      codebert_path=settings.codebert_path,
                      device=settings.device)


def make_llm(cfg: ModelConfig) -> LLMClient:
    return LLMClient(base_url=cfg.base_url, api_key=cfg.api_key, model=cfg.model)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": embedder is not None}


@app.post("/summarize", response_model=SummarizeResponse)
def summarize(req: SummarizeRequest) -> SummarizeResponse:
    llm = make_llm(req.model)
    pipeline = Pipeline(
        translator=Translator(llm, embedder, req.params),
        retriever=retriever,
        extractor=extractor,
        generator=Generator(llm),
    )
    return pipeline.run(req.code, req.language, req.params, req.trace)
