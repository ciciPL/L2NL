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
from app.assets import asset_status, ensure_ready

app = FastAPI(title="Code Summary Backend")

# Resident resources loaded once at import (process start).
embedder = Embedder(load=settings.load_sbert)
retriever = Retriever(settings.corpus_path, limit=settings.corpus_limit,
                      allow_stubs=settings.allow_stubs)
extractor = Extractor(settings.extractor_weights,
                      codebert_path=settings.codebert_path,
                      device=settings.device,
                      allow_stubs=settings.allow_stubs)


def make_llm(cfg: ModelConfig) -> LLMClient:
    return LLMClient(
        base_url=cfg.base_url, api_key=cfg.api_key, model=cfg.model,
        timeout=settings.llm_timeout, max_tokens=settings.llm_max_tokens)


@app.get("/health")
def health() -> dict:
    status = asset_status(settings)
    return {"status": "ok", "model_loaded": embedder is not None, **status}


@app.post("/summarize", response_model=SummarizeResponse)
def summarize(req: SummarizeRequest) -> SummarizeResponse:
    try:
        ensure_ready(settings)
        llm = make_llm(req.model)
        pipeline = Pipeline(
            translator=Translator(llm, embedder, req.params),
            retriever=retriever,
            extractor=extractor,
            generator=Generator(llm),
        )
        return pipeline.run(req.code, req.language, req.params, req.trace)
    except Exception as e:  # noqa: BLE001 - API boundary returns structured failure
        return SummarizeResponse(summary="", trace=None, error=str(e), failed_stage="startup")
