from __future__ import annotations
from pydantic import BaseModel, Field


class CoreBlock(BaseModel):
    text: str
    block_type: str
    prob: float


class Example(BaseModel):
    code: str
    core_blocks: list[CoreBlock] = Field(default_factory=list)
    summary: str
    score: float


class TranslationResult(BaseModel):
    pivot_code: str
    candidates: list[str] = Field(default_factory=list)
    selected_score: float | None = None
    repaired: bool
    fell_back: bool


class Trace(BaseModel):
    translation: TranslationResult
    retrieved: list[Example] = Field(default_factory=list)
    core_blocks: list[CoreBlock] = Field(default_factory=list)
    prompt: str


class ModelConfig(BaseModel):
    base_url: str
    api_key: str = "sk-no-key"
    model: str


class Params(BaseModel):
    k: int = 5
    temperatures: list[float] = Field(default_factory=lambda: [0.0, 0.4, 0.8])
    lambda_: float = Field(default=0.5, alias="lambda")
    threshold: float = 0.5
    max_repair_iters: int = 3

    model_config = {"populate_by_name": True}


class SummarizeRequest(BaseModel):
    code: str
    language: str
    model: ModelConfig
    params: Params = Field(default_factory=Params)
    trace: bool = True


class SummarizeResponse(BaseModel):
    summary: str
    trace: Trace | None = None
    error: str | None = None
    failed_stage: str | None = None
