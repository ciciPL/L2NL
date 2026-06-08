from __future__ import annotations
from app.components.translator import Translator
from app.components.retriever import Retriever
from app.components.extractor import Extractor
from app.components.generator import Generator
from app.schemas import Params, Trace, SummarizeResponse


class Pipeline:
    def __init__(self, translator: Translator, retriever: Retriever,
                 extractor: Extractor, generator: Generator):
        self.translator = translator
        self.retriever = retriever
        self.extractor = extractor
        self.generator = generator

    def run(self, code: str, language: str, params: Params,
            trace: bool) -> SummarizeResponse:
        stage = "translate"
        try:
            translation = self.translator.translate(code, language)

            stage = "retrieve"
            examples = self.retriever.retrieve(translation.pivot_code, params.k)

            stage = "extract"
            core_blocks = self.extractor.extract(
                translation.pivot_code, params.threshold)
            for ex in examples:
                ex.core_blocks = self.extractor.extract(ex.code, params.threshold)

            stage = "generate"
            summary, prompt = self.generator.generate(
                translation.pivot_code, core_blocks, examples)
        except Exception as e:  # noqa: BLE001 - boundary: report which stage broke
            return SummarizeResponse(
                summary="", trace=None, error=str(e), failed_stage=stage)

        trace_obj = None
        if trace:
            trace_obj = Trace(
                translation=translation, retrieved=examples,
                core_blocks=core_blocks, prompt=prompt)
        return SummarizeResponse(summary=summary, trace=trace_obj)
