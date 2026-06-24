from __future__ import annotations
import argparse
from pathlib import Path
from app.config import settings
from app.models.llm import LLMClient
from app.models.embedder import Embedder
from app.components.translator import Translator
from app.components.retriever import Retriever
from app.components.extractor import Extractor
from app.components.generator import Generator
from app.pipeline import Pipeline
from app.schemas import Params
from app.assets import ensure_ready


def make_llm(base_url: str, api_key: str, model: str) -> LLMClient:
    return LLMClient(base_url=base_url, api_key=api_key, model=model)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Low-resource code summary (paper artifact).")
    p.add_argument("--code-file", required=True)
    p.add_argument("--language", default="ruby")
    p.add_argument("--base-url", required=True)
    p.add_argument("--api-key", default="sk-no-key")
    p.add_argument("--model", required=True)
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--no-trace", action="store_true")
    args = p.parse_args(argv)

    code = Path(args.code_file).read_text()
    ensure_ready(settings)
    llm = make_llm(args.base_url, args.api_key, args.model)
    params = Params(k=args.k)
    pipeline = Pipeline(
        translator=Translator(llm, Embedder(load=settings.load_sbert), params),
        retriever=Retriever(settings.corpus_path, limit=settings.corpus_limit,
                            allow_stubs=settings.allow_stubs),
        extractor=Extractor(settings.extractor_weights,
                            codebert_path=settings.codebert_path,
                            device=settings.device,
                            allow_stubs=settings.allow_stubs),
        generator=Generator(llm),
    )
    resp = pipeline.run(code, args.language, params, trace=not args.no_trace)
    if resp.error:
        print(f"[FAILED at {resp.failed_stage}] {resp.error}")
        return
    print("=== SUMMARY ===")
    print(resp.summary)
    if resp.trace:
        print("\n=== PIVOT ===")
        print(resp.trace.translation.pivot_code)
        print(f"\n=== RETRIEVED ({len(resp.trace.retrieved)}) ===")
        for ex in resp.trace.retrieved:
            print(f"  [{ex.score:.2f}] {ex.summary}")


if __name__ == "__main__":
    main()
