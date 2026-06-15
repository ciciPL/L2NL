# L2NL: Bridging the Data Gap for Low-Resource Code Summarization

Source code for the paper **"Bridging the Data Gap: Leveraging High-Resource Pivots for Low-Resource Code Summarization without Parallel Corpora"**.

The pipeline maps low-resource programming-language (LRPL) code into a high-resource pivot language (Python) and uses a translate-then-generate framework to produce natural-language summaries without any LRPL training data.

---

## Directory Layout

| Folder | Paper section | Contents |
|---|---|---|
| `01_data_prep/` | §4.1 | Parquet parsing, low-resource data filtering, LLM-as-judge, train/eval split |
| `02_translation/` | §3.2 | Multi-temperature sampling, AST-based repair, back-translation candidate selection |
| `03_retrieval/` | §3.3 | BM25 retrieval over high-resource corpus |
| `04_core_block/` | §3.4 | ROUGE-L greedy labeling + CodeBERT-based classifier (binary) for core-statement-block extraction |
| `05_summary_gen/` | §3.5 | Structure-guided LLM prompting (zero-shot / few-shot / BM25-retrieved few-shot) |
| `06_evaluation/` | §4.2 | BLEU-4 / ROUGE-L / METEOR / BERTScore / SBERT / AST parsability |
| `07_baselines/` | §4.4 | Baseline reproduction instructions (points to EACS / ESALE upstream repos; CSN Ruby numbers cited from RaxCS report) |
| `09_human_eval/` | §4.2, §5.1 | Statistical tests reported in the paper: paired Wilcoxon (RQ1 human-eval), Wilson 95% CI (failure-rate CI), Student's-t CI (small-n fallback) |
| `10_figures/` | §5.1 (Fig 5) | Standalone matplotlib reproducer of the temperature × back-translation figure (the manuscript's TikZ/pgfplots version uses the same data) |

---

## Reproducing the Paper

### Prerequisites

- Python 3.10
- PyTorch 2.5.1
- A GPU server (single A100 or 2×4090 works)
- Java 8+ (for METEOR jar in `06_evaluation/meteor/`)

### Installation

```bash
pip install -r requirements.txt
# vllm, transformers, sentence-transformers, spacy, language-tool-python, jina-embeddings, tree-sitter, ...
python -m spacy download en_core_web_sm
```

### Pretrained Models (download separately)

The following LLMs are used as translator / summarizer. Download from HuggingFace and place under `model/`:

- `deepseek-ai/deepseek-coder-1.3b-instruct`
- `deepseek-ai/deepseek-coder-6.7b-instruct`
- `meta-llama/Llama-3.1-8B-Instruct`
- `Qwen/Qwen2.5-Coder-14B-Instruct`
- `ByteDance-Seed/Seed-Coder-8B-Instruct`
- `jinaai/jina-code-embeddings-1.5b` (for SBERT)
- `microsoft/codebert-base` (for core-block classifier)

### Datasets (public, download separately)

- **CodeSearchNet (Ruby)** for high-resource pivot corpus and Ruby test:
  https://github.com/github/CodeSearchNet

- **MultiPL-T** for Julia / Lua / OCaml / R / Racket low-resource test sets:
  https://huggingface.co/datasets/nuprl/MultiPL-T

- **PCSD** for core-block classifier training:
  https://github.com/EdinburghNLP/code-docstring-corpus

After download, place the splits under `dataset/{CSN,LowData,PCSD}/`.

### Pipeline

```bash
# 1. Build the cleaned five-language LRPL test set (Julia/Lua/OCaml/R/Racket)
python 01_data_prep/build_LRPL_dataset_valid500_train1000.py
python 01_data_prep/filter.py
python 01_data_prep/isRealNL.py
python 01_data_prep/getNL_LLM_judge_vllm.py

# 2. Cross-language translation (§3.2): LRPL → Python pivot code
python 02_translation/translate_stepBystep_vllm.py        # multi-temperature
python 02_translation/translate_back_2_lrpl_repair.py     # AST validation + iterative repair
python 02_translation/translate_find_bestCode_from_back_vllm.py  # BLEU+SBERT selection

# 3. Train the core-block classifier (§3.4)
python 04_core_block/make_dataset_label.py        # ROUGE-L greedy labels
python 04_core_block/classfier_BM25.py            # train CodeBERT classifier

# 4. Generate summaries (§3.5, full method)
python 05_summary_gen/finalScript_genNL.py

# 5. Evaluate (§4.2)
python 06_evaluation/eval_hf_real_ref.py          # BLEU/ROUGE/METEOR/BERTScore
python 06_evaluation/ast_acc_deepseek.py          # AST Parsability Rate (APR)

# 6. Reproduce Fig 5
python 10_figures/plot_temperature_picture.py
```

---

## Baselines (Paper §4.4)

See `07_baselines/README.md` for the full reproduction guide. Summary:

- **Group A (LLM prompting)** — Zero-shot / Few-shot variants of our own pipeline; obtained by
  toggling the switches in `05_summary_gen/finalScript_genNL.py`.
- **Group B (open-source pretrained)** — EACS (https://github.com/wssun/EACS) and ESALE
  (https://github.com/NTDXYG/ESALE). Use the authors' code directly with our cleaned LRPL test sets.
- **Group C (CSN Ruby published)** — Numbers cited verbatim from the RaxCS paper's Table; no
  re-implementation needed.

The paper is **training-free at inference time** — no fine-tuning of the main pipeline.

---

## Notes

- **CodeBERT vs UniXcoder**: §3.4 uses **CodeBERT** as the classifier encoder. We tested UniXcoder during exploration; not used in the final pipeline.
- **Llama fine-tuning**: early experiments only; deprecated. The final results in the paper use Llama-3.1-8B-Instruct in inference-only mode.
- **Generation scripts**: `05_summary_gen/finalScript_genNL.py` is the canonical entry point; its three `--use_*` toggles select the four RQ5 ablation variants (Full / w/oKey / w/oRet / w/oMod). `gen_NL_sentences_vllm_icse_prompt_adapter.py` is an alternative ICSE-2025 prompt formulation kept for comparison.

---

## Citation

To be added after acceptance.

## License

MIT (or whatever you prefer — please add a `LICENSE` file).

## Contact

Corresponding author: Xingqi Wang — 570619106@qq.com
