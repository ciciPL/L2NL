from __future__ import annotations
import os
import sys
from pathlib import Path
from app.schemas import CoreBlock
from app.components._py_split import split_python_by_structure

# Vendored EASC classifier code (SelectorNet + utils) lives here; added to
# sys.path lazily so `from model import SelectorNet` / `from utils import ...`
# (the research code's flat imports) resolve only when the real model is loaded.
_VENDOR = str(Path(__file__).resolve().parents[2] / "vendor" / "easc")

# Hyperparameters from vendor/easc/classfier_BM25.py + train.py (must match the
# trained checkpoint).
_MAX_WORD = 32
_MAX_STAT = 32
_WORD_HIDDEN = 128
_STAT_HIDDEN = 256

# (parsed dict key, CoreBlock.block_type) in the exact order make_dataset_label.py
# flattens them into cleaned_seqs.
_BLOCK_KEYS = [
    ("loops", "loop"),
    ("conditionals", "conditional"),
    ("assignments", "assignment"),
    ("others", "other"),
]


def build_seq_items(code: str) -> list[tuple[str, str, str]]:
    """Split Python code into (raw_segment, cleaned_seq, block_type) tuples.

    Mirrors make_dataset_label.py: function_def first (as a 'signature' block),
    then loops/conditionals/assignments/others; each cleaned seq is whitespace-
    normalized and lowercased exactly as the classifier was trained on.
    """
    parsed, ok = split_python_by_structure(code)
    if not ok:
        return []
    items: list[tuple[str, str, str]] = []
    func = parsed.get("function_def", "")
    if func and func.strip():
        items.append((func, " ".join(func.split()).lower(), "signature"))
    for key, btype in _BLOCK_KEYS:
        for stmt in parsed.get(key, []):
            cleaned = " ".join(stmt.split()).lower()
            if cleaned.strip():
                items.append((stmt, cleaned, btype))
    return items


class Extractor:
    """Core-statement-block extraction (paper §3.4).

    AST structural split -> CodeBERT/SelectorNet binary classifier. When no
    checkpoint is configured (e.g. the stub shell or local dev without torch),
    falls back to a naive line split so the pipeline still runs.
    """

    def __init__(self, weights_path: str | None = None,
                 codebert_path: str = "microsoft/codebert-base",
                 device: str | None = None):
        self.weights_path = weights_path
        self.codebert_path = codebert_path
        self._device_pref = device
        self._model = None
        self._tokenizer = None
        self._torch = None
        self._dev = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import RobertaConfig, RobertaModel, RobertaTokenizer
        if _VENDOR not in sys.path:
            sys.path.insert(0, _VENDOR)
        from model import SelectorNet  # vendored

        self._torch = torch
        dev = self._device_pref or ("cuda" if torch.cuda.is_available() else "cpu")
        self._dev = torch.device(dev)

        config = RobertaConfig.from_pretrained(self.codebert_path)
        self._tokenizer = RobertaTokenizer.from_pretrained(self.codebert_path)
        encoder = RobertaModel.from_pretrained(self.codebert_path, config=config)
        word_embeddings_weight = encoder.embeddings.word_embeddings.weight

        model = SelectorNet(
            batch_size=1, word_embeddings_weight=word_embeddings_weight,
            word_hidden_size=_WORD_HIDDEN, stat_hidden_size=_STAT_HIDDEN,
            max_word_len=_MAX_WORD, max_stat_len=_MAX_STAT,
            vocab_size=config.vocab_size, embed_size=config.hidden_size,
            num_classes=2, imbalance_loss_fct=True,
        )
        model.load_state_dict(torch.load(self.weights_path, map_location=self._dev))
        model.to(self._dev).eval()
        self._model = model

    def _classify(self, items: list[tuple[str, str, str]], threshold: float
                  ) -> list[CoreBlock]:
        torch = self._torch
        tok = self._tokenizer
        items = items[:_MAX_STAT]

        source_ids, word_masks = [], []
        for _raw, cleaned, _bt in items:
            wt = tok.tokenize(cleaned)[:_MAX_WORD]
            ids = tok.convert_tokens_to_ids(wt)
            wm = [1] * len(ids)
            pad = _MAX_WORD - len(ids)
            ids += [tok.pad_token_id] * pad
            wm += [0] * pad
            source_ids.append(ids)
            word_masks.append(wm)

        stat_masks = [1] * len(items)
        pad = _MAX_STAT - len(items)
        stat_masks += [0] * pad
        source_ids += [[tok.pad_token_id] * _MAX_WORD] * pad
        word_masks += [[0] * _MAX_WORD] * pad

        # The vendored SelectorNet squeezes the batch dim, which collapses shapes
        # when batch_size==1. Run a batch of 2 (the same example twice) and keep
        # the first example's statement predictions.
        si = torch.tensor([source_ids, source_ids], dtype=torch.long, device=self._dev)
        wm = torch.tensor([word_masks, word_masks], dtype=torch.long, device=self._dev)
        sm = torch.tensor([stat_masks, stat_masks], dtype=torch.long, device=self._dev)

        with torch.no_grad():
            _num, active_mask, probs = self._model(si, wm, sm, None)
        active_probs = probs[active_mask][:len(items)]  # first copy's real statements
        p1 = active_probs[:, 1].tolist()                # P(core) per statement

        # Paper §3.4: select blocks whose core probability exceeds the threshold
        # (threshold=0.5 is equivalent to argmax; lower it to be more inclusive).
        out: list[CoreBlock] = []
        for (raw, _cleaned, btype), prob in zip(items, p1):
            if prob >= threshold:
                out.append(CoreBlock(text=raw, block_type=btype, prob=float(prob)))
        # The classifier is weakly calibrated (probabilities cluster near 0.5); for
        # some functions every statement falls just below the threshold. Keep the
        # single most-core block so a parseable function never yields an empty trace.
        if not out and items:
            i = max(range(len(items)), key=lambda j: p1[j])
            raw, _cleaned, btype = items[i]
            out = [CoreBlock(text=raw, block_type=btype, prob=float(p1[i]))]
        return out

    def extract(self, code: str, threshold: float) -> list[CoreBlock]:
        if not self.weights_path or not os.path.exists(self.weights_path):
            return self._fallback(code)
        items = build_seq_items(code)
        if not items:
            return []
        self._ensure_loaded()
        return self._classify(items, threshold)

    @staticmethod
    def _fallback(code: str) -> list[CoreBlock]:
        blocks: list[CoreBlock] = []
        for line in code.splitlines():
            s = line.strip()
            if s:
                blocks.append(CoreBlock(text=s, block_type="other", prob=1.0))
        return blocks
