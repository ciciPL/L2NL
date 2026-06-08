from __future__ import annotations
from app.schemas import CoreBlock


class Extractor:
    """Core-statement-block extraction: AST split + trained binary classifier.

    STUB: splits by non-empty lines and labels each as 'other' with prob 1.0.
    Replace `__init__` to load classifier weights and `extract` with AST semantic
    splitting + classifier scoring + threshold filtering. Signatures are final.
    """

    def __init__(self, weights_path: str | None = None):
        self.weights_path = weights_path

    def extract(self, code: str, threshold: float) -> list[CoreBlock]:
        blocks: list[CoreBlock] = []
        for line in code.splitlines():
            stripped = line.strip()
            if stripped:
                blocks.append(
                    CoreBlock(text=stripped, block_type="other", prob=1.0)
                )
        return blocks
