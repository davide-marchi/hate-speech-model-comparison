from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class SentenceTransformerEncoder(TransformerMixin, BaseEstimator):
    """
    Transformer that converts text into sentence embeddings using Sentence-Transformers.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 32,
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self._model = None

    def fit(self, X: Sequence[str], y: Optional[Sequence[int]] = None):
        self._ensure_model()
        return self

    def transform(self, X: Sequence[str]) -> np.ndarray:
        self._ensure_model()
        sentences = list(X)
        embeddings = self._model.encode(
            sentences,
            batch_size=self.batch_size,
            show_progress_bar=False,
            device=self.device,
            convert_to_numpy=True,
        )
        return embeddings

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "sentence-transformers is required for SentenceTransformerEncoder."
            ) from exc

        self._model = SentenceTransformer(self.model_name, device=self.device)

