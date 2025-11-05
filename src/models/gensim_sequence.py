from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class EmbeddingSequenceTransformer(TransformerMixin, BaseEstimator):
    """
    Convert text into sequences of pretrained word embeddings using Gensim KeyedVectors.

    - Outputs an array of shape (n_samples, max_len, embedding_dim), dtype float32.
    - Tokens without vectors and padding positions are zeros.
    """

    def __init__(
        self,
        model_name: Optional[str] = "glove-wiki-gigaword-100",
        local_path: Optional[str] = None,
        binary: bool = False,
        lowercase: bool = True,
        max_len: int = 64,
        use_gensim_downloader: bool = True,
    ) -> None:
        self.model_name = model_name
        self.local_path = local_path
        self.binary = binary
        self.lowercase = lowercase
        self.max_len = max_len
        self.use_gensim_downloader = use_gensim_downloader
        self.kv = None
        self.embedding_dim_: Optional[int] = None

    def _load_kv(self):
        try:
            from gensim.models import KeyedVectors
        except ImportError as exc:
            raise ImportError("gensim is required for EmbeddingSequenceTransformer.") from exc

        if self.local_path:
            self.kv = KeyedVectors.load_word2vec_format(self.local_path, binary=self.binary)
        else:
            if not self.use_gensim_downloader:
                raise ValueError("No local_path provided and downloader disabled.")
            if not self.model_name:
                raise ValueError("model_name must be set when using downloader.")
            import gensim.downloader as api
            self.kv = api.load(self.model_name)

        if not hasattr(self.kv, "vector_size"):
            raise RuntimeError("Loaded KeyedVectors does not expose vector_size.")
        self.embedding_dim_ = int(self.kv.vector_size)

    def fit(self, X: Sequence[str], y: Optional[Sequence[int]] = None):
        if self.kv is None:
            self._load_kv()
        return self

    def _tokenize(self, text: str) -> Iterable[str]:
        try:
            from gensim.utils import simple_preprocess
            return simple_preprocess(text if isinstance(text, str) else "", deacc=True, min_len=2)
        except Exception:
            if not isinstance(text, str):
                return []
            s = text.lower() if self.lowercase else text
            import re
            return re.findall(r"\b[a-zA-Z]{2,}\b", s)

    def transform(self, X: Sequence[str]) -> np.ndarray:
        if self.kv is None or self.embedding_dim_ is None:
            raise RuntimeError("Transformer not fitted. Call fit before transform.")
        dim = self.embedding_dim_
        out = np.zeros((len(X), self.max_len, dim), dtype=np.float32)
        key_index = getattr(self.kv, "key_to_index", None)
        for i, text in enumerate(X):
            toks = list(self._tokenize(text))
            if self.lowercase:
                toks = [t.lower() for t in toks]
            j = 0
            for t in toks:
                if j >= self.max_len:
                    break
                if (key_index is not None and t in key_index) or t in self.kv:
                    try:
                        out[i, j] = self.kv[t]
                        j += 1
                    except KeyError:
                        continue
            # remainder stays zeros (padding)
        return out

