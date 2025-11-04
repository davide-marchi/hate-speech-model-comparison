from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Sequence

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


TOKEN_PATTERN = re.compile(r"\b[a-z]{2,}\b")


class KeywordRegexClassifier(BaseEstimator, ClassifierMixin):
    """
    Simple keyword-based classifier that assigns labels based on frequent class-specific tokens.
    """

    def __init__(
        self,
        num_keywords: int = 25,
        min_token_length: int = 3,
        exclude_stopwords: bool = True,
    ) -> None:
        self.num_keywords = num_keywords
        self.min_token_length = min_token_length
        self.exclude_stopwords = exclude_stopwords

    def fit(self, X: Sequence[str], y: Sequence[int]) -> "KeywordRegexClassifier":
        X = np.asarray(X)
        y = np.asarray(y)
        self.classes_, counts = np.unique(y, return_counts=True)
        self.default_class_ = int(self.classes_[np.argmax(counts)])

        tokens_per_class: Dict[int, Counter] = defaultdict(Counter)
        for text, label in zip(X, y):
            tokens = self._tokenize(text)
            tokens_per_class[label].update(tokens)

        self.class_keywords_: Dict[int, List[str]] = {}
        for label, counter in tokens_per_class.items():
            most_common = [
                token
                for token, _ in counter.most_common(self.num_keywords)
                if len(token) >= self.min_token_length
            ]
            if self.exclude_stopwords:
                most_common = [
                    token for token in most_common if token not in ENGLISH_STOP_WORDS
                ]
            self.class_keywords_[int(label)] = most_common[: self.num_keywords]

        return self

    def predict(self, X: Sequence[str]) -> np.ndarray:
        if not hasattr(self, "class_keywords_"):
            raise RuntimeError("The classifier must be fitted before calling predict.")

        predictions: List[int] = []
        for text in X:
            tokens = set(self._tokenize(text))
            class_scores = {
                label: sum(1 for token in tokens if token in keywords)
                for label, keywords in self.class_keywords_.items()
            }
            best_label = self._choose_label(class_scores)
            predictions.append(best_label)

        return np.asarray(predictions)

    def _choose_label(self, class_scores: Dict[int, int]) -> int:
        if not class_scores:
            return self.default_class_

        max_score = max(class_scores.values())
        if max_score == 0:
            return self.default_class_

        winning_labels = [
            label for label, score in class_scores.items() if score == max_score
        ]
        return int(sorted(winning_labels)[0])

    def _tokenize(self, text: str) -> Iterable[str]:
        if not isinstance(text, str):
            return []
        lowered = text.lower()
        return TOKEN_PATTERN.findall(lowered)

