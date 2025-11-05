from __future__ import annotations

import re
from collections import Counter
from typing import Iterable, Optional, Sequence

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin

import torch
from torch import nn


TOKEN_PATTERN = re.compile(r"\b[a-z]{2,}\b")


class LSTMSequenceModule(nn.Module):
    """LSTM classifier that consumes sequences of embedding vectors.

    Input: float tensor (batch, seq_len, embedding_dim). Zeros are treated as PAD.
    """

    def __init__(
        self,
        embedding_dim: int = 100,
        hidden_size: int = 64,
        num_classes: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, E)
        out, _ = self.lstm(x)
        # treat rows with all-zeros as padding
        mask = (x.abs().sum(dim=2) != 0)  # (B, T)
        lengths = mask.long().sum(dim=1).clamp(min=1)
        last_idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, out.size(2))
        last_out = out.gather(dim=1, index=last_idx).squeeze(1)
        last_out = self.dropout(last_out)
        return self.fc(last_out)
