from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    average: str = "macro",
) -> Dict[str, float]:
    """Return standard classification metrics."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average=average,
        zero_division=0,
    )
    accuracy = accuracy_score(y_true, y_pred)
    return {
        "accuracy": float(accuracy),
        f"precision_{average}": float(precision),
        f"recall_{average}": float(recall),
        f"f1_{average}": float(f1),
    }


def build_results_frame(metrics: Dict[str, float]) -> pd.DataFrame:
    """Wrap metrics in a single-row dataframe for easier exporting."""
    return pd.DataFrame([metrics])

