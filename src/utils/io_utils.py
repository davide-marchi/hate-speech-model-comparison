from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import pandas as pd


def ensure_dir(path: Path) -> Path:
    """Create a directory if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_csv(data: pd.DataFrame, path: Path) -> Path:
    """Persist a dataframe to CSV, creating parent directories as required."""
    ensure_dir(path.parent)
    data.to_csv(path, index=False)
    return path


def dict_to_frame(record: Dict[str, float]) -> pd.DataFrame:
    """Convert a dictionary to a single-row DataFrame."""
    return pd.DataFrame([record])


def iterable_to_frame(records: Iterable[Dict[str, float]]) -> pd.DataFrame:
    """Convert an iterable of dictionaries to a DataFrame."""
    return pd.DataFrame(list(records))

