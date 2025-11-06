from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import PredefinedSplit, train_test_split


RAW_DATA_PATH = Path("dataset/train_E6oV3lV.csv")
PROCESSED_DIR = Path("dataset/processed")
PROCESSED_PATH = PROCESSED_DIR / "train_clean.csv"

MENTION_PATTERN = re.compile(r"@\w+")
URL_PATTERN = re.compile(r"http\S+|www\.\S+")
NON_ALPHANUM = re.compile(r"[^a-z0-9\s]")


def clean_text(text: str) -> str:
    """Basic text normalisation for the Twitter hate speech dataset."""
    if not isinstance(text, str):
        text = "" if pd.isna(text) else str(text)
    ascii_text = text.encode("ascii", errors="ignore").decode("ascii")
    lowered = ascii_text.lower()
    no_urls = URL_PATTERN.sub(" ", lowered)
    no_mentions = MENTION_PATTERN.sub(" ", no_urls)
    no_hashtags = no_mentions.replace("#", " ")
    without_punct = NON_ALPHANUM.sub(" ", no_hashtags)
    collapsed_spaces = re.sub(r"\s+", " ", without_punct)
    return collapsed_spaces.strip()


def load_dataset(force_refresh: bool = False) -> pd.DataFrame:
    """Load and clean the dataset, caching the cleaned version on disk."""
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Raw dataset not found at {RAW_DATA_PATH}")

    if PROCESSED_PATH.exists() and not force_refresh:
        return pd.read_csv(PROCESSED_PATH)

    df = pd.read_csv(RAW_DATA_PATH)
    if "tweet" not in df.columns or "label" not in df.columns:
        raise ValueError("Expected columns 'tweet' and 'label' in the dataset.")

    df["text"] = df["tweet"].apply(clean_text)
    processed_df = df[["id", "label", "tweet", "text"]].copy()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    processed_df.to_csv(PROCESSED_PATH, index=False)
    return processed_df


@dataclass
class DatasetSplits:
    X_train: pd.Series
    y_train: pd.Series
    X_val: pd.Series
    y_val: pd.Series
    X_test: pd.Series
    y_test: pd.Series

    def combined_train_val(self) -> Tuple[pd.Series, pd.Series, PredefinedSplit]:
        """Return stacked train+validation data with a predefined split for grid-search."""
        X = pd.concat([self.X_train, self.X_val], axis=0).reset_index(drop=True)
        y = pd.concat([self.y_train, self.y_val], axis=0).reset_index(drop=True)
        test_fold = [-1] * len(self.X_train) + [0] * len(self.X_val)
        predefined_split = PredefinedSplit(test_fold=test_fold)
        return X, y, predefined_split


def get_dataset_splits(
    *,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    force_refresh: bool = False,
    text_column: str = "text",
) -> DatasetSplits:
    """Return train/validation/test splits with stratification."""
    if not 0 < test_size < 0.5:
        raise ValueError("test_size must be between 0 and 0.5.")
    if not 0 < val_size < 0.5:
        raise ValueError("val_size must be between 0 and 0.5.")

    df = load_dataset(force_refresh=force_refresh)
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in the dataset.")

    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df["label"],
        random_state=random_state,
    )

    relative_val_size = val_size / (1 - test_size)

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=relative_val_size,
        stratify=train_val_df["label"],
        random_state=random_state,
    )

    def _ensure_text(series: pd.Series) -> pd.Series:
        return series.fillna("").astype(str).reset_index(drop=True)

    return DatasetSplits(
        X_train=_ensure_text(train_df[text_column]),
        y_train=train_df["label"].reset_index(drop=True),
        X_val=_ensure_text(val_df[text_column]),
        y_val=val_df["label"].reset_index(drop=True),
        X_test=_ensure_text(test_df[text_column]),
        y_test=test_df["label"].reset_index(drop=True),
    )
