from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from src.utils.data_utils import load_dataset
from src.utils.io_utils import ensure_dir, save_csv


RESULTS_DIR = Path("results/analysis")


def _length_stats(series: pd.Series) -> Dict[str, float]:
    arr = series.to_numpy()
    return {
        "count": float(series.shape[0]),
        "mean": float(np.mean(arr)) if arr.size else 0.0,
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "min": float(np.min(arr)) if arr.size else 0.0,
        "median": float(np.median(arr)) if arr.size else 0.0,
        "max": float(np.max(arr)) if arr.size else 0.0,
    }


def main() -> None:
    ensure_dir(RESULTS_DIR)
    df = load_dataset(force_refresh=False)

    n_records = len(df)
    n_classes = df["label"].nunique()

    # Sentence lengths (characters and tokens) for raw and cleaned text
    df["tweet_len_chars"] = df["tweet"].fillna("").astype(str).str.len()
    df["text_len_chars"] = df["text"].fillna("").astype(str).str.len()
    df["text_len_tokens"] = df["text"].fillna("").astype(str).str.split().map(len)

    overall_stats = pd.DataFrame([
        {
            "metric": "tweet_len_chars",
            **_length_stats(df["tweet_len_chars"]),
        },
        {
            "metric": "text_len_chars",
            **_length_stats(df["text_len_chars"]),
        },
        {
            "metric": "text_len_tokens",
            **_length_stats(df["text_len_tokens"]),
        },
    ])
    save_csv(overall_stats, RESULTS_DIR / "length_stats_overall.csv")

    # Unified per-class summary: counts/percent + length stats
    counts = df["label"].value_counts().sort_index()
    percents = counts / counts.sum() * 100.0
    per_class_records = []
    for label, group in df.groupby("label"):
        rec = {
            "label": int(label),
            "count": int(len(group)),
            "percent": float(100.0 * len(group) / n_records) if n_records else 0.0,
        }
        rec.update({f"tweet_len_chars_{k}": v for k, v in _length_stats(group["tweet_len_chars"]).items()})
        rec.update({f"text_len_chars_{k}": v for k, v in _length_stats(group["text_len_chars"]).items()})
        rec.update({f"text_len_tokens_{k}": v for k, v in _length_stats(group["text_len_tokens"]).items()})
        per_class_records.append(rec)
    per_class_summary = pd.DataFrame(per_class_records).sort_values("label")
    save_csv(per_class_summary, RESULTS_DIR / "per_class_summary.csv")

    # Compact summary
    summary = pd.DataFrame([
        {
            "n_records": n_records,
            "n_classes": n_classes,
            "avg_tokens": float(df["text_len_tokens"].mean()),
            "median_tokens": float(df["text_len_tokens"].median()),
        }
    ])
    save_csv(summary, RESULTS_DIR / "dataset_summary.csv")

    # Optional: basic histogram of token lengths
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(df["text_len_tokens"], bins=50, color="#3366cc", alpha=0.8)
        ax.set_title("Token length distribution (cleaned text)")
        ax.set_xlabel("# tokens")
        ax.set_ylabel("# tweets")
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / "token_length_hist.png", dpi=150)
        plt.close(fig)
    except Exception:
        # Plotting is optional; skip failures silently
        pass

    print(
        f"Dataset summary saved to {RESULTS_DIR}. "
        f"Records={n_records}, Classes={n_classes}"
    )


if __name__ == "__main__":
    main()
