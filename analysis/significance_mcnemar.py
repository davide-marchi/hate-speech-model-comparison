from __future__ import annotations

from itertools import combinations
from math import comb, lgamma, log, exp
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.utils.io_utils import ensure_dir, save_csv


RESULTS_ROOT = Path("results")
OUT_DIR = RESULTS_ROOT / "analysis"


def _load_predictions() -> Dict[str, pd.DataFrame]:
    """Load per-model predictions from results/*/test_predictions.csv.

    Returns mapping model_name -> dataframe with columns: idx, text, true_label, predicted_label
    """
    pred_map: Dict[str, pd.DataFrame] = {}
    for sub in sorted(RESULTS_ROOT.iterdir()):
        if not sub.is_dir() or sub.name == "analysis":
            continue
        path = sub / "test_predictions.csv"
        if path.exists():
            df = pd.read_csv(path)
            if "true_label" not in df.columns or "predicted_label" not in df.columns:
                continue
            df = df.reset_index(drop=True).reset_index().rename(columns={"index": "idx"})
            pred_map[sub.name] = df[["idx", "text", "true_label", "predicted_label"]]
    if not pred_map:
        raise FileNotFoundError("No test_predictions.csv files found under results/*/")
    return pred_map


def _align_frames(frames: List[pd.DataFrame]) -> List[pd.DataFrame]:
    """Attempt to align by idx; if true labels disagree, fallback to text+true_label with tie-id."""
    # First try by idx
    by_idx = frames
    first_true = by_idx[0]["true_label"].to_numpy()
    if all(np.array_equal(first_true, f["true_label"].to_numpy()) for f in by_idx[1:]):
        return by_idx

    # Fallback: build a composite key text + true_label + within-key order (to disambiguate duplicates)
    aligned: List[pd.DataFrame] = []
    for f in frames:
        g = f.copy()
        g["_dup_order"] = g.groupby(["text", "true_label"]).cumcount()
        g["join_key"] = g["text"].astype(str) + "\u0001" + g["true_label"].astype(str) + "\u0001" + g["_dup_order"].astype(str)
        aligned.append(g)
    # Join keys from first frame
    base = aligned[0][["join_key", "true_label", "predicted_label"]].rename(columns={"predicted_label": "pred_0"})
    merged = base
    for i, f in enumerate(aligned[1:], start=1):
        merged = merged.merge(
            f[["join_key", "predicted_label"]].rename(columns={"predicted_label": f"pred_{i}"}),
            on="join_key",
            how="inner",
        )
    # Rebuild frames from merged result
    if merged.empty:
        raise RuntimeError("Failed to align prediction files across models.")
    out_frames: List[pd.DataFrame] = []
    out_frames.append(pd.DataFrame({"true_label": merged["true_label"], "predicted_label": merged["pred_0"]}))
    for i in range(1, len(frames)):
        out_frames.append(pd.DataFrame({"true_label": merged["true_label"], "predicted_label": merged[f"pred_{i}"]}))
    return out_frames


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value using binomial tail in log-space.

    Avoids overflow/underflow for large n by summing log probabilities.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)

    # log(0.5^n) = -n*log(2)
    log_half_pow = -n * log(2.0)

    # log C(n, i) via lgamma
    def logC(n: int, i: int) -> float:
        return lgamma(n + 1) - lgamma(i + 1) - lgamma(n - i + 1)

    # log-sum-exp over i=0..k of [log C(n,i) - n*log(2)]
    terms = [logC(n, i) + log_half_pow for i in range(0, k + 1)]
    m = max(terms)
    s = sum(exp(t - m) for t in terms)
    log_tail = m + log(s)
    # Two-sided p-value
    p = 2.0 * exp(log_tail)
    if p > 1.0:
        p = 1.0
    return float(p)


def compare_pair(y_true: np.ndarray, y_a: np.ndarray, y_b: np.ndarray) -> Tuple[int, int, float, float]:
    correct_a = (y_a == y_true)
    correct_b = (y_b == y_true)
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    n = b + c
    p = mcnemar_exact(b, c)
    advantage = (b - c) / n if n > 0 else 0.0
    return b, c, advantage, p


def main() -> None:
    ensure_dir(OUT_DIR)
    pred_map = _load_predictions()
    names = list(pred_map.keys())
    frames = [pred_map[n] for n in names]
    frames = _align_frames(frames)

    y_true = frames[0]["true_label"].to_numpy()
    preds = [f["predicted_label"].to_numpy() for f in frames]

    records: List[Dict[str, object]] = []
    for (i, j) in combinations(range(len(names)), 2):
        b, c, advantage, p = compare_pair(y_true, preds[i], preds[j])
        rec = {
            "model_a": names[i],
            "model_b": names[j],
            "b_a_correct_b_wrong": b,
            "c_a_wrong_b_correct": c,
            "n_discordant": b + c,
            "advantage_sign": np.sign(advantage),
            "advantage_frac": advantage,
            "p_value": p,
            "significant_0.05": bool(p < 0.05),
        }
        records.append(rec)

    df = pd.DataFrame(records).sort_values("p_value")
    save_csv(df, OUT_DIR / "significance_mcnemar.csv")
    print(f"Saved McNemar pairwise results to {OUT_DIR / 'significance_mcnemar.csv'}")


if __name__ == "__main__":
    main()
