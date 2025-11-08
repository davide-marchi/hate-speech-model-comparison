from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.utils.io_utils import ensure_dir, save_csv


RESULTS_ROOT = Path("results")
OUT_DIR = RESULTS_ROOT / "analysis"


def _load_all_test_metrics() -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for sub in sorted(RESULTS_ROOT.iterdir()):
        if not sub.is_dir() or sub.name == "analysis":
            continue
        metrics_path = sub / "test_metrics.csv"
        if metrics_path.exists():
            df = pd.read_csv(metrics_path)
            df["results_dir"] = sub.name
            rows.append(df)
    if not rows:
        raise FileNotFoundError("No test_metrics.csv files found under results/*/")
    return pd.concat(rows, ignore_index=True)


def _add_efficiency(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Derived efficiency ratios; handle zeros and missing emissions
    def safe_ratio(num, den):
        try:
            den = float(den)
            if den <= 0:
                return float("nan")
            return float(num) / den
        except Exception:
            return float("nan")

    df["eff_test_f1_per_s"] = [safe_ratio(f1, t) for f1, t in zip(df["f1_macro"], df["test_duration_s"])]
    df["eff_train_cv_f1_per_s"] = [safe_ratio(cv, t) for cv, t in zip(df.get("best_score_cv", [float("nan")]*len(df)), df["train_duration_s"])]
    df["eff_test_f1_per_kg"] = [safe_ratio(f1, e) for f1, e in zip(df["f1_macro"], df.get("test_emissions_kg", [float("nan")]*len(df)))]
    df["eff_train_cv_f1_per_kg"] = [safe_ratio(cv, e) for cv, e in zip(df.get("best_score_cv", [float("nan")]*len(df)), df.get("train_emissions_kg", [float("nan")]*len(df)))]
    return df


def _drop_unwanted_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop only known unwanted columns (e.g., 'selected_template') and
    preserve every other column as-is to avoid surprising downstream changes.
    """
    df = df.copy()
    for col in ["selected_template"]:
        if col in df.columns:
            df = df.drop(columns=[col])
    return df


def main() -> None:
    ensure_dir(OUT_DIR)
    df = _load_all_test_metrics()
    df = _add_efficiency(df)
    df = _drop_unwanted_columns(df)
    # Sort helpful view by test F1 desc then test time asc
    df_sorted = df.sort_values(["f1_macro", "test_duration_s"], ascending=[False, True])
    save_csv(df_sorted, OUT_DIR / "summary.csv")
    print(f"Saved aggregated summary with {len(df_sorted)} rows to {OUT_DIR / 'summary.csv'}")


if __name__ == "__main__":
    main()
