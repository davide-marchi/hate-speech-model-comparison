from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from src.utils.io_utils import ensure_dir, save_csv


RESULTS_ROOT = Path("results")
OUT_DIR = RESULTS_ROOT / "analysis"


def _load_metrics() -> pd.DataFrame:
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


def _pareto_front(f1: np.ndarray, cost: np.ndarray) -> np.ndarray:
    """Boolean mask of non-dominated points (maximize f1, minimize cost).

    A point j is dominated by i if f1[i] >= f1[j] and cost[i] <= cost[j] with at least
    one strict inequality. We keep points that are not dominated by any other point.
    """
    n = len(f1)
    if n == 0:
        return np.array([], dtype=bool)
    on_front = np.ones(n, dtype=bool)
    for i in range(n):
        if not on_front[i]:
            continue
        dominated_by_i = (f1 <= f1[i]) & (cost >= cost[i]) & ((f1 < f1[i]) | (cost > cost[i]))
        dominated_by_i[i] = False
        on_front[dominated_by_i] = False
    return on_front


def _plot_scatter(df: pd.DataFrame, x: str, y: str, title: str, out_path: Path, logx: bool = False) -> None:
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        xs = df[x].to_numpy()
        ys = df[y].to_numpy()
        ax.scatter(xs, ys, c="#1f77b4")
        for _, row in df.iterrows():
            ax.annotate(row.get("model", row.get("results_dir", "")), (row[x], row[y]), textcoords="offset points", xytext=(3, 3), fontsize=8)
        ax.set_xlabel(x.replace("_", " "))
        ax.set_ylabel(y.replace("_", " "))
        ax.set_title(title)
        if logx:
            ax.set_xscale("log")
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    except Exception:
        pass


def _plot_pareto(df: pd.DataFrame, cost_col: str, out_path: Path, title: str) -> None:
    try:
        import matplotlib.pyplot as plt

        valid = df[["f1_macro", cost_col]].dropna()
        if valid.empty:
            return
        f1 = valid["f1_macro"].to_numpy()
        cost = valid[cost_col].to_numpy()
        mask = _pareto_front(f1, cost)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(cost, f1, c="#7f7f7f", label="Models")
        front = valid[mask].sort_values(cost_col)
        ax.plot(front[cost_col], front["f1_macro"], "-o", color="#d62728", label="Pareto front")
        for _, row in valid.iterrows():
            ax.annotate(row.get("model", row.name), (row[cost_col], row["f1_macro"]), textcoords="offset points", xytext=(3, 3), fontsize=8)
        ax.set_xlabel(cost_col.replace("_", " "))
        ax.set_ylabel("f1_macro")
        ax.set_title(title)
        if "emissions" in cost_col:
            ax.set_xscale("log")
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    except Exception:
        pass


def _plot_grouped_bars(df: pd.DataFrame, columns: list[str], labels: list[str], title: str, y_label: str, out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt

        names = df.get("model", df.get("results_dir", df.index.astype(str))).astype(str).tolist()
        x = np.arange(len(names))
        width = 0.38

        vals_a = df[columns[0]].fillna(0.0).to_numpy(dtype=float)
        vals_b = df[columns[1]].fillna(0.0).to_numpy(dtype=float)

        fig, ax = plt.subplots(figsize=(max(6, len(names) * 0.8), 4))
        rects1 = ax.bar(x - width/2, vals_a, width, label=labels[0], color="#1f77b4")
        rects2 = ax.bar(x + width/2, vals_b, width, label=labels[1], color="#ff7f0e")

        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.set_xticks(x, names, rotation=35, ha="right")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    except Exception:
        pass


def _plot_combined_time_emissions(df: pd.DataFrame, out_path: Path) -> None:
    """Single figure with two subplots: top=time, bottom=emissions."""
    try:
        import matplotlib.pyplot as plt

        names = df.get("model", df.get("results_dir", df.index.astype(str))).astype(str).tolist()
        x = np.arange(len(names))
        width = 0.38

        tr_time = df["train_duration_s"].fillna(0.0).to_numpy(dtype=float)
        te_time = df["test_duration_s"].fillna(0.0).to_numpy(dtype=float)
        tr_em = df.get("train_emissions_kg", pd.Series([np.nan]*len(df))).fillna(0.0).to_numpy(dtype=float)
        te_em = df.get("test_emissions_kg", pd.Series([np.nan]*len(df))).fillna(0.0).to_numpy(dtype=float)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(6, len(names) * 0.8), 6), sharex=True)
        # Time subplot
        ax1.bar(x - width/2, tr_time, width, label="Train/Validation time", color="#1f77b4")
        ax1.bar(x + width/2, te_time, width, label="Test time", color="#ff7f0e")
        ax1.set_ylabel("seconds")
        ax1.set_title("Runtimes by model")
        ax1.legend()

        # Emissions subplot
        ax2.bar(x - width/2, tr_em, width, label="Train/Validation emissions", color="#2ca02c")
        ax2.bar(x + width/2, te_em, width, label="Test emissions", color="#d62728")
        ax2.set_ylabel("kg CO2e")
        ax2.set_title("Emissions by model")
        ax2.set_xticks(x, names, rotation=35, ha="right")
        ax2.legend()

        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    except Exception:
        pass


def main() -> None:
    ensure_dir(OUT_DIR)
    df = _load_metrics()
    # Local efficiency computation (mirrors aggregate_results)
    def _add_efficiency_local(df_in: pd.DataFrame) -> pd.DataFrame:
        df_in = df_in.copy()
        def safe_ratio(num, den):
            try:
                den = float(den)
                if den <= 0:
                    return float("nan")
                return float(num) / den
            except Exception:
                return float("nan")
        df_in["eff_test_f1_per_s"] = [safe_ratio(f1, t) for f1, t in zip(df_in["f1_macro"], df_in["test_duration_s"])]
        df_in["eff_train_cv_f1_per_s"] = [safe_ratio(cv, t) for cv, t in zip(df_in.get("best_score_cv", [float("nan")]*len(df_in)), df_in["train_duration_s"])]
        df_in["eff_test_f1_per_kg"] = [safe_ratio(f1, e) for f1, e in zip(df_in["f1_macro"], df_in.get("test_emissions_kg", [float("nan")]*len(df_in)))]
        df_in["eff_train_cv_f1_per_kg"] = [safe_ratio(cv, e) for cv, e in zip(df_in.get("best_score_cv", [float("nan")]*len(df_in)), df_in.get("train_emissions_kg", [float("nan")]*len(df_in)))]
        return df_in

    df2 = _add_efficiency_local(df)

    # Scatter plots: F1 vs time/emissions
    _plot_scatter(df2, x="test_duration_s", y="f1_macro", title="F1 vs test time", out_path=OUT_DIR / "scatter_f1_vs_test_time.png", logx=True)
    if "test_emissions_kg" in df2.columns and df2["test_emissions_kg"].notna().any():
        _plot_scatter(df2, x="test_emissions_kg", y="f1_macro", title="F1 vs test emissions", out_path=OUT_DIR / "scatter_f1_vs_test_emissions.png", logx=True)

    # Pareto fronts
    _plot_pareto(df2, cost_col="test_duration_s", out_path=OUT_DIR / "pareto_test_time.png", title="Pareto front (min time, max F1)")
    if df2["test_emissions_kg"].notna().any():
        _plot_pareto(df2, cost_col="test_emissions_kg", out_path=OUT_DIR / "pareto_test_emissions.png", title="Pareto front (min emissions, max F1)")

    # Grouped bars for emissions (train/test) and durations (train/test)
    name_col = df2.get("model", df2.get("results_dir"))
    df_plot = df2.copy()
    if name_col is None:
        df_plot["model"] = [f"model_{i}" for i in range(len(df_plot))]
    # Keep columns and order stable
    cols_keep = [c for c in ["model", "results_dir", "train_emissions_kg", "test_emissions_kg", "train_duration_s", "test_duration_s"] if c in df_plot.columns]
    df_plot = df_plot[cols_keep]

    # Sort by model name for stable x-axis
    sort_key = "model" if "model" in df_plot.columns else "results_dir"
    df_plot = df_plot.sort_values(sort_key)

    # Emissions bars
    if "train_emissions_kg" in df_plot.columns and "test_emissions_kg" in df_plot.columns:
        _plot_grouped_bars(
            df_plot,
            columns=["train_emissions_kg", "test_emissions_kg"],
            labels=["Train/Validation emissions", "Test emissions"],
            title="Emissions by model",
            y_label="kg CO2e",
            out_path=OUT_DIR / "bars_emissions.png",
        )

    # Duration bars
    _plot_grouped_bars(
        df_plot,
        columns=["train_duration_s", "test_duration_s"],
        labels=["Train/Validation time", "Test time"],
        title="Runtimes by model",
        y_label="seconds",
        out_path=OUT_DIR / "bars_time.png",
    )

    # Combined 2x1 figure
    _plot_combined_time_emissions(df_plot, OUT_DIR / "bars_time_emissions.png")

    print(f"Saved efficiency plots to {OUT_DIR}")


if __name__ == "__main__":
    main()
