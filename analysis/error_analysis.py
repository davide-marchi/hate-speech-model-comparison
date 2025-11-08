from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from src.utils.io_utils import ensure_dir, save_csv


def _mpl_setup() -> None:
    try:
        import matplotlib as mpl
        mpl.rcParams.update(
            {
                "font.family": "serif",
                "font.serif": [
                    "Times New Roman",
                    "Times",
                    "CMU Serif",
                    "DejaVu Serif",
                ],
                "mathtext.fontset": "cm",
                "font.size": 10,
            }
        )
    except Exception:
        pass


RESULTS_ROOT = Path("results")


def analyze_model(results_dir: Path, label_names: List[str] | None = None) -> None:
    preds_path = results_dir / "test_predictions.csv"
    metrics_path = results_dir / "test_metrics.csv"
    if not preds_path.exists() or not metrics_path.exists():
        return

    df_preds = pd.read_csv(preds_path)
    y_true = df_preds["true_label"].to_numpy()
    y_pred = df_preds["predicted_label"].to_numpy()

    # Per-class report
    report = classification_report(y_true, y_pred, target_names=label_names, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose().reset_index().rename(columns={"index": "class"})

    out_dir = results_dir / "analysis"
    ensure_dir(out_dir)
    save_csv(report_df, out_dir / "classification_report.csv")

    # Confusion matrix
    labels = sorted(np.unique(y_true))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"true_{l}" for l in labels], columns=[f"pred_{l}" for l in labels])
    save_csv(cm_df, out_dir / "confusion_matrix.csv")

    # Plot confusion matrix heatmap (optional)
    try:
        import matplotlib.pyplot as plt
        _mpl_setup()

        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        # Use readable class names when provided, in the order of 'labels'
        if label_names is not None and len(label_names) == len(labels):
            name_map = {int(i): str(n) for i, n in enumerate(label_names)}
            x_ticklabels = [name_map.get(int(l), str(l)) for l in labels]
            y_ticklabels = x_ticklabels
        else:
            x_ticklabels = [str(l) for l in labels]
            y_ticklabels = x_ticklabels
        ax.set_xticks(range(len(labels)), x_ticklabels)
        ax.set_yticks(range(len(labels)), y_ticklabels)
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(out_dir / "confusion_matrix.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
    except Exception:
        pass

    # Misclassified samples
    mis = df_preds[df_preds["true_label"] != df_preds["predicted_label"]].copy()
    save_csv(mis, out_dir / "misclassified.csv")


def main() -> None:
    label_names = ["Not hateful", "Hateful"]
    for sub in sorted(RESULTS_ROOT.iterdir()):
        if sub.is_dir() and sub.name != "analysis":
            analyze_model(sub, label_names)
    print("Saved per-model error analysis under results/*/analysis/")


if __name__ == "__main__":
    main()
