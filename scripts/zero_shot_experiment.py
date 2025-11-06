from __future__ import annotations

from pathlib import Path
from typing import Iterator, List, Sequence

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src.utils.data_utils import get_dataset_splits
from src.utils.evaluation_utils import classification_metrics
from src.utils.io_utils import save_csv
from src.utils.tracking_utils import run_with_tracking

RESULTS_DIR = Path("results/zero_shot")
MODEL_NAME = "typeform/distilbert-base-uncased-mnli"
BATCH_SIZE = 16

LABEL_MAP = {
    0: "hate speech",
    1: "offensive language",
    2: "neither offensive nor hateful",
}


def load_pipeline():
    try:
        from transformers import pipeline
    except ImportError as exc:
        raise ImportError("transformers is required for the zero-shot experiment.") from exc

    return pipeline("zero-shot-classification", model=MODEL_NAME)


def batched(iterable: Sequence[str], batch_size: int) -> Iterator[List[str]]:
    for start in range(0, len(iterable), batch_size):
        yield iterable[start : start + batch_size]


def predict_zero_shot(
    classifier,
    texts: Sequence[str],
    candidate_labels: List[str],
    hypothesis_template: str,
    *,
    show_progress: bool = False,
) -> np.ndarray:
    predictions: List[int] = []
    label_to_id = {label.lower(): idx for idx, label in LABEL_MAP.items()}
    normalized_candidates = [label.lower() for label in candidate_labels]
    progress_bar = (
        tqdm(total=len(texts), desc="Zero-shot inference", unit="tweet")
        if show_progress
        else None
    )
    for batch in batched(texts, BATCH_SIZE):
        safe_batch = [text if text.strip() else " " for text in batch]
        outputs = classifier(
            safe_batch,
            candidate_labels=candidate_labels,
            hypothesis_template=hypothesis_template,
            multi_label=False,
        )
        for output in outputs:
            predicted_label = output["labels"][0].lower()
            if predicted_label not in normalized_candidates:
                raise KeyError(
                    f"Predicted label '{predicted_label}' not in candidate labels."
                )
            predictions.append(label_to_id[predicted_label])
        if progress_bar is not None:
            progress_bar.update(len(batch))
    if progress_bar is not None:
        progress_bar.close()
    return np.asarray(predictions)


def main() -> None:
    splits = get_dataset_splits()
    classifier = load_pipeline()

    candidate_labels = list(LABEL_MAP.values())
    templates = [
        "This tweet is {}.",
        "The author of this tweet is using {}.",
    ]

    def evaluate_templates():
        records = []
        for template in templates:
            print(f"[zero-shot] Evaluating hypothesis template: {template}")
            preds = predict_zero_shot(
                classifier,
                list(splits.X_val),
                candidate_labels,
                template,
            )
            metrics = classification_metrics(splits.y_val, preds, average="macro")
            record = {
                "hypothesis_template": template,
                "val_accuracy": metrics["accuracy"],
                "val_f1_macro": metrics["f1_macro"],
            }
            records.append(record)
        return records

    grid_records, train_stats = run_with_tracking(
        "zero_shot_validation",
        RESULTS_DIR,
        evaluate_templates,
    )
    grid_df = pd.DataFrame(grid_records)
    save_csv(grid_df, RESULTS_DIR / "grid_search_results.csv")

    best_row = grid_df.sort_values("val_f1_macro", ascending=False).iloc[0]
    best_template = best_row["hypothesis_template"]

    def evaluate_test():
        return predict_zero_shot(
            classifier,
            list(splits.X_test),
            candidate_labels,
            best_template,
            show_progress=True,
        )

    test_predictions, test_stats = run_with_tracking(
        "zero_shot_testing",
        RESULTS_DIR,
        evaluate_test,
    )

    metrics = classification_metrics(splits.y_test, test_predictions, average="macro")
    results_record = {
        "model": "zero_shot",
        "selected_template": best_template,
        "train_duration_s": train_stats.duration_seconds,
        "train_emissions_kg": train_stats.emissions_kg,
        "test_duration_s": test_stats.duration_seconds,
        "test_emissions_kg": test_stats.emissions_kg,
    }
    results_record.update(metrics)

    metrics_df = pd.DataFrame([results_record])
    save_csv(metrics_df, RESULTS_DIR / "test_metrics.csv")

    predictions_frame = pd.DataFrame(
        {
            "text": splits.X_test.values,
            "true_label": splits.y_test.values,
            "predicted_label": test_predictions,
        }
    )
    save_csv(predictions_frame, RESULTS_DIR / "test_predictions.csv")

    print(
        "Completed zero-shot experiment. "
        f"Template='{best_template}' Test F1_macro={metrics['f1_macro']:.3f}"
    )


if __name__ == "__main__":
    main()
