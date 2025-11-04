from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Tuple

import pandas as pd
from sklearn.model_selection import GridSearchCV

from .data_utils import DatasetSplits
from .evaluation_utils import classification_metrics
from .io_utils import ensure_dir, save_csv
from .tracking_utils import ExecutionStats, run_with_tracking


def _fit_grid_search(
    estimator: Any,
    param_grid: Mapping[str, Iterable[Any]],
    splits: DatasetSplits,
    scoring: str,
    n_jobs: int,
) -> GridSearchCV:
    X_train_val, y_train_val, predefined_split = splits.combined_train_val()
    grid = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring=scoring,
        n_jobs=n_jobs,
        cv=predefined_split,
        refit=True,
        verbose=0,
    )
    grid.fit(X_train_val, y_train_val)
    return grid


def run_grid_search_experiment(
    *,
    model_name: str,
    estimator: Any,
    param_grid: Mapping[str, Iterable[Any]],
    splits: DatasetSplits,
    results_dir: Path,
    scoring: str = "f1_macro",
    n_jobs: int = -1,
) -> Dict[str, Any]:
    """Run grid-search training and evaluation with tracking and persistence."""
    ensure_dir(results_dir)

    train_callable = (
        lambda: _fit_grid_search(
            estimator=estimator,
            param_grid=param_grid,
            splits=splits,
            scoring=scoring,
            n_jobs=n_jobs,
        )
    )
    grid_search, train_stats = run_with_tracking(
        f"{model_name}_training",
        results_dir,
        train_callable,
    )
    assert isinstance(grid_search, GridSearchCV)

    grid_results = pd.DataFrame(grid_search.cv_results_)
    save_csv(grid_results, results_dir / "grid_search_results.csv")

    best_estimator = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_params_frame = pd.DataFrame([best_params])
    save_csv(best_params_frame, results_dir / "best_params.csv")

    predict_callable = lambda: best_estimator.predict(splits.X_test)
    test_predictions, test_stats = run_with_tracking(
        f"{model_name}_testing",
        results_dir,
        predict_callable,
    )

    metrics = classification_metrics(splits.y_test, test_predictions, average="macro")
    results_record: Dict[str, Any] = {
        "model": model_name,
        "best_score_cv": float(grid_search.best_score_),
        "train_duration_s": train_stats.duration_seconds,
        "train_emissions_kg": train_stats.emissions_kg,
        "test_duration_s": test_stats.duration_seconds,
        "test_emissions_kg": test_stats.emissions_kg,
    }
    results_record.update(metrics)

    metrics_frame = pd.DataFrame([results_record])
    save_csv(metrics_frame, results_dir / "test_metrics.csv")

    predictions_frame = pd.DataFrame(
        {
            "text": splits.X_test.values,
            "true_label": splits.y_test.values,
            "predicted_label": test_predictions,
        }
    )
    save_csv(predictions_frame, results_dir / "test_predictions.csv")

    return {
        "grid_search": grid_search,
        "train_stats": train_stats,
        "test_stats": test_stats,
        "metrics": results_record,
        "predictions_frame": predictions_frame,
    }

