from __future__ import annotations

from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.utils.data_utils import get_dataset_splits
from src.utils.experiment_utils import run_grid_search_experiment

RESULTS_DIR = Path("results/svm_tfidf")


def main() -> None:
    splits = get_dataset_splits()

    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer()),
            (
                "svm",
                LinearSVC(
                    class_weight="balanced",
                    max_iter=5000,
                ),
            ),
        ]
    )

    param_grid = {
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "tfidf__min_df": [1, 3],
        "tfidf__max_df": [0.9],
        "tfidf__stop_words": [None, "english"],
        "svm__C": [0.5, 1.0, 2.0],
    }

    experiment = run_grid_search_experiment(
        model_name="svm_tfidf",
        estimator=pipeline,
        param_grid=param_grid,
        splits=splits,
        results_dir=RESULTS_DIR,
    )

    metrics = experiment["metrics"]
    print(f"Completed svm_tfidf experiment. Test F1_macro={metrics['f1_macro']:.3f}")


if __name__ == "__main__":
    main()
