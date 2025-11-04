from __future__ import annotations

from pathlib import Path

from sklearn.pipeline import Pipeline

from src.models.keyword_classifier import KeywordRegexClassifier
from src.utils.data_utils import get_dataset_splits
from src.utils.experiment_utils import run_grid_search_experiment

RESULTS_DIR = Path("results/regex_keyword")


def main() -> None:
    splits = get_dataset_splits()

    pipeline = Pipeline(
        [
            ("clf", KeywordRegexClassifier()),
        ]
    )

    param_grid = {
        "clf__num_keywords": [15, 30, 60],
        "clf__exclude_stopwords": [True, False],
    }

    experiment = run_grid_search_experiment(
        model_name="regex_keyword",
        estimator=pipeline,
        param_grid=param_grid,
        splits=splits,
        results_dir=RESULTS_DIR,
    )

    metrics = experiment["metrics"]
    print(f"Completed regex_keyword experiment. Test F1_macro={metrics['f1_macro']:.3f}")


if __name__ == "__main__":
    main()

