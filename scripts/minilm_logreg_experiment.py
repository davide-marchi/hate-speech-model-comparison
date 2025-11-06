from __future__ import annotations

from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.embedding_transformer import SentenceTransformerEncoder
from src.utils.data_utils import get_dataset_splits
from src.utils.experiment_utils import run_grid_search_experiment

RESULTS_DIR = Path("results/minilm_logreg")


def main() -> None:
    splits = get_dataset_splits()

    pipeline = Pipeline(
        [
            (
                "embed",
                SentenceTransformerEncoder(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    batch_size=128,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    solver="lbfgs",
                ),
            ),
        ]
    )

    param_grid = {
        "clf__C": [0.5, 1.0, 2.0],
        "clf__penalty": ["l2"],
    }

    experiment = run_grid_search_experiment(
        model_name="minilm_logreg",
        estimator=pipeline,
        param_grid=param_grid,
        splits=splits,
        results_dir=RESULTS_DIR,
    )

    metrics = experiment["metrics"]
    print(f"Completed minilm_logreg experiment. Test F1_macro={metrics['f1_macro']:.3f}")


if __name__ == "__main__":
    main()
