from __future__ import annotations

from pathlib import Path

from sklearn.pipeline import Pipeline

try:
    from skorch.classifier import NeuralNetClassifier
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError("skorch is required for the lstm experiment. Install skorch and torch.") from exc

import numpy as np
import torch
from torch import nn

from src.models.gensim_sequence import EmbeddingSequenceTransformer
from src.models.torch_text import LSTMSequenceModule
from src.utils.data_utils import get_dataset_splits
from src.utils.experiment_utils import run_grid_search_experiment

RESULTS_DIR = Path("results/lstm_skorch")


def main() -> None:
    splits = get_dataset_splits()
    labels = splits.y_train.to_numpy()
    classes, counts = np.unique(labels, return_counts=True)
    counts = counts.astype(float)
    counts[counts == 0.0] = 1.0
    weights = counts.sum() / (counts.size * counts)
    class_weights = torch.tensor(weights, dtype=torch.float32)

    # Using GloVe 100d by default; set module__embedding_dim accordingly.
    pipeline = Pipeline(
        [
            (
                "embedseq",
                EmbeddingSequenceTransformer(
                    model_name="glove-wiki-gigaword-100",
                    local_path=None,  # set to local file if offline
                    binary=False,
                    max_len=80,
                ),
            ),
            (
                "clf",
                NeuralNetClassifier(
                    LSTMSequenceModule,
                    module__embedding_dim=100,  # must match chosen vectors
                    module__num_classes=len(classes),
                    criterion=nn.CrossEntropyLoss,
                    criterion__weight=class_weights,
                    iterator_train__shuffle=True,
                    verbose=0,
                ),
            ),
        ]
    )

    param_grid = {
        "embedseq__max_len": [80],
        "clf__module__hidden_size": [128],
        "clf__module__dropout": [0.3],
        "clf__max_epochs": [8],
        "clf__batch_size": [32],
        "clf__lr": [5e-4],
    }

    experiment = run_grid_search_experiment(
        model_name="lstm_skorch",
        estimator=pipeline,
        param_grid=param_grid,
        splits=splits,
        results_dir=RESULTS_DIR,
        scoring="f1_macro",
        n_jobs=1,  # Torch/skorch models: keep single worker for safety
    )

    metrics = experiment["metrics"]
    print(f"Completed lstm_skorch experiment. Test F1_macro={metrics['f1_macro']:.3f}")


if __name__ == "__main__":
    main()
