# Hate Speech Model Comparison

This repository contains an implementation and evaluation of multiple text classifiers for hate speech detection on Twitter. The work follows these requirements (from the original project brief):

- Implement at least two classifiers of your choice.
- Measure both effectiveness and efficiency of the classifiers and run a statistical test.
- Perform an error analysis of your choice.
- Reflect on how likely the observed effectiveness/efficiency is to repeat on other data in the same domain or for other models.
- Produce a short report summarizing the above.

The experiments are run on the training portion of the “Twitter Hate Speech” dataset (Kaggle). The repository includes code to train/evaluate several models, aggregate metrics, generate plots, perform a significance test, and save error analysis artifacts. A concise report is provided in `report/report.pdf`.


## Repository Structure

- `src/`
  - `models/`
    - `keyword_classifier.py` - simple keyword/regex baseline classifier.
    - `embedding_transformer.py` - sentence embeddings via Sentence-Transformers.
    - `gensim_sequence.py` - text-to-sequence of pretrained word embeddings (Gensim).
    - `torch_text.py` - PyTorch LSTM module consuming embedding sequences.
  - `utils/`
    - `data_utils.py` - dataset loading/cleaning and train/val/test splits.
    - `experiment_utils.py` - grid search, evaluation, and result persistence.
    - `evaluation_utils.py` - standard classification metrics.
    - `io_utils.py` - small I/O helpers.
    - `tracking_utils.py` - runtime and (optional) emissions tracking via CodeCarbon.

- `scripts/` - entry points to run each experiment:
  - `regex_experiment.py`
  - `svm_tfidf_experiment.py`
  - `minilm_logreg_experiment.py`
  - `lstm_experiment.py`
  - `zero_shot_experiment.py`

- `analysis/` - post-processing and analysis scripts:
  - `dataset_summary.py` - dataset size/distribution and text length stats.
  - `aggregate_results.py` - consolidate per-model metrics and add efficiency ratios.
  - `error_analysis.py` - per-model confusion matrix, classification report, misclassified samples.
  - `efficiency_plots.py` - scatter, bar, and Pareto-front visualizations.
  - `significance_mcnemar.py` - pairwise McNemar significance test across models.

- `dataset/`
  - `train_E6oV3lV.csv` - raw Kaggle training data (place the CSV here).
  - `processed/train_clean.csv` - cached cleaned data, generated automatically.

- `results/`
  - Per-model subfolders (e.g., `svm_tfidf/`, `minilm_logreg/`, `lstm_skorch/`, `regex_keyword/`, `zero_shot/`) containing:
    - `grid_search_results.csv`, `best_params.csv` (when applicable)
    - `test_metrics.csv`, `test_predictions.csv`
    - `*_training_emissions.csv`, `*_testing_emissions.csv` (if CodeCarbon is available)
    - `analysis/` with `classification_report.csv`, `confusion_matrix.csv`, `confusion_matrix.png`, `misclassified.csv`
  - `analysis/` (aggregated across models): `summary.csv`, `dataset_summary.csv`, `per_class_summary.csv`, and plots.

- `report/`
  - `report.pdf` - rendered report with findings.
  - `report.tex` - LaTeX source (optional recompile).

- `requirements.txt` - Python dependencies.
- `LICENSE` - license of this repository.



## Setup

1) Create a Python environment and install dependencies:

```
pip install -r requirements.txt
```

2) Obtain the dataset: download the training CSV from the Kaggle “Twitter Hate Speech” dataset and place it at:

```
dataset/train_E6oV3lV.csv
```

The first run will create a cleaned cache at `dataset/processed/train_clean.csv`.

Notes:
- Emissions tracking is optional. If the `codecarbon` package is installed (it is included in `requirements.txt`), per-run emissions CSV files are saved next to results. If not available, emissions values will be `None` and the files are omitted.
- Some models download pretrained weights on first use (Sentence-Transformers and Transformers). Ensure network access the first time you run those experiments.


## Data Processing and Splits

`src/utils/data_utils.py` performs light text cleaning (lowercasing, remove URLs and mentions, strip punctuation, collapse whitespace). Splits are stratified with default proportions:

- Test: 20% of the full dataset
- Validation: 10% of the full dataset
- Train: remaining 70%

Splits are reproducible via a fixed `random_state` in the code.


## Running Experiments

Each experiment script trains (or configures) a model, performs grid search on hyperparameters when relevant, evaluates on the test set, and writes outputs under `results/<model_name>/`.

Run any of the following from the repository root (module syntax ensures imports work):

```
python -m scripts.regex_experiment          # keyword/regex baseline
python -m scripts.svm_tfidf_experiment      # linear SVM on TF-IDF
python -m scripts.minilm_logreg_experiment  # sentence embeddings + logistic regression
python -m scripts.lstm_experiment           # LSTM over embedding sequences (requires skorch + torch)
python -m scripts.zero_shot_experiment      # zero-shot classifier (Transformers pipeline)
```

Outputs per model include:
- `test_metrics.csv` with accuracy/macro-F1 and runtime/emissions (if available)
- `test_predictions.csv` with per-example predictions
- `grid_search_results.csv` and `best_params.csv` when grid search is used
- `*_training_emissions.csv` and `*_testing_emissions.csv` if emissions tracking is enabled


## Analysis and Reporting

After running one or more experiments, generate summaries and plots (run from the repository root):

```
python -m analysis.dataset_summary
python -m analysis.aggregate_results
python -m analysis.error_analysis
python -m analysis.efficiency_plots
python -m analysis.significance_mcnemar
```

Key outputs land under `results/analysis/` and under each model’s `analysis/` subfolder. The LaTeX report (`report/report.tex`) references these CSVs and figures; the rendered PDF is at `report/report.pdf`.


## Reproducibility and Efficiency Tracking

- Scoring uses macro-F1 (`f1_macro`) by default for model selection and reporting.
- Grid search uses the train/validation split via a predefined split strategy.
- The `CodeCarbon`-based tracker records estimated emissions for training and testing when available.


## Citing the Data

Please refer to the Kaggle dataset page for terms of use and citation guidance:
- Twitter Hate Speech (training split): https://www.kaggle.com/vkrahul/twitter-hate-speech


## License

See `LICENSE` for the license covering this repository. The dataset is subject to its own license/terms on Kaggle and is not redistributed here.
