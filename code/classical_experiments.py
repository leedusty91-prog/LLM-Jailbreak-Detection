"""Repeated stratified splits for the TF-IDF and frozen-embedding experiments.

The vectoriser is fit inside each run, on that run's training prompts only.
Prompts are checked for overlap before any model is fit. Models are constructed
inside the run so one fit cannot leak state into the next run.
"""

from __future__ import annotations

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit

from metrics import classification_metrics
from splits import LeakageError, normalize_prompt


def run_tfidf_runs(
    frame,
    num_runs: int = 30,
    test_size: float = 0.2,
    max_features: int = 5000,
    random_state: int = 0,
    max_iter: int = 8000,
    prompt_col: str = "prompt",
    label_col: str = "jailbreak",
) -> list[dict]:
    prompts = [str(prompt) for prompt in frame[prompt_col]]
    labels = frame[label_col].astype(int).to_numpy()
    _require_unique_prompts(prompts)
    splitter = StratifiedShuffleSplit(n_splits=num_runs, test_size=test_size, random_state=random_state)
    runs = []
    for run_index, (train_index, test_index) in enumerate(splitter.split(np.zeros(len(labels)), labels)):
        train_prompts = [prompts[index] for index in train_index]
        test_prompts = [prompts[index] for index in test_index]
        _require_disjoint(train_prompts, test_prompts)
        vectorizer = TfidfVectorizer(max_features=max_features)
        x_train = vectorizer.fit_transform(train_prompts)
        x_test = vectorizer.transform(test_prompts)
        runs.extend(
            _fit_models(
                x_train,
                labels[train_index],
                x_test,
                labels[test_index],
                features_name="TF-IDF",
                run_index=run_index,
                max_iter=max_iter,
            )
        )
    return runs


def run_matrix_model_runs(
    features,
    prompts,
    labels,
    features_name: str,
    num_runs: int = 30,
    test_size: float = 0.2,
    random_state: int = 42,
    max_iter: int = 1000,
) -> list[dict]:
    """Repeated splits for a precomputed feature matrix (frozen BERT embeddings)."""
    matrix = np.asarray(features)
    prompt_list = [str(prompt) for prompt in prompts]
    targets = np.asarray(labels).astype(int)
    if len(matrix) != len(prompt_list) or len(matrix) != len(targets):
        raise ValueError("features, prompts, and labels must have the same length")
    _require_unique_prompts(prompt_list)
    splitter = StratifiedShuffleSplit(n_splits=num_runs, test_size=test_size, random_state=random_state)
    runs = []
    for run_index, (train_index, test_index) in enumerate(splitter.split(np.zeros(len(targets)), targets)):
        _require_disjoint(
            [prompt_list[index] for index in train_index],
            [prompt_list[index] for index in test_index],
        )
        runs.extend(
            _fit_models(
                matrix[train_index],
                targets[train_index],
                matrix[test_index],
                targets[test_index],
                features_name=features_name,
                run_index=run_index,
                max_iter=max_iter,
            )
        )
    return runs


def _fit_models(x_train, y_train, x_test, y_test, features_name, run_index, max_iter):
    models = {
        "Logistic Regression": LogisticRegression(
            random_state=run_index,
            solver="saga",
            l1_ratio=0.2,
            max_iter=max_iter,
        ),
        "Extra Trees": ExtraTreesClassifier(n_estimators=100, random_state=run_index, n_jobs=1),
        "LightGBM": LGBMClassifier(n_jobs=1, random_state=run_index, verbosity=-1),
    }
    # Frozen-embedding runs in the original notebook used a plain logistic
    # regression. Keep elastic-net for TF-IDF, where that was the notebook
    # setting, and a larger default max_iter only when the caller asks for it.
    if features_name != "TF-IDF":
        models["Logistic Regression"] = LogisticRegression(
            max_iter=max_iter,
            random_state=run_index,
        )
    rows = []
    for model_name, model in models.items():
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        scores = model.predict_proba(x_test)[:, 1]
        metrics = classification_metrics(y_test, predictions, scores)
        rows.append(
            {
                "features": features_name,
                "model": model_name,
                "auc": metrics["AUC"],
                "accuracy": metrics["Accuracy"],
                "fnr": metrics["FNR"],
                "tpr": metrics["TPR"],
                "f1": metrics["F1"],
                "run": run_index,
            }
        )
    return rows


def _require_unique_prompts(prompts: list[str]) -> None:
    normalized = [normalize_prompt(prompt) for prompt in prompts]
    if any(not prompt for prompt in normalized):
        raise LeakageError("Refusing to train on an empty prompt")
    if len(normalized) != len(set(normalized)):
        raise LeakageError("Training corpus contains duplicate prompts")


def _require_disjoint(train_prompts, test_prompts) -> None:
    shared = set(map(normalize_prompt, train_prompts)) & set(map(normalize_prompt, test_prompts))
    if shared:
        raise LeakageError(f"{len(shared)} prompts appear in both sides of a random split")
