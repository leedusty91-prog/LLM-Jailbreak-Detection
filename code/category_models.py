"""One-vs-rest jailbreak-type classifiers and their full metric rows.

Each type is its own model. Evaluation always scores the model that was fit
on that type's training fold. The labelling notebook previously fit five
LightGBM models and then called the Sudo model to score Simulate Jailbreaking
and Ethical Appeal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from labels import canonicalize, parse_labels
from metrics import category_row, classification_metrics
from splits import normalize_prompt


def one_vs_rest_frame(
    df: pd.DataFrame,
    label_name: str,
    prompt_col: str = "prompt",
    labels_col: str = "Labels",
    random_state: int = 1,
) -> pd.DataFrame:
    """Binary frame for ``label_name`` using exact parsed labels.

    Duplicate prompt strings are merged first so a later random split cannot
    place the same prompt on both sides. The positive class is the named
    jailbreak type.
    """
    canonical = canonicalize(label_name)
    grouped: dict[str, list[str]] = {}
    originals: dict[str, str] = {}
    order: list[str] = []
    for prompt, raw_labels in zip(df[prompt_col], df[labels_col]):
        key = normalize_prompt(prompt)
        if not key:
            continue
        if key not in grouped:
            grouped[key] = []
            originals[key] = prompt if isinstance(prompt, str) else str(prompt)
            order.append(key)
        for label in parse_labels(raw_labels):
            if label not in grouped[key]:
                grouped[key].append(label)
    frame = pd.DataFrame(
        {
            "prompt": [originals[key] for key in order],
            "Jailbreak": [1 if canonical in grouped[key] else 0 for key in order],
        }
    )
    return frame.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def fit_and_evaluate_precomputed(
    jailbreak_type: str,
    features,
    labels,
    model_factory,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Fit ``model_factory()`` on one stratified fold and score that same fold."""
    matrix = np.asarray(features)
    targets = np.asarray(list(labels)).astype(int)
    _check_both_classes(jailbreak_type, targets)
    x_train, x_test, y_train, y_test = train_test_split(
        matrix,
        targets,
        test_size=test_size,
        random_state=random_state,
        stratify=targets,
    )
    model = model_factory()
    model.fit(x_train, y_train)
    row = _score(jailbreak_type, model, x_test, y_test)
    row["n_positive_total"] = int((targets == 1).sum())
    row["n_total"] = int(len(targets))
    return model, row


def fit_and_evaluate_texts(
    jailbreak_type: str,
    prompts,
    labels,
    model_factory,
    test_size: float = 0.2,
    random_state: int = 42,
    max_features: int = 5000,
):
    """TF-IDF one-vs-rest. The vectoriser is fit on the training fold only."""
    texts = np.asarray(list(prompts))
    targets = np.asarray(list(labels)).astype(int)
    _check_both_classes(jailbreak_type, targets)
    text_train, text_test, y_train, y_test = train_test_split(
        texts,
        targets,
        test_size=test_size,
        random_state=random_state,
        stratify=targets,
    )
    vectorizer = TfidfVectorizer(max_features=max_features)
    x_train = vectorizer.fit_transform(text_train)
    x_test = vectorizer.transform(text_test)
    model = model_factory()
    model.fit(x_train, y_train)
    row = _score(jailbreak_type, model, x_test, y_test)
    row["n_positive_total"] = int((targets == 1).sum())
    row["n_total"] = int(len(targets))
    return model, vectorizer, row


def _score(jailbreak_type: str, model, features, y_test) -> dict:
    predictions = model.predict(features)
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(features)[:, 1]
    else:
        scores = model.decision_function(features)
    return category_row(jailbreak_type, classification_metrics(y_test, predictions, scores))


def _check_both_classes(jailbreak_type: str, targets: np.ndarray) -> None:
    classes, counts = np.unique(targets, return_counts=True)
    if len(classes) < 2 or counts.min() < 2:
        raise ValueError(
            f"{jailbreak_type} needs at least two examples of each class to stratify, "
            f"found counts {dict(zip(classes.tolist(), counts.tolist()))}"
        )
