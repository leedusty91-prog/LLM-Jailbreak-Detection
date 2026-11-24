import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from category_models import fit_and_evaluate_texts, one_vs_rest_frame
from labels import EVALUATED_JAILBREAK_TYPES
from metrics import classification_metrics, trainer_compute_metrics


class _Prediction:
    def __init__(self, labels, logits):
        self.label_ids = np.asarray(labels)
        self.predictions = np.asarray(logits)


def test_accuracy_hides_a_missed_rare_class_and_full_metrics_do_not():
    y_true = [0] * 158 + [1] * 10
    y_pred = [0] * 168
    scores = [0.1] * 168
    metrics = classification_metrics(y_true, y_pred, scores)
    assert metrics["Accuracy"] > 0.9
    assert metrics["F1"] == 0.0
    assert metrics["TPR"] == 0.0
    assert metrics["FNR"] == 1.0
    assert metrics["n_positive"] == 10
    assert set(metrics) >= {"Accuracy", "F1", "AUC", "TPR", "FNR"}


def test_trainer_metrics_include_auc_tpr_and_fnr():
    labels = [0, 0, 1, 1]
    logits = [[2.0, 0.0], [1.5, 0.2], [0.1, 1.4], [0.2, 2.0]]
    result = trainer_compute_metrics(_Prediction(labels, logits))
    assert set(result) == {"accuracy", "f1", "auc", "tpr", "fnr"}
    assert result["accuracy"] == 1.0
    assert result["f1"] == 1.0
    assert result["tpr"] == 1.0
    assert result["fnr"] == 0.0
    assert result["auc"] == 1.0


def test_category_frames_and_models_record_full_metrics_for_rare_types():
    frame = pd.read_csv("data/df_jailbreak_categ.csv")
    ethical = one_vs_rest_frame(frame, "Ethical Appeal")
    assert int(ethical["Jailbreak"].sum()) == 10
    assert len(ethical) == frame["prompt"].map(lambda value: str(value).strip()).nunique()
    contradiction = one_vs_rest_frame(
        pd.DataFrame({"prompt": ["a", "b"], "Labels": ["Contradiction", "['Contrastive']"]}),
        "Contrastive",
    )
    assert contradiction.set_index("prompt").loc["a", "Jailbreak"] == 0
    assert contradiction.set_index("prompt").loc["b", "Jailbreak"] == 1

    rows = []
    for label in EVALUATED_JAILBREAK_TYPES:
        prepared = one_vs_rest_frame(frame, label)
        _model, _vectorizer, row = fit_and_evaluate_texts(
            label,
            prepared["prompt"],
            prepared["Jailbreak"],
            lambda: LogisticRegression(max_iter=200),
            random_state=42,
        )
        assert row["n_positive_total"] == int(prepared["Jailbreak"].sum())
        for key in ("accuracy", "f1", "auc", "tpr", "fnr"):
            assert key in row
            assert row[key] is not None
        rows.append(row)
    ethical_row = next(row for row in rows if row["jailbreak_type"] == "Ethical Appeal")
    assert ethical_row["n_positive_total"] == 10
    assert ethical_row["n_total"] == 166
