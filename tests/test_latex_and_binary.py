import pandas as pd

from binary_data import build_binary_dataset, load_binary_frames
from classical_experiments import run_matrix_model_runs, run_tfidf_runs
from latex_tables import (
    binary_detection_table,
    export_results,
    jailbreak_type_table,
    novel_jailbreak_table,
    population_std,
)
from splits import normalize_prompt, overlap_count


def test_population_std_matches_numpy_default():
    assert population_std([1, 3]) == 1.0
    assert population_std([0.5, 0.5]) == 0.0


def test_binary_table_has_mean_and_std_columns():
    runs = [
        {"features": "TF-IDF", "model": "Logistic Regression", "auc": 0.954, "accuracy": 0.887, "fnr": 0.151, "tpr": 0.848},
        {"features": "TF-IDF", "model": "Logistic Regression", "auc": 0.956, "accuracy": 0.889, "fnr": 0.149, "tpr": 0.850},
        {"features": "Tokens", "model": "BD-LSTM", "auc": 0.854, "accuracy": 0.843, "fnr": 0.120, "tpr": 0.933},
    ]
    tex = binary_detection_table(runs, caption="Known jailbreaks 100%.", label="tab:binary-detection")
    assert tex.startswith("% Requires \\usepackage{booktabs}\n\\begin{table}[t]")
    assert r"\caption{Known jailbreaks 100\%.}" in tex
    assert "Features & Model & \\multicolumn{2}{c}{AUC}" in tex
    assert "Mean & Std & Mean & Std & Mean & Std & Mean & Std" in tex
    assert "TF-IDF & Logistic Regression & 0.955 & 0.001 & 0.888 & 0.001 & 0.150 & 0.001 & 0.849 & 0.001 \\\\" in tex
    assert "Tokens & BD-LSTM & 0.854 & 0.000" in tex
    assert tex.strip().endswith(r"\end{table}")


def test_novel_and_category_tables_escape_and_keep_column_order():
    novel = novel_jailbreak_table(
        [
            {"held_out_type": "Character Roleplay", "auc": 0.99, "accuracy": 0.99, "fnr": 0.04, "tpr": 0.96},
            {"held_out_type": "A_B&C", "auc": None, "accuracy": 1, "fnr": 0, "tpr": 1},
        ]
    )
    assert "Held-out Jailbreak Type & AUC & Accuracy & FNR & TPR \\\\" in novel
    assert "Character Roleplay & 0.990 & 0.990 & 0.040 & 0.960 \\\\" in novel
    assert r"A\_B\&C & -- & 1.000 & 0.000 & 1.000 \\" in novel
    category = jailbreak_type_table(
        [
            {
                "jailbreak_type": "Ethical Appeal",
                "accuracy": 0.94,
                "f1": 0.2,
                "auc": 0.7,
                "tpr": 0.4,
                "fnr": 0.6,
            }
        ]
    )
    assert "Jailbreak Type & Accuracy & F1 & AUC & TPR & FNR \\\\" in category
    assert "Ethical Appeal & 0.940 & 0.200 & 0.700 & 0.400 & 0.600 \\\\" in category


def test_export_writes_csv_json_and_tex(tmp_path):
    paths = export_results(
        "unit_binary",
        "binary",
        [
            {"features": "BERT", "model": "BERT", "auc": 0.5, "accuracy": 0.5, "fnr": 0.5, "tpr": 0.5, "run": 1},
        ],
        results_directory=tmp_path,
    )
    assert paths["csv"].read_text(encoding="utf-8").splitlines()[0] == (
        "Features,Model,AUC Mean,AUC Std,Accuracy Mean,Accuracy Std,FNR Mean,FNR Std,TPR Mean,TPR Std"
    )
    assert '"kind": "binary"' in paths["json"].read_text(encoding="utf-8")
    assert "BERT & BERT & 0.500 & 0.000" in paths["tex"].read_text(encoding="utf-8")


def test_build_binary_dataset_dedupes_and_blocks_augmentation_collisions(tmp_path):
    jailbreak = pd.DataFrame(
        {"prompt": ["Hello Jail", "Hello Jail", "Second Jail"], "jailbreak": [True, True, True]}
    )
    regular = pd.DataFrame(
        {"prompt": ["hello jail", "Safe One", "Safe Two", "Safe Three", "Safe Four"], "jailbreak": [False] * 5}
    )

    def preprocess(text):
        return str(text).upper()

    def augment(text):
        if text == "SAFE ONE":
            return "SAFE TWO"  # collides with a prompt that may be in either split
        return text + " EXTRA"

    train, test = build_binary_dataset(
        jailbreak,
        regular,
        preprocess_fn=preprocess,
        augment_fn=augment,
        regular_cap=None,
        test_size=0.4,
        random_state=1,
    )
    assert overlap_count(train, test) == 0
    assert not train["prompt"].map(normalize_prompt).duplicated().any()
    combined = pd.concat([train, test], ignore_index=True)
    hello = combined.loc[combined["prompt"] == "HELLO JAIL"]
    assert len(hello) == 1
    assert int(hello["jailbreak"].iloc[0]) == 1
    assert train["prompt"].str.isupper().all()
    assert test["prompt"].str.isupper().all()
    assert "SAFE TWO" in set(combined["prompt"])
    # The augmentation that copied an existing prompt was dropped; the extra
    # tokens are new training text and are absent from the test split.
    extras = [prompt for prompt in train["prompt"] if prompt.endswith(" EXTRA")]
    assert extras
    assert set(extras).isdisjoint(set(test["prompt"]))


def test_checked_in_binary_files_have_no_train_test_overlap():
    train, test = load_binary_frames()
    assert overlap_count(train, test) == 0
    assert not train["prompt"].map(normalize_prompt).duplicated().any()
    assert not test["prompt"].map(normalize_prompt).duplicated().any()
    assert len(train) > 1000
    assert len(test) > 100


def test_repeated_splits_reject_duplicate_prompts_and_export(tmp_path):
    prompts = [f"jailbreak prompt number {i}" for i in range(40)] + [f"ordinary prompt number {i}" for i in range(40)]
    labels = [1] * 40 + [0] * 40
    frame = pd.DataFrame({"prompt": prompts, "jailbreak": labels})
    runs = run_tfidf_runs(frame, num_runs=2, max_iter=200, random_state=0)
    assert len(runs) == 6
    assert {row["features"] for row in runs} == {"TF-IDF"}
    assert {row["model"] for row in runs} == {"Logistic Regression", "Extra Trees", "LightGBM"}
    matrix = [[float(index == position % 8), float((index + 1) == position % 8)] for index, position in enumerate(range(80))]
    embedding_runs = run_matrix_model_runs(
        matrix, prompts, labels, features_name="BERT", num_runs=1, max_iter=200, random_state=0
    )
    assert {row["features"] for row in embedding_runs} == {"BERT"}
    export_results("tfidf_smoke", "binary", runs, results_directory=tmp_path)
    try:
        run_tfidf_runs(pd.concat([frame, frame.iloc[[0]]], ignore_index=True), num_runs=1, max_iter=50)
    except Exception as exc:
        assert "duplicate" in str(exc).lower()
    else:
        raise AssertionError("duplicate prompts should not be trained")
