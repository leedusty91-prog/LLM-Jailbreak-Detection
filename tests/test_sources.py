import json
from pathlib import Path

ACTIVE_NOTEBOOKS = [
    "code/00_preprocess_dataset.ipynb",
    "code/02_TFIDF_ML_Models-2.ipynb",
    "code/03_BD_LSTM_Model-2.ipynb",
    "code/04_LR_ET_LGMB_BERT.ipynb",
    "code/05_BERT_Fine_Tuned_Models.ipynb",
    "code/06___Label_all.ipynb",
    "code/06_Train_Jailbreak_Category_Model.ipynb",
    "code/07_Novel_Jailbreak_Detection.ipynb",
]


def _sources(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    chunks = []
    for cell in notebook["cells"]:
        chunks.append("".join(cell.get("source", [])))
        assert cell.get("outputs", []) == [] or cell["cell_type"] == "markdown"
    return "\n".join(chunks)


def test_training_notebooks_are_runnable_json_without_colab_paths():
    for name in ACTIVE_NOTEBOOKS:
        path = Path(name)
        source = _sources(path)
        assert "/content/" not in source
        assert "gs://" not in source
        assert "google.colab" not in source
    exporters = [
        name
        for name in ACTIVE_NOTEBOOKS
        if name
        not in {
            "code/00_preprocess_dataset.ipynb",
            "code/05_BERT_Fine_Tuned_Models.ipynb",
        }
    ]
    for name in exporters:
        assert "export_results" in _sources(Path(name))
    bert_script = Path("code/05_BERT_Fine_Tuned_Models.py").read_text(encoding="utf-8")
    assert "export_results" in bert_script


def test_known_evaluation_bugs_are_gone_from_source():
    label_all = _sources(Path("code/06___Label_all.ipynb"))
    assert "et_sud.predict" not in label_all
    assert "len(df_input)" not in label_all
    assert "len()" not in label_all
    assert "LGBMClassifier" in label_all
    assert "fit_and_evaluate" in label_all

    category = _sources(Path("code/06_Train_Jailbreak_Category_Model.ipynb"))
    assert "eval_f1" in category
    assert "eval_auc" in category
    assert "parse_labels" in category or "one_vs_rest_frame" in category
    assert "'accuracy': eval_results['eval_accuracy']" not in category

    novel = _sources(Path("code/07_Novel_Jailbreak_Detection.ipynb"))
    assert "build_heldout_splits" in novel
    assert "sampled_regular" not in novel
    assert "NON_JAILBREAK_RATIO" in novel

    bert = Path("code/05_BERT_Fine_Tuned_Models.py").read_text(encoding="utf-8")
    assert "gs://" not in bert
    assert "train_df.drop(train_sample.index)" not in bert
    assert "split_train_val" in bert
    assert "testing_data_binary.csv" in bert

    preprocess = _sources(Path("code/00_preprocess_dataset.ipynb"))
    assert "build_binary_dataset" in preprocess
    assert "write_binary_splits" in preprocess
