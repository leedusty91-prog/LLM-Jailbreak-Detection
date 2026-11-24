# Jailbreak Experiments

## Process

To reproduce the results run the following notebooks and scripts.

The first two scripts prepare the data sets and sumamrise them.

* [00_preprocess_dataset.ipynb](00_preprocess_dataset.ipynb) Prepare all data.
* [01_summarize_dataset.py](01_summarize_dataset.py) Generate some summary tables.

The next scripts run 30 iterations training models to distinguish jailbreaks from regular prompts. Paths are resolved from the repository root (`data/`, `results/`). Override them with `JAILBREAK_DATA_DIR` and `JAILBREAK_RESULTS_DIR` if needed. `NUM_RUNS` and `EPOCHS` shorten a smoke run. Each script writes `results/<experiment>_metrics.csv`, `.json`, and a booktabs `.tex` table.

* [02_TFIDF_ML_Models-2.ipynb](02_TFIDF_ML_Models-2.ipynb) Repeated TF-IDF runs, plus an optional frozen-BERT section (`RUN_BERT_EMBEDDINGS=0` skips it).
* [03_BD_LSTM_Model-2.ipynb](03_BD_LSTM_Model-2.ipynb) Bidirectional LSTM. Early stopping uses a validation slice of the training file; metrics are computed on the held-out test file.
* [04_LR_ET_LGMB_BERT.ipynb](04_LR_ET_LGMB_BERT.ipynb) Logistic regression, Extra Trees, and LightGBM on frozen BERT embeddings.
* [05_BERT_Fine_Tuned_Models.py](05_BERT_Fine_Tuned_Models.py) End-to-end BERT fine-tuning. Validation is a disjoint slice of the training file; the reported metrics are on the held-out test file.

The next script trains a model to categorise jailbreaks into the recognised types.

* [06_Train_Jailbreak_Category_Model.ipynb](06_Train_Jailbreak_Category_Model.ipynb)

The next scripts run the Novel Jailbreak Detection experiment which determines the
expected accuracy of a jailbreak detection model working on new jailbreak types.

* [07_Novel_Jailbreak_Detection.ipynb](07_Novel_Jailbreak_Detection.ipynb)

The final notebooks runs KeyBERT and produces the visualisations analysing the
word usage in jailbreaks and non-jailbreaks.

* [08_KeyBERT_Analysis.ipynb](08_KeyBERT_Analysis.ipynb)

Shared helpers live next to the notebooks: `labels.py` (canonical type parser), `splits.py` and `novel_splits.py` (disjoint splits), `metrics.py`, and `latex_tables.py`. `pytest` covers the parser, the split invariants, and the table layout.


## Models

This project involves testin and using 5 different models 
for both binary and multi-class classification.

 1. Logistic Regression
 2. Light GBM
 3. Extra Trees
 4. Bi-LSTM
 5. BERT


### Archive

This refers to earlier versions

To run the code, download all files and upload them on colab. The dataset needs to be uploaded separately. Change train and test data import variables to the location where data is stored on your device and run the code cell by cell.

- **BERT_01** : Run BERT on binary jailbreak classification problem.
- **BERT_30_Runs** : Run BERT 30 training and validation loops.
- **BERT_primary_category** : Classify binary "Character Roleplay" category.
- **LR_ET_LGBM** : Run Logistic Regression, Extra Trees and LightBGM on binary jailbreak classification problem.
- **LR_ET_LGBM_BERT** : Run Logistic Regression, Extra Trees and LightBGM on binary jailbreak classification problem using BERT feature extraction.
- **Classwise** : Run multiple tests on each category of jailbreak prompts on a separate training and validation dataframe.
- **lstm_01** : Run LSTM on binary jailbreak classification problem.
- **preprocess_dataset** : fetch and preprocess data for binary jailbreak classification.
