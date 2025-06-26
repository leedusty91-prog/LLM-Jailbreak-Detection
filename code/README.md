# Jailbreak Experiments

## Process

To reproduce the results run the following notebooks and scripts.

The first two scripts prepare the data sets and sumamrise them.

* [00_preprocess_dataset.ipynb](00_preprocess_dataset.ipynb) Prepare all data.
* [01_summarize_dataset.py](01_summarize_dataset.py) Generate some summary tables.

The next 4 scripts run 30 iterations training models to distinguish jailbreaks from regular prompts.

* [02_TFIDF_ML_Models.ipynb](02_TFIDF_ML_Models.ipynb) 30 runs of ML models using TFIDF for features.
* [03_BD_LSTM_Model.ipynb]() Train a Bi-Directional LSTM model on raw tokens
* [04_BERT_Embedding_ML_Models.ipynb]() 30 Runs of ML models using BERT emebeddings.
* [05_BERT_Fine_Tuned_Models.py](05_BERT_Fine_Tuned_Models.py) 30 runs of an end-to-end BERT classifier fine-tuned on the detection task

The next script trains a model to categorise jailbreaks into the recognised types.

* [06_Train_Jailbreak_Category_Model.ipynb](06_Train_Jailbreak_Category_Model.ipynb)

The next scripts run the Novel Jailbreak Detection experiment which determines the
expected accuracy of a jailbreak detection model working on new jailbreak types.

* [07_Novel_Jailbreak_Detection.ipynb](07_Novel_Jailbreak_Detection.ipynb)

The final notebooks runs KeyBERT and produces the visualisations analysing the
word usage in jailbreaks and non-jailbreaks.

* [08_KeyBERT_Analysis.ipynb](08_KeyBERT_Analysis.ipynb)

Note: Most of these notebooks were executed inside Google Colab.
However, some of the longer running training sessions were done inside
a Google Compute instance.


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
