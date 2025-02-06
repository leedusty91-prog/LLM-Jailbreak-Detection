# Models

There are five models currently being tested for binary classification of Jailbreak vs Non Jailbreak text.

 1. Logistic Regression
 2. Light GBM
 3. Extra Trees
 4. Bi-LSTM
 5. BERT

To run the code, download all files and upload them on colab. The dataset needs to be uploaded separately. Change train and test data import variables to the location where data is stored on your device and run the code cell by cell.

- **BERT_01** : Run BERT on binary jailbreak classification problem.
- **BERT_30_Runs** : Run BERT 30 training and validation loops.
- **BERT_primary_category** : Classify binary "Character Roleplay" category.
- **LR_ET_LGBM** : Run Logistic Regression, Extra Trees and LightBGM on binary jailbreak classification problem.
- **LR_ET_LGBM_BERT** : Run Logistic Regression, Extra Trees and LightBGM on binary jailbreak classification problem using BERT feature extraction.
- **Classwise** : Run multiple tests on each category of jailbreak prompts on a separate training and validation dataframe.
- **lstm_01** : Run LSTM on binary jailbreak classification problem.
- **preprocess_dataset** : fetch and preprocess data for binary jailbreak classification.
