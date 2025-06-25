
# LLM Jailbreak vs. Non-Jailbreak Prompts

Datasets used for experiments.

### File Descriptions

- **`Folder : binary_data.csv`**: This file contains the categorized, preprocessed and augumented data. It has been split into training and testing. The file includes the following columns:

1. **`prompt`**: The text of the prompt provided to the LLM.
2. **`category`** or **`jailbreak`**: A binary variable indicating the classification of the prompt:
   - `1`: Jailbreak Prompt
   - `0`: Non-Jailbreak Prompt

- **Jailbreak_Prompts_DAN and Regular_Prompts_DAN :** Contain examples of jailbreak prompts sourced from [this repository](https://github.com/verazuo/jailbreak_llms).
- **df_jailbreak_categ :** Contains jailbreak prompts manually labelled from DAN repository, processed and formatted for training classifiers.
- **all_labelled_cleaned :** Contains results data frame from applying [category classifiers](../code/label_all.ipynb) to jailbreak_prompts_DAN.
- 
