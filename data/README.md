# LLM Jailbreak vs. Non-Jailbreak Prompts

Datasets used for categorizing and analyzing jailbreak (JB) and non-jailbreak (non-JB).

### File Description

- **`binary_data.csv`**: This file contains the categorized, preprocessed and augumented data. It has been split into training and testing.
The file includes the following columns:

1. **`prompt`**: The text of the prompt provided to the LLM.
2. **`category`** or **`jailbreak`**: A binary variable indicating the classification of the prompt:
   - `1`: Jailbreak Prompt
   - `0`: Non-Jailbreak Prompt
