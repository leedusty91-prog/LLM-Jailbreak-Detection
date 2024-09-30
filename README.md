# LLM Jailbreak Detection

A set of experiments for detecting attempted jailbreaks in LLM prompting

## TODO LIST

* Compile all datasets (jailbreak and non-jailbreak)
* Categorise all jailbreaks in the types and patterns
* Implement Machine Learning Experiment
** Decide on Pre-Trained BERT Models


## Data

We combine datasets from these two repositories:
* Jailbreak Examples:  https://github.com/verazuo/jailbreak_llms
* Non-Jailbreak Examples: https://github.com/Zjh-819/LLMDataHub

These need to be processed to create a classification dataset to distinguish
between jailbreak and non-jailbreak prompts.


## Experiments

Initial experiment is to determine how well we can distinguish jailbreak
prompts from other queries. We will experiment with multiple text feature extraction
and machine learning techniques. These will be benchmarked against fine tuning
transformer models.

Later experiments will look at identifying the specific elements of input text
that result in a jailbreak. This will involve sequence to sequence models that
extract the sub-string of the input that results in the jailbreak.

## References/Papers

We use the jailbreak classification scheme from the paper
[Jailbreaking ChatGPT via Prompt Engineering: AnEmpirical Study](https://www.researchgate.net/publication/370981802_Jailbreaking_ChatGPT_via_Prompt_Engineering_An_Empirical_Study) as our basis for understanding the types of adversarial attacks.


## Results

[Paper being drafted in Overleaf](https://www.overleaf.com/project/6673a4a0ac3dbbc682b12f27)


