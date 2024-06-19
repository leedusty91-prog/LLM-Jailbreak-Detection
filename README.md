# LLM Jailbreak Detection

A set of experiments for detecting attempted jailbreaks in LLM prompting

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

https://www.researchgate.net/publication/370981802_Jailbreaking_ChatGPT_via_Prompt_Engineering_An_Empirical_Study

https://arxiv.org/abs/2402.13457

https://arxiv.org/html/2402.16192v2

https://arxiv.org/html/2402.14872v1

https://github.com/ThuCCSLab/JailbreakEval

## Results

Paper being drafted in Overleaf 
[Read it here](https://www.overleaf.com/project/667239dab5b15503128e776c)



