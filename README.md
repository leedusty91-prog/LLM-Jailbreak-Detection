# LLM Jailbreak Detection

A set of experiments for detecting attempted jailbreaks in LLM prompting

## Setup

The script [SETUP.sh](SETUP.sh) will create a virtual environment,
install the required packages and download the NLTK data used in the
preprocessing. 

It will install Jupyter and configure the Jupyter installation to use the 
virtual environment as the kernel.

When loading the Notebooks be sure to select the environment `venv` so
that all of the libraries and data are available.


## Data

We combine datasets from these two repositories:
* Jailbreak Examples:  https://github.com/verazuo/jailbreak_llms
* Non-Jailbreak Examples: https://github.com/Zjh-819/LLMDataHub

These need to be processed to create a classification dataset to distinguish
between jailbreak and non-jailbreak prompts.


## Citations


@inproceedings{detecting_llm_jailbreak_bert_2026,
  author    = {Hawkins, John and Pramar, Aditya and Beard, Rodney and Chandra, Rohitash},
  title     = {Detecting {LLM} Jailbreak Prompts with Fine-Tuned {BERT} Models,
  booktitle = {AI 2026: Advances in Artificial Intelligence},
  series    = {Lecture Notes in Computer Science},
  volume    = {17354},
  publisher = {Springer},
  year      = {2026}
}

## References/Papers

We use the jailbreak classification scheme from the paper
[Jailbreaking ChatGPT via Prompt Engineering: AnEmpirical Study](https://www.researchgate.net/publication/370981802_Jailbreaking_ChatGPT_via_Prompt_Engineering_An_Empirical_Study) as our basis for understanding the types of adversarial attacks.


## Results

[Next Paper being drafted in Overleaf](https://www.overleaf.com/project/6673a4a0ac3dbbc682b12f27)


## TODO LIST

* Expand the training data to include more recent datasets
* Expand the labeled set of Jailbreak types
* Apply Jailbreak Type Classification to all unlabelled jailbreaks
* * Update statistics about the number of labelled points in Journal Version of the paper
* Re-Run the Novel Jailbreak Experiments on the full dataset (with new and re-labeled data)
* * Insert results into paper
* RERun [KeyBERT](https://maartengr.github.io/KeyBERT/api/keybert.html) over test data using trained BERT.
* * Look for common keywords in either jailbreaks or jailbreak categories
* * Look at clusters of jailbreak keywords based on similairity.
* * -- Visualise with color by Jailbreak Class - Does KeyBERT recover the observed categories.

