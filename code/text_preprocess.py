"""Text preprocessing shared by the novel-jailbreak experiment.

Jailbreak prompts used to be left raw while non-jailbreak prompts were stemmed
and stop-worded. A model can then separate the classes by the presence of
stems rather than by the jailbreak strategy. ``stem_preprocess`` is the single
function applied to both classes.
"""

from __future__ import annotations

_STATE: dict = {}


def stem_preprocess(text) -> str:
    """Lower-case via the Porter stemmer and drop English stop words."""
    if text is None:
        return ""
    stemmer, stop_words = _stemmer_and_stops()
    tokens = []
    for word in str(text).split():
        if word.lower() in stop_words:
            continue
        tokens.append(stemmer.stem(word))
    return " ".join(tokens)


def _stemmer_and_stops():
    if "stemmer" not in _STATE:
        import nltk
        from nltk.corpus import stopwords
        from nltk.stem import PorterStemmer

        try:
            words = stopwords.words("english")
        except LookupError:
            nltk.download("stopwords", quiet=True)
            words = stopwords.words("english")
        _STATE["stemmer"] = PorterStemmer()
        _STATE["stop"] = set(words)
    return _STATE["stemmer"], _STATE["stop"]
