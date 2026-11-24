"""Build and repair the binary jailbreak-detection splits.

``00_preprocess_dataset`` used to shuffle with an unseeded sample, split, and
then lemmatise. Duplicate source prompts, and distinct prompts that collapse
to the same string, can therefore land in both ``training_data_binary.csv`` and
``testing_data_binary.csv``. The checked-in files had 37 shared prompts.
``build_binary_dataset`` deduplicates on the preprocessed text before the
split and drops augmented rows that collide with either side. It never writes
the raw source tables.
"""

from __future__ import annotations

import random
import re
import string

import pandas as pd
from sklearn.model_selection import train_test_split

from paths import data_path
from splits import (
    LeakageError,
    assert_no_prompt_overlap,
    coerce_binary_label,
    normalize_prompt,
    repair_binary_frames,
)


def load_binary_frames(
    train_path=None,
    test_path=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path = train_path or data_path("binary_data", "training_data_binary.csv")
    test_path = test_path or data_path("binary_data", "testing_data_binary.csv")
    return pd.read_csv(train_path), pd.read_csv(test_path)


def load_clean_binary_corpus(train_path=None, test_path=None) -> tuple[pd.DataFrame, dict]:
    """Return the deduplicated union of the binary train and test files."""
    train_df, test_df = load_binary_frames(train_path, test_path)
    train_df, test_df, report = repair_binary_frames(train_df, test_df)
    corpus = pd.concat([train_df, test_df], ignore_index=True)
    assert_no_prompt_overlap(train_df, test_df)
    if corpus["prompt"].map(normalize_prompt).duplicated().any():
        raise LeakageError("Combined binary corpus still contains duplicate prompts")
    return corpus, report


def write_binary_splits(train_df: pd.DataFrame, test_df: pd.DataFrame, train_path=None, test_path=None):
    """Write derived binary splits after checking they do not overlap."""
    assert_no_prompt_overlap(train_df, test_df)
    for name, frame in ("train", train_df), ("test", test_df):
        if frame["prompt"].map(normalize_prompt).duplicated().any():
            raise LeakageError(f"{name} split contains duplicate prompts")
    train_path = train_path or data_path("binary_data", "training_data_binary.csv")
    test_path = test_path or data_path("binary_data", "testing_data_binary.csv")
    train_path.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    return train_path, test_path


def build_binary_dataset(
    jailbreak_df: pd.DataFrame,
    regular_df: pd.DataFrame,
    preprocess_fn,
    augment_fn=None,
    regular_cap: int | None = 2000,
    test_size: float = 0.2,
    random_state: int = 42,
    prompt_col: str = "prompt",
    label_col: str = "jailbreak",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a leak-free binary dataset from raw prompt tables.

    Inputs are not modified and are not written back. ``regular_cap`` keeps the
    historical cap of 2000 non-jailbreak prompts (applied after dropping
    prompts that also occur in the jailbreak table). Pass ``None`` to keep
    every remaining non-jailbreak prompt.
    """
    jailbreak_prompts = _unique_processed(
        jailbreak_df, preprocess_fn, prompt_col, positive=True, label_col=label_col
    )
    regular_prompts = _unique_processed(
        regular_df, preprocess_fn, prompt_col, positive=False, label_col=label_col
    )
    jailbreak_keys = set(jailbreak_prompts)
    regular_prompts = [prompt for prompt in regular_prompts if prompt not in jailbreak_keys]
    if regular_cap is not None:
        regular_prompts = regular_prompts[:regular_cap]
    if not jailbreak_prompts or not regular_prompts:
        raise ValueError("Both classes need at least one prompt after deduplication")

    pool = pd.DataFrame(
        {
            "prompt": jailbreak_prompts + regular_prompts,
            "jailbreak": [1] * len(jailbreak_prompts) + [0] * len(regular_prompts),
        }
    )
    train_df, test_df = train_test_split(
        pool,
        test_size=test_size,
        random_state=random_state,
        stratify=pool["jailbreak"],
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    if augment_fn is not None:
        train_df = _append_augmentations(train_df, test_df, augment_fn)
    assert_no_prompt_overlap(train_df, test_df)
    return train_df, test_df


def default_text_preprocess(text: str) -> str:
    """The binary-detection cleaner: lower-case, strip URLs and punctuation, drop stop words, lemmatise."""
    import nltk
    from nltk.corpus import stopwords, wordnet
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize

    _ensure_nltk()
    cleaned = str(text).lower()
    cleaned = re.sub(r"http\S+|www\S+|https\S+", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\@\w+|\#", "", cleaned)
    cleaned = cleaned.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(cleaned)
    stop_words = set(stopwords.words("english"))
    tokens = [token for token in tokens if token not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tagged = nltk.pos_tag(tokens)
    lemmas = [lemmatizer.lemmatize(word, _wordnet_pos(tag)) for word, tag in tagged]
    return " ".join(lemmas)


def synonym_replacement(text: str, rng: random.Random | None = None) -> str:
    """Replace some tokens with a WordNet lemma name. ``rng`` makes a rerun reproducible."""
    from nltk.corpus import wordnet

    _ensure_nltk()
    rng = rng or random.Random()
    words = str(text).split()
    replaced = []
    for word in words:
        synsets = wordnet.synsets(word)
        if synsets and rng.random() < 0.5:
            replaced.append(synsets[0].lemmas()[0].name())
        else:
            replaced.append(word)
    return " ".join(replaced)


def seeded_synonym_replacement(seed: int = 42):
    rng = random.Random(seed)

    def _replace(text: str) -> str:
        return synonym_replacement(text, rng=rng)

    return _replace


def _unique_processed(df, preprocess_fn, prompt_col, positive: bool, label_col: str) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    labels = df[label_col] if label_col in df.columns else None
    for index, raw in enumerate(df[prompt_col]):
        if labels is not None:
            label = coerce_binary_label(labels.iloc[index])
            if bool(label) != positive:
                continue
        processed = normalize_prompt(preprocess_fn(raw))
        if not processed or processed in seen:
            continue
        seen.add(processed)
        ordered.append(processed)
    return ordered


def _append_augmentations(train_df, test_df, augment_fn) -> pd.DataFrame:
    blocked = set(train_df["prompt"].map(normalize_prompt)) | set(test_df["prompt"].map(normalize_prompt))
    extra_prompts = []
    extra_labels = []
    for prompt, label in zip(train_df["prompt"], train_df["jailbreak"]):
        augmented = normalize_prompt(augment_fn(prompt))
        if not augmented or augmented in blocked:
            continue
        blocked.add(augmented)
        extra_prompts.append(augmented)
        extra_labels.append(int(label))
    if not extra_prompts:
        return train_df
    extra = pd.DataFrame({"prompt": extra_prompts, "jailbreak": extra_labels})
    return pd.concat([train_df, extra], ignore_index=True)


def _wordnet_pos(tag: str):
    from nltk.corpus import wordnet

    if tag.startswith("J"):
        return wordnet.ADJ
    if tag.startswith("V"):
        return wordnet.VERB
    if tag.startswith("N"):
        return wordnet.NOUN
    if tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


def _ensure_nltk() -> None:
    import nltk

    for resource in ("punkt", "punkt_tab", "stopwords", "wordnet", "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng"):
        try:
            nltk.data.find(_nltk_path(resource))
        except LookupError:
            nltk.download(resource, quiet=True)


def _nltk_path(resource: str) -> str:
    if resource in {"punkt", "punkt_tab"}:
        return f"tokenizers/{resource}"
    if resource in {"averaged_perceptron_tagger", "averaged_perceptron_tagger_eng"}:
        return f"taggers/{resource}"
    if resource == "wordnet":
        return "corpora/wordnet"
    return f"corpora/{resource}"
