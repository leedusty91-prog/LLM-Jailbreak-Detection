"""Held-out jailbreak-type splits with disjoint non-jailbreak prompts.

The previous sampler drew non-jailbreak prompts twice with
``random_state=1``. Pandas implements that seed as one permutation, so the
smaller draw is a prefix of the larger one. For every evaluated type except
Character Roleplay the entire non-jailbreak test set was contained in
training. Character Roleplay is the majority type, so the relationship flips
and almost every non-jailbreak training prompt also appears in the test set.

``build_heldout_splits`` preprocesses both classes with the same function,
collapses duplicate strings, and samples non-jailbreak prompts from one
permutation that is then cut into two disjoint blocks.

``ratio`` is the number of non-jailbreak prompts per jailbreak prompt, computed
separately on the train side and the test side. The historical notebooks
hardcoded 10. That is higher than the roughly 1.4:1 balance of the binary
detection files. It remains the default so a rerun can be compared with the
original protocol; pass another positive number to change it. The paper text
describes matching the source corpus. Use
``ratio_matching_corpus`` when that is the protocol you want.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from labels import canonicalize, parse_labels
from splits import LeakageError, assert_no_prompt_overlap, normalize_prompt
from text_preprocess import stem_preprocess

# Historical notebook setting. See the module docstring.
DEFAULT_NON_JAILBREAK_RATIO = 10.0


@dataclass(frozen=True)
class SplitInfo:
    held_out_type: str
    ratio: float
    n_train_jailbreak: int
    n_test_jailbreak: int
    n_train_regular: int
    n_test_regular: int
    n_regular_available: int
    prompt_overlap: int

    def as_dict(self) -> dict:
        return asdict(self)


def ratio_matching_corpus(n_regular: int, n_jailbreak: int) -> float:
    """Non-jailbreak prompts per jailbreak prompt in a source corpus."""
    if n_jailbreak <= 0:
        raise ValueError("n_jailbreak must be positive")
    return n_regular / n_jailbreak


def legacy_nonjailbreak_overlap(
    regular_prompts,
    n_train: int,
    n_test: int,
    random_state: int = 1,
) -> int:
    """Overlap created by sampling both splits with the same seed.

    This reproduces the notebook bug and is used to document it. It is not
    used to build evaluation data.
    """
    if n_train < 0 or n_test < 0:
        raise ValueError("sample sizes must be non-negative")
    series = pd.Series(list(regular_prompts))
    if n_train > len(series) or n_test > len(series):
        raise ValueError("sample size exceeds the number of regular prompts")
    train = set(series.sample(n=n_train, random_state=random_state))
    test = set(series.sample(n=n_test, random_state=random_state))
    return len(train & test)


def build_heldout_splits(
    jailbreak_df: pd.DataFrame,
    regular_df: pd.DataFrame,
    label_name: str,
    ratio: float = DEFAULT_NON_JAILBREAK_RATIO,
    random_state: int = 1,
    preprocess_fn=None,
    prompt_col: str = "prompt",
    labels_col: str = "Labels",
    regular_prompt_col: str = "prompt",
) -> tuple[pd.DataFrame, pd.DataFrame, SplitInfo]:
    """Build one novel-jailbreak train/test pair.

    Prompts whose parsed labels include ``label_name`` are the test jailbreaks.
    Every other jailbreak prompt is a training jailbreak. Non-jailbreak prompts
    are sampled without replacement into the two sides. Both classes are passed
    through ``preprocess_fn`` (stemming and stop-word removal by default).
    """
    if ratio <= 0:
        raise ValueError(f"ratio must be positive, got {ratio}")
    if preprocess_fn is None:
        preprocess_fn = stem_preprocess
    held_out_type = canonicalize(label_name)
    if not held_out_type:
        raise ValueError("label_name is empty")

    jailbreak_groups = _collapse_jailbreaks(
        jailbreak_df, preprocess_fn, prompt_col, labels_col
    )
    test_jailbreaks = [prompt for prompt, labels in jailbreak_groups if held_out_type in labels]
    train_jailbreaks = [prompt for prompt, labels in jailbreak_groups if held_out_type not in labels]
    if not test_jailbreaks:
        raise ValueError(f"No jailbreak prompts are labelled {held_out_type!r}")
    if not train_jailbreaks:
        raise ValueError(f"Holding out {held_out_type!r} leaves no jailbreak training prompts")

    jailbreak_text = {prompt for prompt, _labels in jailbreak_groups}
    regular_prompts = _collapse_regular(
        regular_df, preprocess_fn, regular_prompt_col, blocked=jailbreak_text
    )
    n_test_regular = _scaled_count(len(test_jailbreaks), ratio)
    n_train_regular = _scaled_count(len(train_jailbreaks), ratio)
    needed = n_train_regular + n_test_regular
    if needed > len(regular_prompts):
        raise ValueError(
            f"Need {needed} unique non-jailbreak prompts at ratio {ratio} "
            f"for {held_out_type}, but {len(regular_prompts)} remain after "
            "preprocessing and removing prompts that collide with jailbreaks. "
            "Lower the ratio or add non-jailbreak prompts."
        )

    rng = np.random.RandomState(random_state)
    chosen = [regular_prompts[index] for index in rng.permutation(len(regular_prompts))[:needed]]
    regular_train = chosen[:n_train_regular]
    regular_test = chosen[n_train_regular:]
    train_df = _pack(train_jailbreaks, regular_train, rng)
    test_df = _pack(test_jailbreaks, regular_test, rng)
    assert_no_prompt_overlap(train_df, test_df)
    info = SplitInfo(
        held_out_type=held_out_type,
        ratio=float(ratio),
        n_train_jailbreak=len(train_jailbreaks),
        n_test_jailbreak=len(test_jailbreaks),
        n_train_regular=len(regular_train),
        n_test_regular=len(regular_test),
        n_regular_available=len(regular_prompts),
        prompt_overlap=0,
    )
    return train_df, test_df, info


def _scaled_count(n_jailbreak: int, ratio: float) -> int:
    count = int(round(ratio * n_jailbreak))
    if n_jailbreak > 0 and count < 1:
        return 1
    return count


def _collapse_jailbreaks(df, preprocess_fn, prompt_col, labels_col):
    grouped: dict[str, list[str]] = {}
    order: list[str] = []
    for raw_prompt, raw_labels in zip(df[prompt_col], df[labels_col]):
        processed = _processed(raw_prompt, preprocess_fn)
        if not processed:
            continue
        if processed not in grouped:
            grouped[processed] = []
            order.append(processed)
        for label in parse_labels(raw_labels):
            if label not in grouped[processed]:
                grouped[processed].append(label)
    return [(prompt, grouped[prompt]) for prompt in order]


def _collapse_regular(df, preprocess_fn, prompt_col, blocked: set[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for raw_prompt in df[prompt_col]:
        processed = _processed(raw_prompt, preprocess_fn)
        if not processed or processed in blocked or processed in seen:
            continue
        seen.add(processed)
        ordered.append(processed)
    return ordered


def _processed(raw_prompt, preprocess_fn) -> str:
    processed = preprocess_fn(raw_prompt)
    return normalize_prompt(processed)


def _pack(jailbreaks: list[str], regulars: list[str], rng: np.random.RandomState) -> pd.DataFrame:
    rows = [{"prompt": prompt, "Jailbreak": 1} for prompt in jailbreaks]
    rows.extend({"prompt": prompt, "Jailbreak": 0} for prompt in regulars)
    frame = pd.DataFrame(rows, columns=["prompt", "Jailbreak"])
    if frame.empty:
        raise LeakageError("Refusing to build an empty split")
    order = rng.permutation(len(frame))
    frame = frame.iloc[order].reset_index(drop=True)
    if frame["prompt"].duplicated().any():
        raise LeakageError("A split contains duplicate prompts")
    return frame
