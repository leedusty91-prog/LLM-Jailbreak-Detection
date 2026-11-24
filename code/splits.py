"""Leak-free prompt splits.

The fine-tuning script used to do this::

    train_sample = train_df.sample(frac=0.8, random_state=run).reset_index(drop=True)
    val_sample = train_df.drop(train_sample.index)

``reset_index`` replaces the sampled row ids with ``0 .. n_train-1``, so
``drop`` deletes the first ``n_train`` rows of ``train_df`` rather than the
sampled rows. Those two frames share most of their prompts. ``split_train_val``
draws a permutation and slices it, and it refuses to return overlapping text.
"""

from __future__ import annotations

import math
from collections import OrderedDict

import numpy as np
import pandas as pd


class LeakageError(ValueError):
    """Raised when a prompt would be scored after it was used for training."""


def normalize_prompt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def coerce_binary_label(value) -> int:
    if isinstance(value, str):
        text = value.strip().casefold()
        if text in {"1", "true", "yes", "jailbreak"}:
            return 1
        if text in {"0", "false", "no", "benign"}:
            return 0
        raise ValueError(f"Unrecognised binary label: {value!r}")
    if isinstance(value, (bool, np.bool_, int, np.integer)):
        return int(value)
    if isinstance(value, float):
        if math.isnan(value):
            raise ValueError("Missing binary label")
        return int(value)
    raise ValueError(f"Unrecognised binary label: {value!r}")


def prompt_set(frame: pd.DataFrame, prompt_col: str = "prompt") -> set[str]:
    return {prompt for prompt in frame[prompt_col].map(normalize_prompt) if prompt}


def overlap_count(*frames: pd.DataFrame, prompt_col: str = "prompt") -> int:
    if len(frames) < 2:
        return 0
    shared = prompt_set(frames[0], prompt_col)
    for frame in frames[1:]:
        shared &= prompt_set(frame, prompt_col)
    return len(shared)


def assert_no_prompt_overlap(*frames: pd.DataFrame, prompt_col: str = "prompt") -> None:
    """Raise ``LeakageError`` if any prompt string is in more than one frame."""
    owner: dict[str, int] = {}
    collisions: list[str] = []
    for index, frame in enumerate(frames):
        for prompt in frame[prompt_col].map(normalize_prompt):
            if not prompt:
                continue
            previous = owner.get(prompt)
            if previous is None:
                owner[prompt] = index
            elif previous != index:
                collisions.append(prompt)
    if collisions:
        raise LeakageError(
            f"{len(set(collisions))} prompts appear in more than one split"
        )


def split_train_val(
    df: pd.DataFrame,
    frac: float = 0.8,
    random_state: int = 0,
    prompt_col: str = "prompt",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split ``df`` into train and validation sets with disjoint prompts.

    Duplicate prompt strings are collapsed before the split so the same text
    cannot land in both sides. ``frac`` is the fraction of unique prompts
    assigned to training. At least one prompt is left for validation.
    """
    if not 0 < frac < 1:
        raise ValueError(f"frac must be between 0 and 1, got {frac}")
    if prompt_col not in df.columns:
        raise KeyError(prompt_col)

    working = df.copy()
    working["_norm_prompt"] = working[prompt_col].map(normalize_prompt)
    working = working.loc[working["_norm_prompt"] != ""].copy()
    working = working.drop_duplicates(subset=["_norm_prompt"], keep="first")
    working = working.reset_index(drop=True)
    n_rows = len(working)
    if n_rows < 2:
        raise ValueError("Need at least two unique prompts to split train and validation")

    n_train = int(round(n_rows * frac))
    n_train = min(max(n_train, 1), n_rows - 1)
    permutation = np.random.RandomState(random_state).permutation(n_rows)
    train = working.iloc[permutation[:n_train]].drop(columns=["_norm_prompt"]).reset_index(drop=True)
    validation = working.iloc[permutation[n_train:]].drop(columns=["_norm_prompt"]).reset_index(drop=True)
    assert_no_prompt_overlap(train, validation, prompt_col=prompt_col)
    return train, validation


def repair_binary_frames(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    prompt_col: str = "prompt",
    label_col: str = "jailbreak",
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Drop duplicate and cross-split prompts from the binary detection files.

    A prompt that appears in both files with the same label is removed from
    the test file and kept once in training. A prompt whose rows disagree on
    the label is removed from both files. Empty prompts are dropped. Raw
    source datasets are not read or written here; only these two frames change.
    """
    grouped: OrderedDict[str, dict] = OrderedDict()

    def absorb(frame: pd.DataFrame, split: str) -> int:
        kept = 0
        for prompt, label in zip(frame[prompt_col], frame[label_col]):
            key = normalize_prompt(prompt)
            if not key:
                continue
            kept += 1
            slot = grouped.get(key)
            if slot is None:
                slot = {
                    "labels": set(),
                    "prompt": prompt if isinstance(prompt, str) else str(prompt),
                    "splits": set(),
                }
                grouped[key] = slot
            slot["labels"].add(coerce_binary_label(label))
            slot["splits"].add(split)
        return kept

    train_nonempty = absorb(train_df, "train")
    test_nonempty = absorb(test_df, "test")

    train_keys = [
        normalize_prompt(prompt)
        for prompt in train_df[prompt_col]
        if normalize_prompt(prompt)
    ]
    test_keys = [
        normalize_prompt(prompt)
        for prompt in test_df[prompt_col]
        if normalize_prompt(prompt)
    ]
    shared = [key for key, slot in grouped.items() if slot["splits"] == {"train", "test"}]
    shared_same = [key for key in shared if len(grouped[key]["labels"]) == 1]
    shared_conflict = [key for key in shared if len(grouped[key]["labels"]) != 1]
    ambiguous = [key for key, slot in grouped.items() if len(slot["labels"]) != 1]

    train_rows = []
    test_rows = []
    ambiguous_examples = []
    for key, slot in grouped.items():
        if len(slot["labels"]) != 1:
            if len(ambiguous_examples) < 20:
                ambiguous_examples.append(
                    {
                        "prompt_prefix": slot["prompt"][:80],
                        "labels": sorted(slot["labels"]),
                        "splits": sorted(slot["splits"]),
                    }
                )
            continue
        label = next(iter(slot["labels"]))
        row = {prompt_col: slot["prompt"], label_col: label}
        if "train" in slot["splits"]:
            train_rows.append(row)
        elif "test" in slot["splits"]:
            test_rows.append(row)

    train_out = pd.DataFrame(train_rows, columns=[prompt_col, label_col])
    test_out = pd.DataFrame(test_rows, columns=[prompt_col, label_col])
    assert_no_prompt_overlap(train_out, test_out, prompt_col=prompt_col)
    report = {
        "train_rows_before": int(len(train_df)),
        "test_rows_before": int(len(test_df)),
        "train_nonempty_rows_before": int(train_nonempty),
        "test_nonempty_rows_before": int(test_nonempty),
        "train_rows_after": int(len(train_out)),
        "test_rows_after": int(len(test_out)),
        "dropped_train_duplicate_rows": int(len(train_keys) - len(set(train_keys))),
        "dropped_test_duplicate_rows": int(len(test_keys) - len(set(test_keys))),
        "shared_prompts_before": int(len(shared)),
        "shared_prompts_same_label": int(len(shared_same)),
        "shared_prompts_conflicting_label": int(len(shared_conflict)),
        "removed_ambiguous_prompts": int(len(ambiguous)),
        "removed_from_test_because_overlap": int(len(shared_same)),
        "ambiguous_examples": ambiguous_examples,
        "overlap_after": 0,
    }
    return train_out, test_out, report


def split_train_val_test(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    val_fraction: float = 0.2,
    random_state: int = 0,
    prompt_col: str = "prompt",
    label_col: str = "jailbreak",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Repair train/test leakage, then cut a validation slice from training only.

    The returned test frame is the held-out file after duplicate removal. It is
    not used to choose epochs. Validation is ``val_fraction`` of the unique
    training prompts.
    """
    if not 0 < val_fraction < 1:
        raise ValueError(f"val_fraction must be between 0 and 1, got {val_fraction}")
    cleaned_train, cleaned_test, report = repair_binary_frames(
        train_df, test_df, prompt_col=prompt_col, label_col=label_col
    )
    train_part, val_part = split_train_val(
        cleaned_train,
        frac=1.0 - val_fraction,
        random_state=random_state,
        prompt_col=prompt_col,
    )
    assert_no_prompt_overlap(train_part, val_part, cleaned_test, prompt_col=prompt_col)
    return train_part, val_part, cleaned_test, report
