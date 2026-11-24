"""Canonical parser for jailbreak-type label strings.

The labelled files do not share a format:

* ``data/df_jailbreak_categ.csv`` stores Python list literals,
  e.g. ``['Character Roleplay', 'Superior Model']``.
* ``data/all_labelled_cleaned.csv`` stores quoted names without brackets,
  e.g. ``'Character Roleplay', 'Superior Model'``.
* ``data/jailbreak_prompts_DAN.csv`` stores plain comma-separated names,
  e.g. ``Character Roleplay, Superior Model``, and uses whitespace for
  unlabelled rows. Some values also have a trailing space
  (``Ethical Appeal ``).

Membership used to be ``label_name in labels_string``. That happens to succeed
for the current five evaluated names, and it is also wrong for any name that
is a substring of another (``Contrastive`` / ``Contradiction``, or a future
``Jailbreak`` label inside ``Non-Jailbreak`` and ``Simulate Jailbreaking``).
``parse_labels`` turns every format into an ordered tuple of exact labels.
"""

from __future__ import annotations

import ast
import math
from typing import Iterable, Sequence

CANONICAL_LABELS: tuple[str, ...] = (
    "Character Roleplay",
    "Assumed Responsibility",
    "Research Experiment",
    "Text Continuation",
    "Logical Reasoning",
    "Program Execution",
    "Translation",
    "Superior Model",
    "Sudo Mode",
    "Simulate Jailbreaking",
    "Ethical Appeal",
    "Contrastive",
    "Contradiction",
    "Complexity",
    "Gameplay",
    "Non-Jailbreak",
)

# Types used for the category classifiers and the novel-jailbreak experiment.
EVALUATED_JAILBREAK_TYPES: tuple[str, ...] = (
    "Character Roleplay",
    "Superior Model",
    "Sudo Mode",
    "Simulate Jailbreaking",
    "Ethical Appeal",
)

_LOOKUP = {name.casefold(): name for name in CANONICAL_LABELS}


def canonicalize(label: str) -> str:
    """Return the canonical spelling of ``label`` when it is a known type."""
    cleaned = _clean_token(label)
    if not cleaned:
        return ""
    return _LOOKUP.get(cleaned.casefold(), cleaned)


def parse_labels(value) -> tuple[str, ...]:
    """Parse one label cell into unique canonical names, in source order."""
    if _is_missing(value):
        return ()
    if isinstance(value, (list, tuple, set)):
        tokens = list(value)
    else:
        text = str(value).strip()
        if text == "" or text.casefold() in {"nan", "none", "<na>"}:
            return ()
        tokens = _parse_text(text)
    seen: list[str] = []
    for token in tokens:
        name = canonicalize(str(token))
        if name and name not in seen:
            seen.append(name)
    return tuple(seen)


def has_label(value, label_name: str) -> bool:
    """Exact membership test. Substring matches do not count."""
    name = canonicalize(label_name)
    if not name:
        return False
    return name in parse_labels(value)


def selected_flags(value, vocabulary: Sequence[str]) -> list[bool]:
    """Checkbox state for a labelling UI, using exact parsed labels."""
    present = set(parse_labels(value))
    return [canonicalize(name) in present for name in vocabulary]


def _is_missing(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return False


def _parse_text(text: str) -> list:
    # List literals and quoted tuples are valid Python literals. Plain
    # comma-separated names are not, and fall through to a split.
    if text[0] in "[('\"{":
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            parsed = None
        else:
            if isinstance(parsed, str):
                return [parsed]
            if isinstance(parsed, (list, tuple, set)):
                return list(parsed)
    if "," in text:
        return [_clean_token(part) for part in text.split(",")]
    return [_clean_token(text)]


def _clean_token(token: str) -> str:
    cleaned = str(token).strip()
    cleaned = cleaned.strip("[](){}")
    cleaned = cleaned.strip().strip("'\"")
    return cleaned.strip()


def iter_label_names(values: Iterable) -> list[str]:
    """Flatten parsed labels, preserving first-seen order."""
    ordered: list[str] = []
    for value in values:
        for name in parse_labels(value):
            if name not in ordered:
                ordered.append(name)
    return ordered
