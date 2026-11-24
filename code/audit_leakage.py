"""Measure the evaluation leaks and, with --write-binary, repair the derived splits.

The raw prompt tables are never written. The binary train/test CSVs are derived
files. Repair keeps one copy of a prompt that leaked into both with the same
label (it stays in training) and drops prompts whose labels disagree.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_CODE_DIR = Path(__file__).resolve().parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

import pandas as pd

from binary_data import load_binary_frames, write_binary_splits
from labels import EVALUATED_JAILBREAK_TYPES, has_label
from novel_splits import DEFAULT_NON_JAILBREAK_RATIO, build_heldout_splits, legacy_nonjailbreak_overlap
from paths import data_path, results_path
from splits import repair_binary_frames
from text_preprocess import stem_preprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--write-binary",
        action="store_true",
        help="Rewrite data/binary_data/*.csv with the repaired frames.",
    )
    args = parser.parse_args()

    train_df, test_df = load_binary_frames()
    cleaned_train, cleaned_test, repair_report = repair_binary_frames(train_df, test_df)
    if args.write_binary:
        write_binary_splits(cleaned_train, cleaned_test)
        repair_report["wrote_binary_splits"] = True
    else:
        repair_report["wrote_binary_splits"] = False

    jailbreak = pd.read_csv(data_path("all_labelled_cleaned.csv"))
    regular = pd.read_csv(data_path("regular_prompts_DAN.csv"))
    category = pd.read_csv(data_path("df_jailbreak_categ.csv"))
    novel = []
    for label in EVALUATED_JAILBREAK_TYPES:
        raw_test = int(jailbreak["Labels"].map(lambda value, name=label: has_label(value, name)).sum())
        raw_train = len(jailbreak) - raw_test
        legacy = legacy_nonjailbreak_overlap(
            regular["prompt"],
            n_train=int(DEFAULT_NON_JAILBREAK_RATIO * raw_train),
            n_test=int(DEFAULT_NON_JAILBREAK_RATIO * raw_test),
            random_state=1,
        )
        _train, _test, info = build_heldout_splits(
            jailbreak,
            regular,
            label,
            ratio=DEFAULT_NON_JAILBREAK_RATIO,
            random_state=1,
            preprocess_fn=stem_preprocess,
        )
        novel.append(
            {
                "held_out_type": label,
                "legacy_nonjailbreak_overlap": int(legacy),
                "legacy_nonjailbreak_test_size": int(DEFAULT_NON_JAILBREAK_RATIO * raw_test),
                **info.as_dict(),
            }
        )

    category_counts = {
        label: int(category["Labels"].map(lambda value, name=label: has_label(value, name)).sum())
        for label in EVALUATED_JAILBREAK_TYPES
    }
    payload = {
        "binary_repair": repair_report,
        "category_positive_counts_in_df_jailbreak_categ": category_counts,
        "category_rows": int(len(category)),
        "novel_splits": novel,
        "notes": [
            "legacy_nonjailbreak_overlap is the old sampler: both splits call sample(random_state=1).",
            "prompt_overlap is the repaired held-out split and must be 0.",
            "Ethical Appeal has 10 positives in the 168-row hand-labelled file.",
        ],
    }
    path = results_path("leakage_audit.json")
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {path}")
    print(json.dumps(repair_report, indent=2))
    for row in novel:
        print(
            f"{row['held_out_type']}: legacy overlap {row['legacy_nonjailbreak_overlap']} "
            f"of {row['legacy_nonjailbreak_test_size']} test regulars; "
            f"repaired overlap {row['prompt_overlap']}"
        )


if __name__ == "__main__":
    main()
