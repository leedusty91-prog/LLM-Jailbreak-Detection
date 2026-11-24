import pandas as pd

from splits import (
    LeakageError,
    assert_no_prompt_overlap,
    overlap_count,
    repair_binary_frames,
    split_train_val,
    split_train_val_test,
)


def _frame(prompts, labels=None):
    labels = labels if labels is not None else [0] * len(prompts)
    return pd.DataFrame({"prompt": prompts, "jailbreak": labels})


def test_reset_index_drop_overlaps_and_permutation_split_does_not():
    frame = _frame([f"prompt {i}" for i in range(50)], [i % 2 for i in range(50)])
    buggy_train = frame.sample(frac=0.8, random_state=1).reset_index(drop=True)
    buggy_val = frame.drop(buggy_train.index)
    assert overlap_count(buggy_train, buggy_val) > 0

    train, validation = split_train_val(frame, frac=0.8, random_state=1)
    assert overlap_count(train, validation) == 0
    assert set(train["prompt"]) | set(validation["prompt"]) == set(frame["prompt"])
    assert len(train) + len(validation) == len(frame)


def test_split_collapses_duplicate_prompt_text():
    frame = _frame(["same", "same", "other"], [1, 1, 0])
    train, validation = split_train_val(frame, frac=0.5, random_state=0)
    assert overlap_count(train, validation) == 0
    assert len(train) + len(validation) == 2


def test_repair_removes_shared_prompts_from_test_and_conflicts_from_both():
    train = _frame(["keep", "shared", "conflict", "dup", "dup"], [1, 1, 1, 0, 0])
    test = _frame(["shared", "conflict", "only-test", "only-test"], [1, 0, 0, 0])
    cleaned_train, cleaned_test, report = repair_binary_frames(train, test)
    assert set(cleaned_train["prompt"]) == {"keep", "shared", "dup"}
    assert set(cleaned_test["prompt"]) == {"only-test"}
    assert overlap_count(cleaned_train, cleaned_test) == 0
    assert report["shared_prompts_before"] == 2
    assert report["shared_prompts_same_label"] == 1
    assert report["shared_prompts_conflicting_label"] == 1
    assert report["removed_from_test_because_overlap"] == 1
    again_train, again_test, again = repair_binary_frames(cleaned_train, cleaned_test)
    assert len(again_train) == len(cleaned_train)
    assert len(again_test) == len(cleaned_test)
    assert again["overlap_after"] == 0
    assert_no_prompt_overlap(cleaned_train, cleaned_test)


def test_three_way_split_is_disjoint():
    train = _frame([f"t{i}" for i in range(20)] + ["leak"], [i % 2 for i in range(21)])
    test = _frame(["leak", "e1", "e2"], [0, 1, 0])
    train_part, val_part, test_part, _report = split_train_val_test(
        train, test, val_fraction=0.25, random_state=3
    )
    assert "leak" not in set(test_part["prompt"])
    assert overlap_count(train_part, val_part) == 0
    assert overlap_count(train_part, test_part) == 0
    assert overlap_count(val_part, test_part) == 0
    assert len(train_part) > 0 and len(val_part) > 0 and len(test_part) > 0


def test_assert_rejects_overlap():
    try:
        assert_no_prompt_overlap(_frame(["a"]), _frame(["a"]))
    except LeakageError:
        return
    raise AssertionError("expected LeakageError")
