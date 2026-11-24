import pandas as pd

from labels import EVALUATED_JAILBREAK_TYPES
from novel_splits import (
    DEFAULT_NON_JAILBREAK_RATIO,
    build_heldout_splits,
    legacy_nonjailbreak_overlap,
    ratio_matching_corpus,
)
from text_preprocess import stem_preprocess


def _identity(text):
    return str(text).strip().lower()


def test_legacy_sampler_nests_the_smaller_draw():
    prompts = [f"regular {i}" for i in range(300)]
    assert legacy_nonjailbreak_overlap(prompts, n_train=80, n_test=25, random_state=1) == 25
    assert DEFAULT_NON_JAILBREAK_RATIO == 10.0
    assert ratio_matching_corpus(14, 10) == 1.4


def test_heldout_split_uses_one_preprocessor_and_disjoint_prompts():
    seen = []

    def preprocess(text):
        seen.append(text)
        return "stem:" + _identity(text)

    jailbreak = pd.DataFrame(
        {
            "prompt": [
                "Role Alpha",
                "Role Alpha",
                "Sudo Beta",
                "Both Gamma",
                "Plain Delta",
            ],
            "Labels": [
                "['Character Roleplay']",
                "'Character Roleplay', 'Superior Model'",
                "Sudo Mode",
                "'Character Roleplay', 'Sudo Mode'",
                "Ethical Appeal ",
            ],
        }
    )
    regular = pd.DataFrame({"prompt": [f"Regular {i}" for i in range(30)] + ["Role Alpha"]})
    train, test, info = build_heldout_splits(
        jailbreak,
        regular,
        "Character Roleplay",
        ratio=2,
        random_state=1,
        preprocess_fn=preprocess,
    )
    assert info.prompt_overlap == 0
    assert set(train["prompt"]).isdisjoint(set(test["prompt"]))
    assert (test["prompt"] == "stem:role alpha").any()
    assert (test["prompt"] == "stem:both gamma").any()
    assert (train["prompt"] == "stem:role alpha").sum() == 0
    assert (train["prompt"] == "stem:sudo beta").any()
    assert (train["prompt"] == "stem:plain delta").any()
    assert train["prompt"].str.startswith("stem:").all()
    assert test["prompt"].str.startswith("stem:").all()
    assert info.n_test_regular == 2 * info.n_test_jailbreak
    assert info.n_train_regular == 2 * info.n_train_jailbreak
    assert "Role Alpha" in seen and "Regular 0" in seen


def test_ratio_is_configurable_and_samples_do_not_overlap_when_sizes_differ():
    jailbreak = pd.DataFrame(
        {
            "prompt": [f"jb {i}" for i in range(6)],
            "Labels": ["Ethical Appeal"] * 2 + ["Sudo Mode"] * 4,
        }
    )
    regular = pd.DataFrame({"prompt": [f"reg {i}" for i in range(40)]})
    train, test, info = build_heldout_splits(
        jailbreak, regular, "Ethical Appeal", ratio=1, random_state=1, preprocess_fn=_identity
    )
    wide_train, wide_test, wide = build_heldout_splits(
        jailbreak, regular, "Ethical Appeal", ratio=3, random_state=1, preprocess_fn=_identity
    )
    assert info.n_test_regular == info.n_test_jailbreak
    assert wide.n_test_regular == 3 * wide.n_test_jailbreak
    assert set(train["prompt"]).isdisjoint(set(test["prompt"]))
    assert set(wide_train["prompt"]).isdisjoint(set(wide_test["prompt"]))
    assert len(wide_test) > len(test)


def test_real_novel_splits_have_no_prompt_overlap():
    jailbreak = pd.read_csv("data/all_labelled_cleaned.csv")
    regular = pd.read_csv("data/regular_prompts_DAN.csv")
    for label in EVALUATED_JAILBREAK_TYPES:
        raw_test = jailbreak["Labels"].map(lambda value, name=label: name in str(value)).sum()
        raw_train = len(jailbreak) - raw_test
        legacy = legacy_nonjailbreak_overlap(
            regular["prompt"],
            n_train=10 * int(raw_train),
            n_test=10 * int(raw_test),
            random_state=1,
        )
        assert legacy > 0
        train, test, info = build_heldout_splits(
            jailbreak,
            regular,
            label,
            ratio=DEFAULT_NON_JAILBREAK_RATIO,
            random_state=1,
            preprocess_fn=stem_preprocess,
        )
        assert info.prompt_overlap == 0
        assert set(train["prompt"]).isdisjoint(set(test["prompt"]))
        assert not train["prompt"].duplicated().any()
        assert not test["prompt"].duplicated().any()
        assert info.n_train_regular == 10 * info.n_train_jailbreak
        assert info.n_test_regular == 10 * info.n_test_jailbreak
        sample = next(prompt for prompt in jailbreak["prompt"] if stem_preprocess(prompt))
        processed = stem_preprocess(sample)
        assert processed in set(train["prompt"]) or processed in set(test["prompt"])
        assert (train["Jailbreak"] == 1).sum() == info.n_train_jailbreak
        assert (test["Jailbreak"] == 1).sum() == info.n_test_jailbreak
