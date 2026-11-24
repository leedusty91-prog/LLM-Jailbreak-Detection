import pandas as pd

from labels import (
    CANONICAL_LABELS,
    EVALUATED_JAILBREAK_TYPES,
    has_label,
    parse_labels,
    selected_flags,
)


def test_parses_all_three_on_disk_formats():
    assert parse_labels("['Character Roleplay', 'Superior Model']") == (
        "Character Roleplay",
        "Superior Model",
    )
    assert parse_labels("'Character Roleplay', 'Superior Model'") == (
        "Character Roleplay",
        "Superior Model",
    )
    assert parse_labels("Character Roleplay, Superior Model") == (
        "Character Roleplay",
        "Superior Model",
    )
    assert parse_labels("Ethical Appeal ") == ("Ethical Appeal",)
    assert parse_labels(" ") == ()
    assert parse_labels(float("nan")) == ()
    assert parse_labels(None) == ()


def test_membership_is_exact_not_substring():
    assert has_label("Contradiction", "Contrastive") is False
    assert has_label("['Contrastive']", "Contrastive") is True
    assert has_label("Non-Jailbreak", "Simulate Jailbreaking") is False
    assert has_label("Simulate Jailbreaking", "Non-Jailbreak") is False
    assert has_label("Sudo Mode", "Sudo") is False
    assert has_label("'Character Roleplay', 'Superior Model'", "Sudo Mode") is False
    assert has_label("'Character Roleplay', 'Superior Model'", "Superior Model") is True


def test_checkbox_flags_follow_the_parser():
    vocabulary = ["Contrastive", "Contradiction", "Ethical Appeal"]
    flags = selected_flags("Contradiction, Ethical Appeal ", vocabulary)
    assert flags == [False, True, True]


def test_real_label_files_round_trip_and_match_known_counts():
    expected = {
        "data/df_jailbreak_categ.csv": {
            "rows": 168,
            "empty": 0,
            "Character Roleplay": 112,
            "Superior Model": 38,
            "Sudo Mode": 22,
            "Simulate Jailbreaking": 34,
            "Ethical Appeal": 10,
        },
        "data/all_labelled_cleaned.csv": {
            "rows": 998,
            "empty": 0,
            "Character Roleplay": 880,
            "Superior Model": 114,
            "Sudo Mode": 50,
            "Simulate Jailbreaking": 51,
            "Ethical Appeal": 25,
        },
        "data/jailbreak_prompts_DAN.csv": {
            "rows": 1405,
            "empty": 1318,
            "Character Roleplay": 56,
            "Superior Model": 19,
            "Sudo Mode": 11,
            "Simulate Jailbreaking": 17,
            "Ethical Appeal": 5,
        },
    }
    for path, counts in expected.items():
        frame = pd.read_csv(path)
        assert len(frame) == counts["rows"]
        parsed = frame["Labels"].map(parse_labels)
        assert int(parsed.map(len).eq(0).sum()) == counts["empty"]
        for name in parsed.explode().dropna().unique():
            assert name in CANONICAL_LABELS
        for label_name in EVALUATED_JAILBREAK_TYPES:
            assert int(parsed.map(lambda labels, name=label_name: name in labels).sum()) == counts[label_name]
            # The five evaluated names are not substrings of each other, so the
            # old test agrees here. The parser still rejects partial names.
            substring = frame["Labels"].map(
                lambda value, name=label_name: isinstance(value, str) and name in value
            ).sum()
            assert int(substring) == counts[label_name]
