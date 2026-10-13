"""Tests for batchtextreview.labels."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from batchtextreview.labels import (
    INVALID,
    canonical_labels,
    clean_output,
    is_missing,
    label_key,
    normalise_label,
    parse_json_output,
    resolve_labels,
)

SENTIMENT = ["positive", "negative", "mixed", "neutral"]
SCORES = ["1", "2", "3", "4", "5"]


# --------------------------------------------------------------------------
# is_missing
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [None, float("nan"), np.nan, np.float64("nan"), pd.NA, pd.NaT, "", "   ", "\n\t "],
)
def test_is_missing_true(value):
    assert is_missing(value)


@pytest.mark.parametrize(
    "value", ["nan", "NA", "None", "0", 0, 0.0, False, "x", [], ["a"], np.array([np.nan])]
)
def test_is_missing_false(value):
    assert not is_missing(value)


# --------------------------------------------------------------------------
# clean_output
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, ""),
        (float("nan"), ""),
        ("  Positive \n", "Positive"),
        (42, "42"),
        # complete blocks, any case, multiline, several of them
        ("<think>\nhmm, negative?\n</think>\n\nPositive", "Positive"),
        ("<THINK>a</THINK>Pos<Think>b</tHiNk>itive", "Positive"),
        ("<think></think>Neutral", "Neutral"),
        # a template opened the block: only the close tag appears
        ("the user wants... negative</think>\nPositive", "Positive"),
        ("a</think>b</think> Positive", "Positive"),
        # unterminated: generation stopped while thinking
        ("<think>still reasoning about whether it is positive", ""),
        ("<think>", ""),
        ("Positive <think>second thoughts", "Positive"),
        ("dangling</think>Answer <think>cut off", "Answer"),
        # whitespace inside the tags
        ("< think >x</ think >Yes", "Yes"),
        # only think handling - nothing else is touched
        ("**Positive.**", "**Positive.**"),
        ("I think it's positive", "I think it's positive"),
    ],
)
def test_clean_output(raw, expected):
    assert clean_output(raw) == expected


# --------------------------------------------------------------------------
# label_key
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        (float("nan"), None),
        (pd.NA, None),
        ("", None),
        ("  ", None),
        ("Positive", "positive"),
        ("  POSITIVE  ", "positive"),
        ("**Negative**", "negative"),
        ("`neutral`", "neutral"),
        ("_mixed_", "mixed"),
        ('"positive."', "positive"),
        ("“Neutral”", "neutral"),
        ("(positive)", "positive"),
        ("- positive", "positive"),
        ("**- positive**", "positive"),
        ("ｐｏｓｉｔｉｖｅ！", "positive"),  # fullwidth -> NFKC
        ("posi\u200btive", "positive"),  # zero-width space
        ("very   \n positive", "very positive"),
        ("not_mentioned", "not mentioned"),
        ("Not-Mentioned", "not mentioned"),
        ("not mentioned", "not mentioned"),
        ("a__b", "a b"),
        ("Sentiment: neutral", "sentiment: neutral"),
        # numbers
        (3, "3"),
        (3.0, "3"),
        (np.float64(3.0), "3"),
        (np.int64(3), "3"),
        (np.float32(2.0), "2"),
        (2.5, "2.5"),
        ("3", "3"),
        ("3.0", "3"),
        ("3.00", "3"),
        ("3.5", "3.5"),
        ("03", "03"),  # IDs keep leading zeros
        (True, "true"),
        # punctuation that is part of the label survives
        ("C++", "c++"),
        ("C#", "c#"),
        ("N/A", "n/a"),
        ("n/a.", "n/a"),
        ("-1", "-1"),
        ("+1", "+1"),
        ("50%", "50%"),
        # nothing but punctuation keeps its spelling instead of vanishing
        ("?", "?"),
        ("...", "..."),
    ],
)
def test_label_key(value, expected):
    assert label_key(value) == expected


@pytest.mark.parametrize(
    "value",
    ["**- Positive**", "Not_Mentioned", "ǅ", "İstanbul", "ℌello", "“n/a.”", "3.0", "...", "K"],
)
def test_label_key_is_idempotent(value):
    once = label_key(value)
    assert label_key(once) == once


def test_label_key_float_gold_matches_string():
    assert label_key(3.0) == label_key("3") == label_key(np.float64(3))


# --------------------------------------------------------------------------
# normalise_label
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # exact (after cleaning)
        ("Positive", "positive"),
        ("**Negative**", "negative"),
        ("NEUTRAL.", "neutral"),
        ("<think>could be negative</think>Positive", "positive"),
        ("reasoning... negative</think> Mixed", "mixed"),
        ("<think>I believe it is positive", None),
        # prefixes
        ("Sentiment: neutral", "neutral"),
        ("Answer: Positive", "positive"),
        ("Final answer: **Negative**", "negative"),
        ("The sentiment of this text is: mixed", "mixed"),
        ("Label: Sentiment: positive", "positive"),
        # containment
        ("The sentiment is positive.", "positive"),
        ("I'd say **negative** overall", "negative"),
        ("It is _positive_", "positive"),
        ("Positive - the customer thanks the team.", "positive"),
        # negation
        ("Not positive", None),
        ("isn't positive", None),
        ("isn’t positive", None),
        ("never positive", None),
        ("It is not very positive", None),
        ("This is not a negative review; it is positive.", "positive"),
        ("Positive, not negative", "positive"),
        ("neither positive nor negative", None),
        # negating phrases and fillers that once let the label through
        ("Not at all positive", None),
        ("not even remotely positive", None),
        ("hardly positive", None),
        ("barely positive", None),
        ("scarcely positive", None),
        ("far from positive", None),
        ("anything but positive", None),
        ("Far from negative, it is positive.", "positive"),
        # ambiguity is never guessed away
        ("positive/negative", None),
        ("Positive or negative", None),
        # several labels mentioned, no containment rule applies -> None
        ("Mixed (positive and negative)", None),
        ("Mixed: positive and negative", None),
        # nothing recognisable
        ("I cannot tell", None),
        ("", None),
        (None, None),
        (float("nan"), None),
        # "knot" isn't a negator; "positively" isn't "positive"
        ("knot positive", "positive"),
        ("positively", None),
    ],
)
def test_normalise_label_sentiment(raw, expected):
    assert normalise_label(raw, SENTIMENT) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("3", "3"),
        (3, "3"),
        (3.0, "3"),
        (np.float64(4.0), "4"),
        ("3.0", "3"),
        ("3/5", "3"),
        ("3 / 5", "3"),
        ("3 out of 5", "3"),
        ("Urgency: 2/5", "2"),
        ("Rating: 4 out of 5", "4"),
        ("I'd rate it 4 out of 5.", "4"),
        ("5/10", "5"),
        ("**2**", "2"),
        # markdown / brackets around a score form
        ("**4**/5", "4"),
        ("4/**5**", "4"),
        ("Rating: **4** out of 5", "4"),
        ("4 (out of 5)", "4"),
        ("I'd rate it **4**/5 overall.", "4"),
        ("Score (1-5): 4", "4"),
        ("Urgency (1/5 scale): 2", "2"),
        ("Level 3.", "3"),
        ("Somewhere between 2 and 3", None),
        ("3-4", None),
        ("2.5", None),
        ("7/10", None),  # numerator not an allowed label
        ("10:30", None),
    ],
)
def test_normalise_label_scores(raw, expected):
    assert normalise_label(raw, SCORES) == expected


def test_longer_label_beats_contained_one():
    allowed = ["positive", "very positive"]
    assert normalise_label("very positive", allowed) == "very positive"
    assert normalise_label("Very Positive!", allowed) == "very positive"
    assert normalise_label("The text is very positive.", allowed) == "very positive"
    assert normalise_label("positive", allowed) == "positive"
    assert normalise_label("Somewhat positive", allowed) == "positive"


def test_labels_with_punctuation():
    allowed = ["c++", "c", "c#", "n/a"]
    assert normalise_label("C++", allowed) == "c++"
    assert normalise_label("I'd use c++.", allowed) == "c++"
    assert normalise_label("C", allowed) == "c"
    assert normalise_label("c#", allowed) == "c#"
    assert normalise_label("N/A", allowed) == "n/a"
    assert normalise_label("The answer is n/a", allowed) == "n/a"
    assert normalise_label("c++ or c#", allowed) is None


def test_underscore_and_space_spellings_are_one_label():
    allowed = ["not_mentioned", "mentioned"]
    assert canonical_labels(allowed) == ["not mentioned", "mentioned"]
    assert normalise_label("not mentioned", allowed) == "not mentioned"
    assert normalise_label("Not_Mentioned", allowed) == "not mentioned"
    assert normalise_label("NOT-MENTIONED", allowed) == "not mentioned"
    assert normalise_label("The topic is not mentioned.", allowed) == "not mentioned"
    assert normalise_label("mentioned", allowed) == "mentioned"
    # Without the longer label, "not mentioned" is a negated hit.
    assert normalise_label("not mentioned", ["mentioned"]) is None


def test_allowed_labels_are_canonicalised():
    allowed = ["Positive", "NEGATIVE", " positive ", ""]
    assert canonical_labels(allowed) == ["positive", "negative"]
    assert normalise_label("positive", allowed) == "positive"
    assert normalise_label("Negative", allowed) == "negative"
    assert canonical_labels(None) == []


def test_yes_no_labels():
    allowed = ["yes", "no"]
    assert normalise_label("Yes.", allowed) == "yes"
    assert normalise_label("NO", allowed) == "no"
    assert normalise_label("No, it does not.", allowed) == "no"
    assert normalise_label("Yes - the parcel was lost.", allowed) == "yes"
    assert normalise_label("Yes and no", allowed) is None


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # The answer comes first; "no" further on is an ordinary word.
        ("Yes, the customer has had no reply.", "yes"),
        ("Yes - no one returned his calls.", "yes"),
        ("Yes. No response was given.", "yes"),
        ("Yes; there is no doubt.", "yes"),
        ("Yes: no one called back.", "yes"),
        ("Yes! No one called back.", "yes"),
        ("Yes – no one called back.", "yes"),
        ("Yes—no one called back.", "yes"),
        ("**Yes**, no one replied.", "yes"),
        ('"Yes", no one replied.', "yes"),
        ("Yes\n\nThe customer has had no", "yes"),  # cut off at max_new_tokens
        ("Answer: Yes, no one called.", "yes"),
        ("Answer:\nYes\nNo one replied.", "yes"),
        ("No, no one was at fault.", "no"),
        # Nothing but more answers after the first: still ambiguous.
        ("Yes, no", None),
        ("Yes. No.", None),
        ("Yes - no", None),
        ("Yes\nNo", None),
        ("Yes, no, maybe", None),
        ("Yes? No one knows.", None),  # "?" isn't a clause break
    ],
)
def test_leading_answer_wins_over_later_function_words(raw, expected):
    assert normalise_label(raw, ["yes", "no"]) == expected
    assert resolve_labels([raw], ["yes", "no"]) == [expected or INVALID]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Complaint - and other issues.", "complaint"),
        ("Other: asks about other things", "other"),
        ("Other, request", None),
        ("Request, query or other", None),
    ],
)
def test_leading_answer_intent(raw, expected):
    intent = ["complaint", "query", "request", "praise", "other"]
    assert normalise_label(raw, intent) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Neutral, no strong positive or negative words.", "neutral"),
        ("Positive, not negative", "positive"),
        # A list of options, or the first label not set off by a clause break.
        ("Mixed, positive and negative", None),
        ("Mixed - both positive and negative", None),
        ("Positive, negative, neutral", None),
        ("Positive and negative aspects", None),
        ("Positive? Negative? Hard to say.", None),
        ("Very positive, mostly.", "very positive"),
    ],
)
def test_leading_answer_sentiment(raw, expected):
    assert normalise_label(raw, [*SENTIMENT, "very positive"]) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("4. Fairly urgent", "4"),
        ("3, maybe 4", None),
        ("3 - 4", None),
        ("2.5", None),  # a decimal point is no clause break
        ("3,5", None),
    ],
)
def test_leading_answer_scores(raw, expected):
    assert normalise_label(raw, SCORES) == expected


@pytest.mark.parametrize(
    "raw", ["Non-urgent", "This is non-urgent.", "NON-URGENT", "non urgent", "non‑urgent"]
)
def test_non_prefix_negates(raw):
    assert normalise_label(raw, ["urgent", "routine"]) is None
    # A label that is itself "non-urgent" still matches.
    assert normalise_label(raw, ["urgent", "non-urgent"]) == "non urgent"


def test_non_prefix_negates_preset_labels():
    assert normalise_label("Non-negative", SENTIMENT) is None
    assert normalise_label("Non-complaint", ["complaint", "query", "other"]) is None
    assert normalise_label("Non-urgent, routine.", ["urgent", "routine"]) == "routine"
    # Only the word "non": "nonetheless" negates nothing.
    assert normalise_label("Nonetheless urgent", ["urgent", "routine"]) == "urgent"


def test_gold_values():
    # golds go through the same function: floats, missing values, strings
    assert normalise_label(3.0, SCORES) == "3"
    assert normalise_label("3", SCORES) == "3"
    for missing in (None, float("nan"), pd.NA, "", "   "):
        assert normalise_label(missing, SCORES) is None
        assert normalise_label(missing) is None
    # without a label set: plain key, "nan" text is a real value
    assert normalise_label("nan") == "nan"
    assert normalise_label(3.0) == "3"
    assert normalise_label("  The Answer.  ") == "the answer"


def test_degenerate_repeated_output_is_resolved_quickly():
    # A model stuck in a loop; containment once compared every hit with every
    # other one (about 17 s for this input).
    assert normalise_label("4 " * 12_000, SCORES) == "4"
    assert normalise_label("positive " * 6_000, SENTIMENT) == "positive"


def test_negation_is_only_looked_for_just_before_a_hit():
    assert normalise_label("lorem ipsum " * 2_000 + "not positive", SENTIMENT) is None
    assert normalise_label("not " * 2_000 + "positive", SENTIMENT) is None
    assert normalise_label("not negative " + "x " * 100 + "positive", SENTIMENT) == "positive"
    # The look-back window can start inside a word: "knot" is still no negator.
    assert normalise_label("knot " + '"' * 60 + "positive", SENTIMENT) == "positive"
    assert normalise_label("not " + '"' * 60 + "positive", SENTIMENT) is None


def test_label_set_cache_does_not_mix_equal_values_of_other_types():
    # True == 1 and hash(True) == hash(1): a shared cache slot gave [1] the
    # canonical labels of [True].
    assert canonical_labels([True]) == ["true"]
    assert canonical_labels([1]) == ["1"]
    assert normalise_label(True, [True, False]) == "true"
    assert normalise_label(1, [1, 0]) == "1"
    assert normalise_label("1", [1, 0]) == "1"


def test_empty_allowed_resolves_nothing():
    assert normalise_label("positive", []) is None


def test_unhashable_allowed_entries_do_not_break():
    assert normalise_label("positive", [["positive"], "positive"]) == "positive"


# --------------------------------------------------------------------------
# resolve_labels
# --------------------------------------------------------------------------


def test_resolve_labels():
    outputs = ["Positive", "", "   ", None, float("nan"), "no idea", "Sentiment: mixed", 3]
    assert resolve_labels(outputs, ["Positive", "Negative", "Mixed"]) == [
        "positive",
        "",
        "",
        "",
        "",
        INVALID,
        "mixed",
        INVALID,
    ]


def test_resolve_labels_matches_normalise_label():
    outputs = ["3/5", "Urgency: 4", "2 or 3", 5.0]
    expected = [normalise_label(o, SCORES) or INVALID for o in outputs]
    assert resolve_labels(outputs, SCORES) == expected == ["3", "4", INVALID, "5"]


def test_resolve_labels_accepts_a_series():
    series = pd.Series(["Positive", None, "negative"], index=[10, 20, 30])
    assert resolve_labels(series, SENTIMENT) == ["positive", "", "negative"]


# --------------------------------------------------------------------------
# parse_json_output
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('{"a": 1}', {"a": 1}),
        ("[1, 2]", [1, 2]),
        ('  {"a": [1, {"b": null}]}  ', {"a": [1, {"b": None}]}),
        # fences
        ('```json\n{"people": ["Ann"]}\n```', {"people": ["Ann"]}),
        ('```\n{"a": 1}\n```', {"a": 1}),
        ('```JSON {"a": 1}```', {"a": 1}),
        ('Here you go:\n```json\n{"a": 1}\n```\nHope that helps!', {"a": 1}),
        ('```json\n{"a": 1}', {"a": 1}),  # closing fence cut off
        # prose around
        ('Sure! The result is {"a": 1}. Anything else?', {"a": 1}),
        ('Result: [{"x": 1}, {"x": 2}] done', [{"x": 1}, {"x": 2}]),
        # braces / brackets / quotes inside strings
        ('Output: {"text": "a } b { c ] [", "n": 2} end', {"text": "a } b { c ] [", "n": 2}),
        ('{"q": "she said \\"hi}\\""} trailing', {"q": 'she said "hi}"'}),
        ('{"path": "C:\\\\dir\\\\"} x', {"path": "C:\\dir\\"}),
        # a bracketed aside before the real object is skipped
        ('Note [see below]: {"a": 1}', {"a": 1}),
        # think blocks are removed first
        ('<think>{"draft": true}</think>{"final": true}', {"final": True}),
        # unicode
        ('{"name": "Zoë"}', {"name": "Zoë"}),
        # an object beats a list echoed before it
        ('Aspects: ["food", "service"]\n{"food": "positive"}', {"food": "positive"}),
        ('[1, 2] then {"a": 1}', {"a": 1}),
        # a stray "{" in prose is passed over
        ('Here is { a broken thing and then {"a": 1}', {"a": 1}),
        ('Here is [1, 2, {"a": 1} and more', {"a": 1}),
        # a list comes back when there is no object
        ('Keywords: ["a", "b"]. See [ref] too.', ["a", "b"]),
    ],
)
def test_parse_json_output(raw, expected):
    assert parse_json_output(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        None,
        "",
        "no json here",
        "42",
        '"just a string"',
        '{"a": 1',  # truncated
        "{not: json}",
        "[1, 2",
        "{]",
        float("nan"),
    ],
)
def test_parse_json_output_none(raw):
    assert parse_json_output(raw) is None


@pytest.mark.parametrize(
    "raw",
    [
        # Cut off at max_new_tokens: none of the values inside is the answer.
        '{"people": ["Ann Lee", "Bob Ray"], "organisations": ["BPS"], "locations": ["Leeds", "York',
        '{"pii": ["John Smith", "07700 900123"]',
        '{"pii": ["John Smith", "07700 900123"],\n',
        '```json\n{"people": ["Ann"], "organisations": ["BPS"], "locations": ["Le',
        '{"text": "see [1, 2] for',
        '{"text": "line one\nline two [1, 2]',
        '{"a": [1], "b": tr',
        '{"a": [1], "b": 1.',
        '{"a": [1], "b": -',
        '{"a": [1], "b": 1e+',
        'Aspects: ["food", "service"]\n{"food": "positive", "service": "neg',
        # Closed but not JSON: its nested values aren't the answer either.
        '{"people": ["John"], "organisations": [], "locations": [], "dates": [],}',
        "{'people': ['Ann'], 'organisations': [], 'locations': [], 'dates': []}",
        '{people: ["Ann"], organisations: []}',
        '{"a": [1] // a comment\n}',
    ],
)
def test_parse_json_output_cut_off_or_malformed_object_is_none(raw):
    assert parse_json_output(raw) is None


def test_parse_json_output_skips_many_bracketed_asides():
    assert parse_json_output("[x] " * 5000 + '{"a": 1}') == {"a": 1}


def test_parse_json_output_stray_closers_do_not_use_up_the_scan_budget():
    # Each "[ref n}" stops at its mismatched "}"; charging the rest of the
    # text for each of them once gave up before reaching the object.
    pad = "lorem ipsum " * 2000
    text = "".join(f"[ref {i}}} " for i in range(90)) + pad + '{"label": "positive"}'
    assert parse_json_output(text) == {"label": "positive"}


def test_parse_json_output_pathological_input_is_bounded():
    # Every "{" opens a span that never closes; must give up, not hang.
    assert parse_json_output("{" * 50_000) is None
    assert parse_json_output('{"a": ' * 20_000) is None
    assert parse_json_output("[}" * 200_000) is None
