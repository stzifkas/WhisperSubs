"""Tests for span-based selective transcript refinement."""
import pytest
from backend.span_classifier import (
    CATEGORY_ACRONYM,
    CATEGORY_DATE,
    CATEGORY_MEASUREMENT,
    CATEGORY_NUMBER,
    CATEGORY_PROPER_NOUN,
    TextSpan,
    annotate_transcript,
    classify_spans,
    token_word_confidences,
    _align_confidences,
)

LOW = 0.55    # default low-confidence threshold
HIGH_RISK = 0.80  # default high-risk threshold


# ── Helpers ───────────────────────────────────────────────────────────────────

def _lp(token: str, logprob: float) -> dict:
    return {"token": token, "logprob": logprob, "bytes": []}


def _spans(transcript: str, logprobs=None) -> list[TextSpan]:
    return classify_spans(transcript, logprobs)


def _annotate(transcript: str, logprobs=None, low=LOW, high_risk=HIGH_RISK) -> str | None:
    spans = classify_spans(transcript, logprobs)
    return annotate_transcript(transcript, spans, low, high_risk)


# ── token_word_confidences ────────────────────────────────────────────────────

def test_twc_empty():
    assert token_word_confidences([]) == []


def test_twc_single_word():
    result = token_word_confidences([_lp("Hello", -0.1)])
    assert len(result) == 1
    word, conf = result[0]
    assert word == "Hello"
    assert conf is not None and 0 < conf <= 1


def test_twc_splits_on_leading_space():
    lps = [_lp("Hello", -0.1), _lp(" world", -0.2)]
    result = token_word_confidences(lps)
    assert len(result) == 2
    assert result[0][0] == "Hello"
    assert result[1][0] == "world"


def test_twc_multi_token_word():
    # "car" + "bon" + "ara" (no leading spaces) → one word
    lps = [_lp("car", -0.3), _lp("bon", -0.2), _lp("ara", -0.1)]
    result = token_word_confidences(lps)
    assert len(result) == 1
    assert result[0][0] == "carbonara"


def test_twc_averages_logprobs():
    import math
    lps = [_lp("fast", -0.2), _lp("food", -0.0)]  # second token starts without space → same word? no space
    # Actually "food" doesn't start with space so it's same word as "fast"
    result = token_word_confidences(lps)
    assert len(result) == 1
    expected = round(math.exp((-0.2 + 0.0) / 2), 4)
    assert result[0][1] == expected


# ── _align_confidences ────────────────────────────────────────────────────────

def test_align_exact_match():
    tw = ["Hello", "world"]
    tok = [("Hello", 0.9), ("world", 0.7)]
    result = _align_confidences(tw, tok)
    assert result == [0.9, 0.7]


def test_align_length_mismatch_returns_none_for_unmatched():
    tw = ["Hello", "beautiful", "world"]
    tok = [("Hello", 0.9), ("world", 0.7)]
    result = _align_confidences(tw, tok)
    assert result[0] == 0.9   # matched
    assert result[2] == 0.7   # matched
    assert result[1] is None  # unmatched insertion


def test_align_punctuation_stripped():
    tw = ["Hello,", "world."]
    tok = [("Hello", 0.8), ("world", 0.6)]
    result = _align_confidences(tw, tok)
    assert result == [0.8, 0.6]


def test_align_empty_tokens():
    tw = ["Hello", "world"]
    result = _align_confidences(tw, [])
    assert result == [None, None]


# ── classify_spans ────────────────────────────────────────────────────────────

class TestCleanTranscript:
    """High-confidence, no high-risk content → nothing to repair."""

    def test_returns_one_span_per_word(self):
        spans = _spans("the cat sat")
        assert len(spans) == 3

    def test_high_confidence_common_words_not_high_risk(self):
        spans = _spans("the cat sat on the mat")
        for s in spans:
            assert not s.is_high_risk

    def test_high_confidence_no_repair_needed(self):
        # Build logprobs with high confidence (logprob ≈ 0)
        text = "the quick brown fox"
        lps = [_lp(" " + w if i > 0 else w, -0.02) for i, w in enumerate(text.split())]
        spans = classify_spans(text, lps)
        assert all(not s.should_refine(LOW, HIGH_RISK) for s in spans)

    def test_annotate_returns_none_for_clean_transcript(self):
        text = "the quick brown fox"
        lps = [_lp(" " + w if i > 0 else w, -0.01) for i, w in enumerate(text.split())]
        assert _annotate(text, lps) is None


class TestNoisyTranscript:
    """Low-confidence tokens should be marked for repair."""

    def test_low_confidence_spans_flagged(self):
        text = "hello world"
        lps = [_lp("hello", -3.0), _lp(" world", -3.0)]  # exp(-3) ≈ 0.05 → very low
        spans = classify_spans(text, lps)
        assert all(s.should_refine(LOW, HIGH_RISK) for s in spans)

    def test_mixed_confidence_only_low_flagged(self):
        text = "the brigate ross met here"
        # "the" and "met" and "here" → high conf; "brigate" and "ross" → low conf
        lps = [
            _lp("the", -0.01),
            _lp(" brigate", -2.5),
            _lp(" ross", -2.8),
            _lp(" met", -0.05),
            _lp(" here", -0.02),
        ]
        spans = classify_spans(text, lps)
        repair_words = {s.text for s in spans if s.should_refine(LOW, HIGH_RISK)}
        assert "brigate" in repair_words
        assert "ross" in repair_words
        assert "the" not in repair_words
        assert "met" not in repair_words

    def test_annotate_wraps_consecutive_low_conf_spans(self):
        text = "the brigate ross met"
        lps = [
            _lp("the", -0.01),
            _lp(" brigate", -2.5),
            _lp(" ross", -2.8),
            _lp(" met", -0.05),
        ]
        result = _annotate(text, lps)
        assert result is not None
        assert "[REPAIR: brigate ross]" in result
        assert result.startswith("the ")
        assert result.endswith(" met")

    def test_annotate_non_consecutive_creates_separate_markers(self):
        text = "the brigate met ross here"
        lps = [
            _lp("the", -0.01),
            _lp(" brigate", -2.5),
            _lp(" met", -0.05),
            _lp(" ross", -2.8),
            _lp(" here", -0.02),
        ]
        result = _annotate(text, lps)
        assert result is not None
        assert result.count("[REPAIR:") == 2


class TestEntityHeavyTranscript:
    """Proper nouns and acronyms flagged even at moderate confidence."""

    def test_acronym_detected(self):
        spans = _spans("the NATO alliance")
        cats = {s.text: s.categories for s in spans}
        assert CATEGORY_ACRONYM in cats["NATO"]
        assert not cats["the"]
        assert not cats["alliance"]

    def test_acronym_at_moderate_conf_flagged(self):
        text = "the NATO summit"
        lps = [_lp("the", -0.01), _lp(" NATO", -0.5), _lp(" summit", -0.1)]
        # exp(-0.5) ≈ 0.61 → above LOW (0.55) but below HIGH_RISK (0.80) and it's an acronym
        spans = classify_spans(text, lps)
        nato = next(s for s in spans if s.text == "NATO")
        assert nato.is_high_risk
        assert nato.should_refine(LOW, HIGH_RISK)

    def test_proper_noun_mid_sentence_flagged(self):
        # "Berlusconi" is mid-sentence (not first word)
        spans = _spans("former prime minister Berlusconi spoke")
        by_word = {s.text: s for s in spans}
        assert CATEGORY_PROPER_NOUN in by_word["Berlusconi"].categories

    def test_proper_noun_sentence_initial_not_flagged(self):
        # First word capitalized but sentence-initial → not a proper noun
        spans = _spans("The meeting was held in Rome")
        by_word = {s.text: s for s in spans}
        assert not by_word["The"].is_high_risk

    def test_proper_noun_after_period_not_flagged(self):
        spans = _spans("It ended. The minister left.")
        by_word = {s.text: s for s in spans}
        # "The" after period → sentence-initial, not flagged
        assert not by_word["The"].is_high_risk

    def test_high_confidence_proper_noun_protected(self):
        text = "the Rome summit"
        lps = [_lp("the", -0.01), _lp(" Rome", -0.1), _lp(" summit", -0.05)]
        # exp(-0.1) ≈ 0.905 → above HIGH_RISK threshold → not repaired
        spans = classify_spans(text, lps)
        rome = next(s for s in spans if s.text == "Rome")
        assert rome.is_high_risk
        assert not rome.should_refine(LOW, HIGH_RISK)

    def test_no_logprobs_proper_noun_still_flagged(self):
        # Without logprobs, high-risk spans are still flagged
        spans = _spans("former president Moro was kidnapped")
        by_word = {s.text: s for s in spans}
        assert by_word["Moro"].should_refine(LOW, HIGH_RISK)
        assert not by_word["former"].should_refine(LOW, HIGH_RISK)


class TestNumberDateTranscript:
    """Numbers, measurements, and dates are flagged as high-risk."""

    def test_plain_number_detected(self):
        spans = _spans("there were 42 people")
        by_word = {s.text: s for s in spans}
        assert CATEGORY_NUMBER in by_word["42"].categories

    def test_percentage_detected(self):
        spans = _spans("inflation rose 3.5%")
        by_word = {s.text: s for s in spans}
        # 3.5% matches the measurement pattern (% is a unit); high-risk either way
        assert by_word["3.5%"].is_high_risk

    def test_measurement_detected(self):
        spans = _spans("the package weighs 500mg")
        by_word = {s.text: s for s in spans}
        assert CATEGORY_MEASUREMENT in by_word["500mg"].categories

    def test_date_numeric_detected(self):
        spans = _spans("born on 12/03/1975")
        by_word = {s.text: s for s in spans}
        assert CATEGORY_DATE in by_word["12/03/1975"].categories

    def test_month_name_detected(self):
        spans = _spans("it happened in January")
        by_word = {s.text: s for s in spans}
        assert CATEGORY_DATE in by_word["January"].categories

    def test_number_at_moderate_conf_repaired(self):
        text = "at 14:00 on March"
        lps = [
            _lp("at", -0.05),
            _lp(" 14:00", -0.6),   # exp(-0.6) ≈ 0.55 → below HIGH_RISK, high-risk → repaired
            _lp(" on", -0.03),
            _lp(" March", -0.6),
        ]
        spans = classify_spans(text, lps)
        by_word = {s.text: s for s in spans}
        assert by_word["14:00"].should_refine(LOW, HIGH_RISK)
        assert by_word["March"].should_refine(LOW, HIGH_RISK)

    def test_number_high_confidence_protected(self):
        text = "the number 42 appears"
        lps = [
            _lp("the", -0.01),
            _lp(" number", -0.02),
            _lp(" 42", -0.05),   # exp(-0.05) ≈ 0.95 → above HIGH_RISK → protected
            _lp(" appears", -0.02),
        ]
        spans = classify_spans(text, lps)
        by_word = {s.text: s for s in spans}
        assert not by_word["42"].should_refine(LOW, HIGH_RISK)

    def test_annotate_number_date_transcript(self):
        text = "the meeting on March 15 at 14:00"
        # No logprobs — only high-risk spans flagged
        result = _annotate(text)
        assert result is not None
        assert "[REPAIR:" in result
        # "the", "meeting", "on", "at" should be unmodified
        assert "the meeting on" in result


# ── annotate_transcript edge cases ───────────────────────────────────────────

def test_annotate_empty_transcript():
    assert _annotate("") is None


def test_annotate_all_protected_returns_none():
    text = "the cat sat"
    lps = [_lp(t, -0.01) for t in ["the", " cat", " sat"]]
    assert _annotate(text, lps) is None


def test_annotate_all_repairable_single_marker():
    text = "brigate ross curcio"
    lps = [_lp(t, -3.0) for t in ["brigate", " ross", " curcio"]]
    result = _annotate(text, lps)
    assert result == "[REPAIR: brigate ross curcio]"


def test_annotate_preserves_surrounding_text():
    text = "the wrong word here"
    lps = [
        _lp("the", -0.01),
        _lp(" wrong", -3.0),
        _lp(" word", -3.0),
        _lp(" here", -0.01),
    ]
    result = _annotate(text, lps)
    assert result == "the [REPAIR: wrong word] here"
