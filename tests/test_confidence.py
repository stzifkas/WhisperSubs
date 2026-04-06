"""Unit tests for confidence parsing and propagation."""
import math
import pytest
from backend.confidence import parse_confidence, confidence_level
from backend.translation_graph import _build_context_block, SessionContext


# ── parse_confidence ──────────────────────────────────────────────────────────

def test_parse_confidence_returns_none_for_none():
    assert parse_confidence(None) is None


def test_parse_confidence_returns_none_for_empty_list():
    assert parse_confidence([]) is None


def test_parse_confidence_perfect_confidence():
    # logprob of 0 → probability 1.0 → score 1.0
    result = parse_confidence([{"token": "Hello", "logprob": 0.0, "bytes": []}])
    assert result == 1.0


def test_parse_confidence_single_token():
    logprob = -0.5
    result = parse_confidence([{"token": "Hi", "logprob": logprob, "bytes": []}])
    assert result == round(math.exp(logprob), 4)


def test_parse_confidence_multiple_tokens_geometric_mean():
    logprobs_list = [
        {"token": "Hello", "logprob": -0.2, "bytes": []},
        {"token": " world", "logprob": -0.8, "bytes": []},
    ]
    expected = round(math.exp((-0.2 + -0.8) / 2), 4)
    assert parse_confidence(logprobs_list) == expected


def test_parse_confidence_very_low():
    # Very low logprob → near-zero confidence
    logprobs_list = [{"token": "x", "logprob": -10.0, "bytes": []}]
    result = parse_confidence(logprobs_list)
    assert result is not None
    assert result < 0.01


def test_parse_confidence_skips_missing_logprob_key():
    # Entry without a logprob key should be ignored
    logprobs_list = [
        {"token": "Hello", "bytes": []},  # no logprob
        {"token": " world", "logprob": -0.5, "bytes": []},
    ]
    expected = round(math.exp(-0.5), 4)
    assert parse_confidence(logprobs_list) == expected


def test_parse_confidence_all_missing_returns_none():
    logprobs_list = [{"token": "x", "bytes": []}, {"token": "y", "bytes": []}]
    assert parse_confidence(logprobs_list) is None


def test_parse_confidence_result_in_unit_interval():
    logprobs_list = [
        {"token": t, "logprob": lp, "bytes": []}
        for t, lp in [("The", -0.1), (" cat", -0.3), (" sat", -1.2)]
    ]
    result = parse_confidence(logprobs_list)
    assert result is not None
    assert 0.0 < result <= 1.0


# ── confidence_level ──────────────────────────────────────────────────────────

def test_confidence_level_none_is_unknown():
    assert confidence_level(None) == "unknown"


def test_confidence_level_high():
    assert confidence_level(0.85) == "high"
    assert confidence_level(1.0) == "high"
    assert confidence_level(0.99) == "high"


def test_confidence_level_medium():
    assert confidence_level(0.55) == "medium"
    assert confidence_level(0.70) == "medium"
    assert confidence_level(0.84) == "medium"


def test_confidence_level_low():
    assert confidence_level(0.0) == "low"
    assert confidence_level(0.3) == "low"
    assert confidence_level(0.54) == "low"


# ── context block propagation ─────────────────────────────────────────────────

def _make_ctx(**kwargs) -> SessionContext:
    base: SessionContext = {
        "summary": "",
        "glossary": {},
        "recent_pairs": [],
        "covered_up_to": 0,
        "target_language": "",
        "_updating": False,
    }
    base.update(kwargs)
    return base


def test_context_block_empty_ctx_no_confidence():
    assert _build_context_block(_make_ctx()) == ""


def test_context_block_low_confidence_adds_warning():
    block = _build_context_block(_make_ctx(), confidence=0.3)
    assert "LOW" in block
    assert "thorough" in block.lower()


def test_context_block_high_confidence_adds_note():
    block = _build_context_block(_make_ctx(), confidence=0.9)
    assert "HIGH" in block
    assert "unambiguous" in block.lower()


def test_context_block_medium_confidence_no_level_note():
    block = _build_context_block(_make_ctx(), confidence=0.7)
    # Medium confidence: no special note injected, block empty when no ctx
    assert "LOW" not in block
    assert "HIGH" not in block


def test_context_block_unknown_confidence_no_level_note():
    block = _build_context_block(_make_ctx(), confidence=None)
    assert "LOW" not in block
    assert "HIGH" not in block


def test_context_block_includes_summary_and_glossary():
    ctx = _make_ctx(summary="A debate about politics", glossary={"Moro": "Μόρο"})
    block = _build_context_block(ctx)
    assert "A debate about politics" in block
    assert "Moro=Μόρο" in block


def test_context_block_low_confidence_with_glossary():
    ctx = _make_ctx(summary="Cooking show", glossary={"carbonara": "καρμπονάρα"})
    block = _build_context_block(ctx, confidence=0.2)
    assert "LOW" in block
    assert "Cooking show" in block
    assert "carbonara=καρμπονάρα" in block
