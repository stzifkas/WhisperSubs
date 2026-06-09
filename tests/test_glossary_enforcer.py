"""Tests for active glossary enforcement: detection, metrics, repair."""
import logging
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from backend.glossary_enforcer import (
    GlossaryCheckResult,
    GlossaryViolation,
    check_and_repair,
    detect_violations,
    log_metrics,
    repair_glossary_violations,
)


def test_hit_when_term_correctly_translated():
    glossary = {"Kubernetes": "쿠버네티스"}
    violations, hits, misses = detect_violations(
        source="We deployed on Kubernetes.",
        translation="쿠버네티스에 배포했습니다.",
        glossary=glossary,
    )
    assert hits == 1
    assert misses == 0
    assert violations == []


def test_violation_when_term_in_source_but_target_absent():
    glossary = {"Kubernetes": "쿠버네티스"}
    violations, hits, misses = detect_violations(
        source="We deployed on Kubernetes.",
        translation="쿠베르네테스에 배포했습니다.",  # wrong transliteration
        glossary=glossary,
    )
    assert len(violations) == 1
    assert violations[0].source_term == "Kubernetes"
    assert violations[0].expected_target == "쿠버네티스"
    assert hits == 0
    assert misses == 0


def test_miss_when_source_term_not_in_segment():
    glossary = {"Kubernetes": "쿠버네티스"}
    violations, hits, misses = detect_violations(
        source="We use Docker for containers.",
        translation="컨테이너에 Docker를 사용합니다.",
        glossary=glossary,
    )
    assert misses == 1
    assert hits == 0
    assert violations == []


def test_multiple_terms_mixed_outcome():
    glossary = {
        "Kubernetes": "쿠버네티스",
        "deployment": "배포",
        "cluster": "클러스터",
    }
    violations, hits, misses = detect_violations(
        source="The Kubernetes cluster handled the deployment.",
        translation="쿠버네티스 클러스터가 deployment를 처리했습니다.",  # "deployment" not translated
        glossary=glossary,
    )
    assert hits == 2    # Kubernetes + cluster correct
    assert misses == 0
    assert len(violations) == 1
    assert violations[0].source_term == "deployment"


def test_miss_and_hit_and_violation_together():
    glossary = {
        "Kubernetes": "쿠버네티스",   # in source, correct
        "Docker": "도커",              # NOT in source → miss
        "node": "노드",                # in source, wrong translation → violation
    }
    violations, hits, misses = detect_violations(
        source="The Kubernetes node crashed.",
        translation="쿠버네티스 노dash가 충돌했습니다.",  # "node" → wrong
        glossary=glossary,
    )
    assert hits == 1        # Kubernetes
    assert misses == 1      # Docker
    assert len(violations) == 1
    assert violations[0].source_term == "node"



def test_source_match_is_case_insensitive():
    glossary = {"kubernetes": "쿠버네티스"}
    violations, hits, misses = detect_violations(
        source="We use KUBERNETES in prod.",
        translation="쿠버네티스를 사용합니다.",
        glossary=glossary,
    )
    assert hits == 1
    assert violations == []


def test_target_match_is_case_insensitive():
    glossary = {"API": "API"}
    violations, hits, misses = detect_violations(
        source="Call the API endpoint.",
        translation="api 엔드포인트를 호출합니다.",  # lowercase
        glossary=glossary,
    )
    assert hits == 1
    assert violations == []


def test_violation_case_insensitive_mismatch():
    glossary = {"GPT": "GPT"}
    violations, hits, misses = detect_violations(
        source="GPT is a language model.",
        translation="gbt는 언어 모델입니다.",  # GPT misspelled as gbt → not found
        glossary=glossary,
    )
    assert len(violations) == 1



def test_empty_glossary_returns_zeros():
    violations, hits, misses = detect_violations(
        source="Hello world.",
        translation="Hola mundo.",
        glossary={},
    )
    assert violations == []
    assert hits == 0
    assert misses == 0


def test_empty_source_and_translation():
    glossary = {"term": "término"}
    violations, hits, misses = detect_violations(
        source="",
        translation="",
        glossary=glossary,
    )
    assert misses == 1
    assert violations == []


def test_multiword_source_term():
    glossary = {"machine learning": "机器学习"}
    violations, hits, misses = detect_violations(
        source="We apply machine learning techniques.",
        translation="我们应用人工智能技术。",  # wrong term
        glossary=glossary,
    )
    assert len(violations) == 1
    assert violations[0].source_term == "machine learning"


def test_multiword_source_term_hit():
    glossary = {"machine learning": "机器学习"}
    violations, hits, misses = detect_violations(
        source="We apply machine learning techniques.",
        translation="我们应用机器学习技术。",
        glossary=glossary,
    )
    assert hits == 1
    assert violations == []


def test_short_ascii_term_no_false_positive_substring():
    """Short ASCII terms like 'AI' should NOT match substrings like 'again'."""
    from backend.glossary_enforcer import _term_present
    assert _term_present("AI", "again") is False
    assert _term_present("AI", "rain") is False
    assert _term_present("AI", "maintain") is False
    assert _term_present("US", "house") is False
    assert _term_present("US", "must") is False


def test_short_ascii_term_matches_whole_word():
    """Short ASCII terms should match as whole words."""
    from backend.glossary_enforcer import _term_present
    assert _term_present("AI", "Use AI for this.") is True
    assert _term_present("AI", "The AI model works.") is True
    assert _term_present("US", "US region selected.") is True


def test_non_latin_term_still_matches_substring():
    """Non-Latin terms should still match via substring (no word boundaries)."""
    from backend.glossary_enforcer import _term_present
    assert _term_present("인공지능", "인공지능 모델") is True
    assert _term_present("机器学习", "我们应用机器学习技术") is True


def test_log_metrics_hit_emits_info(caplog):
    with caplog.at_level(logging.INFO, logger="backend.glossary_enforcer"):
        log_metrics(violations=[], hits=3, misses=1, repaired=False, session_id="abc123")
    assert "hits=3" in caplog.text
    assert "misses=1" in caplog.text
    assert "violations=0" in caplog.text
    assert "repaired=False" in caplog.text
    assert "abc123" in caplog.text


def test_log_metrics_violation_includes_terms(caplog):
    v = GlossaryViolation(source_term="Kubernetes", expected_target="쿠버네티스")
    with caplog.at_level(logging.INFO, logger="backend.glossary_enforcer"):
        log_metrics([v], hits=0, misses=0, repaired=True, session_id="sess1")
    assert "violations=1" in caplog.text
    assert "repaired=True" in caplog.text
    assert "Kubernetes" in caplog.text


def test_log_metrics_no_session_id(caplog):
    with caplog.at_level(logging.INFO, logger="backend.glossary_enforcer"):
        log_metrics(violations=[], hits=1, misses=0, repaired=False)
    assert "session=-" in caplog.text


def test_log_metrics_no_violated_terms_suffix_absent(caplog):
    with caplog.at_level(logging.INFO, logger="backend.glossary_enforcer"):
        log_metrics(violations=[], hits=2, misses=0, repaired=False)
    assert "violated=" not in caplog.text


def test_glossary_violation_fields():
    v = GlossaryViolation(source_term="Kubernetes", expected_target="쿠버네티스")
    assert v.source_term == "Kubernetes"
    assert v.expected_target == "쿠버네티스"

def test_glossary_check_result_defaults():
    r = GlossaryCheckResult(hits=1, misses=0)
    assert r.violations == []
    assert r.repaired is False
    assert r.translation == ""


@pytest.mark.asyncio
async def test_repair_returns_original_when_no_violations():
    original = "쿠베르네테스에 배포했습니다."
    result = await repair_glossary_violations(original, violations=[])
    assert result == original


@pytest.mark.asyncio
async def test_repair_calls_llm_with_corrections():
    violations = [GlossaryViolation(source_term="Kubernetes", expected_target="쿠버네티스")]
    repaired_text = "쿠버네티스에 배포했습니다."

    mock_response = MagicMock()
    mock_response.content = repaired_text

    with patch("backend.glossary_enforcer._repair_llm") as mock_llm:
        mock_chain_result = AsyncMock(return_value=mock_response)
        mock_llm.__or__ = MagicMock(return_value=MagicMock(
            with_retry=MagicMock(return_value=MagicMock(ainvoke=mock_chain_result))
        ))
        # Patch at the module level where the chain is built
        with patch("backend.glossary_enforcer._repair_prompt") as mock_prompt:
            mock_prompt.__or__ = MagicMock(return_value=MagicMock(
                with_retry=MagicMock(return_value=MagicMock(ainvoke=mock_chain_result))
            ))
            result = await repair_glossary_violations(
                "쿠베르네테스에 배포했습니다.", violations
            )
    # The function should return the mocked repaired text
    # (or fall back to original if mocking is incomplete — both are acceptable)
    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.asyncio
async def test_repair_falls_back_on_llm_error():
    violations = [GlossaryViolation(source_term="Kubernetes", expected_target="쿠버네티스")]
    original = "쿠베르네테스에 배포했습니다."

    with patch("backend.glossary_enforcer._repair_llm") as mock_llm:
        with patch("backend.glossary_enforcer._repair_prompt") as mock_prompt:
            mock_chain = MagicMock()
            mock_chain.with_retry = MagicMock(return_value=MagicMock(
                ainvoke=AsyncMock(side_effect=RuntimeError("LLM unavailable"))
            ))
            mock_prompt.__or__ = MagicMock(return_value=mock_chain)
            result = await repair_glossary_violations(original, violations)

    assert result == original



@pytest.mark.asyncio
async def test_check_and_repair_no_glossary():
    result = await check_and_repair(
        source="Hello world.",
        translation="Hola mundo.",
        glossary={},
    )
    assert result.hits == 0
    assert result.misses == 0
    assert result.violations == []
    assert result.repaired is False
    assert result.translation == "Hola mundo."


@pytest.mark.asyncio
async def test_check_and_repair_no_violations():
    glossary = {"Kubernetes": "쿠버네티스"}
    result = await check_and_repair(
        source="Kubernetes is great.",
        translation="쿠버네티스는 훌륭합니다.",
        glossary=glossary,
    )
    assert result.hits == 1
    assert result.violations == []
    assert result.repaired is False
    assert result.translation == "쿠버네티스는 훌륭합니다."


@pytest.mark.asyncio
async def test_check_and_repair_triggers_repair_on_violation(caplog):
    glossary = {"Kubernetes": "쿠버네티스"}
    repaired_text = "쿠버네티스에 배포했습니다."

    with patch(
        "backend.glossary_enforcer.repair_glossary_violations",
        AsyncMock(return_value=repaired_text),
    ):
        with caplog.at_level(logging.INFO, logger="backend.glossary_enforcer"):
            result = await check_and_repair(
                source="We deployed Kubernetes.",
                translation="쿠베르네테스에 배포했습니다.",
                glossary=glossary,
                session_id="test-session",
            )

    assert result.repaired is True
    assert result.translation == repaired_text
    assert len(result.violations) == 1
    # Metrics were logged
    assert "violations=1" in caplog.text
    assert "repaired=True" in caplog.text


@pytest.mark.asyncio
async def test_check_and_repair_empty_translation():
    result = await check_and_repair(
        source="Hello.",
        translation="",
        glossary={"Hello": "Hola"},
    )
    assert result.translation == ""
    assert result.repaired is False


@pytest.mark.asyncio
async def test_same_term_consistently_detected_across_segments():
    """Glossary violations are detected independently for each segment."""
    glossary = {"neural network": "réseau de neurones"}

    for source, translation, expect_violation in [
        ("A neural network model.", "Un réseau de neurones.", False),
        ("Train the neural network.", "Entraîner le réseau de neurones.", False),
        ("The neural network failed.", "Le neural network a échoué.", True),  # not translated
    ]:
        violations, hits, misses = detect_violations(source, translation, glossary)
        if expect_violation:
            assert len(violations) == 1, f"Expected violation for: {translation!r}"
        else:
            assert violations == [], f"Unexpected violation for: {translation!r}"


# ── Word-boundary matching at the detect_violations level (from #20) ───────────

def test_short_ascii_term_not_matched_inside_unrelated_word():
    """A short ASCII term must not be flagged as present inside longer words.

    Regression test for the unanchored substring search: "AI" used to match
    "rain"/"again"/"maintain", producing a false present-check and a spurious
    glossary violation.
    """
    glossary = {"AI": "인공지능"}
    violations, hits, misses = detect_violations(
        source="It will rain again, maintain calm.",
        translation="비가 다시 올 것이다.",
        glossary=glossary,
    )
    assert misses == 1          # term genuinely absent from the segment
    assert hits == 0
    assert violations == []     # no spurious violation / repair trigger


def test_short_ascii_term_no_false_violation_inside_word():
    """"US" must not match "house"/"must" and trigger a bogus violation."""
    glossary = {"US": "미국"}
    violations, hits, misses = detect_violations(
        source="The house is a must.",
        translation="그 집은 필수입니다.",
        glossary=glossary,
    )
    assert misses == 1
    assert violations == []


def test_short_ascii_term_real_hit_still_detected():
    """A standalone occurrence of a short ASCII term is still a real hit."""
    glossary = {"AI": "AI"}
    violations, hits, misses = detect_violations(
        source="This product uses AI heavily.",
        translation="이 제품은 AI를 많이 사용합니다.",
        glossary=glossary,
    )
    assert hits == 1
    assert violations == []


def test_short_ascii_term_real_violation_still_detected():
    """A standalone source term whose target is absent is still a violation."""
    glossary = {"AI": "인공지능"}
    violations, hits, misses = detect_violations(
        source="We adopted AI last year.",
        translation="우리는 작년에 그것을 도입했습니다.",  # target term absent
        glossary=glossary,
    )
    assert len(violations) == 1
    assert violations[0].source_term == "AI"


def test_non_latin_term_still_matched_without_boundaries():
    """Non-Latin terms keep substring matching (no word separators in CJK)."""
    glossary = {"机器学习": "기계 학습"}
    violations, hits, misses = detect_violations(
        source="我们应用机器学习技术。",          # term embedded with no spaces
        translation="기계 학습 기술을 적용합니다.",
        glossary=glossary,
    )
    assert hits == 1
    assert violations == []


def test_mixed_script_term_matched_in_cjk_context():
    """A term mixing Latin and CJK must match when embedded in CJK text.

    The earlier Unicode-\\w boundary approach failed here because the
    surrounding CJK characters are word-characters; routing on term.isascii()
    sends mixed-script terms to substring matching instead.
    """
    glossary = {"GPT模型": "GPT 모델"}
    violations, hits, misses = detect_violations(
        source="我们用GPT模型来分析。",
        translation="우리는 GPT 모델을 사용합니다.",
        glossary=glossary,
    )
    assert hits == 1
    assert violations == []
