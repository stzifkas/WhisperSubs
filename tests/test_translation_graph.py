import pytest
from unittest.mock import AsyncMock, MagicMock
from backend.translation_graph import route_translation, refine_node, translate_node, glossary_check_node
from backend.glossary_enforcer import (
    GlossaryCheckResult,
    GlossaryViolation,
)


def test_route_translation_skips_when_no_target_language():

    state = {
        "target_language": "",
    }

    assert route_translation(state) == "__end__"

def test_route_translation_goes_to_translate():

    state = {
        "target_language": "fr",
    }

    assert route_translation(state) == "translate"

@pytest.mark.asyncio
async def test_refine_node_skips_llm_when_no_repair(monkeypatch):

    monkeypatch.setattr(
        "backend.translation_graph.annotate_transcript",
        lambda *_a, **_k: None,
    )

    llm = AsyncMock()

    monkeypatch.setattr(
           "backend.translation_graph._refine_llm",
            llm,
    )

    state = {
        "raw_transcript": "hello",
        "session_context": {},
        "logprobs": None,
        "confidence": None,
        "refinement_low_threshold": 0.55,
        "refinement_high_risk_threshold":0.8,
        "correction_aggressiveness": "medium",
    }

    result = await refine_node(state)

    assert result == {
        "refined_source": "hello"

    }

    llm.with_structured_output.assert_not_called()


@pytest.mark.asyncio
async def test_translate_node_returns_translation(monkeypatch):

    class FakeResult:
        content = " Bonjour "

    chain = MagicMock()
    chain.with_retry.return_value = chain
    chain.ainvoke = AsyncMock (
    return_value=FakeResult()
)

    prompt = MagicMock()
    prompt.__or__.return_value = chain

    monkeypatch.setattr(
        "backend.translation_graph._translate_prompt",
        prompt,
    )

    state = {
        "refined_source": "hello",
        "target_language": "fr",
        "session_context": {
            "glossary": {},
        },
    }

    result = await translate_node(state)

    assert result == {
        "translation": "Bonjour"
    }

    chain.ainvoke.assert_called_once()


@pytest.mark.asyncio
async def test_glossary_check_node_returns_repaired_translation(monkeypatch):

    fake = AsyncMock()

    fake.return_value = GlossaryCheckResult(
        hits=0,
        misses=0,
        repaired=True,
        translation="Bonjour",
        violations=[
            GlossaryViolation(
                source_term="AI",
                expected_target="IA",
            )
        ],
    )

    monkeypatch.setattr(
        "backend.translation_graph.check_and_repair",
        fake,
    )

    state = {
        "translation": "hello",
        "refined_source": "AI",
        "session_context": {
            "glossary": {
                "AI": "IA"
            }
        },
    }

    result = await glossary_check_node(state)

    assert result == {
        "translation": "Bonjour",
        "glossary_violations": [
            {
                "source_term": "AI",
                "expected_target": "IA",
            }
        ],
    }

    fake.assert_awaited_once()