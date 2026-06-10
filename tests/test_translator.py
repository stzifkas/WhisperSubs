import pytest
from unittest.mock import AsyncMock

from backend.translator import process_and_stream


class FakeChunk:
    def __init__(self, content):
        self.content = content


def make_session():
    return {
        "summary": "",
        "glossary": {},
        "recent_pairs": [],
    }


async def run_process(monkeypatch, events):

    async def fake_stream():
        for event in events:
            yield event

    fake_graph = AsyncMock()
    fake_graph.astream_events = (
        lambda *_a, **_k: fake_stream()
    )

    monkeypatch.setattr(
        "backend.translator.translation_graph",
        fake_graph,
    )

    websocket = AsyncMock()

    result = await process_and_stream(
        raw="hello",
        target_language="fr",
        whisper_language="en",
        session_context=make_session(),
        websocket=websocket,
    )

    return result, websocket


@pytest.mark.asyncio
async def test_process_stream_emits_subtitle_update(monkeypatch):

    result, websocket = await run_process(
        monkeypatch,
        [
            {
                "event": "on_chain_end",
                "name": "refine",
                "data": {
                    "output": {
                        "refined_source": "Hello refined"
                    }
                },
            }
        ],
    )

    websocket.send_json.assert_any_call({
        "type": "subtitle_update",
        "id": 0,
        "text": "Hello refined",
        "lang": "en",
    })

    assert result.source == "Hello refined"


@pytest.mark.asyncio
async def test_process_stream_emits_translation_delta(monkeypatch):

    result, websocket = await run_process(
        monkeypatch,
        [
            {
                "event": "on_chat_model_stream",
                "metadata": {
                    "langgraph_node": "translate"
                },
                "data": {
                    "chunk": FakeChunk("Hel")
                },
            },
            {
                "event": "on_chat_model_stream",
                "metadata": {
                    "langgraph_node": "translate"
                },
                "data": {
                    "chunk": FakeChunk("lo")
                },
            },
        ],
    )

    assert result.translation == "Hello"

    websocket.send_json.assert_any_call({
        "type": "translation_delta",
        "delta": "Hel",
    })

    websocket.send_json.assert_any_call({
        "type": "translation_delta",
        "delta": "lo",
    })


@pytest.mark.asyncio
async def test_process_stream_prefers_glossary_translation(monkeypatch):

    result, websocket = await run_process(
        monkeypatch,
        [
            {
                "event": "on_chat_model_stream",
                "metadata": {
                    "langgraph_node": "translate"
                },
                "data": {
                    "chunk": FakeChunk("Helo")
                },
            },
            {
                "event": "on_chain_end",
                "name": "glossary_check",
                "data": {
                    "output": {
                        "translation": "Hello"
                    }
                },
            },
        ],
    )

    assert result.translation == "Hello"

    websocket.send_json.assert_any_call({
        "type": "translation_delta",
        "delta": "Helo",
    })