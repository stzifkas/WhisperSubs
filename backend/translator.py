"""Translation pipeline: thin wrapper around the LangGraph translation graph.

Streams translation tokens through the WebSocket as they are generated,
then returns the final ProcessResult for SRT storage.
"""
import logging
from dataclasses import dataclass, field

from fastapi import WebSocket

from .translation_graph import SessionContext, TranslationState, translation_graph

logger = logging.getLogger(__name__)


@dataclass
class ProcessResult:
    source: str
    translation: str = ""


def make_session_context() -> SessionContext:
    return SessionContext(
        summary="",
        glossary={},
        recent_pairs=[],
        covered_up_to=0,
        target_language="",
        _updating=False,
    )


async def process_and_stream(
    raw: str,
    target_language: str,
    whisper_language: str,
    session_context: SessionContext,
    websocket: WebSocket,
) -> ProcessResult:
    """Run the refine → translate graph, streaming translation tokens to the WebSocket.

    The `transcript` message is sent to the WebSocket as soon as the refine node
    completes (before translation streaming begins), so the frontend can create its
    pending caption block before translation deltas start arriving.

    Returns ProcessResult with the final refined source and translation.
    """
    if not raw.strip():
        return ProcessResult(source=raw)

    has_context = bool(session_context.get("summary") or session_context.get("glossary"))

    # Nothing to do. skip the API call entirely
    if not target_language and not has_context:
        return ProcessResult(source=raw)

    # Handle target language change - reset context memory
    prev_lang = session_context.get("target_language", "")
    if prev_lang and prev_lang != target_language:
        session_context["summary"] = ""
        session_context["glossary"] = {}
        session_context["covered_up_to"] = 0
        session_context["recent_pairs"] = []
    session_context["target_language"] = target_language

    state: TranslationState = {
        "raw_transcript": raw,
        "target_language": target_language,
        "whisper_language": whisper_language,
        "session_context": session_context,
        "refined_source": "",
        "translation": "",
    }

    refined_source = raw
    translation_parts: list[str] = []
    transcript_sent = False

    async for event in translation_graph.astream_events(state, version="v2"):
        etype = event["event"]
        name = event.get("name", "")
        node = event.get("metadata", {}).get("langgraph_node", "")

        # When the refine node finishes, send transcript immediately so the
        # frontend creates pendingBlock before translation tokens start arriving.
        if etype == "on_chain_end" and name == "refine":
            output = event.get("data", {}).get("output", {})
            if isinstance(output, dict):
                refined_source = output.get("refined_source", raw) or raw
            if not transcript_sent:
                try:
                    await websocket.send_json({
                        "type": "transcript",
                        "text": refined_source,
                        "lang": whisper_language or "auto",
                    })
                    transcript_sent = True
                except Exception:
                    pass

        # Stream translation tokens as they arrive from the translate node.
        # Use metadata["langgraph_node"]. event["name"] is the LLM class name, not the node name.
        elif etype == "on_chat_model_stream" and node == "translate":
            chunk = event.get("data", {}).get("chunk")
            if chunk and chunk.content:
                translation_parts.append(chunk.content)
                try:
                    await websocket.send_json({
                        "type": "translation_delta",
                        "delta": chunk.content,
                    })
                except Exception:
                    pass

    translation = "".join(translation_parts).strip()

    # Fallback: if graph exited before refine_node's on_chain_end fired
    if not transcript_sent:
        try:
            await websocket.send_json({
                "type": "transcript",
                "text": refined_source,
                "lang": whisper_language or "auto",
            })
        except Exception:
            pass

    session_context["recent_pairs"].append((refined_source, translation))

    return ProcessResult(source=refined_source, translation=translation)
