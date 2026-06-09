"""Translation pipeline: thin wrapper around the LangGraph translation graph.

Streams translation tokens through the WebSocket as they are generated,
then returns the final ProcessResult for SRT storage.

WebSocket events emitted by this module:
  subtitle_update   {id, text, lang}   — refined source text (after refine_node)
  translation_delta {delta}            — streaming translation token
"""
import logging
from dataclasses import dataclass

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
    segment_id: int = 0,
    confidence: float | None = None,
    logprobs: list[dict] | None = None,
    refinement_low_threshold: float = 0.55,
    refinement_high_risk_threshold: float = 0.80,
    correction_aggressiveness: str = "medium",
) -> ProcessResult:
    """Run the refine → translate → glossary_check graph, streaming to the WebSocket.

    Emits:
    - ``subtitle_update`` as soon as the refine node completes, so the frontend
      can update the tentative caption block before translation tokens arrive.
    - ``translation_delta`` for each streaming token from the translate node.

    The caller (main.py) is responsible for sending ``subtitle_tentative`` before
    calling this function and ``subtitle_commit`` after it returns.

    Returns ProcessResult with the final refined source and (post-repair) translation.
    """
    if not raw.strip():
        return ProcessResult(source=raw)

    has_context = bool(session_context.get("summary") or session_context.get("glossary"))

    # Nothing to do — skip the LLM entirely; emit subtitle_update so the frontend
    # can apply the raw text to the tentative block.
    if not target_language and not has_context:
        try:
            await websocket.send_json({
                "type": "subtitle_update",
                "id": segment_id,
                "text": raw,
                "lang": whisper_language or "auto",
            })
        except Exception:
            pass
        return ProcessResult(source=raw)

    # Handle target language change — reset context memory
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
        "confidence": confidence,
        "logprobs": logprobs,
        "refinement_low_threshold": refinement_low_threshold,
        "refinement_high_risk_threshold": refinement_high_risk_threshold,
        "correction_aggressiveness": correction_aggressiveness,
        "refined_source": "",
        "translation": "",
        "glossary_violations": [],
    }

    refined_source = raw
    translation_parts: list[str] = []
    update_sent = False
    # Authoritative translation from glossary_check_node (may be repaired).
    glossary_checked_translation: str | None = None

    async for event in translation_graph.astream_events(state, version="v2"):
        etype = event["event"]
        name = event.get("name", "")
        node = event.get("metadata", {}).get("langgraph_node", "")

        # refine_node finished → send subtitle_update so the frontend can
        # replace the tentative raw text with the refined source immediately,
        # before translation streaming begins.
        if etype == "on_chain_end" and name == "refine":
            output = event.get("data", {}).get("output", {})
            if isinstance(output, dict):
                refined_source = output.get("refined_source", raw) or raw
            if not update_sent:
                try:
                    await websocket.send_json({
                        "type": "subtitle_update",
                        "id": segment_id,
                        "text": refined_source,
                        "lang": whisper_language or "auto",
                    })
                    update_sent = True
                except Exception:
                    pass

        # translate_node streaming — forward each token.
        # Use metadata["langgraph_node"]; event["name"] is the LLM class name.
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

        # glossary_check_node finished — capture authoritative (possibly repaired) translation.
        elif etype == "on_chain_end" and name == "glossary_check":
            output = event.get("data", {}).get("output", {})
            if isinstance(output, dict):
                glossary_checked_translation = output.get("translation", "")

    # Prefer the post-repair translation; fall back to streamed tokens if the
    # glossary_check node didn't run (i.e. no target language was set).
    translation = (
        glossary_checked_translation
        if glossary_checked_translation is not None
        else "".join(translation_parts).strip()
    )

    # Fallback: if the graph exited before refine_node's on_chain_end fired.
    if not update_sent:
        try:
            await websocket.send_json({
                "type": "subtitle_update",
                "id": segment_id,
                "text": refined_source,
                "lang": whisper_language or "auto",
            })
        except Exception:
            pass

    session_context["recent_pairs"].append((refined_source, translation))

    return ProcessResult(source=refined_source, translation=translation)
