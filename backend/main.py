import asyncio
import base64
import json
import logging
import re
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import websockets
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from openai import AsyncOpenAI
from pydantic import BaseModel

from . import config
from .confidence import confidence_level, parse_confidence
from .context_extractor import maybe_update_context
from .interpreter_modes import (
    InterpreterMode,
    SessionPolicy,
)
from .revision_policy import SubtitleState, log_decision
from .translator import ProcessResult, make_session_context, process_and_stream
from .translation_graph import SessionContext
from .vector_store import add_segment, remove_store, search_segments

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"

SWEEP_INTERVAL_S = 60  # how often the sweeper checks for expired sessions

# Fire-and-forget background tasks. asyncio keeps only a weak reference to a
# task, so a bare `create_task(...)` whose result is discarded can be garbage
# collected mid-execution. Retain strong references here until they finish.
_background_tasks: set[asyncio.Task] = set()


def _spawn(coro) -> asyncio.Task:
    """Schedule a background task and keep a strong reference until it finishes."""
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return task


@asynccontextmanager
async def lifespan(_app: FastAPI):
    sweeper = asyncio.create_task(_session_sweeper())
    try:
        yield
    finally:
        sweeper.cancel()
        try:
            await sweeper
        except asyncio.CancelledError:
            pass


app = FastAPI(title="WhisperSubs", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

_openai_client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)

# ── Session storage ───────────────────────────────────────────────────────────

class SRTSegment:
    __slots__ = ("start_s", "end_s", "text", "confidence", "state")

    def __init__(
        self,
        start_s: float,
        end_s: float,
        text: str,
        confidence: float | None = None,
        state: SubtitleState = SubtitleState.TENTATIVE,
    ):
        self.start_s = start_s
        self.end_s = end_s
        self.text = text
        self.confidence = confidence  # geometric mean token probability; None = unknown
        self.state = state


# SRT store: session_id → list of SRTSegment
_srt_sessions: dict[str, list[SRTSegment]] = {}

# Translation context per session
_session_contexts: dict[str, SessionContext] = {}

# Session interpreter policies (mode profile + revision tracker)
_session_policies: dict[str, SessionPolicy] = {}


class ChatSession:
    def __init__(self):
        self.history: list[dict] = []
        self.summary: str = ""
        self.summary_covered_up_to: int = 0


_chat_sessions: dict[str, ChatSession] = {}

# Disconnected sessions are retained until this monotonic deadline, then purged.
# Keeping them alive lets SRT download and chat keep working after a stream ends.
_session_expiry: dict[str, float] = {}


def _purge_session(session_id: str) -> None:
    """Drop all state for one session in a single place."""
    _srt_sessions.pop(session_id, None)
    _chat_sessions.pop(session_id, None)
    _session_contexts.pop(session_id, None)
    _session_policies.pop(session_id, None)
    _session_expiry.pop(session_id, None)
    remove_store(session_id)


async def _session_sweeper() -> None:
    """Purge sessions whose retention window has elapsed."""
    while True:
        await asyncio.sleep(SWEEP_INTERVAL_S)
        now = time.monotonic()
        expired = [sid for sid, deadline in _session_expiry.items() if now >= deadline]
        for sid in expired:
            _purge_session(sid)
            logger.info("Session %s expired and purged", sid)


# ── Helpers ───────────────────────────────────────────────────────────────────

_GARBAGE_RE = re.compile(r'^[\s#\-_=\.\,\!\?\*\~\[\]\(\)]+$')

def _is_garbage(text: str) -> bool:
    """Filter Whisper hallucinations: '###', '...', lone punctuation, etc."""
    stripped = text.strip()
    return not stripped or bool(_GARBAGE_RE.match(stripped)) or len(stripped) < 2


def _fmt_srt_time(s: float) -> str:
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = int(s % 60)
    ms = int((s % 1) * 1000)
    return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"


def _build_srt(entries: list[SRTSegment]) -> str:
    blocks = []
    for i, seg in enumerate(entries, 1):
        blocks.append(f"{i}\n{_fmt_srt_time(seg.start_s)} --> {_fmt_srt_time(seg.end_s)}\n{seg.text}\n")
    return "\n".join(blocks)


async def _maybe_summarize(session_id: str) -> None:
    srt = _srt_sessions.get(session_id, [])
    chat = _chat_sessions.get(session_id)
    if chat is None:
        return

    KEEP_RECENT = 10
    SUMMARIZE_THRESHOLD = 20

    if len(srt) - chat.summary_covered_up_to < SUMMARIZE_THRESHOLD + KEEP_RECENT:
        return

    compress_up_to = len(srt) - KEEP_RECENT
    chunk = srt[chat.summary_covered_up_to:compress_up_to]
    text_block = "\n".join(
        f"[{_fmt_srt_time(seg.start_s)}-{_fmt_srt_time(seg.end_s)}] {seg.text}" for seg in chunk
    )
    prompt = (
        "You are maintaining a rolling notes scratchpad for an ongoing live transcription.\n"
        "Merge the previous summary with the new transcript entries into a single concise summary.\n"
        "Preserve key topics, names, facts, decisions, and anything a viewer might want to ask about.\n\n"
        + (f"Previous summary:\n{chat.summary}\n\n" if chat.summary else "")
        + f"New transcript entries:\n{text_block}"
    )
    try:
        resp = await _openai_client.chat.completions.create(
            model=config.CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
            temperature=0.3,
        )
        chat.summary = resp.choices[0].message.content.strip()
        chat.summary_covered_up_to = compress_up_to
        logger.info("Summary updated for session %s (covered %d entries)", session_id, compress_up_to)
    except Exception as exc:
        logger.warning("Summarization failed for session %s: %s", session_id, exc)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/srt/{session_id}")
async def download_srt(session_id: str):
    entries = _srt_sessions.get(session_id)
    if not entries:
        return PlainTextResponse("Session not found or empty", status_code=404)
    return PlainTextResponse(
        _build_srt(entries),
        headers={"Content-Disposition": f'attachment; filename="session_{session_id[:8]}.srt"'},
    )


class ChatRequest(BaseModel):
    message: str


@app.post("/chat/{session_id}")
async def chat_endpoint(session_id: str, req: ChatRequest):
    srt = _srt_sessions.get(session_id)
    chat = _chat_sessions.get(session_id)
    if srt is None or chat is None:
        raise HTTPException(status_code=404, detail="Session not found")

    _spawn(_maybe_summarize(session_id))

    relevant = await search_segments(session_id, req.message, k=6)
    recent_text = "\n".join(relevant) or "(no transcript yet)"

    tx_ctx = _session_contexts.get(session_id)
    system_content = (
        "You are a helpful live assistant. The user is watching/listening to something "
        "and you have access to the live transcript. Answer questions concisely and helpfully.\n\n"
        + (f"Summary of earlier content:\n{chat.summary}\n\n" if chat.summary else "")
        + (f"Topic/translation context:\n{tx_ctx['summary']}\n\n" if tx_ctx and tx_ctx.get("summary") else "")
        + f"Most relevant transcript segments:\n{recent_text}"
    )

    chat.history.append({"role": "user", "content": req.message})
    messages = [{"role": "system", "content": system_content}] + chat.history[-20:]

    try:
        resp = await _openai_client.chat.completions.create(
            model=config.CHAT_MODEL,
            messages=messages,
            max_tokens=512,
            temperature=0.7,
        )
        reply = resp.choices[0].message.content.strip()
    except Exception as exc:
        logger.error("Chat error for session %s: %s", session_id, exc)
        raise HTTPException(status_code=500, detail=str(exc))

    chat.history.append({"role": "assistant", "content": reply})
    return {"reply": reply}


@app.get("/")
async def index():
    return FileResponse(str(FRONTEND_DIR / "index.html"))


# ── WebSocket endpoint ────────────────────────────────────────────────────────

@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    session_id = str(uuid.uuid4())
    _srt_sessions[session_id] = []
    _chat_sessions[session_id] = ChatSession()
    _session_contexts[session_id] = make_session_context()
    _session_policies[session_id] = SessionPolicy.create(
        InterpreterMode(config.DEFAULT_INTERPRETER_MODE)
    )
    logger.info("Client connected, session=%s", session_id)
    await websocket.send_json({"type": "session_id", "id": session_id})

    target_language = config.TARGET_LANGUAGE
    whisper_language = config.WHISPER_LANGUAGE
    vad_threshold = config.NO_SPEECH_THRESHOLD        # 0–1, higher = stricter
    silence_duration_ms = config.VAD_SILENCE_DURATION_MS  # ms of silence before turn ends
    session_start = time.monotonic()
    last_transcript_end_s = 0.0

    realtime_url = f"wss://api.openai.com/v1/realtime?model={config.REALTIME_MODEL}"

    try:
        async with websockets.connect(
            realtime_url,
            additional_headers={
                "Authorization": f"Bearer {config.OPENAI_API_KEY}",
                "OpenAI-Beta": "realtime=v1",
            },
        ) as openai_ws:

            async def send_session_update():
                cfg: dict = {
                    "input_audio_format": "pcm16",
                    "input_audio_transcription": {"model": config.WHISPER_MODEL},
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": vad_threshold,
                        "prefix_padding_ms": 300,
                        "silence_duration_ms": max(200, silence_duration_ms),
                    },
                    "include": ["item.input_audio_transcription.logprobs"],
                }
                if whisper_language:
                    cfg["input_audio_transcription"]["language"] = whisper_language
                await openai_ws.send(json.dumps({"type": "session.update", "session": cfg}))

            await send_session_update()

            async def receiver():
                nonlocal target_language, whisper_language, vad_threshold, silence_duration_ms
                try:
                    while True:
                        message = await websocket.receive()

                        if "bytes" in message and message["bytes"]:
                            audio_b64 = base64.b64encode(message["bytes"]).decode()
                            try:
                                await openai_ws.send(json.dumps({
                                    "type": "input_audio_buffer.append",
                                    "audio": audio_b64,
                                }))
                            except websockets.exceptions.ConnectionClosed:
                                break

                        elif "text" in message and message["text"]:
                            try:
                                data = json.loads(message["text"])
                                if data.get("type") == "config":
                                    target_language = data.get("target_language", target_language)
                                    if "whisper_language" in data:
                                        whisper_language = data["whisper_language"]
                                    if "no_speech_threshold" in data:
                                        vad_threshold = float(data["no_speech_threshold"])
                                    if "silence_threshold" in data:
                                        silence_duration_ms = int(float(data["silence_threshold"]))
                                    try:
                                        await send_session_update()
                                    except websockets.exceptions.ConnectionClosed:
                                        break
                                    logger.info(
                                        "Config: lang=%r src=%r vad=%.2f silence_ms=%d",
                                        target_language, whisper_language, vad_threshold, silence_duration_ms,
                                    )
                                elif data.get("type") == "mode":
                                    raw_mode = data.get("mode", "")
                                    try:
                                        new_mode = InterpreterMode(raw_mode)
                                        policy = _session_policies.get(session_id)
                                        if policy:
                                            policy.switch_to(new_mode)
                                            logger.info(
                                                "Mode switched to %r for session=%s",
                                                new_mode.value, session_id,
                                            )
                                    except ValueError:
                                        logger.warning("Unknown interpreter mode: %r", raw_mode)
                            except json.JSONDecodeError:
                                pass

                except WebSocketDisconnect:
                    pass
                finally:
                    await openai_ws.close()

            async def openai_listener():
                nonlocal last_transcript_end_s
                try:
                    async for raw in openai_ws:
                        event = json.loads(raw)
                        etype = event.get("type", "")

                        if etype == "conversation.item.input_audio_transcription.delta":
                            delta = event.get("delta", "")
                            if delta:
                                try:
                                    await websocket.send_json({"type": "transcript_delta", "delta": delta})
                                except (WebSocketDisconnect, websockets.exceptions.ConnectionClosed):
                                    break

                        elif etype == "conversation.item.input_audio_transcription.completed":
                            transcript = event.get("transcript", "").strip()
                            if transcript and not _is_garbage(transcript):
                                confidence = parse_confidence(event.get("logprobs"))
                                end_s = time.monotonic() - session_start
                                seg_start_s = last_transcript_end_s
                                srt = _srt_sessions[session_id]
                                srt.append(SRTSegment(seg_start_s, end_s, transcript, confidence))
                                segment_index = len(srt) - 1
                                last_transcript_end_s = end_s
                                logger.info(
                                    "Transcript seg=%d confidence=%.3f (%s): %r",
                                    segment_index,
                                    confidence if confidence is not None else float("nan"),
                                    confidence_level(confidence),
                                    transcript[:60],
                                )

                                # ── Revision policy + auto-lock notifications ──
                                policy = _session_policies.get(session_id)
                                if policy:
                                    newly_locked = policy.tracker.register(segment_index)
                                    # Notify frontend of any segments that were just
                                    # locked out of the revision window.
                                    for locked_idx in newly_locked:
                                        try:
                                            await websocket.send_json({
                                                "type": "subtitle_lock",
                                                "id": locked_idx,
                                            })
                                        except (WebSocketDisconnect, websockets.exceptions.ConnectionClosed):
                                            break
                                    decision = policy.tracker.decide_revision(segment_index)
                                    log_decision(decision, policy.tracker.mode, session_id)
                                else:
                                    decision = None

                                try:
                                    ctx = _session_contexts.get(session_id)

                                    if decision is None or decision.allowed:
                                        # ── Revision allowed: show tentative, then refine + translate ──
                                        await websocket.send_json({
                                            "type": "subtitle_tentative",
                                            "id": segment_index,
                                            "text": transcript,
                                            "lang": whisper_language or "auto",
                                        })

                                        profile = policy.profile if policy else None
                                        result = await process_and_stream(
                                            transcript, target_language, whisper_language,
                                            ctx, websocket,
                                            segment_id=segment_index,
                                            confidence=confidence,
                                            logprobs=event.get("logprobs"),
                                            refinement_low_threshold=(
                                                profile.refinement_low_threshold if profile
                                                else config.REFINEMENT_LOW_CONFIDENCE_THRESHOLD
                                            ),
                                            refinement_high_risk_threshold=(
                                                profile.refinement_high_risk_threshold if profile
                                                else config.REFINEMENT_HIGH_RISK_THRESHOLD
                                            ),
                                            correction_aggressiveness=(
                                                profile.correction_aggressiveness if profile
                                                else "medium"
                                            ),
                                        )
                                        final_state = SubtitleState.STABLE
                                        if policy:
                                            policy.tracker.transition(segment_index, SubtitleState.STABLE)

                                        # Commit: finalizes text + translation in the frontend block.
                                        await websocket.send_json({
                                            "type": "subtitle_commit",
                                            "id": segment_index,
                                            "text": result.source,
                                            "translation": result.translation or "",
                                            "lang": whisper_language or "auto",
                                        })

                                    else:
                                        # ── Revision blocked: commit immediately as locked ──
                                        result = ProcessResult(source=transcript)
                                        final_state = SubtitleState.LOCKED
                                        if policy:
                                            policy.tracker.transition(segment_index, SubtitleState.LOCKED)
                                        await websocket.send_json({
                                            "type": "subtitle_commit",
                                            "id": segment_index,
                                            "text": transcript,
                                            "translation": "",
                                            "lang": whisper_language or "auto",
                                            "locked": True,
                                        })

                                    # Update SRT with final text + state
                                    srt[-1].text = result.source
                                    srt[-1].state = final_state

                                    # Background: context memory + vector store
                                    if ctx:
                                        _spawn(maybe_update_context(ctx))
                                    _spawn(add_segment(
                                        session_id,
                                        result.source,
                                        {"start_s": seg_start_s, "end_s": end_s,
                                         "confidence": confidence, "state": final_state.value},
                                    ))

                                except (WebSocketDisconnect, websockets.exceptions.ConnectionClosed):
                                    break

                        elif etype == "error":
                            err = event.get("error", {}).get("message", "OpenAI Realtime error")
                            logger.error("OpenAI Realtime error: %s", event)
                            try:
                                await websocket.send_json({"type": "error", "message": err})
                            except (WebSocketDisconnect, websockets.exceptions.ConnectionClosed):
                                pass

                except websockets.exceptions.ConnectionClosed:
                    pass
                except Exception as e:
                    logger.error("OpenAI listener error: %s", e)

            await asyncio.gather(receiver(), openai_listener())

    except Exception as exc:
        logger.error("Session error: %s", exc)
        try:
            await websocket.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass
    finally:
        # Retain session state for a while so SRT download and chat keep working;
        # the sweeper purges it (vector store included) once the window elapses.
        _session_expiry[session_id] = time.monotonic() + config.SESSION_TTL_S
        logger.info(
            "Client disconnected, session=%s (retained %ds)",
            session_id, int(config.SESSION_TTL_S),
        )
