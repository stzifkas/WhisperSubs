import asyncio
import base64
import json
import logging
import time
import uuid
from pathlib import Path

import websockets
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from openai import AsyncOpenAI
from pydantic import BaseModel

from . import config
from .translator import Translator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"

app = FastAPI(title="WhisperSubs")
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

_openai_client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)
_translator = Translator(_openai_client)

# ── Session storage ───────────────────────────────────────────────────────────

# SRT store: session_id → list of (start_s, end_s, text)
_srt_sessions: dict[str, list[tuple[float, float, str]]] = {}


class ChatSession:
    def __init__(self):
        self.history: list[dict] = []
        self.summary: str = ""
        self.summary_covered_up_to: int = 0


_chat_sessions: dict[str, ChatSession] = {}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _fmt_srt_time(s: float) -> str:
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = int(s % 60)
    ms = int((s % 1) * 1000)
    return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"


def _build_srt(entries: list[tuple[float, float, str]]) -> str:
    blocks = []
    for i, (start_s, end_s, text) in enumerate(entries, 1):
        blocks.append(f"{i}\n{_fmt_srt_time(start_s)} --> {_fmt_srt_time(end_s)}\n{text}\n")
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
        f"[{_fmt_srt_time(s)}-{_fmt_srt_time(e)}] {t}" for s, e, t in chunk
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

    asyncio.create_task(_maybe_summarize(session_id))

    recent = srt[-10:]
    recent_text = "\n".join(
        f"[{_fmt_srt_time(s)}-{_fmt_srt_time(e)}] {t}" for s, e, t in recent
    ) or "(no transcript yet)"

    system_content = (
        "You are a helpful live assistant. The user is watching/listening to something "
        "and you have access to the live transcript. Answer questions concisely and helpfully.\n\n"
        + (f"Summary of earlier content:\n{chat.summary}\n\n" if chat.summary else "")
        + f"Most recent transcript:\n{recent_text}"
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
    logger.info("Client connected, session=%s", session_id)
    await websocket.send_json({"type": "session_id", "id": session_id})

    target_language = config.TARGET_LANGUAGE
    whisper_language = config.WHISPER_LANGUAGE
    vad_threshold = config.NO_SPEECH_THRESHOLD        # 0–1, higher = stricter
    silence_duration_ms = int(config.SILENCE_RMS_THRESHOLD)  # ms of silence before turn ends
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
                                except Exception:
                                    break

                        elif etype == "conversation.item.input_audio_transcription.completed":
                            transcript = event.get("transcript", "").strip()
                            if transcript:
                                end_s = time.monotonic() - session_start
                                _srt_sessions[session_id].append((last_transcript_end_s, end_s, transcript))
                                last_transcript_end_s = end_s

                                try:
                                    await websocket.send_json({
                                        "type": "transcript",
                                        "text": transcript,
                                        "lang": whisper_language or "auto",
                                    })
                                    if target_language:
                                        translation = await _translator.translate(transcript, target_language)
                                        if translation:
                                            await websocket.send_json({"type": "translation", "text": translation})
                                except Exception:
                                    break

                        elif etype == "error":
                            err = event.get("error", {}).get("message", "OpenAI Realtime error")
                            logger.error("OpenAI Realtime error: %s", event)
                            try:
                                await websocket.send_json({"type": "error", "message": err})
                            except Exception:
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
        logger.info("Client disconnected, session=%s", session_id)
