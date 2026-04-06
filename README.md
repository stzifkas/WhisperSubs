# WhisperSubs

Real-time speech transcription and translation in your browser, powered by OpenAI's Realtime API.

Stream audio from any browser tab or microphone and get live subtitles with sub-second latency — with optional word-by-word streaming translation into 30+ languages, SRT export, and a floating subtitle overlay you can drag over any video player.

![WhisperSubs screenshot](docs/screenshot.png)

---

## Features

- **Real-time transcription** — word-by-word streaming via OpenAI Realtime API + `gpt-4o-transcribe`
- **Streaming translation** — translation tokens appear word-by-word as they are generated (LangGraph pipeline)
- **Context-aware refinement** — rolling topic summary and glossary improve transcription accuracy over time
- **Semantic chat** — ask questions about what's being said; answers are retrieved by semantic similarity across the entire session, not just the last few lines
- **Floating subtitle overlay** — detachable popup window you can position over any video on your desktop
- **SRT export** — download every session as a timestamped `.srt` subtitle file
- **Source language selection** — lock to a specific language or let the model auto-detect
- **VAD controls** — tune speech sensitivity and silence duration via the advanced panel
- **Tab or mic capture** — capture any browser tab's audio or your microphone

---

## Requirements

- An [OpenAI API key](https://platform.openai.com/api-keys) with access to the Realtime API
- Docker (recommended) **or** Python 3.12+
- A modern Chromium-based browser (for tab audio capture via `getDisplayMedia`)

---

## Quick start (Docker)

```bash
git clone https://github.com/stzifkas/whispersubs.git
cd whispersubs

cp .env.example .env
# edit .env and set OPENAI_API_KEY

docker compose up --build
```

Open `http://localhost:8000` in your browser.

To run in the background:

```bash
docker compose up -d --build
docker compose logs -f   # follow logs
docker compose down      # stop
```

---

## Manual setup (without Docker)

```bash
git clone https://github.com/stzifkas/whispersubs.git
cd whispersubs

uv venv .venv && source .venv/bin/activate   # or: python3 -m venv .venv
uv pip install -r requirements.txt           # or: pip install -r requirements.txt

cp .env.example .env
# edit .env and set OPENAI_API_KEY

uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## Usage

1. Select **Source** (Tab audio or Microphone) and optionally pick a **Spoken language**
2. Pick a **Translate to** language if you want live translation
3. Click **Start** — for tab audio, select a tab in Chrome's dialog and check *Share tab audio*
4. Captions stream in real time. Click **⧉** for the floating subtitle overlay
5. Click **💬** to open the live chat panel and ask questions about the content
6. Click **↓ SRT** to download the session transcript as a subtitle file

---

## Configuration

All settings have sensible defaults. Override via `.env`:

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | **Required** |
| `REALTIME_MODEL` | `gpt-4o-realtime-preview` | Realtime API transcription model |
| `WHISPER_MODEL` | `gpt-4o-transcribe` | Transcription engine |
| `TRANSLATION_MODEL` | `gpt-4o-mini` | Refinement + translation model |
| `CHAT_MODEL` | `gpt-4o-mini` | Live chat model |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embeddings for semantic chat retrieval |
| `WHISPER_LANGUAGE` | *(auto)* | Lock source language (e.g. `en`, `es`) |
| `TARGET_LANGUAGE` | *(none)* | Default translation target |
| `NO_SPEECH_THRESHOLD` | `0.6` | VAD speech sensitivity (0–1) |
| `SILENCE_RMS_THRESHOLD` | `200` | Silence duration before turn ends (ms) |

---

## Architecture

```
Browser
  └─ MediaStream (tab / mic) at 24 kHz
       └─ ScriptProcessor → raw PCM16 binary frames
            └─ WebSocket /ws (FastAPI)
                 ├─ WebSocket → OpenAI Realtime API
                 │    └─ transcript deltas / completed events
                 ├─ LangGraph pipeline (per completed turn)
                 │    ├─ refine_node   — fix transcription errors (function_calling)
                 │    └─ translate_node — stream tokens word-by-word to browser
                 ├─ Background: context extractor (rolling summary + glossary)
                 ├─ Background: vector store (embeddings for semantic chat)
                 ├─ In-memory SRT store (per session)
                 └─ Chat endpoint (semantic retrieval + rolling summary + GPT)
```

---

## License

MIT
