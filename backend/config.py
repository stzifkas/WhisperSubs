import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key, "
        "or export OPENAI_API_KEY in the environment."
    )
TARGET_LANGUAGE: str = os.getenv("TARGET_LANGUAGE", "")
REALTIME_MODEL: str = os.getenv("REALTIME_MODEL", "gpt-4o-realtime-preview")
WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "gpt-4o-transcribe")
WHISPER_LANGUAGE: str = os.getenv("WHISPER_LANGUAGE", "")  # e.g. "en", "el" — leave empty for auto-detect

# Server VAD turn detection (both overridable per-session from the UI)
VAD_SILENCE_DURATION_MS: int = int(os.getenv("VAD_SILENCE_DURATION_MS", "500"))  # ms of silence before a turn ends
NO_SPEECH_THRESHOLD: float = float(os.getenv("NO_SPEECH_THRESHOLD", "0.5"))  # 0.0–1.0, higher = stricter
TRANSLATION_MODEL: str = os.getenv("TRANSLATION_MODEL", "gpt-4o-mini")
CHAT_MODEL: str = os.getenv("CHAT_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# Selective refinement thresholds (span-level)
# Spans with token confidence below LOW are always repaired.
# High-risk spans (proper nouns, acronyms, numbers, dates) are repaired when
# confidence is below HIGH_RISK (a looser threshold to catch misheard names/numbers).
REFINEMENT_LOW_CONFIDENCE_THRESHOLD: float = float(os.getenv("REFINEMENT_LOW_CONFIDENCE_THRESHOLD", "0.55"))
REFINEMENT_HIGH_RISK_THRESHOLD: float = float(os.getenv("REFINEMENT_HIGH_RISK_THRESHOLD", "0.80"))

# Interpreter mode: fast | balanced | broadcast | precision
# Single named preset controlling all policy parameters — refinement thresholds,
# correction aggressiveness, and the subtitle revision budget window.
DEFAULT_INTERPRETER_MODE: str = os.getenv("DEFAULT_INTERPRETER_MODE", "balanced")

# How long to retain a disconnected session's state (SRT, chat, context, vectors)
# so downloads and post-stream chat keep working. After this, it is purged.
SESSION_TTL_S: float = float(os.getenv("SESSION_TTL_S", "3600"))
