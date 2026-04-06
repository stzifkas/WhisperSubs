import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY: str = os.environ["OPENAI_API_KEY"]
TARGET_LANGUAGE: str = os.getenv("TARGET_LANGUAGE", "")
CHUNK_DURATION_S: int = int(os.getenv("CHUNK_DURATION_S", "5"))
CHUNK_OVERLAP_S: int = int(os.getenv("CHUNK_OVERLAP_S", "1"))
REALTIME_MODEL: str = os.getenv("REALTIME_MODEL", "gpt-4o-realtime-preview")
WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "gpt-4o-transcribe")
WHISPER_LANGUAGE: str = os.getenv("WHISPER_LANGUAGE", "")  # e.g. "en", "el" — leave empty for auto-detect
SILENCE_RMS_THRESHOLD: float = float(os.getenv("SILENCE_RMS_THRESHOLD", "200"))  # 0–32767 scale
NO_SPEECH_THRESHOLD: float = float(os.getenv("NO_SPEECH_THRESHOLD", "0.6"))  # 0.0–1.0
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
# Controls all policy parameters (refinement thresholds, correction aggressiveness,
# revision budget) as a single named preset.
DEFAULT_INTERPRETER_MODE: str = os.getenv("DEFAULT_INTERPRETER_MODE", "balanced")

# Subtitle revision policy
# Mode controls the default budget window: strict | balanced | relaxed
REVISION_MODE: str = os.getenv("REVISION_MODE", "balanced")
# Optional overrides (0 = use mode default)
_REVISION_MAX_AGE_S  = float(os.getenv("REVISION_MAX_AGE_S", "0"))
_REVISION_MAX_SEGS   = int(os.getenv("REVISION_MAX_SEGMENTS_BACK", "0"))
REVISION_MAX_AGE_S: float | None          = _REVISION_MAX_AGE_S  or None
REVISION_MAX_SEGMENTS_BACK: int | None    = _REVISION_MAX_SEGS   or None
MAX_QUEUE_DEPTH: int = int(os.getenv("MAX_QUEUE_DEPTH", "3"))
WHISPER_RPM: int = int(os.getenv("WHISPER_RPM", "50"))
