import io
import wave
import struct
import math
from dataclasses import dataclass

from openai import AsyncOpenAI

from .rate_limiter import AsyncTokenBucket
from . import config


@dataclass
class TranscriptResult:
    text: str           # deduplicated clean text
    raw_text: str       # full Whisper output
    detected_lang: str


def _is_silent(wav_bytes: bytes, threshold: float = config.SILENCE_RMS_THRESHOLD) -> bool:
    """Return True if the audio RMS energy is below the threshold."""
    try:
        with wave.open(io.BytesIO(wav_bytes)) as wf:
            n_frames = wf.getnframes()
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            if sampwidth != 2:  # only handle 16-bit PCM
                return False
            raw = wf.readframes(n_frames)
        samples = struct.unpack(f"<{len(raw) // 2}h", raw)
        if not samples:
            return True
        rms = math.sqrt(sum(s * s for s in samples) / len(samples))
        return rms < threshold
    except Exception:
        return False


def _find_overlap_offset(previous_tail: str, new_text: str, min_match: int = 10) -> int:
    """Return the character index in new_text where deduplicated content starts.

    We look for the longest suffix of previous_tail that appears as a prefix of new_text.
    If no overlap found, returns 0 (use full new_text).
    """
    prev = previous_tail.strip().lower()
    new = new_text.strip().lower()

    # Try progressively shorter suffixes of previous_tail
    for length in range(min(len(prev), 80), min_match - 1, -1):
        suffix = prev[-length:]
        if new.startswith(suffix):
            # Find end of this overlap in the original (case-preserved) new_text
            return length

    return 0


class Transcriber:
    def __init__(self, client: AsyncOpenAI, rate_limiter: AsyncTokenBucket):
        self._client = client
        self._rate_limiter = rate_limiter

    async def transcribe(
        self,
        wav_bytes: bytes,
        previous_tail: str,
        silence_threshold: float = config.SILENCE_RMS_THRESHOLD,
        no_speech_threshold: float = config.NO_SPEECH_THRESHOLD,
        whisper_language: str = config.WHISPER_LANGUAGE,
    ) -> TranscriptResult | None:
        if _is_silent(wav_bytes, silence_threshold):
            return None

        await self._rate_limiter.acquire()

        is_gpt4o = config.WHISPER_MODEL.startswith("gpt-4o")

        kwargs: dict = dict(
            model=config.WHISPER_MODEL,
            file=("audio.wav", io.BytesIO(wav_bytes), "audio/wav"),
            response_format="json" if is_gpt4o else "verbose_json",
        )
        if not is_gpt4o:
            kwargs["temperature"] = 0
        if previous_tail:
            kwargs["prompt"] = previous_tail[-224:]
        if whisper_language:
            kwargs["language"] = whisper_language

        response = await self._client.audio.transcriptions.create(**kwargs)

        raw_text: str = response.text.strip()
        detected_lang: str = getattr(response, "language", "unknown")

        # no_speech_prob filtering only available with verbose_json (whisper-1)
        segments = getattr(response, "segments", None) or []
        if segments:
            avg_no_speech = sum(getattr(s, "no_speech_prob", 0) for s in segments) / len(segments)
            if avg_no_speech > no_speech_threshold:
                return None

        # Deduplicate overlap
        offset = _find_overlap_offset(previous_tail, raw_text)
        clean_text = raw_text[offset:].strip() if offset > 0 else raw_text

        return TranscriptResult(
            text=clean_text,
            raw_text=raw_text,
            detected_lang=detected_lang,
        )
