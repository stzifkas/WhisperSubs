"""Confidence scoring from OpenAI Realtime API logprobs.

Logprobs are optional — all callers must handle a None return gracefully.
"""
import math
from typing import Any


def parse_confidence(logprobs: list[dict[str, Any]] | None) -> float | None:
    """Convert a list of token logprob dicts to a [0, 1] confidence score.

    Uses the geometric mean of token probabilities: exp(mean(logprob values)).
    A score of 1.0 means the model was certain about every token; near 0 means
    very low confidence.

    Returns None when logprobs are absent or contain no usable values — callers
    should treat None as "confidence unknown" and apply the same logic as medium.
    """
    if not logprobs:
        return None
    values = [
        entry["logprob"]
        for entry in logprobs
        if isinstance(entry.get("logprob"), (int, float))
    ]
    if not values:
        return None
    mean_logprob = sum(values) / len(values)
    return round(math.exp(mean_logprob), 4)


def confidence_level(confidence: float | None) -> str:
    """Map a [0, 1] score (or None) to a named tier.

    Tiers:
      high    >= 0.85  — model was sure; minimal correction expected
      medium  >= 0.55  — normal uncertainty; standard refinement
      low      < 0.55  — significant uncertainty; aggressive correction needed
      unknown   None   — logprobs unavailable; treat as medium
    """
    if confidence is None:
        return "unknown"
    if confidence >= 0.85:
        return "high"
    if confidence >= 0.55:
        return "medium"
    return "low"
