"""Policy-based span classification for selective transcript refinement.

Splits a transcript into word-level TextSpans and marks each as protected
or repairable based on token-level confidence and high-risk content categories.

Only repairable spans are sent to the refinement LLM; protected spans pass
through unchanged, saving both latency and cost.
"""
import math
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any

# ── High-risk category constants ──────────────────────────────────────────────

CATEGORY_PROPER_NOUN = "proper_noun"
CATEGORY_ACRONYM = "acronym"
CATEGORY_NUMBER = "number"
CATEGORY_MEASUREMENT = "measurement"
CATEGORY_DATE = "date"

# ── Regex patterns for high-risk categories ───────────────────────────────────

_ACRONYM_RE = re.compile(r'^[A-Z]{2,}s?$')

_NUMBER_RE = re.compile(r'^[+\-]?\d[\d,. ]*%?$')

_MEASUREMENT_RE = re.compile(
    r'^\d+(?:[.,]\d+)?\s*'
    r'(?:km|m|cm|mm|kg|g|mg|µg|l|ml|s|ms|min|h|°C|°F|°|%|Hz|kHz|MHz|GHz|V|W|kW|MW)$',
    re.IGNORECASE,
)

_DATE_NUMERIC_RE = re.compile(
    r'^\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}$'   # 12/31/2024 or 31-12-2024
    r'|^\d{4}-\d{2}-\d{2}$'                   # 2024-01-31
)

_MONTH_RE = re.compile(
    r'^(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?'
    r'|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)$',
    re.IGNORECASE,
)

# Words that are capitalized at sentence start but are not proper nouns
_FUNCTION_WORDS: frozenset[str] = frozenset({
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'must', 'shall', 'not', 'no', 'yes', 'this',
    'that', 'these', 'those', 'it', 'its', 'he', 'she', 'we', 'they',
    'i', 'you', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his',
    'our', 'their', 'what', 'which', 'who', 'how', 'when', 'where', 'why',
    'all', 'each', 'every', 'some', 'any', 'also', 'so', 'if', 'as',
    'just', 'then', 'than', 'about', 'into', 'after', 'before', 'now',
    'up', 'down', 'out', 'there', 'here', 'because', 'while', 'through',
    'more', 'very', 'even', 'still', 'only', 'well', 'back', 'off', 'both',
    'such', 'too', 'over', 'under', 'again', 'between', 'against', 'during',
    'without', 'within', 'along', 'following', 'across', 'behind', 'beyond',
    'plus', 'except', 'until', 'around', 'however', 'therefore', 'thus',
    'although', 'since', 'unless', 'whether', 'whenever', 'wherever',
})

# Sentence-ending punctuation — a capital word after one of these is sentence-initial
_SENTENCE_ENDS = frozenset({'.', '?', '!', '\n'})


# ── Core data type ────────────────────────────────────────────────────────────

@dataclass
class TextSpan:
    """A single whitespace-delimited word in the transcript, with metadata."""

    text: str                                      # original word text (may include punctuation)
    start: int                                     # char offset in transcript
    end: int                                       # char offset in transcript
    token_confidence: float | None                 # mean token probability; None = unknown
    categories: frozenset[str] = field(default_factory=frozenset)

    @property
    def is_high_risk(self) -> bool:
        return bool(self.categories)

    def should_refine(self, low_threshold: float, high_risk_threshold: float) -> bool:
        """True if this span should be sent to the refinement LLM.

        Decision rules (in order):
        1. Low confidence  → always repair
        2. High-risk content at sub-threshold confidence → repair
        3. High-risk content with no confidence info → repair (conservative)
        4. Everything else → protect
        """
        if self.token_confidence is not None:
            if self.token_confidence < low_threshold:
                return True
            if self.is_high_risk and self.token_confidence < high_risk_threshold:
                return True
            return False
        # No confidence data available — repair high-risk spans only
        return self.is_high_risk


# ── Category detection ────────────────────────────────────────────────────────

def _categorize_word(word: str, char_start: int, full_text: str) -> frozenset[str]:
    """Return the set of high-risk categories that apply to this word."""
    clean = word.strip(".,!?;:\"'()[]{}-")
    if not clean:
        return frozenset()

    # Check patterns in priority order (most specific first)
    if _ACRONYM_RE.match(clean):
        return frozenset({CATEGORY_ACRONYM})

    if _MEASUREMENT_RE.match(clean):
        return frozenset({CATEGORY_MEASUREMENT})

    if _DATE_NUMERIC_RE.match(clean) or _MONTH_RE.match(clean):
        return frozenset({CATEGORY_DATE})

    if _NUMBER_RE.match(clean):
        return frozenset({CATEGORY_NUMBER})

    # Proper noun heuristic: starts uppercase, not a function word, not sentence-initial
    if (
        len(clean) >= 2
        and clean[0].isupper()
        and clean.lower() not in _FUNCTION_WORDS
        and char_start > 0
        and full_text[char_start - 1] not in _SENTENCE_ENDS
    ):
        return frozenset({CATEGORY_PROPER_NOUN})

    return frozenset()


# ── Token → word confidence alignment ────────────────────────────────────────

def token_word_confidences(
    logprobs: list[dict[str, Any]],
) -> list[tuple[str, float | None]]:
    """Group logprob tokens into word-level (word_text, confidence) pairs.

    Whisper tokens that begin a new word start with a space character.
    Tokens within the same word are concatenated and their logprobs averaged.
    """
    if not logprobs:
        return []

    groups: list[list[dict]] = []
    current: list[dict] = []
    for entry in logprobs:
        tok = entry.get("token", "")
        if tok.startswith(" ") and current:
            groups.append(current)
            current = []
        current.append(entry)
    if current:
        groups.append(current)

    result: list[tuple[str, float | None]] = []
    for group in groups:
        word = "".join(e.get("token", "") for e in group).strip()
        lps = [e["logprob"] for e in group if isinstance(e.get("logprob"), (int, float))]
        conf = round(math.exp(sum(lps) / len(lps)), 4) if lps else None
        if word:
            result.append((word, conf))
    return result


def _align_confidences(
    transcript_words: list[str],
    token_words: list[tuple[str, float | None]],
) -> list[float | None]:
    """Align token-group confidences to transcript words via sequence matching.

    Strips punctuation and lowercases before comparison so minor normalization
    differences don't break alignment. Unmatched positions get None.
    """
    if not token_words:
        return [None] * len(transcript_words)

    def _norm(w: str) -> str:
        return re.sub(r'[^\w]', '', w).lower()

    tw_norm = [_norm(w) for w in transcript_words]
    tok_norm = [_norm(w) for w, _ in token_words]

    confidences: list[float | None] = [None] * len(transcript_words)
    sm = SequenceMatcher(None, tok_norm, tw_norm, autojunk=False)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == 'equal':
            for ti, wi in zip(range(i1, i2), range(j1, j2)):
                confidences[wi] = token_words[ti][1]
    return confidences


# ── Public API ────────────────────────────────────────────────────────────────

def classify_spans(
    transcript: str,
    logprobs: list[dict[str, Any]] | None,
) -> list[TextSpan]:
    """Return word-level TextSpans annotated with confidence and risk categories.

    When logprobs is None, confidence is set to None for all spans and only
    high-risk categories drive the repair decision.
    """
    word_matches = list(re.finditer(r'\S+', transcript))
    if not word_matches:
        return []

    if logprobs:
        tok_words = token_word_confidences(logprobs)
        confs = _align_confidences([m.group() for m in word_matches], tok_words)
    else:
        confs = [None] * len(word_matches)

    spans: list[TextSpan] = []
    for match, conf in zip(word_matches, confs):
        word, start, end = match.group(), match.start(), match.end()
        cats = _categorize_word(word, start, transcript)
        spans.append(TextSpan(
            text=word,
            start=start,
            end=end,
            token_confidence=conf,
            categories=cats,
        ))
    return spans


def annotate_transcript(
    transcript: str,
    spans: list[TextSpan],
    low_threshold: float,
    high_risk_threshold: float,
) -> str | None:
    """Wrap repairable spans with [REPAIR: ...] markers.

    Consecutive repairable spans are merged into a single marker so the LLM
    sees them in context. Returns None when nothing needs repair — the caller
    should skip the LLM call entirely.
    """
    repairable = [s for s in spans if s.should_refine(low_threshold, high_risk_threshold)]
    if not repairable:
        return None

    # Merge consecutive repairable spans (only whitespace between them)
    groups: list[list[TextSpan]] = []
    current_group: list[TextSpan] = [repairable[0]]
    for span in repairable[1:]:
        between = transcript[current_group[-1].end:span.start]
        if between.strip() == "":
            current_group.append(span)
        else:
            groups.append(current_group)
            current_group = [span]
    groups.append(current_group)

    parts: list[str] = []
    prev = 0
    for group in groups:
        g_start, g_end = group[0].start, group[-1].end
        parts.append(transcript[prev:g_start])
        parts.append(f"[REPAIR: {transcript[g_start:g_end]}]")
        prev = g_end
    parts.append(transcript[prev:])
    return "".join(parts)
