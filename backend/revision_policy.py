"""Subtitle revision policy engine.

Manages bounded subtitle instability by assigning each segment a lifecycle
state and deciding whether a revision is still within the allowed budget.

State machine
─────────────

  TENTATIVE ──► STABLE ──► LOCKED
      │                       ▲
      └───────────────────────┘
          (budget exceeded or auto-lock)

  • TENTATIVE  — segment just received from Realtime API; not yet refined.
  • STABLE     — refinement applied within the revision window.
  • LOCKED     — outside the revision budget; text is frozen permanently.

Revision is allowed iff:
  state ∈ {TENTATIVE, STABLE}
  AND age_s ≤ budget.max_age_s
  AND distance ≤ budget.max_segments_back

Mode-specific defaults
──────────────────────

  strict    max_age_s=5   max_segments_back=1
  balanced  max_age_s=15  max_segments_back=3   (default)
  relaxed   max_age_s=60  max_segments_back=10
"""
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

logger = logging.getLogger(__name__)


# ── Public enums ──────────────────────────────────────────────────────────────

class SubtitleState(str, Enum):
    TENTATIVE = "tentative"
    STABLE    = "stable"
    LOCKED    = "locked"


class RevisionMode(str, Enum):
    STRICT   = "strict"
    BALANCED = "balanced"
    RELAXED  = "relaxed"


# ── Budget & records ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RevisionBudget:
    max_age_s: float          # max seconds since first shown; older → locked
    max_segments_back: int    # max distance from latest segment; further → locked


DEFAULT_BUDGETS: dict[RevisionMode, RevisionBudget] = {
    RevisionMode.STRICT:   RevisionBudget(max_age_s=5.0,  max_segments_back=1),
    RevisionMode.BALANCED: RevisionBudget(max_age_s=15.0, max_segments_back=3),
    RevisionMode.RELAXED:  RevisionBudget(max_age_s=60.0, max_segments_back=10),
}

RevisionReason = Literal[
    "within_window",    # allowed
    "too_old",          # blocked: age > max_age_s
    "too_far_back",     # blocked: distance > max_segments_back
    "already_locked",   # blocked: terminal state
    "not_registered",   # blocked: index unknown
]


@dataclass
class _SegmentRecord:
    index: int
    state: SubtitleState
    created_at: float   # time.monotonic() at registration


@dataclass
class RevisionDecision:
    segment_index: int
    allowed: bool
    reason: RevisionReason
    state_before: SubtitleState
    age_s: float
    distance: int       # current_index - segment_index


# ── Tracker ───────────────────────────────────────────────────────────────────

class SubtitleRevisionTracker:
    """Per-session state machine tracking subtitle segment lifecycle.

    Usage:
        tracker = SubtitleRevisionTracker(RevisionMode.BALANCED)

        # When a new transcript segment arrives:
        newly_locked = tracker.register(segment_index)

        # Before applying refinement:
        decision = tracker.decide_revision(segment_index)
        if decision.allowed:
            ... apply refinement ...
            tracker.transition(segment_index, SubtitleState.STABLE)
        else:
            tracker.transition(segment_index, SubtitleState.LOCKED)
    """

    def __init__(
        self,
        mode: RevisionMode = RevisionMode.BALANCED,
        budget: RevisionBudget | None = None,
    ) -> None:
        self.mode = mode
        self.budget = budget or DEFAULT_BUDGETS[mode]
        self._records: dict[int, _SegmentRecord] = {}
        self._current_index: int = -1

    # ── Mutating methods ──────────────────────────────────────────────────────

    def register(self, index: int) -> list[int]:
        """Register a new segment as TENTATIVE.

        Advances the current-index pointer and auto-locks segments that are
        now outside the budget window.

        Returns a list of segment indices that were newly locked by this call.
        """
        self._records[index] = _SegmentRecord(
            index=index,
            state=SubtitleState.TENTATIVE,
            created_at=time.monotonic(),
        )
        self._current_index = index
        return self._auto_lock()

    def transition(self, index: int, new_state: SubtitleState) -> SubtitleState:
        """Attempt a state transition.

        LOCKED is a terminal state — any attempt to transition away from LOCKED
        is silently ignored. Transitioning TO LOCKED is always allowed.

        Returns the resulting state.
        """
        record = self._records.get(index)
        if record is None:
            raise KeyError(f"Segment {index} not registered in revision tracker")
        if record.state is SubtitleState.LOCKED and new_state is not SubtitleState.LOCKED:
            return SubtitleState.LOCKED
        record.state = new_state
        return new_state

    # ── Query methods ─────────────────────────────────────────────────────────

    def decide_revision(self, index: int) -> RevisionDecision:
        """Return a full decision with reason and telemetry fields."""
        record = self._records.get(index)
        now = time.monotonic()

        if record is None:
            return RevisionDecision(
                segment_index=index, allowed=False, reason="not_registered",
                state_before=SubtitleState.LOCKED, age_s=0.0, distance=0,
            )

        age_s = now - record.created_at
        distance = max(0, self._current_index - index)

        if record.state is SubtitleState.LOCKED:
            return RevisionDecision(
                segment_index=index, allowed=False, reason="already_locked",
                state_before=SubtitleState.LOCKED, age_s=age_s, distance=distance,
            )

        if age_s > self.budget.max_age_s:
            return RevisionDecision(
                segment_index=index, allowed=False, reason="too_old",
                state_before=record.state, age_s=age_s, distance=distance,
            )

        if distance > self.budget.max_segments_back:
            return RevisionDecision(
                segment_index=index, allowed=False, reason="too_far_back",
                state_before=record.state, age_s=age_s, distance=distance,
            )

        return RevisionDecision(
            segment_index=index, allowed=True, reason="within_window",
            state_before=record.state, age_s=age_s, distance=distance,
        )

    def can_revise(self, index: int) -> bool:
        return self.decide_revision(index).allowed

    def get_state(self, index: int) -> SubtitleState | None:
        record = self._records.get(index)
        return record.state if record else None

    # ── Internal ──────────────────────────────────────────────────────────────

    def _auto_lock(self) -> list[int]:
        """Lock all non-locked segments outside the current budget window."""
        newly_locked: list[int] = []
        now = time.monotonic()
        for idx, record in self._records.items():
            if record.state is SubtitleState.LOCKED:
                continue
            age = now - record.created_at
            distance = self._current_index - idx
            if age > self.budget.max_age_s or distance > self.budget.max_segments_back:
                record.state = SubtitleState.LOCKED
                newly_locked.append(idx)
        return newly_locked


# ── Logging helper ────────────────────────────────────────────────────────────

def log_decision(decision: RevisionDecision, mode: RevisionMode, session_id: str = "") -> None:
    """Emit a structured INFO log line for a revision decision."""
    verb = "ALLOWED" if decision.allowed else "BLOCKED"
    sid = f"session={session_id[:8]} " if session_id else ""
    logger.info(
        "revision_policy: %sseg=%d %s (reason=%s, state=%s, age=%.1fs, dist=%d, mode=%s)",
        sid,
        decision.segment_index,
        verb,
        decision.reason,
        decision.state_before.value,
        decision.age_s,
        decision.distance,
        mode.value,
    )
