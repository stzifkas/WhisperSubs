"""Interpreter mode profiles.

Each mode is a named bundle of policy parameters that trades off latency,
stability, and correction aggressiveness. Switching modes during a session
takes effect on the next completed transcript segment.

Mode summary
────────────
  fast       Low latency. Minimal correction; only very uncertain spans repaired.
             Tight revision window. Good for fast-moving live events.

  balanced   Default. Moderate correction of uncertain spans and high-risk terms.
             Standard revision window. Good for most use cases.

  broadcast  Stability-first. Text locks quickly; only obvious errors corrected.
             Minimal on-screen instability. Good for live TV / streaming.

  precision  Accuracy-first. Aggressive correction of uncertain and high-risk spans.
             Wide revision window. Good for transcription, subtitling review.
"""
from dataclasses import dataclass
from enum import Enum

from .revision_policy import DEFAULT_BUDGETS, RevisionBudget, RevisionMode, SubtitleRevisionTracker


class InterpreterMode(str, Enum):
    FAST      = "fast"
    BALANCED  = "balanced"
    BROADCAST = "broadcast"
    PRECISION = "precision"


@dataclass(frozen=True)
class ModeProfile:
    mode: InterpreterMode
    description: str

    # Revision policy
    revision_mode: RevisionMode           # controls budget window

    # Span refinement thresholds
    refinement_low_threshold: float       # spans below this are always repaired
    refinement_high_risk_threshold: float # high-risk spans (names/numbers) below this are repaired

    # Correction style passed to refine_node
    correction_aggressiveness: str        # "low" | "medium" | "high"

    # Frontend commit delay — how long to wait for translation before showing
    # the source caption on its own (milliseconds)
    commit_delay_ms: int


MODE_PROFILES: dict[InterpreterMode, ModeProfile] = {
    InterpreterMode.FAST: ModeProfile(
        mode=InterpreterMode.FAST,
        description="Minimal latency. Fixes only highly uncertain spans.",
        revision_mode=RevisionMode.STRICT,
        refinement_low_threshold=0.35,
        refinement_high_risk_threshold=0.60,
        correction_aggressiveness="low",
        commit_delay_ms=500,
    ),
    InterpreterMode.BALANCED: ModeProfile(
        mode=InterpreterMode.BALANCED,
        description="Default. Balances speed, accuracy, and stability.",
        revision_mode=RevisionMode.BALANCED,
        refinement_low_threshold=0.55,
        refinement_high_risk_threshold=0.80,
        correction_aggressiveness="medium",
        commit_delay_ms=2000,
    ),
    InterpreterMode.BROADCAST: ModeProfile(
        mode=InterpreterMode.BROADCAST,
        description="Maximum stability. Text locks quickly; only obvious errors fixed.",
        revision_mode=RevisionMode.STRICT,
        refinement_low_threshold=0.40,
        refinement_high_risk_threshold=0.65,
        correction_aggressiveness="low",
        commit_delay_ms=800,
    ),
    InterpreterMode.PRECISION: ModeProfile(
        mode=InterpreterMode.PRECISION,
        description="Maximum accuracy. Aggressive correction; wide revision window.",
        revision_mode=RevisionMode.RELAXED,
        refinement_low_threshold=0.70,
        refinement_high_risk_threshold=0.92,
        correction_aggressiveness="high",
        commit_delay_ms=3000,
    ),
}

DEFAULT_MODE = InterpreterMode.BALANCED


@dataclass
class SessionPolicy:
    """Runtime policy for one WebSocket session. Mutable — switches atomically."""

    mode: InterpreterMode
    profile: ModeProfile
    tracker: SubtitleRevisionTracker

    @classmethod
    def create(cls, mode: InterpreterMode = DEFAULT_MODE) -> "SessionPolicy":
        profile = MODE_PROFILES[mode]
        tracker = SubtitleRevisionTracker(
            mode=profile.revision_mode,
            budget=DEFAULT_BUDGETS[profile.revision_mode],
        )
        return cls(mode=mode, profile=profile, tracker=tracker)

    def switch_to(self, new_mode: InterpreterMode) -> None:
        """Switch to a new mode. Resets the revision tracker (existing segment
        history is discarded — the new tracker starts fresh).
        """
        self.mode = new_mode
        self.profile = MODE_PROFILES[new_mode]
        self.tracker = SubtitleRevisionTracker(
            mode=self.profile.revision_mode,
            budget=DEFAULT_BUDGETS[self.profile.revision_mode],
        )
