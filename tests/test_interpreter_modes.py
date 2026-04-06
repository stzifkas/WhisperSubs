"""Tests for interpreter mode profiles and SessionPolicy."""
import pytest

from backend.interpreter_modes import (
    DEFAULT_MODE,
    InterpreterMode,
    MODE_PROFILES,
    ModeProfile,
    SessionPolicy,
)
from backend.revision_policy import (
    DEFAULT_BUDGETS,
    RevisionMode,
    SubtitleState,
)


# ── MODE_PROFILES coverage ────────────────────────────────────────────────────

def test_all_modes_have_profiles():
    for mode in InterpreterMode:
        assert mode in MODE_PROFILES


def test_profile_types():
    for mode, profile in MODE_PROFILES.items():
        assert isinstance(profile, ModeProfile)
        assert profile.mode is mode
        assert isinstance(profile.description, str)
        assert isinstance(profile.revision_mode, RevisionMode)
        assert 0.0 <= profile.refinement_low_threshold <= 1.0
        assert 0.0 <= profile.refinement_high_risk_threshold <= 1.0
        assert profile.correction_aggressiveness in ("low", "medium", "high")
        assert profile.commit_delay_ms > 0


def test_default_mode_is_balanced():
    assert DEFAULT_MODE is InterpreterMode.BALANCED


# ── Mode ordering / relative invariants ───────────────────────────────────────

def test_precision_has_higher_thresholds_than_fast():
    fast = MODE_PROFILES[InterpreterMode.FAST]
    precision = MODE_PROFILES[InterpreterMode.PRECISION]
    assert precision.refinement_low_threshold > fast.refinement_low_threshold
    assert precision.refinement_high_risk_threshold > fast.refinement_high_risk_threshold


def test_precision_has_longer_commit_delay_than_fast():
    assert (
        MODE_PROFILES[InterpreterMode.PRECISION].commit_delay_ms
        > MODE_PROFILES[InterpreterMode.FAST].commit_delay_ms
    )


def test_broadcast_uses_strict_revision_mode():
    assert MODE_PROFILES[InterpreterMode.BROADCAST].revision_mode is RevisionMode.STRICT


def test_precision_uses_relaxed_revision_mode():
    assert MODE_PROFILES[InterpreterMode.PRECISION].revision_mode is RevisionMode.RELAXED


def test_fast_correction_aggressiveness_is_low():
    assert MODE_PROFILES[InterpreterMode.FAST].correction_aggressiveness == "low"


def test_precision_correction_aggressiveness_is_high():
    assert MODE_PROFILES[InterpreterMode.PRECISION].correction_aggressiveness == "high"


def test_balanced_correction_aggressiveness_is_medium():
    assert MODE_PROFILES[InterpreterMode.BALANCED].correction_aggressiveness == "medium"


def test_high_risk_threshold_always_above_low_threshold():
    for profile in MODE_PROFILES.values():
        assert profile.refinement_high_risk_threshold >= profile.refinement_low_threshold


# ── SessionPolicy.create ──────────────────────────────────────────────────────

def test_create_uses_requested_mode():
    policy = SessionPolicy.create(InterpreterMode.PRECISION)
    assert policy.mode is InterpreterMode.PRECISION


def test_create_wires_correct_profile():
    for mode in InterpreterMode:
        policy = SessionPolicy.create(mode)
        assert policy.profile is MODE_PROFILES[mode]


def test_create_initialises_tracker_with_profile_revision_mode():
    policy = SessionPolicy.create(InterpreterMode.BROADCAST)
    assert policy.tracker.mode is RevisionMode.STRICT


def test_create_default_mode():
    policy = SessionPolicy.create()
    assert policy.mode is DEFAULT_MODE


# ── SessionPolicy.switch_to ───────────────────────────────────────────────────

def test_switch_to_updates_mode():
    policy = SessionPolicy.create(InterpreterMode.FAST)
    policy.switch_to(InterpreterMode.PRECISION)
    assert policy.mode is InterpreterMode.PRECISION


def test_switch_to_updates_profile():
    policy = SessionPolicy.create(InterpreterMode.FAST)
    policy.switch_to(InterpreterMode.PRECISION)
    assert policy.profile is MODE_PROFILES[InterpreterMode.PRECISION]


def test_switch_to_resets_tracker():
    policy = SessionPolicy.create(InterpreterMode.RELAXED if hasattr(InterpreterMode, 'RELAXED') else InterpreterMode.BALANCED)
    old_tracker = policy.tracker
    policy.switch_to(InterpreterMode.FAST)
    assert policy.tracker is not old_tracker


def test_switch_to_fresh_tracker_has_no_history():
    policy = SessionPolicy.create(InterpreterMode.BALANCED)
    for i in range(5):
        policy.tracker.register(i)
    policy.switch_to(InterpreterMode.PRECISION)
    # New tracker should not know about old segments
    assert policy.tracker.get_state(0) is None


def test_switch_updates_tracker_revision_mode():
    policy = SessionPolicy.create(InterpreterMode.FAST)
    policy.switch_to(InterpreterMode.PRECISION)
    assert policy.tracker.mode is RevisionMode.RELAXED


# ── Tracker inherits mode budget ──────────────────────────────────────────────

def test_fast_policy_tracker_uses_strict_budget():
    policy = SessionPolicy.create(InterpreterMode.FAST)
    strict_budget = DEFAULT_BUDGETS[RevisionMode.STRICT]
    assert policy.tracker.budget.max_segments_back == strict_budget.max_segments_back
    assert policy.tracker.budget.max_age_s == strict_budget.max_age_s


def test_precision_policy_tracker_allows_more_segments_back_than_fast():
    fast = SessionPolicy.create(InterpreterMode.FAST)
    precision = SessionPolicy.create(InterpreterMode.PRECISION)
    assert (
        precision.tracker.budget.max_segments_back
        > fast.tracker.budget.max_segments_back
    )


# ── InterpreterMode enum ──────────────────────────────────────────────────────

def test_interpreter_mode_values_are_strings():
    for mode in InterpreterMode:
        assert isinstance(mode.value, str)


def test_interpreter_mode_constructible_from_string():
    assert InterpreterMode("balanced") is InterpreterMode.BALANCED
    assert InterpreterMode("fast") is InterpreterMode.FAST
    assert InterpreterMode("broadcast") is InterpreterMode.BROADCAST
    assert InterpreterMode("precision") is InterpreterMode.PRECISION


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        InterpreterMode("turbo")
