"""Tests for the subtitle revision policy engine."""
import time
import pytest
from unittest.mock import patch

from backend.revision_policy import (
    DEFAULT_BUDGETS,
    RevisionBudget,
    RevisionMode,
    SubtitleRevisionTracker,
    SubtitleState,
    log_decision,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _tracker(mode: RevisionMode = RevisionMode.BALANCED, **budget_overrides) -> SubtitleRevisionTracker:
    default = DEFAULT_BUDGETS[mode]
    budget = RevisionBudget(
        max_age_s=budget_overrides.get("max_age_s", default.max_age_s),
        max_segments_back=budget_overrides.get("max_segments_back", default.max_segments_back),
    )
    return SubtitleRevisionTracker(mode=mode, budget=budget)


# ── State machine: registration & initial state ───────────────────────────────

def test_new_segment_is_tentative():
    t = _tracker()
    t.register(0)
    assert t.get_state(0) is SubtitleState.TENTATIVE


def test_unregistered_segment_returns_none():
    t = _tracker()
    assert t.get_state(99) is None


def test_register_multiple_advances_current_index():
    t = _tracker()
    t.register(0)
    t.register(1)
    t.register(2)
    assert t.get_state(2) is SubtitleState.TENTATIVE


# ── State machine: transitions ────────────────────────────────────────────────

def test_transition_tentative_to_stable():
    t = _tracker()
    t.register(0)
    result = t.transition(0, SubtitleState.STABLE)
    assert result is SubtitleState.STABLE
    assert t.get_state(0) is SubtitleState.STABLE


def test_transition_tentative_to_locked():
    t = _tracker()
    t.register(0)
    result = t.transition(0, SubtitleState.LOCKED)
    assert result is SubtitleState.LOCKED
    assert t.get_state(0) is SubtitleState.LOCKED


def test_transition_stable_to_locked():
    t = _tracker()
    t.register(0)
    t.transition(0, SubtitleState.STABLE)
    result = t.transition(0, SubtitleState.LOCKED)
    assert result is SubtitleState.LOCKED


def test_locked_is_terminal_cannot_go_to_stable():
    t = _tracker()
    t.register(0)
    t.transition(0, SubtitleState.LOCKED)
    result = t.transition(0, SubtitleState.STABLE)  # attempt to escape
    assert result is SubtitleState.LOCKED
    assert t.get_state(0) is SubtitleState.LOCKED


def test_locked_is_terminal_cannot_go_to_tentative():
    t = _tracker()
    t.register(0)
    t.transition(0, SubtitleState.LOCKED)
    t.transition(0, SubtitleState.TENTATIVE)
    assert t.get_state(0) is SubtitleState.LOCKED


def test_transition_unregistered_raises():
    t = _tracker()
    with pytest.raises(KeyError):
        t.transition(99, SubtitleState.STABLE)


# ── Revision decisions: allowed path ─────────────────────────────────────────

def test_revision_allowed_for_new_segment():
    t = _tracker()
    t.register(0)
    assert t.can_revise(0) is True


def test_revision_allowed_within_segment_window():
    t = _tracker(max_segments_back=3)
    for i in range(4):
        t.register(i)
    # segment 1 is 3 back from current (3) → exactly at limit → allowed
    assert t.can_revise(1) is True


def test_decision_reason_within_window():
    t = _tracker()
    t.register(0)
    d = t.decide_revision(0)
    assert d.allowed is True
    assert d.reason == "within_window"


# ── Revision decisions: blocked paths ────────────────────────────────────────

def test_revision_blocked_for_locked_segment():
    t = _tracker()
    t.register(0)
    t.transition(0, SubtitleState.LOCKED)
    assert t.can_revise(0) is False


def test_decision_reason_already_locked():
    t = _tracker()
    t.register(0)
    t.transition(0, SubtitleState.LOCKED)
    d = t.decide_revision(0)
    assert d.reason == "already_locked"
    assert d.state_before is SubtitleState.LOCKED


def test_revision_blocked_too_far_back():
    t = _tracker(max_segments_back=2)
    for i in range(4):
        t.register(i)
    # segment 0 is 3 back from current (3) → exceeds max_segments_back=2
    assert t.can_revise(0) is False


def test_decision_reason_too_far_back():
    # Auto-lock runs inside register(), so to observe the raw too_far_back
    # reason we advance _current_index directly without triggering auto-lock.
    t = _tracker(max_segments_back=2)
    t.register(0)
    t._current_index = 5           # 5 back, but _auto_lock not called
    d = t.decide_revision(0)
    assert d.reason == "too_far_back"
    assert d.distance == 5


def test_revision_blocked_too_old():
    t = _tracker(max_age_s=0.05)   # 50ms window
    t.register(0)
    time.sleep(0.06)                # exceed the window
    assert t.can_revise(0) is False


def test_decision_reason_too_old():
    t = _tracker(max_age_s=0.05)
    t.register(0)
    time.sleep(0.06)
    d = t.decide_revision(0)
    assert d.reason == "too_old"
    assert d.age_s > 0.05


def test_revision_blocked_unregistered():
    t = _tracker()
    assert t.can_revise(99) is False
    assert t.decide_revision(99).reason == "not_registered"


# ── Auto-lock on new segment arrival ─────────────────────────────────────────

def test_auto_lock_locks_old_segments_by_distance():
    t = _tracker(max_segments_back=2)
    for i in range(5):
        t.register(i)
    # Current is 4; segments 0 and 1 are 4 and 3 back → locked
    assert t.get_state(0) is SubtitleState.LOCKED
    assert t.get_state(1) is SubtitleState.LOCKED
    # segment 2 is exactly 2 back → should still be tentative
    assert t.get_state(2) is not SubtitleState.LOCKED


def test_auto_lock_returns_newly_locked_indices():
    t = _tracker(max_segments_back=1)
    t.register(0)
    t.register(1)
    newly_locked = t.register(2)   # segment 0 is now 2 back → locked
    assert 0 in newly_locked


def test_auto_lock_does_not_double_count_already_locked():
    t = _tracker(max_segments_back=1)
    t.register(0)
    t.transition(0, SubtitleState.LOCKED)   # already locked
    t.register(1)
    newly_locked = t.register(2)
    assert 0 not in newly_locked            # was already locked, not "newly" locked


def test_auto_lock_by_age():
    t = _tracker(max_age_s=0.05, max_segments_back=100)
    t.register(0)
    time.sleep(0.06)
    t.register(1)  # auto-lock triggered: segment 0 is now too old
    assert t.get_state(0) is SubtitleState.LOCKED
    assert t.get_state(1) is SubtitleState.TENTATIVE


# ── Mode-specific budgets ─────────────────────────────────────────────────────

def test_strict_mode_has_tight_budget():
    budget = DEFAULT_BUDGETS[RevisionMode.STRICT]
    assert budget.max_age_s <= 10
    assert budget.max_segments_back <= 2


def test_balanced_mode_has_medium_budget():
    budget = DEFAULT_BUDGETS[RevisionMode.BALANCED]
    assert budget.max_age_s > DEFAULT_BUDGETS[RevisionMode.STRICT].max_age_s
    assert budget.max_segments_back > DEFAULT_BUDGETS[RevisionMode.STRICT].max_segments_back


def test_relaxed_mode_has_loose_budget():
    budget = DEFAULT_BUDGETS[RevisionMode.RELAXED]
    assert budget.max_age_s > DEFAULT_BUDGETS[RevisionMode.BALANCED].max_age_s
    assert budget.max_segments_back > DEFAULT_BUDGETS[RevisionMode.BALANCED].max_segments_back


def test_strict_mode_blocks_second_segment_back():
    # strict: max_segments_back=1 → segment 0 blocked when segment 2 arrives
    t = _tracker(RevisionMode.STRICT)
    t.register(0)
    t.register(1)
    t.register(2)
    assert t.can_revise(0) is False


def test_relaxed_mode_allows_many_segments_back():
    t = _tracker(RevisionMode.RELAXED)
    budget = DEFAULT_BUDGETS[RevisionMode.RELAXED]
    for i in range(budget.max_segments_back + 1):
        t.register(i)
    # The last registered segment should still be allowed
    assert t.can_revise(budget.max_segments_back) is True


# ── Locked text never rewritten ───────────────────────────────────────────────

def test_locked_text_never_rewritten_scenario():
    """Simulate a full register → revise → lock → re-revision-attempt cycle."""
    t = _tracker(max_segments_back=1)

    # Segment 0: registered, refined, locked by segment 2 arriving
    t.register(0)
    assert t.can_revise(0)
    t.transition(0, SubtitleState.STABLE)

    t.register(1)
    assert t.can_revise(0)   # still within 1-back window

    t.register(2)            # segment 0 is now 2 back → auto-locked
    assert not t.can_revise(0)
    assert t.get_state(0) is SubtitleState.LOCKED

    # Any further transition attempt on segment 0 must be rejected
    result = t.transition(0, SubtitleState.TENTATIVE)
    assert result is SubtitleState.LOCKED
    assert t.get_state(0) is SubtitleState.LOCKED


# ── Decision fields (telemetry) ───────────────────────────────────────────────

def test_decision_contains_distance():
    t = _tracker(max_segments_back=5)
    t.register(0)
    t.register(1)
    t.register(2)
    d = t.decide_revision(0)
    assert d.distance == 2


def test_decision_contains_age():
    t = _tracker()
    t.register(0)
    time.sleep(0.01)
    d = t.decide_revision(0)
    assert d.age_s >= 0.01


def test_decision_contains_state_before():
    t = _tracker()
    t.register(0)
    t.transition(0, SubtitleState.STABLE)
    d = t.decide_revision(0)
    assert d.state_before is SubtitleState.STABLE


# ── log_decision emits INFO ───────────────────────────────────────────────────

def test_log_decision_emits_info(caplog):
    import logging
    t = _tracker()
    t.register(0)
    d = t.decide_revision(0)
    with caplog.at_level(logging.INFO, logger="backend.revision_policy"):
        log_decision(d, t.mode, session_id="abc123")
    assert "ALLOWED" in caplog.text
    assert "seg=0" in caplog.text
    assert "balanced" in caplog.text


def test_log_decision_blocked_emits_info(caplog):
    import logging
    # Use too_old path (time-based) — sidestep auto-lock with direct _current_index
    t = _tracker(max_segments_back=100, max_age_s=0.01)
    t.register(0)
    time.sleep(0.02)
    d = t.decide_revision(0)
    with caplog.at_level(logging.INFO, logger="backend.revision_policy"):
        log_decision(d, t.mode)
    assert "BLOCKED" in caplog.text
    assert "too_old" in caplog.text
