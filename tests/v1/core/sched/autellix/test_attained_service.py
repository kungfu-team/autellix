# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the Autellix per-request attained-service tracker."""

from hypothesis import given
from hypothesis import strategies as st

from vllm.v1.core.sched.autellix.attained_service import AttainedServiceTracker


def test_get_unknown_returns_zero():
    tracker = AttainedServiceTracker()
    assert tracker.get("missing") == 0.0


def test_record_step_default_amount_is_one():
    tracker = AttainedServiceTracker()
    tracker.record_step("r")
    assert tracker.get("r") == 1.0
    tracker.record_step("r")
    assert tracker.get("r") == 2.0


def test_record_step_custom_amount_accumulates():
    tracker = AttainedServiceTracker()
    tracker.record_step("r", 3.0)
    tracker.record_step("r", 2.0)
    assert tracker.get("r") == 5.0


def test_record_step_does_not_reset_across_preemption():
    tracker = AttainedServiceTracker()
    # Recomputed steps after a preemption are still served, so accrual only ever
    # grows; the tracker exposes no reset path.
    tracker.record_step("r")
    tracker.record_step("r")
    before = tracker.get("r")
    tracker.record_step("r")
    assert tracker.get("r") == before + 1.0
    assert not hasattr(tracker, "reset")


def test_requests_are_independent():
    tracker = AttainedServiceTracker()
    tracker.record_step("a", 2.0)
    tracker.record_step("b", 5.0)
    assert tracker.get("a") == 2.0
    assert tracker.get("b") == 5.0


def test_pop_removes_and_returns_accrued():
    tracker = AttainedServiceTracker()
    tracker.record_step("r", 4.0)
    assert tracker.pop("r") == 4.0
    # Popped, so it is now unknown again.
    assert tracker.get("r") == 0.0


def test_pop_unknown_returns_zero():
    tracker = AttainedServiceTracker()
    assert tracker.pop("missing") == 0.0


@given(
    amounts=st.lists(st.floats(min_value=0.0, max_value=1e6), min_size=1, max_size=50)
)
def test_record_step_accumulates_monotonically(amounts):
    tracker = AttainedServiceTracker()
    previous = 0.0
    expected = 0.0
    for amount in amounts:
        tracker.record_step("r", amount)
        current = tracker.get("r")
        assert current >= previous
        previous = current
        # Mirror the tracker's own left-fold so the comparison is bit-exact and
        # does not depend on float summation order.
        expected += amount
    assert tracker.pop("r") == expected
