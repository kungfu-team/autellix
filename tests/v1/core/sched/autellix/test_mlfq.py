# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the Autellix multi-level feedback queue mechanics."""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from vllm.v1.core.sched.autellix.mlfq import MlfqBinner

_SERVICE = st.floats(
    min_value=0.0, max_value=1e12, allow_nan=False, allow_infinity=False
)


def test_geometric_thresholds_binning():
    # first_quantum=1, ratio=2, K=4 -> cumulative thresholds [1, 3, 7].
    binner = MlfqBinner(num_queues=4, first_quantum=1.0, growth_ratio=2.0)
    assert binner.bin(0.0) == 0
    assert binner.bin(0.5) == 0
    assert binner.bin(1.0) == 1
    assert binner.bin(2.0) == 1
    assert binner.bin(3.0) == 2
    assert binner.bin(6.9) == 2
    assert binner.bin(7.0) == 3
    assert binner.bin(1000.0) == 3


def test_explicit_thresholds_binning():
    binner = MlfqBinner(num_queues=4, thresholds=[10.0, 20.0, 30.0])
    assert binner.bin(0.0) == 0
    assert binner.bin(9.999) == 0
    assert binner.bin(10.0) == 1
    assert binner.bin(19.9) == 1
    assert binner.bin(20.0) == 2
    assert binner.bin(29.9) == 2
    assert binner.bin(30.0) == 3
    assert binner.bin(1e9) == 3


def test_bin_boundaries_are_lower_inclusive():
    binner = MlfqBinner(num_queues=4, thresholds=[10.0, 20.0, 30.0])
    # Each threshold belongs to the higher-index (more-served) bin.
    assert binner.bin(10.0 - 1e-9) == 0
    assert binner.bin(10.0) == 1
    assert binner.bin(20.0) == 2
    assert binner.bin(30.0) == 4 - 1


def test_top_bin_is_unbounded():
    binner = MlfqBinner(num_queues=3, thresholds=[5.0, 9.0])
    assert binner.bin(9.0) == 2
    assert binner.bin(1e18) == 2


def test_single_queue_maps_everything_to_zero():
    binner = MlfqBinner(num_queues=1, first_quantum=1.0, growth_ratio=2.0)
    assert binner.bin(0.0) == 0
    assert binner.bin(1e9) == 0


def test_single_queue_explicit_empty_thresholds():
    binner = MlfqBinner(num_queues=1, thresholds=[])
    assert binner.bin(42.0) == 0


def test_growth_ratio_one_gives_equal_quanta():
    # ratio == 1 is valid (boundary); quanta are equal, thresholds [2, 4, 6].
    binner = MlfqBinner(num_queues=4, first_quantum=2.0, growth_ratio=1.0)
    assert binner.bin(1.9) == 0
    assert binner.bin(2.0) == 1
    assert binner.bin(4.0) == 2
    assert binner.bin(6.0) == 3


@pytest.mark.parametrize("num_queues", [0, -1])
def test_invalid_num_queues_raises(num_queues):
    with pytest.raises(ValueError):
        MlfqBinner(num_queues=num_queues, first_quantum=1.0, growth_ratio=2.0)


def test_invalid_growth_ratio_raises():
    with pytest.raises(ValueError):
        MlfqBinner(num_queues=4, first_quantum=1.0, growth_ratio=0.5)


@pytest.mark.parametrize("first_quantum", [0.0, -1.0])
def test_invalid_first_quantum_raises(first_quantum):
    with pytest.raises(ValueError):
        MlfqBinner(num_queues=4, first_quantum=first_quantum, growth_ratio=2.0)


def test_invalid_thresholds_length_raises():
    with pytest.raises(ValueError):
        MlfqBinner(num_queues=4, thresholds=[1.0, 2.0])


def test_invalid_thresholds_non_positive_raises():
    with pytest.raises(ValueError):
        MlfqBinner(num_queues=3, thresholds=[-1.0, 2.0])


def test_invalid_thresholds_not_strictly_increasing_raises():
    with pytest.raises(ValueError):
        MlfqBinner(num_queues=3, thresholds=[10.0, 5.0])


def test_invalid_thresholds_equal_values_raises():
    with pytest.raises(ValueError):
        MlfqBinner(num_queues=3, thresholds=[10.0, 10.0])


def test_demote_increments_by_one():
    binner = MlfqBinner(num_queues=4, first_quantum=1.0, growth_ratio=2.0)
    assert binner.demote(0) == 1
    assert binner.demote(1) == 2
    assert binner.demote(2) == 3


def test_demote_saturates_at_last_queue():
    binner = MlfqBinner(num_queues=4, first_quantum=1.0, growth_ratio=2.0)
    assert binner.demote(3) == 3
    assert binner.demote(10) == 3


def test_anti_starvation_above_beta():
    binner = MlfqBinner(num_queues=4, first_quantum=1.0, growth_ratio=2.0)
    assert binner.anti_starvation(total_wait=10.0, total_service=2.0, beta=3.0) is True


def test_anti_starvation_at_beta():
    binner = MlfqBinner(num_queues=4, first_quantum=1.0, growth_ratio=2.0)
    assert binner.anti_starvation(total_wait=6.0, total_service=2.0, beta=3.0) is True


def test_anti_starvation_below_beta():
    binner = MlfqBinner(num_queues=4, first_quantum=1.0, growth_ratio=2.0)
    assert binner.anti_starvation(total_wait=2.0, total_service=2.0, beta=3.0) is False


def test_anti_starvation_zero_service_with_wait():
    binner = MlfqBinner(num_queues=4, first_quantum=1.0, growth_ratio=2.0)
    # No service yet but waiting -> starving, and no ZeroDivisionError.
    assert binner.anti_starvation(total_wait=5.0, total_service=0.0, beta=3.0) is True


def test_anti_starvation_zero_service_zero_wait():
    binner = MlfqBinner(num_queues=4, first_quantum=1.0, growth_ratio=2.0)
    assert binner.anti_starvation(total_wait=0.0, total_service=0.0, beta=3.0) is False


def test_anti_starvation_zero_service_below_beta_is_not_starving():
    binner = MlfqBinner(num_queues=4, first_quantum=1.0, growth_ratio=2.0)
    # T = 0 is floored to 1.0, so the ratio is W / 1.0 = 2.0 < beta = 3.0.
    assert binner.anti_starvation(total_wait=2.0, total_service=0.0, beta=3.0) is False


def test_anti_starvation_small_service_is_floored_to_one():
    binner = MlfqBinner(num_queues=4, first_quantum=1.0, growth_ratio=2.0)
    # 0 < T < 1 is floored to 1.0, so 5.0 / max(1.0, 0.5) = 5.0 < beta = 8.0.
    assert binner.anti_starvation(total_wait=5.0, total_service=0.5, beta=8.0) is False


def test_outranks():
    binner = MlfqBinner(num_queues=4, first_quantum=1.0, growth_ratio=2.0)
    assert binner.outranks(waiting_bin=0, running_bin=1) is True
    assert binner.outranks(waiting_bin=1, running_bin=1) is False
    assert binner.outranks(waiting_bin=2, running_bin=1) is False


@given(service=_SERVICE)
def test_bin_is_deterministic_pure_function(service):
    config = dict(num_queues=4, thresholds=[1.0, 3.0, 7.0])
    first = MlfqBinner(**config)
    second = MlfqBinner(**config)
    assert first.bin(service) == first.bin(service)
    assert first.bin(service) == second.bin(service)


@given(a=_SERVICE, b=_SERVICE)
def test_bin_is_monotonic_non_decreasing(a, b):
    lo, hi = sorted((a, b))
    binner = MlfqBinner(num_queues=5, first_quantum=1.0, growth_ratio=2.0)
    assert binner.bin(lo) <= binner.bin(hi)


@given(num_queues=st.integers(min_value=1, max_value=8), service=_SERVICE)
def test_bin_is_always_in_range(num_queues, service):
    binner = MlfqBinner(num_queues=num_queues, first_quantum=1.0, growth_ratio=2.0)
    assert 0 <= binner.bin(service) <= num_queues - 1
