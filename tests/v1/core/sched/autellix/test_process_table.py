# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the Autellix per-program process table."""

from hypothesis import given
from hypothesis import strategies as st

from vllm.v1.core.sched.autellix.process_table import ProcessTable, ProgramState


def test_program_state_defaults():
    state = ProgramState(program_id="p")
    assert state.program_id == "p"
    assert state.service == 0.0
    assert state.max_critical_path == 0.0
    assert state.total_wait == 0.0
    assert state.last_arrival == 0.0
    assert state.last_completion == 0.0
    assert state.active_call_ids == set()


def test_program_state_active_call_ids_not_shared():
    a = ProgramState(program_id="a")
    b = ProgramState(program_id="b")
    a.active_call_ids.add("call")
    assert b.active_call_ids == set()


def test_get_unknown_returns_none():
    table = ProcessTable(ttl=10.0)
    assert table.get("missing") is None


def test_get_or_create_creates_once():
    table = ProcessTable(ttl=10.0)
    first = table.get_or_create("p")
    second = table.get_or_create("p")
    assert first is second
    assert table.get("p") is first


def test_add_service_sum_rule():
    table = ProcessTable(ttl=10.0)
    table.add_service("p", 1.0)
    table.add_service("p", 2.5)
    assert table.get_or_create("p").service == 3.5


def test_add_service_creates_program():
    table = ProcessTable(ttl=10.0)
    table.add_service("new", 4.0)
    state = table.get("new")
    assert state is not None
    assert state.service == 4.0


def test_add_service_ignores_negative():
    table = ProcessTable(ttl=10.0)
    table.add_service("p", 5.0)
    table.add_service("p", -3.0)
    assert table.get_or_create("p").service == 5.0


def test_update_critical_path_max_rule():
    table = ProcessTable(ttl=10.0)
    table.update_critical_path("p", start_priority=2.0, call_service=3.0)
    assert table.get_or_create("p").max_critical_path == 5.0
    # A smaller candidate must not lower the scalar.
    table.update_critical_path("p", start_priority=1.0, call_service=1.0)
    assert table.get_or_create("p").max_critical_path == 5.0
    # A larger candidate raises it.
    table.update_critical_path("p", start_priority=4.0, call_service=4.0)
    assert table.get_or_create("p").max_critical_path == 8.0


def test_add_wait_accumulates():
    table = ProcessTable(ttl=10.0)
    table.add_wait("p", 1.0)
    table.add_wait("p", 0.5)
    assert table.get_or_create("p").total_wait == 1.5


def test_register_call_tracks_active_and_arrival():
    table = ProcessTable(ttl=10.0)
    table.register_call("p", "c1", arrival_time=3.0)
    table.register_call("p", "c2", arrival_time=4.0)
    state = table.get_or_create("p")
    assert state.active_call_ids == {"c1", "c2"}
    assert state.last_arrival == 4.0


def test_complete_call_removes_active_and_sets_completion():
    table = ProcessTable(ttl=10.0)
    table.register_call("p", "c1", arrival_time=1.0)
    table.complete_call("p", "c1", completion_time=9.0)
    state = table.get_or_create("p")
    assert state.active_call_ids == set()
    assert state.last_completion == 9.0


def test_complete_call_unknown_call_id_is_safe():
    table = ProcessTable(ttl=10.0)
    table.register_call("p", "c1", arrival_time=1.0)
    # Completing a call id that was never registered must not raise.
    table.complete_call("p", "unknown", completion_time=5.0)
    state = table.get_or_create("p")
    assert state.active_call_ids == {"c1"}
    assert state.last_completion == 5.0


def test_complete_call_is_idempotent():
    table = ProcessTable(ttl=10.0)
    table.register_call("p", "c1", arrival_time=1.0)
    table.complete_call("p", "c1", completion_time=5.0)
    table.complete_call("p", "c1", completion_time=6.0)
    state = table.get_or_create("p")
    assert state.active_call_ids == set()
    assert state.last_completion == 6.0


def test_gc_evicts_idle_and_expired():
    table = ProcessTable(ttl=10.0)
    table.register_call("p", "c1", arrival_time=0.0)
    table.complete_call("p", "c1", completion_time=1.0)
    evicted = table.gc(now=11.0)
    assert evicted == ["p"]
    assert table.get("p") is None


def test_gc_keeps_active_program():
    table = ProcessTable(ttl=10.0)
    table.register_call("p", "c1", arrival_time=0.0)
    # Never completed: still active, so never evicted even far in the future.
    assert table.gc(now=1_000_000.0) == []
    assert table.get("p") is not None


def test_gc_keeps_idle_but_unexpired():
    table = ProcessTable(ttl=10.0)
    table.register_call("p", "c1", arrival_time=0.0)
    table.complete_call("p", "c1", completion_time=5.0)
    # 5.0 + 10.0 = 15.0 > 12.0, so not yet expired.
    assert table.gc(now=12.0) == []
    assert table.get("p") is not None


def test_gc_evicts_only_expired_subset():
    table = ProcessTable(ttl=10.0)
    table.register_call("stale", "c1", arrival_time=0.0)
    table.complete_call("stale", "c1", completion_time=1.0)
    table.register_call("fresh", "c2", arrival_time=0.0)
    table.complete_call("fresh", "c2", completion_time=20.0)
    table.register_call("busy", "c3", arrival_time=0.0)
    evicted = table.gc(now=15.0)
    assert evicted == ["stale"]
    assert table.get("stale") is None
    assert table.get("fresh") is not None
    assert table.get("busy") is not None


def test_gc_on_boundary_is_inclusive():
    table = ProcessTable(ttl=10.0)
    table.register_call("p", "c1", arrival_time=0.0)
    table.complete_call("p", "c1", completion_time=2.0)
    # last_completion + ttl == now exactly -> evicted (<= boundary).
    assert table.gc(now=12.0) == ["p"]


def test_gc_empty_table():
    table = ProcessTable(ttl=10.0)
    assert table.gc(now=5.0) == []


@given(amounts=st.lists(st.floats(min_value=0.0, max_value=1e6), max_size=50))
def test_service_is_monotonic_non_decreasing(amounts):
    table = ProcessTable(ttl=10.0)
    previous = 0.0
    running_total = 0.0
    for amount in amounts:
        table.add_service("p", amount)
        current = table.get_or_create("p").service
        assert current >= previous
        previous = current
        running_total += amount
    assert table.get_or_create("p").service == running_total


@given(
    updates=st.lists(
        st.tuples(
            st.floats(min_value=-1e6, max_value=1e6),
            st.floats(min_value=-1e6, max_value=1e6),
        ),
        max_size=50,
    )
)
def test_critical_path_never_decreases(updates):
    table = ProcessTable(ttl=10.0)
    previous = 0.0
    for start_priority, call_service in updates:
        table.update_critical_path("p", start_priority, call_service)
        current = table.get_or_create("p").max_critical_path
        assert current >= previous
        previous = current
