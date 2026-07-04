# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Behavioural tests for the program-aware critical-path scheduler (ATLAS).

These exercise the real vLLM v1 scheduler stack (mirroring
``tests/v1/core/utils.py::create_scheduler``) composed with the Phase-0 policy
core, so they live in the standard test tree rather than the shielded pure-core
subtree. Each test drives the scheduler with mocked ``ModelRunnerOutput`` steps,
exactly like ``tests/v1/core/test_scheduler.py`` and the PLAS/MLFQ tests.

ATLAS is PLAS with two changes (POLICY_REFERENCE.md §2, paper Eq. 2 + Alg. 1):

* the program scalar is the **max critical path** (not the cumulative sum), and
* each call snapshots that scalar at arrival as its *start priority*, which is
  used in the max-update ``max(prev, start_priority + call_service)`` when the
  call completes.

The defining behaviour vs PLAS: two *concurrent* calls of one program that both
inherit scalar ``M`` and attain service ``t_a`` / ``t_b`` push the program
scalar to ``M + max(t_a, t_b)`` (the critical path), not ``M + t_a + t_b`` (the
sum) -- so a program's parallel threads share a bin and a straggler does not
starve its siblings, all without any explicit parent/DAG tracking.
"""

import time
from collections import deque

import pytest
import torch

from tests.v1.core.utils import EOS_TOKEN_ID
from vllm.config import (
    CacheConfig,
    ModelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.autellix.atlas_scheduler import ATLASScheduler
from vllm.v1.core.sched.autellix.plas_scheduler import PLASScheduler
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.request_queue import PriorityRequestQueue, SchedulingPolicy
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

pytestmark = pytest.mark.cpu_test

_none_hash_initialized = False


def _make_vllm_config(
    max_num_seqs: int,
    max_num_batched_tokens: int,
    num_blocks: int,
    block_size: int,
    policy: str,
    long_prefill_token_threshold: int = 0,
    max_model_len: int | None = None,
    async_scheduling: bool = False,
) -> VllmConfig:
    model_config = ModelConfig(
        model="facebook/opt-125m",
        trust_remote_code=True,
        dtype="float16",
        seed=42,
    )
    if max_model_len is None:
        max_model_len = max_num_batched_tokens
    scheduler_config = SchedulerConfig(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        max_model_len=max_model_len,
        long_prefill_token_threshold=long_prefill_token_threshold,
        enable_chunked_prefill=True,
        is_encoder_decoder=model_config.is_encoder_decoder,
        policy=policy,
        async_scheduling=async_scheduling,
        watermark=0.0,
    )
    cache_config = CacheConfig(
        block_size=block_size,
        gpu_memory_utilization=0.9,
        cache_dtype="auto",
        enable_prefix_caching=False,
    )
    vllm_config = VllmConfig(
        scheduler_config=scheduler_config,
        model_config=model_config,
        cache_config=cache_config,
    )
    cache_config.num_gpu_blocks = num_blocks
    return vllm_config


def _make_kv_cache_config(num_blocks: int, block_size: int) -> KVCacheConfig:
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"],
                FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            )
        ],
    )


def create_atlas_scheduler(
    max_num_seqs: int = 16,
    max_num_batched_tokens: int = 8192,
    num_blocks: int = 10000,
    block_size: int = 16,
    long_prefill_token_threshold: int = 0,
    max_model_len: int | None = None,
    async_scheduling: bool = False,
) -> ATLASScheduler:
    """Build an ``ATLASScheduler`` with the stock (FCFS) config default.

    Leaving ``policy`` at its ``"fcfs"`` default proves the scheduler
    self-configures priority ordering on construction. ``async_scheduling``
    toggles ``SchedulerConfig.async_scheduling`` so the ``AsyncScheduler``
    run-ahead code path is active.
    """
    vllm_config = _make_vllm_config(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        num_blocks=num_blocks,
        block_size=block_size,
        policy="fcfs",
        long_prefill_token_threshold=long_prefill_token_threshold,
        max_model_len=max_model_len,
        async_scheduling=async_scheduling,
    )
    return ATLASScheduler(
        vllm_config=vllm_config,
        kv_cache_config=_make_kv_cache_config(num_blocks, block_size),
        log_stats=True,
        structured_output_manager=StructuredOutputManager(vllm_config),
        block_size=block_size,
        hash_block_size=block_size,
    )


def create_plas_scheduler(
    max_num_seqs: int = 16,
    max_num_batched_tokens: int = 8192,
    num_blocks: int = 10000,
    block_size: int = 16,
) -> PLASScheduler:
    """Build a ``PLASScheduler`` for the ATLAS-vs-PLAS coincidence test."""
    vllm_config = _make_vllm_config(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        num_blocks=num_blocks,
        block_size=block_size,
        policy="fcfs",
    )
    return PLASScheduler(
        vllm_config=vllm_config,
        kv_cache_config=_make_kv_cache_config(num_blocks, block_size),
        log_stats=True,
        structured_output_manager=StructuredOutputManager(vllm_config),
        block_size=block_size,
        hash_block_size=block_size,
    )


def make_request(
    req_id: str,
    program_id: str | None = None,
    thread_id: str | None = None,
    num_tokens: int = 4,
    max_tokens: int = 16,
    arrival_time: float | None = None,
    with_extra_args: bool = True,
    block_size: int = 16,
) -> Request:
    """Build a ``Request`` carrying optional program/thread ids in extra_args."""
    global _none_hash_initialized
    if not _none_hash_initialized:
        init_none_hash(sha256)
        _none_hash_initialized = True
    block_hasher = get_request_block_hasher(block_size, sha256)

    extra_args: dict | None = None
    if with_extra_args:
        extra_args = {}
        if program_id is not None:
            extra_args["program_id"] = program_id
        if thread_id is not None:
            extra_args["thread_id"] = thread_id
    sampling_params = SamplingParams(
        ignore_eos=True,
        max_tokens=max_tokens,
        extra_args=extra_args,
    )
    sampling_params.update_from_generation_config({}, EOS_TOKEN_ID)

    # Distinct prompt per request id keeps prefixes from colliding.
    fill = (abs(hash(req_id)) % 100) + 1
    return Request(
        request_id=req_id,
        prompt_token_ids=[fill] * num_tokens,
        sampling_params=sampling_params,
        pooling_params=None,
        arrival_time=arrival_time,
        block_hasher=block_hasher,
    )


def _step(scheduler: Scheduler, token: int = 100) -> SchedulerOutput:
    """Run one schedule + mocked model output step; return SchedulerOutput."""
    output = scheduler.schedule()
    req_ids = list(output.num_scheduled_tokens.keys())
    sampled: list[list[int]] = []
    for rid in req_ids:
        request = scheduler.requests[rid]
        # Only requests past their prefill sample a token this step.
        sampled.append([token] if not request.is_prefill_chunk else [])
    model_output = ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={rid: i for i, rid in enumerate(req_ids)},
        sampled_token_ids=sampled,
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    scheduler.update_from_output(output, model_output)
    return output


def _run_to_completion(
    scheduler: Scheduler, req_id: str, token: int = 100, max_steps: int = 200
) -> int:
    """Drive steps until ``req_id`` finishes; return #steps it was scheduled."""
    steps_scheduled = 0
    for _ in range(max_steps):
        if req_id not in scheduler.requests:
            return steps_scheduled
        output = _step(scheduler, token)
        if req_id in output.num_scheduled_tokens:
            steps_scheduled += 1
    raise AssertionError(f"request {req_id} did not finish in {max_steps} steps")


def _run_sequential(
    scheduler: Scheduler, program_id: str, calls: list[tuple[str, int]]
) -> list[int]:
    """Add + fully complete each call in order; return the priorities assigned."""
    priorities: list[int] = []
    for i, (req_id, max_tokens) in enumerate(calls):
        request = make_request(
            req_id,
            program_id=program_id,
            max_tokens=max_tokens,
            arrival_time=float(i + 1),
        )
        scheduler.add_request(request)
        priorities.append(request.priority)
        _run_to_completion(scheduler, req_id)
    return priorities


# --------------------------------------------------------------------------- #
# Self-configuration
# --------------------------------------------------------------------------- #


def test_self_configures_priority_ordering():
    """__init__ flips ordering to PRIORITY and builds the D5 core."""
    scheduler = create_atlas_scheduler()

    assert scheduler.policy == SchedulingPolicy.PRIORITY
    assert isinstance(scheduler.waiting, PriorityRequestQueue)
    assert isinstance(scheduler.skipped_waiting, PriorityRequestQueue)

    # D5 boundaries (0, 2, 4, 8, 16, 32, 64, inf) → 7 queues.
    binner = scheduler.binner
    assert [binner.bin(s) for s in (0, 1, 2, 3, 4, 8, 16, 32, 64, 1000)] == [
        0,
        0,
        1,
        1,
        2,
        3,
        4,
        5,
        6,
        6,
    ]


# --------------------------------------------------------------------------- #
# Gate: critical-path LAS ordering
# --------------------------------------------------------------------------- #


def test_las_admits_least_critical_path_program_first():
    """B (zero critical path) is admitted before A (long critical path)."""
    scheduler = create_atlas_scheduler(max_num_seqs=1)
    scheduler.process_table.get_or_create("A").max_critical_path = 100.0

    call_a = make_request("a1", program_id="A", arrival_time=1.0)
    call_b = make_request("b1", program_id="B", arrival_time=2.0)
    scheduler.add_request(call_a)  # arrives first
    scheduler.add_request(call_b)  # arrives later

    assert call_a.priority == 6  # bin(100)
    assert call_b.priority == 0  # bin(0)

    output = scheduler.schedule()
    scheduled = [r.req_id for r in output.scheduled_new_reqs]
    assert scheduled == ["b1"], "ATLAS admits the least-critical-path program first"
    assert [r.request_id for r in scheduler.running] == ["b1"]


# --------------------------------------------------------------------------- #
# Gate: start-priority inheritance
# --------------------------------------------------------------------------- #


def test_start_priority_inherits_max_critical_path():
    """A new call's priority is the bin of the program's max critical path."""
    scheduler = create_atlas_scheduler(max_num_seqs=1)
    scheduler.process_table.get_or_create("P").max_critical_path = 5.0

    call = make_request("c", program_id="P", arrival_time=1.0)
    scheduler.add_request(call)

    assert call.priority == scheduler.binner.bin(5.0)
    assert call.priority == 2  # bin(5) with thresholds (2, 4, 8, ...)
    # The scalar is snapshotted for use in the max-update at completion.
    assert scheduler._req_to_start_scalar["c"] == 5.0


# --------------------------------------------------------------------------- #
# Gate: parallel calls of one program share a bin (grouped)
# --------------------------------------------------------------------------- #


def test_parallel_calls_of_one_program_share_bin():
    """Concurrent calls inherit the SAME scalar, so they share a priority bin.

    This is the anti-starvation point of ATLAS: a straggler thread cannot starve
    its siblings because they were all binned at the same program scalar.
    """
    scheduler = create_atlas_scheduler(max_num_seqs=4)
    scheduler.process_table.get_or_create("P").max_critical_path = 5.0

    a = make_request("a", program_id="P", thread_id="t1", arrival_time=1.0)
    b = make_request("b", program_id="P", thread_id="t2", arrival_time=2.0)
    scheduler.add_request(a)
    scheduler.add_request(b)

    assert a.priority == b.priority == 2  # both bin(5)
    assert scheduler._req_to_start_scalar["a"] == 5.0
    assert scheduler._req_to_start_scalar["b"] == 5.0


# --------------------------------------------------------------------------- #
# Gate: max-update rule with exact values across several calls
# --------------------------------------------------------------------------- #


def test_max_update_rule_exact_values_across_calls():
    """Each completion sets max_critical_path = max(prev, start + call_service)."""
    scheduler = create_atlas_scheduler(max_num_seqs=1)

    # Call 1 (fresh program): start 0, service 3 → max(0, 0 + 3) = 3.
    c1 = make_request("c1", program_id="P", max_tokens=3, arrival_time=1.0)
    scheduler.add_request(c1)
    assert c1.priority == 0  # bin(0)
    _run_to_completion(scheduler, "c1")
    assert scheduler.process_table.get("P").max_critical_path == 3.0

    # Call 2: start 3 → bin(3) = 1, service 2 → max(3, 3 + 2) = 5.
    c2 = make_request("c2", program_id="P", max_tokens=2, arrival_time=2.0)
    scheduler.add_request(c2)
    assert c2.priority == 1
    _run_to_completion(scheduler, "c2")
    assert scheduler.process_table.get("P").max_critical_path == 5.0

    # Call 3: start 5 → bin(5) = 2, service 4 → max(5, 5 + 4) = 9.
    c3 = make_request("c3", program_id="P", max_tokens=4, arrival_time=3.0)
    scheduler.add_request(c3)
    assert c3.priority == 2
    _run_to_completion(scheduler, "c3")
    assert scheduler.process_table.get("P").max_critical_path == 9.0


def test_fold_applies_max_rule_and_keeps_prev_when_smaller():
    """The fold helper applies max(prev, start + service), keeping the larger.

    Directly seeds the per-call maps so the keep-prev branch is exercised
    deterministically: natural scheduling cannot fold a small-total call after a
    larger sibling, since completion order follows service.
    """
    scheduler = create_atlas_scheduler()
    table = scheduler.process_table
    table.get_or_create("P")

    # First fold grows the scalar: max(0, 0 + 5) = 5.
    scheduler._req_to_pid["c1"] = "P"
    scheduler._req_to_start_scalar["c1"] = 0.0
    scheduler.attained_service.record_step("c1", 5.0)
    table.register_call("P", "c1", 0.0)
    scheduler._fold_completed_calls(["c1"], 1.0)
    assert table.get("P").max_critical_path == 5.0

    # Second fold has a smaller total (0 + 2 = 2 < 5), so the max keeps prev.
    scheduler._req_to_pid["c2"] = "P"
    scheduler._req_to_start_scalar["c2"] = 0.0
    scheduler.attained_service.record_step("c2", 2.0)
    table.register_call("P", "c2", 0.0)
    scheduler._fold_completed_calls(["c2"], 2.0)
    assert table.get("P").max_critical_path == 5.0  # kept, not 7


# --------------------------------------------------------------------------- #
# Gate: critical-path (max), NOT sum, for concurrent calls
# --------------------------------------------------------------------------- #


def test_concurrent_calls_use_max_not_sum():
    """Two concurrent calls push the scalar to M + max(t_a, t_b), not the sum.

    Worked example (paper Eq. 2, the program's critical path C_program is the max
    over its threads of start_priority + attained service): program P sits at
    scalar M = 4. Two sibling calls arrive concurrently, both inheriting start
    priority 4, and attain t_a = 5 and t_b = 2 decode steps. ATLAS folds each
    with the max rule, so the program scalar becomes max(4 + 5, 4 + 2) = 9 -- the
    critical path. PLAS's sum rule would instead give 4 + 5 + 2 = 11. This is the
    defining ATLAS-vs-PLAS divergence.
    """
    scheduler = create_atlas_scheduler(max_num_seqs=4)
    m = 4.0
    scheduler.process_table.get_or_create("P").max_critical_path = m

    a = make_request("a", program_id="P", max_tokens=5, arrival_time=1.0)
    b = make_request("b", program_id="P", max_tokens=2, arrival_time=2.0)
    scheduler.add_request(a)
    scheduler.add_request(b)
    assert a.priority == b.priority  # concurrent siblings grouped (both bin(4))

    # The shorter sibling (b, t_b = 2) completes first.
    for _ in range(2):
        _step(scheduler)
    assert "b" not in scheduler.requests
    assert "a" in scheduler.requests
    assert scheduler.process_table.get("P").max_critical_path == 6.0  # max(4, 4+2)

    # The longer sibling (a, t_a = 5) completes and takes the max.
    _run_to_completion(scheduler, "a")
    max_cp = scheduler.process_table.get("P").max_critical_path
    assert max_cp == 9.0  # max(6, 4 + 5)
    assert max_cp == m + max(5.0, 2.0)  # critical path == M + max(t_a, t_b)
    assert max_cp != m + 5.0 + 2.0  # NOT the PLAS sum (would be 11)


# --------------------------------------------------------------------------- #
# Gate: coincides with PLAS for sequential (single-threaded) calls
# --------------------------------------------------------------------------- #


def test_sequential_calls_coincide_with_plas():
    """With no concurrency ATLAS's max collapses to PLAS's sum (Σ t_k).

    A single-threaded program issues its calls one after another, so each call
    inherits the full scalar left by the previous one and the max-update always
    increases -- degrading exactly to PLAS's cumulative-sum behaviour.
    """
    calls = [("s1", 3), ("s2", 2), ("s3", 4)]
    atlas = create_atlas_scheduler(max_num_seqs=1)
    plas = create_plas_scheduler(max_num_seqs=1)

    atlas_priorities = _run_sequential(atlas, "P", calls)
    plas_priorities = _run_sequential(plas, "P", calls)

    total = float(3 + 2 + 4)
    assert atlas.process_table.get("P").max_critical_path == total  # 9 == Σ t_k
    # The ATLAS critical-path scalar equals the PLAS cumulative service, and the
    # inherited start priorities coincide call-for-call.
    assert (
        atlas.process_table.get("P").max_critical_path
        == plas.process_table.get("P").service
    )
    assert atlas_priorities == plas_priorities


# --------------------------------------------------------------------------- #
# Gate: missing program_id degrades gracefully to per-request
# --------------------------------------------------------------------------- #


def test_missing_program_id_does_not_crash_and_is_per_request():
    """A request without a program_id falls back to per-request tracking."""
    scheduler = create_atlas_scheduler(max_num_seqs=1)

    no_args = make_request(
        "solo", program_id=None, with_extra_args=False, max_tokens=2, arrival_time=1.0
    )
    assert no_args.sampling_params.extra_args is None
    scheduler.add_request(no_args)
    assert no_args.priority == 0  # fresh per-request program → bin(0)
    _run_to_completion(scheduler, "solo")

    # Folded under the request id acting as its own single-call program.
    solo = scheduler.process_table.get("solo")
    assert solo is not None
    assert solo.max_critical_path == 2.0  # max(0, 0 + 2)

    # extra_args present but without a program_id key → same graceful fallback.
    partial = make_request(
        "solo2", program_id=None, with_extra_args=True, max_tokens=2, arrival_time=2.0
    )
    assert partial.sampling_params.extra_args == {}
    scheduler.add_request(partial)
    assert partial.priority == 0
    _run_to_completion(scheduler, "solo2")
    assert scheduler.process_table.get("solo2").max_critical_path == 2.0


# --------------------------------------------------------------------------- #
# Gate: thread_id is optional metadata (captured, never branched on)
# --------------------------------------------------------------------------- #


def test_thread_id_is_captured_but_does_not_change_scheduling():
    """thread_id is optional metadata: captured for debugging, never decisive.

    Concurrent calls with distinct thread_ids get the SAME priority they would
    get with no thread_id -- the max rule plus start-scalar inheritance already
    yield the critical path without any DAG/parent tracking.
    """
    scheduler = create_atlas_scheduler(max_num_seqs=4)
    scheduler.process_table.get_or_create("P").max_critical_path = 5.0

    a = make_request("a", program_id="P", thread_id="t1", arrival_time=1.0)
    b = make_request("b", program_id="P", thread_id="t2", arrival_time=2.0)
    scheduler.add_request(a)
    scheduler.add_request(b)
    assert scheduler._req_to_thread_id["a"] == "t1"
    assert scheduler._req_to_thread_id["b"] == "t2"
    assert a.priority == b.priority == 2  # same scalar → same bin

    # A call with no thread_id gets the same priority (thread_id is not consulted).
    c = make_request("c", program_id="P", thread_id=None, arrival_time=3.0)
    scheduler.add_request(c)
    assert "c" not in scheduler._req_to_thread_id
    assert c.priority == 2

    # thread ids are released on completion (no leak).
    _run_to_completion(scheduler, "a")
    assert "a" not in scheduler._req_to_thread_id


# --------------------------------------------------------------------------- #
# Gate: no state leak on normal completion AND abort
# --------------------------------------------------------------------------- #


def test_completed_call_releases_all_state():
    """On normal completion every per-call map entry is released."""
    scheduler = create_atlas_scheduler(max_num_seqs=1)
    call = make_request("r", program_id="R", max_tokens=2, arrival_time=1.0)
    scheduler.add_request(call)
    assert "r" in scheduler.process_table.get("R").active_call_ids
    assert scheduler._req_to_start_scalar["r"] == 0.0

    _run_to_completion(scheduler, "r")

    assert scheduler.process_table.get("R").active_call_ids == set()
    assert "r" not in scheduler._req_to_pid
    assert "r" not in scheduler._req_to_start_scalar
    assert scheduler.attained_service.get("r") == 0.0
    assert scheduler.process_table.get("R").max_critical_path == 2.0


def test_abort_mid_flight_folds_max_and_deregisters():
    """A call aborted mid-flight folds its critical path and frees all state.

    Aborts arrive via ``finish_requests`` (client disconnect / timeout), which
    bypasses ``update_from_output`` -- the next ``schedule()`` resets
    ``finished_req_ids`` before the fold would run. Without a dedicated hook the
    program state, req->pid entry, start-scalar snapshot, and attained-service
    entry leak permanently (mirrors the Phase 1 abort fix).
    """
    scheduler = create_atlas_scheduler(max_num_seqs=1)
    scheduler.process_table.get_or_create("AB").max_critical_path = 4.0
    call = make_request("ab", program_id="AB", max_tokens=50, arrival_time=1.0)
    scheduler.add_request(call)

    # Accrue three decode steps of service, then abort mid-flight.
    for _ in range(3):
        _step(scheduler)
    assert scheduler.attained_service.get("ab") == 3.0
    assert call.status == RequestStatus.RUNNING

    aborted = scheduler.finish_requests("ab", RequestStatus.FINISHED_ABORTED)
    assert ("ab", call.client_index) in aborted

    program = scheduler.process_table.get("AB")
    assert program is not None
    # (a) max rule on abort: max(4, start(4) + service(3)) = 7.
    assert program.max_critical_path == 7.0
    # (b) req->pid map cleaned up.
    assert "ab" not in scheduler._req_to_pid
    # (c) start-scalar snapshot cleaned up.
    assert "ab" not in scheduler._req_to_start_scalar
    # (d) attained-service entry popped.
    assert scheduler.attained_service.get("ab") == 0.0
    # (e) call removed from the program's active set.
    assert "ab" not in program.active_call_ids
    # (f) the now-idle program is GC-eligible (previously un-evictable).
    evicted = scheduler.process_table.gc(time.time() + 1e9)
    assert "AB" in evicted
    assert scheduler.process_table.get("AB") is None


def test_normal_completion_then_abort_does_not_double_fold():
    """Folding is idempotent: a normal completion is never re-folded by abort."""
    scheduler = create_atlas_scheduler(max_num_seqs=1)
    call = make_request("dc", program_id="DC", max_tokens=3, arrival_time=1.0)
    scheduler.add_request(call)
    _run_to_completion(scheduler, "dc")
    assert scheduler.process_table.get("DC").max_critical_path == 3.0

    # A late abort of the already-finished id is a no-op (the base skips it).
    aborted = scheduler.finish_requests("dc", RequestStatus.FINISHED_ABORTED)
    assert aborted == []
    assert scheduler.process_table.get("DC").max_critical_path == 3.0

    # Re-folding the same id is guarded by the popped req->pid entry, so the max
    # is not re-applied on the repeat (idempotency).
    scheduler._fold_completed_calls(["dc"], time.time())
    assert scheduler.process_table.get("DC").max_critical_path == 3.0


# --------------------------------------------------------------------------- #
# Gate: quantum demotion (Algorithm 1 lines 21-24) layered on the arrival bin
# --------------------------------------------------------------------------- #


def test_call_demotes_after_exhausting_arrival_bin_quantum():
    """The arrival bin (critical path) sets the level; quanta then demote it."""
    scheduler = create_atlas_scheduler(max_num_seqs=1)
    scheduler.process_table.get_or_create("Q").max_critical_path = 5.0

    call = make_request("q1", program_id="Q", max_tokens=200, arrival_time=1.0)
    scheduler.add_request(call)
    assert call.priority == 2  # bin(5)
    assert scheduler._call_state["q1"].quantum_remaining == 4  # queue_quanta[2]

    # Queue 2 quantum == 4: four decode steps demote to queue 3.
    for _ in range(4):
        _step(scheduler)
    assert call.priority == 3
    assert scheduler._call_state["q1"].quantum_remaining == 8  # queue_quanta[3]


# --------------------------------------------------------------------------- #
# Gate: wait accrual — a call's waited steps fold into the program's W_p
# --------------------------------------------------------------------------- #


def test_waiting_steps_fold_into_program_total_wait_on_completion():
    """A call's accrued wait window folds into ProcessTable.total_wait (W_p)."""
    scheduler = create_atlas_scheduler(max_num_seqs=1)
    # Deep-critical-path waiter (bin 6): cannot outrank or promote past the
    # fresh blocker within this test's horizon, so its wait is deterministic.
    scheduler.process_table.get_or_create("W").max_critical_path = 100.0

    blocker = make_request("blk", program_id="B", max_tokens=5, arrival_time=1.0)
    waiter = make_request("w1", program_id="W", max_tokens=2, arrival_time=2.0)
    scheduler.add_request(blocker)
    scheduler.add_request(waiter)

    for _ in range(5):  # blocker's 5 decode steps
        _step(scheduler)
    assert "blk" not in scheduler.requests
    assert scheduler._call_state["w1"].wait_window == 5

    _run_to_completion(scheduler, "w1")

    assert scheduler.process_table.get("W").total_wait == 6.0  # 5 + admission pass


# --------------------------------------------------------------------------- #
# Gate: program-level anti-starvation promotion (Algorithm 1 lines 25-31)
# --------------------------------------------------------------------------- #


def test_promotion_uses_critical_path_program_windows():
    """Promotion fires on (W_p + W_c) / max(1, T_p + T_c) with T_p = max rule.

    With W_p=20, T_p=max_critical_path=3 and beta=8 the ratio crosses at
    W_c=4 ((20+4)/3 == 8), well before the call-level-only threshold (W_c=8
    for a zero-service call) -- proving ATLAS consults the critical-path
    program windows.
    """
    scheduler = create_atlas_scheduler(max_num_seqs=1)
    program = scheduler.process_table.get_or_create("P")
    program.max_critical_path = 3.0  # T_p (ATLAS max rule)
    program.total_wait = 20.0  # W_p

    blocker = make_request("blk", program_id="B", max_tokens=500, arrival_time=0.0)
    scheduler.add_request(blocker)
    scheduler.schedule()  # admit the blocker so the P call stays waiting
    call = make_request("p1", program_id="P", max_tokens=2, arrival_time=1.0)
    scheduler.add_request(call)
    assert call.priority == 1  # bin(3)

    # W_c = 3: (20 + 3) / 3 = 7.67 < 8 -> not yet promoted.
    for _ in range(3):
        scheduler._accrue_wait_and_promote()
    assert call.priority == 1

    # W_c = 4: (20 + 4) / 3 = 8.0 >= 8 -> promoted to the top level.
    scheduler._accrue_wait_and_promote()
    assert call.priority == 0

    # Call-level windows reset; program-level windows persist (paper §4.2.2).
    state = scheduler._call_state["p1"]
    assert state.wait_window == 0
    assert state.service_window == 0
    assert state.quantum_remaining == scheduler.queue_quanta[0]
    assert program.total_wait == 20.0
    assert program.max_critical_path == 3.0


# --------------------------------------------------------------------------- #
# Gate: proactive preemption is REPORTED to the model runner (slot invariant)
# --------------------------------------------------------------------------- #


def test_proactive_preemption_is_reported_in_scheduler_output():
    """A proactively preempted call must appear in ``preempted_req_ids``.

    Same invariant as MLFQ/PLAS: the v2 model runner frees persistent-batch
    slots only for ids in ``finished_req_ids`` or ``preempted_req_ids``; an
    unreported proactive victim leaks a slot and kills the engine with
    "No free indices".
    """
    scheduler = create_atlas_scheduler(max_num_seqs=1)
    low = make_request("low", program_id="OLD", max_tokens=200, arrival_time=1.0)
    scheduler.add_request(low)
    for _ in range(7):  # demote the running call below the top level
        _step(scheduler)
    assert low.priority == 3
    assert low.status == RequestStatus.RUNNING

    high = make_request("high", program_id="NEW", max_tokens=5, arrival_time=8.0)
    scheduler.add_request(high)
    assert high.priority == 0

    output = scheduler.schedule()

    assert scheduler.requests["low"].status == RequestStatus.PREEMPTED
    assert "low" in (output.preempted_req_ids or set())
    assert scheduler.proactive_preemption_count == 1


def test_v2_worker_slot_accounting_never_overflows_under_churn():
    """Simulated v2-worker slot occupancy never exceeds ``max_num_seqs``.

    Mirrors ``gpu/model_runner.py`` order of operations per step: free slots
    for ``finished_req_ids | preempted_req_ids``, then (re-)add slots for
    ``scheduled_new_reqs``. Sustained fresh-program arrivals against a full
    batch of demoted calls keeps the proactive-preemption path hot.
    """
    max_num_seqs = 2
    scheduler = create_atlas_scheduler(max_num_seqs=max_num_seqs)
    scheduler.use_v2_model_runner = True

    slots: set[str] = set()

    def step_with_slot_accounting(token: int = 100) -> None:
        output = scheduler.schedule()
        freed = output.finished_req_ids | (output.preempted_req_ids or set())
        for req_id in freed:
            slots.discard(req_id)
        for new_req in output.scheduled_new_reqs:
            slots.add(new_req.req_id)
        assert len(slots) <= max_num_seqs, (
            f"worker slot overflow: {sorted(slots)} exceeds {max_num_seqs}"
        )
        req_ids = list(output.num_scheduled_tokens.keys())
        sampled = [
            [token] if not scheduler.requests[rid].is_prefill_chunk else []
            for rid in req_ids
        ]
        model_output = ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={rid: i for i, rid in enumerate(req_ids)},
            sampled_token_ids=sampled,
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        )
        scheduler.update_from_output(output, model_output)

    for i in range(2):
        scheduler.add_request(
            make_request(
                f"long{i}", program_id="LONG", max_tokens=400, arrival_time=0.0
            )
        )
    for _ in range(8):  # demote the long calls so fresh programs outrank them
        step_with_slot_accounting()

    for i in range(200):
        scheduler.add_request(
            make_request(
                f"f{i}", program_id=f"P{i}", max_tokens=2, arrival_time=8.0 + i
            )
        )
        step_with_slot_accounting()

    assert scheduler.proactive_preemption_count > 0


# --------------------------------------------------------------------------- #
# Decode-step proxy under chunked prefill
# --------------------------------------------------------------------------- #


def test_chunked_prefill_accrues_only_on_decode_steps():
    """Non-final prefill chunks accrue no service; decode steps each add one."""
    scheduler = create_atlas_scheduler(
        max_num_seqs=1,
        long_prefill_token_threshold=8,  # cap prefill at 8 tokens/step
        max_model_len=256,
    )
    call = make_request(
        "cp", program_id="CP", num_tokens=20, max_tokens=3, arrival_time=1.0
    )
    scheduler.add_request(call)

    # Drive the prefill chunks: service must stay 0 while still prefilling.
    prefill_chunk_steps = 0
    while True:
        _step(scheduler)
        if scheduler.requests["cp"].is_prefill_chunk:
            prefill_chunk_steps += 1
            assert scheduler.attained_service.get("cp") == 0.0
        else:
            break
    assert prefill_chunk_steps >= 1, "prompt was not chunked across steps"
    # The step that finishes prefill (and samples the first token) accrues one.
    assert scheduler.attained_service.get("cp") == 1.0

    # Finish decoding; the folded critical path excludes non-final prefill chunks.
    _run_to_completion(scheduler, "cp")
    assert scheduler.process_table.get("CP").max_critical_path == 3.0  # max(0, 0 + 3)
    assert call.num_output_tokens == 3


# --------------------------------------------------------------------------- #
# Async scheduling (run-ahead) harness
#
# The sync ``_step`` helper is lockstep (schedule then immediately deliver), so
# ``num_output_placeholders`` nets to zero every step. To exercise the genuine
# async path we keep outputs in flight in a deque and only deliver the oldest
# before scheduling the next, mirroring ``tests/v1/core/test_async_scheduler.py``.
# --------------------------------------------------------------------------- #


def _capture_prefill(scheduler: Scheduler, output: SchedulerOutput) -> dict[str, bool]:
    """Snapshot each scheduled request's prefill-chunk flag at schedule time."""
    return {
        req_id: scheduler.requests[req_id].is_prefill_chunk
        for req_id in output.num_scheduled_tokens
        if req_id in scheduler.requests
    }


def _async_model_output(
    output: SchedulerOutput, prefill_flags: dict[str, bool], token: int = 100
) -> ModelRunnerOutput:
    """Mocked output: one sampled token per decode step, none for prefill chunks."""
    req_ids = list(output.num_scheduled_tokens.keys())
    sampled = [[token] if not prefill_flags.get(rid, False) else [] for rid in req_ids]
    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={rid: i for i, rid in enumerate(req_ids)},
        sampled_token_ids=sampled,
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )


def _run_async(
    scheduler: Scheduler, token: int = 100, depth: int = 2, max_iters: int = 2000
) -> None:
    """Prime a depth-``depth`` async pipeline over all requests, then drain it."""
    sched_outputs: deque[tuple[SchedulerOutput, dict[str, bool]]] = deque()
    for _ in range(depth):
        output = scheduler.schedule()
        if not output.num_scheduled_tokens:
            break
        sched_outputs.append((output, _capture_prefill(scheduler, output)))

    iters = 0
    while sched_outputs:
        iters += 1
        if iters > max_iters:
            raise AssertionError("async pipeline did not drain")
        output, flags = sched_outputs.popleft()
        scheduler.update_from_output(output, _async_model_output(output, flags, token))
        nxt = scheduler.schedule()
        if nxt.num_scheduled_tokens:
            sched_outputs.append((nxt, _capture_prefill(scheduler, nxt)))


def _run_sequential_async(
    scheduler: Scheduler, program_id: str, calls: list[tuple[str, int]]
) -> list[int]:
    """Add + fully complete each call in order under async; return priorities."""
    priorities: list[int] = []
    for i, (req_id, max_tokens) in enumerate(calls):
        request = make_request(
            req_id,
            program_id=program_id,
            max_tokens=max_tokens,
            arrival_time=float(i + 1),
        )
        scheduler.add_request(request)
        priorities.append(request.priority)
        _run_async(scheduler)
    return priorities


# --------------------------------------------------------------------------- #
# Gate: the scheduler is an AsyncScheduler subclass (async stays ENABLED)
# --------------------------------------------------------------------------- #


def test_atlas_is_async_scheduler_subclass():
    """ATLAS subclasses AsyncScheduler, so vLLM keeps async scheduling enabled."""
    assert issubclass(ATLASScheduler, AsyncScheduler)
    assert isinstance(create_atlas_scheduler(), AsyncScheduler)
    scheduler = create_atlas_scheduler(async_scheduling=True)
    assert hasattr(scheduler, "_spec_token_placeholders")
    assert scheduler.pp_size == 1


def test_async_scheduling_reserves_output_placeholders():
    """Running two steps ahead reserves placeholders -- proof async is ENABLED."""
    scheduler = create_atlas_scheduler(max_num_seqs=1, async_scheduling=True)
    call = make_request("p", program_id="P", max_tokens=10, arrival_time=1.0)
    scheduler.add_request(call)

    scheduler.schedule()
    assert call.num_output_placeholders == 1
    scheduler.schedule()
    assert call.num_output_placeholders == 2


# --------------------------------------------------------------------------- #
# Gate (async): max-update rule is exact across sequential calls under run-ahead
# --------------------------------------------------------------------------- #


def test_max_update_rule_under_async():
    """Sequential completions set max(prev, start + service) exactly under async."""
    scheduler = create_atlas_scheduler(max_num_seqs=1, async_scheduling=True)

    priorities = _run_sequential_async(
        scheduler, "P", [("c1", 3), ("c2", 2), ("c3", 4)]
    )
    # start 0 -> bin(0)=0; start 3 -> bin(3)=1; start 5 -> bin(5)=2.
    assert priorities == [0, 1, 2]
    # max(0,0+3)=3; max(3,3+2)=5; max(5,5+4)=9.
    assert scheduler.process_table.get("P").max_critical_path == 9.0


def test_concurrent_calls_use_max_not_sum_under_async():
    """Two concurrent siblings push the scalar to M + max(t_a, t_b) under async.

    The defining ATLAS-vs-PLAS divergence must survive run-ahead: program P sits
    at M = 4; siblings attain t_a = 5 and t_b = 2, so the max rule gives 9 -- not
    the PLAS sum 11.
    """
    scheduler = create_atlas_scheduler(max_num_seqs=4, async_scheduling=True)
    m = 4.0
    scheduler.process_table.get_or_create("P").max_critical_path = m

    a = make_request("a", program_id="P", max_tokens=5, arrival_time=1.0)
    b = make_request("b", program_id="P", max_tokens=2, arrival_time=2.0)
    scheduler.add_request(a)
    scheduler.add_request(b)
    assert a.priority == b.priority  # concurrent siblings grouped (both bin(4))

    _run_async(scheduler)  # run both siblings to completion under run-ahead

    max_cp = scheduler.process_table.get("P").max_critical_path
    assert max_cp == 9.0  # max(4, 4 + 5, 4 + 2)
    assert max_cp == m + max(5.0, 2.0)  # critical path
    assert max_cp != m + 5.0 + 2.0  # NOT the PLAS sum (would be 11)


def test_abort_folds_max_under_async():
    """A mid-flight abort under run-ahead folds the max rule exactly once."""
    scheduler = create_atlas_scheduler(max_num_seqs=1, async_scheduling=True)
    scheduler.process_table.get_or_create("AB").max_critical_path = 4.0
    call = make_request("ab", program_id="AB", max_tokens=50, arrival_time=1.0)
    scheduler.add_request(call)

    sched_outputs: deque[tuple[SchedulerOutput, dict[str, bool]]] = deque()
    for _ in range(2):
        output = scheduler.schedule()
        sched_outputs.append((output, _capture_prefill(scheduler, output)))
    output, flags = sched_outputs.popleft()
    scheduler.update_from_output(output, _async_model_output(output, flags))
    assert call.status == RequestStatus.RUNNING
    accrued = scheduler.attained_service.get("ab")
    assert accrued >= 1.0

    aborted = scheduler.finish_requests("ab", RequestStatus.FINISHED_ABORTED)
    assert ("ab", call.client_index) in aborted

    program = scheduler.process_table.get("AB")
    assert program is not None
    # max(4, start(4) + accrued) folded exactly once.
    assert program.max_critical_path == max(4.0, 4.0 + accrued)
    assert "ab" not in scheduler._req_to_pid
    assert "ab" not in scheduler._req_to_start_scalar
    assert scheduler.attained_service.get("ab") == 0.0
    assert "ab" not in program.active_call_ids
