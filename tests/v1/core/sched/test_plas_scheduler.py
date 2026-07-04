# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Behavioural tests for the program-aware LAS scheduler (PLAS).

These exercise the real vLLM v1 scheduler stack (via ``create_scheduler``-style
config construction) composed with the Phase-0 policy core, so they live in the
standard test tree rather than the shielded pure-core subtree. Each test drives
the scheduler with mocked ``ModelRunnerOutput`` steps, exactly like
``tests/v1/core/test_scheduler.py``.
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


def create_plas_scheduler(
    max_num_seqs: int = 16,
    max_num_batched_tokens: int = 8192,
    num_blocks: int = 10000,
    block_size: int = 16,
    long_prefill_token_threshold: int = 0,
    max_model_len: int | None = None,
    async_scheduling: bool = False,
) -> PLASScheduler:
    """Build a ``PLASScheduler`` with the stock (FCFS) config default.

    Leaving ``policy`` at its ``"fcfs"`` default proves the scheduler
    self-configures priority ordering on construction. ``async_scheduling`` toggles
    ``SchedulerConfig.async_scheduling`` so the ``AsyncScheduler`` run-ahead code
    path (output placeholders) is active.
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
    return PLASScheduler(
        vllm_config=vllm_config,
        kv_cache_config=_make_kv_cache_config(num_blocks, block_size),
        log_stats=True,
        structured_output_manager=StructuredOutputManager(vllm_config),
        block_size=block_size,
        hash_block_size=block_size,
    )


def create_fcfs_scheduler(
    max_num_seqs: int = 16,
    max_num_batched_tokens: int = 8192,
    num_blocks: int = 10000,
    block_size: int = 16,
) -> Scheduler:
    """Build a stock FCFS ``Scheduler`` for behavioural contrast."""
    vllm_config = _make_vllm_config(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        num_blocks=num_blocks,
        block_size=block_size,
        policy="fcfs",
    )
    return Scheduler(
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
    num_tokens: int = 4,
    max_tokens: int = 16,
    arrival_time: float | None = None,
    with_extra_args: bool = True,
    block_size: int = 16,
) -> Request:
    """Build a ``Request`` carrying an optional ``program_id`` in extra_args."""
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


# --------------------------------------------------------------------------- #
# Self-configuration
# --------------------------------------------------------------------------- #


def test_self_configures_priority_ordering():
    """__init__ flips ordering to PRIORITY and builds the D5 core."""
    scheduler = create_plas_scheduler()

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
# Gate: least-attained-service ordering
# --------------------------------------------------------------------------- #


def test_las_admits_least_served_program_first():
    """B (zero service) is admitted before A (high service), unlike FCFS."""
    scheduler = create_plas_scheduler(max_num_seqs=1)
    # Program A has accrued a lot of prior service; B is brand new.
    scheduler.process_table.get_or_create("A").service = 100.0

    call_a = make_request("a1", program_id="A", arrival_time=1.0)
    call_b = make_request("b1", program_id="B", arrival_time=2.0)
    scheduler.add_request(call_a)  # arrives first
    scheduler.add_request(call_b)  # arrives later

    assert call_a.priority == 6  # bin(100)
    assert call_b.priority == 0  # bin(0)

    output = scheduler.schedule()
    scheduled = [r.req_id for r in output.scheduled_new_reqs]
    assert scheduled == ["b1"], "PLAS must admit the least-served program first"
    assert [r.request_id for r in scheduler.running] == ["b1"]

    # FCFS contrast: same arrival order → the earlier arrival (A) wins.
    fcfs = create_fcfs_scheduler(max_num_seqs=1)
    fcfs.add_request(make_request("a1", program_id="A", arrival_time=1.0))
    fcfs.add_request(make_request("b1", program_id="B", arrival_time=2.0))
    fcfs_output = fcfs.schedule()
    fcfs_scheduled = [r.req_id for r in fcfs_output.scheduled_new_reqs]
    assert fcfs_scheduled == ["a1"], "FCFS would admit the earlier arrival first"


def test_interleaved_programs_reorder_vs_fcfs():
    """Two heavy + two light calls: PLAS runs the light program's calls."""
    scheduler = create_plas_scheduler(max_num_seqs=2)
    scheduler.process_table.get_or_create("HEAVY").service = 100.0

    reqs = [
        ("h1", "HEAVY", 1.0),
        ("l1", "LIGHT", 2.0),
        ("h2", "HEAVY", 3.0),
        ("l2", "LIGHT", 4.0),
    ]
    for rid, pid, t in reqs:
        scheduler.add_request(make_request(rid, program_id=pid, arrival_time=t))

    output = scheduler.schedule()
    admitted = {r.req_id for r in output.scheduled_new_reqs}
    assert admitted == {"l1", "l2"}, "PLAS prioritises the least-served program"

    # FCFS admits by arrival order: h1 then l1.
    fcfs = create_fcfs_scheduler(max_num_seqs=2)
    for rid, pid, t in reqs:
        fcfs.add_request(make_request(rid, program_id=pid, arrival_time=t))
    fcfs_admitted = {r.req_id for r in fcfs.schedule().scheduled_new_reqs}
    assert fcfs_admitted == {"h1", "l1"}
    assert fcfs_admitted != admitted


# --------------------------------------------------------------------------- #
# Gate: service accrual (sum rule) + decode-step proxy
# --------------------------------------------------------------------------- #


def test_service_accrues_with_sum_rule_on_completion():
    """A program's service grows by the decode-step proxy on each completion."""
    scheduler = create_plas_scheduler(max_num_seqs=1)

    # First call: 3 output tokens => 3 decode steps of service.
    call1 = make_request("p_call1", program_id="P", max_tokens=3, arrival_time=1.0)
    scheduler.add_request(call1)
    assert call1.priority == 0  # fresh program => bin(0)
    _run_to_completion(scheduler, "p_call1")

    state = scheduler.process_table.get("P")
    assert state is not None
    assert state.service == 3.0
    assert call1.num_output_tokens == 3  # proxy == decode tokens generated

    # Second call inherits the program's accrued service as its start priority.
    call2 = make_request("p_call2", program_id="P", max_tokens=2, arrival_time=2.0)
    scheduler.add_request(call2)
    assert call2.priority == 1  # bin(3) => queue 1 (thresholds cross 2)
    _run_to_completion(scheduler, "p_call2")

    # Sum rule: 3 + 2 == 5.
    assert scheduler.process_table.get("P").service == 5.0


# --------------------------------------------------------------------------- #
# Gate: quantum demotion (Algorithm 1 lines 21-24) layered on the arrival bin
# --------------------------------------------------------------------------- #


def test_fresh_call_demotes_one_level_after_each_queue_quantum():
    """A fresh call walks down the queue ladder as it exhausts each quantum."""
    scheduler = create_plas_scheduler(max_num_seqs=1)
    call = make_request("d1", program_id="D", max_tokens=200, arrival_time=1.0)
    scheduler.add_request(call)
    assert call.priority == 0  # fresh program => bin(0)

    # Queue 0 quantum == 1: a single decode step demotes to queue 1.
    _step(scheduler)
    assert call.priority == 1
    assert scheduler._call_state["d1"].quantum_remaining == 2  # queue_quanta[1]

    # Queue 1 quantum == 2: two decode steps demote to queue 2.
    _step(scheduler)
    assert call.priority == 1
    _step(scheduler)
    assert call.priority == 2
    assert scheduler._call_state["d1"].quantum_remaining == 4  # queue_quanta[2]


def test_call_starts_at_arrival_bin_with_that_bins_quantum():
    """The arrival bin (program service) sets both the level and the quantum."""
    scheduler = create_plas_scheduler(max_num_seqs=1)
    scheduler.process_table.get_or_create("Q").service = 5.0

    call = make_request("q_call", program_id="Q", max_tokens=200, arrival_time=1.0)
    scheduler.add_request(call)
    assert call.priority == 2  # bin(5)
    assert scheduler._call_state["q_call"].quantum_remaining == 4  # queue_quanta[2]

    # Queue 2 quantum == 4: four decode steps demote to queue 3.
    for _ in range(4):
        _step(scheduler)
    assert call.priority == 3
    assert scheduler._call_state["q_call"].quantum_remaining == 8  # queue_quanta[3]


def test_demotion_saturates_at_bottom_queue():
    """Demotion never moves a call past the last queue (index K-1)."""
    scheduler = create_plas_scheduler(max_num_seqs=1)
    call = make_request("s1", program_id="S", max_tokens=300, arrival_time=1.0)
    scheduler.add_request(call)

    # 1 + 2 + 4 + 8 + 16 + 32 == 63 decode steps reach the bottom queue.
    for _ in range(63):
        _step(scheduler)
    assert call.priority == 6
    for _ in range(64):  # exhausting the bottom quantum keeps it there
        _step(scheduler)
    assert call.priority == 6


# --------------------------------------------------------------------------- #
# Gate: KV-pressure preemption victim is the most-served program's call
# --------------------------------------------------------------------------- #


def test_kv_pressure_preempts_most_served_program_call():
    """Under block pressure the victim is the most-served program's call.

    Mirrors the block math of ``test_priority_scheduling_preemption`` but the
    priorities come from program service, not manual assignment.
    """
    block_size = 16
    num_blocks = 6  # 1 null block => 5 usable
    num_tokens = block_size * 2  # exactly 2 blocks

    scheduler = create_plas_scheduler(
        max_num_seqs=3,
        max_num_batched_tokens=200,
        num_blocks=num_blocks,
        block_size=block_size,
    )
    # SERVED program is heavily served => high priority value (low priority).
    scheduler.process_table.get_or_create("SERVED").service = 100.0

    served = make_request(
        "served",
        program_id="SERVED",
        num_tokens=num_tokens,
        arrival_time=1.0,
        block_size=block_size,
    )
    scheduler.add_request(served)
    assert served.priority == 6
    _step(scheduler)  # served prefills (2 blocks), decodes to 33 tokens

    fresh = make_request(
        "fresh",
        program_id="FRESH",
        num_tokens=num_tokens,
        arrival_time=2.0,
        block_size=block_size,
    )
    scheduler.add_request(fresh)
    assert fresh.priority == 0
    # served gets its 3rd block, fresh admitted => 5 used, 0 free.
    output = scheduler.schedule()
    assert any(r.req_id == "fresh" for r in output.scheduled_new_reqs)
    assert len(scheduler.running) == 2
    model_output = ModelRunnerOutput(
        req_ids=["served", "fresh"],
        req_id_to_index={"served": 0, "fresh": 1},
        sampled_token_ids=[[101], [100]],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    scheduler.update_from_output(output, model_output)

    # fresh needs a 3rd block, 0 free => preempt the most-served program's call.
    scheduler.schedule()
    assert scheduler.requests["served"].status == RequestStatus.PREEMPTED
    assert any(r.request_id == "fresh" for r in scheduler.running)


# --------------------------------------------------------------------------- #
# Gate: missing program_id degrades gracefully to per-request (~FCFS)
# --------------------------------------------------------------------------- #


def test_missing_program_id_does_not_crash_and_is_per_request():
    """Requests without a program_id fall back to per-request tracking."""
    scheduler = create_plas_scheduler(max_num_seqs=1)

    # extra_args is None entirely.
    no_args = make_request(
        "solo", program_id=None, with_extra_args=False, max_tokens=2, arrival_time=1.0
    )
    assert no_args.sampling_params.extra_args is None
    scheduler.add_request(no_args)
    assert no_args.priority == 0  # fresh per-request program => bin(0)
    _run_to_completion(scheduler, "solo")
    # Folded under the request id acting as its own program.
    solo_state = scheduler.process_table.get("solo")
    assert solo_state is not None
    assert solo_state.service == 2.0

    # extra_args present but without a program_id key.
    partial = make_request(
        "solo2", program_id=None, with_extra_args=True, max_tokens=2, arrival_time=2.0
    )
    assert partial.sampling_params.extra_args == {}
    scheduler.add_request(partial)
    assert partial.priority == 0
    _run_to_completion(scheduler, "solo2")
    assert scheduler.process_table.get("solo2").service == 2.0


# --------------------------------------------------------------------------- #
# Program table hygiene
# --------------------------------------------------------------------------- #


def test_completed_call_is_deregistered_from_program():
    """On completion the call leaves the program's active set + req map."""
    scheduler = create_plas_scheduler(max_num_seqs=1)
    call = make_request("r_call", program_id="R", max_tokens=2, arrival_time=1.0)
    scheduler.add_request(call)
    assert "r_call" in scheduler.process_table.get("R").active_call_ids
    _run_to_completion(scheduler, "r_call")
    assert scheduler.process_table.get("R").active_call_ids == set()
    assert "r_call" not in scheduler._req_to_pid


# --------------------------------------------------------------------------- #
# Gate: wait accrual — a call's waited steps fold into the program's W_p
# --------------------------------------------------------------------------- #


def test_waiting_steps_fold_into_program_total_wait_on_completion():
    """A call's accrued wait window folds into ProcessTable.total_wait (W_p)."""
    scheduler = create_plas_scheduler(max_num_seqs=1)
    # Heavily-served waiter (bin 6): it cannot outrank or promote past the
    # fresh blocker within this test's horizon, so its wait is deterministic.
    scheduler.process_table.get_or_create("W").service = 100.0

    blocker = make_request("blk", program_id="B", max_tokens=5, arrival_time=1.0)
    waiter = make_request("w1", program_id="W", max_tokens=2, arrival_time=2.0)
    scheduler.add_request(blocker)
    scheduler.add_request(waiter)

    # Blocker runs 5 decode steps (steps 1-5); the waiter accrues one unit of
    # wait per schedule() pass while queued, including the pass that admits it.
    for _ in range(5):
        _step(scheduler)
    assert "blk" not in scheduler.requests
    assert scheduler._call_state["w1"].wait_window == 5

    _run_to_completion(scheduler, "w1")

    assert scheduler.process_table.get("W").total_wait == 6.0  # 5 + admission pass


def test_waiting_steps_fold_into_program_total_wait_on_abort():
    """An aborted waiting call folds its wait window into W_p (no leak)."""
    scheduler = create_plas_scheduler(max_num_seqs=1)
    scheduler.process_table.get_or_create("W").service = 100.0

    blocker = make_request("blk", program_id="B", max_tokens=50, arrival_time=1.0)
    waiter = make_request("w1", program_id="W", max_tokens=2, arrival_time=2.0)
    scheduler.add_request(blocker)
    scheduler.add_request(waiter)

    for _ in range(3):
        _step(scheduler)
    assert scheduler._call_state["w1"].wait_window == 3

    scheduler.finish_requests("w1", RequestStatus.FINISHED_ABORTED)

    assert scheduler.process_table.get("W").total_wait == 3.0
    assert "w1" not in scheduler._call_state


# --------------------------------------------------------------------------- #
# Gate: program-level anti-starvation promotion (Algorithm 1 lines 25-31)
# --------------------------------------------------------------------------- #


def test_promotion_uses_program_level_windows():
    """Promotion fires on (W_p + W_c) / max(1, T_p + T_c) >= beta, not W_c/T_c.

    With W_p=20, T_p=3 and beta=8 the ratio crosses at W_c=4
    ((20+4)/3 == 8), well before the call-level-only threshold (W_c=8 for
    a zero-service call) -- proving the program windows are consulted.
    """
    scheduler = create_plas_scheduler(max_num_seqs=1)
    program = scheduler.process_table.get_or_create("P")
    program.service = 3.0  # T_p
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
    assert scheduler.waiting.peek_request().request_id == "p1"


def test_promotion_resets_only_call_level_windows():
    """Promotion resets W_c/T_c but leaves the program's W_p/T_p intact.

    Resetting the program windows would immediately re-starve the program's
    other calls (paper §4.2.2); they must persist so sibling calls promote
    together.
    """
    scheduler = create_plas_scheduler(max_num_seqs=1)
    program = scheduler.process_table.get_or_create("P")
    program.service = 3.0
    program.total_wait = 20.0

    blocker = make_request("blk", program_id="B", max_tokens=500, arrival_time=0.0)
    scheduler.add_request(blocker)
    scheduler.schedule()
    call = make_request("p1", program_id="P", max_tokens=2, arrival_time=1.0)
    scheduler.add_request(call)

    for _ in range(4):  # drive past the beta threshold (see companion test)
        scheduler._accrue_wait_and_promote()
    state = scheduler._call_state["p1"]
    assert call.priority == 0

    # Call-level windows reset...
    assert state.queue_index == 0
    assert state.wait_window == 0
    assert state.service_window == 0
    assert state.quantum_remaining == scheduler.queue_quanta[0]
    # ...program-level windows persist.
    assert program.total_wait == 20.0
    assert program.service == 3.0


# --------------------------------------------------------------------------- #
# Gate: externally-aborted calls are folded, not leaked
# --------------------------------------------------------------------------- #


def test_abort_mid_flight_folds_service_and_deregisters():
    """A call aborted mid-flight folds its service and frees all its state.

    Aborts arrive via ``finish_requests`` (client disconnect / timeout), which
    bypasses ``update_from_output`` -- the next ``schedule()`` resets
    ``finished_req_ids`` before the fold would run. Without a dedicated hook the
    program state, req->pid entry, and attained-service entry leak permanently
    (``active_call_ids`` keeps the aborted id, so the program is never GC'd) and
    the dropped service wrongly lowers the program's later calls' priority.
    """
    scheduler = create_plas_scheduler(max_num_seqs=1)
    call = make_request("ab_call", program_id="AB", max_tokens=50, arrival_time=1.0)
    scheduler.add_request(call)

    # Accrue three decode steps of service, then abort mid-flight.
    for _ in range(3):
        _step(scheduler)
    assert scheduler.attained_service.get("ab_call") == 3.0
    assert call.status == RequestStatus.RUNNING

    aborted = scheduler.finish_requests("ab_call", RequestStatus.FINISHED_ABORTED)
    assert ("ab_call", call.client_index) in aborted

    program = scheduler.process_table.get("AB")
    assert program is not None
    # (a) accrued service folded into the program.
    assert program.service == 3.0
    # (b) req->pid map cleaned up.
    assert "ab_call" not in scheduler._req_to_pid
    # (c) attained-service entry popped.
    assert scheduler.attained_service.get("ab_call") == 0.0
    # (d) call removed from the program's active set.
    assert "ab_call" not in program.active_call_ids
    # (e) the now-idle program is GC-eligible (previously un-evictable).
    evicted = scheduler.process_table.gc(time.time() + 1e9)
    assert "AB" in evicted
    assert scheduler.process_table.get("AB") is None


def test_normal_completion_then_abort_does_not_double_fold():
    """Folding is idempotent: a normal completion is never re-folded by abort."""
    scheduler = create_plas_scheduler(max_num_seqs=1)
    call = make_request("dc_call", program_id="DC", max_tokens=3, arrival_time=1.0)
    scheduler.add_request(call)
    _run_to_completion(scheduler, "dc_call")
    assert scheduler.process_table.get("DC").service == 3.0

    # A late abort of the already-finished id is a no-op (the base skips it),
    # so its service is not double-counted.
    aborted = scheduler.finish_requests("dc_call", RequestStatus.FINISHED_ABORTED)
    assert aborted == []
    assert scheduler.process_table.get("DC").service == 3.0

    # Folding the same id a second time is guarded by the popped req->pid entry,
    # so no service is added on the repeat (idempotency).
    scheduler._fold_completed_calls(["dc_call"], time.time())
    assert scheduler.process_table.get("DC").service == 3.0


# --------------------------------------------------------------------------- #
# Decode-step proxy under chunked prefill
# --------------------------------------------------------------------------- #


def test_chunked_prefill_accrues_only_on_decode_steps():
    """Non-final prefill chunks accrue no service; decode steps each add one."""
    scheduler = create_plas_scheduler(
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

    # Finish decoding; total service excludes the non-final prefill chunks.
    _run_to_completion(scheduler, "cp")
    assert scheduler.process_table.get("CP").service == 3.0
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
    """Prime a depth-``depth`` async pipeline over all requests, then drain it.

    Scheduling ``depth`` steps before delivering any output forces the scheduler
    to reserve ``num_output_placeholders`` and run ahead; the drain then delivers
    each in-flight step and schedules one more until the pipeline empties.
    """
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


# --------------------------------------------------------------------------- #
# Gate: the scheduler is an AsyncScheduler subclass (async stays ENABLED)
# --------------------------------------------------------------------------- #


def test_plas_is_async_scheduler_subclass():
    """PLAS subclasses AsyncScheduler, so vLLM keeps async scheduling enabled.

    A ``Scheduler`` (non-async) subclass triggers the documented
    ``config/scheduler.py`` degradation path ("async scheduling being disabled").
    """
    assert issubclass(PLASScheduler, AsyncScheduler)
    assert isinstance(create_plas_scheduler(), AsyncScheduler)
    # Async-only bookkeeping the base initialiser must have set up.
    scheduler = create_plas_scheduler(async_scheduling=True)
    assert hasattr(scheduler, "_spec_token_placeholders")
    assert scheduler.pp_size == 1


def test_async_scheduling_reserves_output_placeholders():
    """Running two steps ahead reserves placeholders -- proof async is ENABLED.

    Under a plain ``Scheduler`` base no placeholders are reserved and the second
    schedule cannot run ahead (it would find nothing to schedule for the
    in-flight request), so this is the definitive async-vs-degraded check.
    """
    scheduler = create_plas_scheduler(max_num_seqs=1, async_scheduling=True)
    call = make_request("p", program_id="P", max_tokens=10, arrival_time=1.0)
    scheduler.add_request(call)

    scheduler.schedule()  # prefill + reserve the first decode's output
    assert call.num_output_placeholders == 1
    scheduler.schedule()  # genuinely run one decode step ahead
    assert call.num_output_placeholders == 2
    assert call.num_computed_tokens == call.num_prompt_tokens + 1


# --------------------------------------------------------------------------- #
# Gate (async): sum-rule accrual + start-priority inheritance under run-ahead
# --------------------------------------------------------------------------- #


def test_service_accrues_with_sum_rule_under_async():
    """The decode-step proxy and sum rule are exact under async run-ahead.

    The base async over-schedule guard means #(non-prefill decode steps) still
    equals max_tokens, so accrual is unchanged by pipelining; the fold happens in
    the same ``update_from_output`` that delivers the stopping token, so it is
    neither missed nor doubled.
    """
    scheduler = create_plas_scheduler(max_num_seqs=1, async_scheduling=True)

    call1 = make_request("p_call1", program_id="P", max_tokens=3, arrival_time=1.0)
    scheduler.add_request(call1)
    assert call1.priority == 0  # fresh program => bin(0)
    _run_async(scheduler)
    assert call1.num_output_tokens == 3
    assert scheduler.process_table.get("P").service == 3.0

    # Second call inherits the folded service as its start priority.
    call2 = make_request("p_call2", program_id="P", max_tokens=2, arrival_time=2.0)
    scheduler.add_request(call2)
    assert call2.priority == 1  # bin(3) => queue 1
    _run_async(scheduler)
    assert scheduler.process_table.get("P").service == 5.0  # sum rule 3 + 2


def test_las_admits_least_served_first_under_async():
    """LAS ordering (least-served program first) holds with async enabled."""
    scheduler = create_plas_scheduler(max_num_seqs=1, async_scheduling=True)
    scheduler.process_table.get_or_create("A").service = 100.0

    scheduler.add_request(make_request("a1", program_id="A", arrival_time=1.0))
    scheduler.add_request(make_request("b1", program_id="B", arrival_time=2.0))

    output = scheduler.schedule()
    scheduled = [r.req_id for r in output.scheduled_new_reqs]
    assert scheduled == ["b1"], "PLAS must admit the least-served program first"


def test_abort_mid_flight_folds_service_under_async():
    """A mid-flight abort under run-ahead folds service exactly once, no leak."""
    scheduler = create_plas_scheduler(max_num_seqs=1, async_scheduling=True)
    call = make_request("ab", program_id="AB", max_tokens=50, arrival_time=1.0)
    scheduler.add_request(call)

    # Prime a depth-2 pipeline and deliver a couple of steps (run-ahead active).
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
    assert program.service == accrued  # folded exactly once on abort
    assert "ab" not in scheduler._req_to_pid
    assert scheduler.attained_service.get("ab") == 0.0
    assert "ab" not in program.active_call_ids
