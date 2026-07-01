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
from vllm.v1.core.sched.autellix.plas_scheduler import PLASScheduler
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
) -> PLASScheduler:
    """Build a ``PLASScheduler`` with the stock (FCFS) config default.

    Leaving ``policy`` at its ``"fcfs"`` default proves the scheduler
    self-configures priority ordering on construction.
    """
    vllm_config = _make_vllm_config(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        num_blocks=num_blocks,
        block_size=block_size,
        policy="fcfs",
        long_prefill_token_threshold=long_prefill_token_threshold,
        max_model_len=max_model_len,
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


def _step(scheduler: Scheduler, token: int = 100) -> "object":
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
# Gate: priority stability across a call's lifetime
# --------------------------------------------------------------------------- #


def test_call_priority_is_stable_through_lifetime():
    """A call's priority is fixed at arrival and never changes mid-call."""
    scheduler = create_plas_scheduler(max_num_seqs=1)
    scheduler.process_table.get_or_create("Q").service = 5.0

    call = make_request("q_call", program_id="Q", max_tokens=6, arrival_time=1.0)
    scheduler.add_request(call)
    initial_priority = call.priority
    assert initial_priority == 2  # bin(5)

    seen = set()
    for _ in range(50):
        if "q_call" not in scheduler.requests:
            break
        _step(scheduler)
        seen.add(call.priority)
    assert seen == {initial_priority}, "priority must not change mid-call"
    assert call.priority == initial_priority


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
