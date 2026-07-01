# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Behavioural tests for the program-agnostic MLFQ scheduler.

These exercise the real vLLM v1 scheduler stack (mirroring
``tests/v1/core/utils.py::create_scheduler``) composed with the Phase-0 policy
core, so they live in the standard test tree rather than the shielded pure-core
subtree. Each test drives the scheduler with mocked ``ModelRunnerOutput`` steps,
exactly like ``tests/v1/core/test_scheduler.py`` and the PLAS tests.

The MLFQ policy is FastServe-style: every new call starts in the top queue Q1
(``priority == 0``); a call is demoted one level once it exhausts its queue's
decode-step quantum; a starving waiting call is promoted back to Q1; and a
higher-priority waiting call proactively preempts the worst running call when the
batch is full. See ``notes/autellix_scheduling/POLICY_REFERENCE.md`` §3-4.
"""

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
from vllm.v1.core.sched.autellix.mlfq_scheduler import MLFQScheduler
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


def create_mlfq_scheduler(
    max_num_seqs: int = 16,
    max_num_batched_tokens: int = 8192,
    num_blocks: int = 10000,
    block_size: int = 16,
    long_prefill_token_threshold: int = 0,
    max_model_len: int | None = None,
    num_queues: int = 7,
    queue_quanta: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64),
    beta: float = 8.0,
    max_proactive_preemptions_per_step: int = 1,
) -> MLFQScheduler:
    """Build an ``MLFQScheduler`` with the stock (FCFS) config default.

    Leaving ``policy`` at its ``"fcfs"`` default proves the scheduler
    self-configures priority ordering on construction. The MLFQ constants are
    forwarded so tests can exercise the tunable-constant (ablation) path.
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
    return MLFQScheduler(
        vllm_config=vllm_config,
        kv_cache_config=_make_kv_cache_config(num_blocks, block_size),
        log_stats=True,
        structured_output_manager=StructuredOutputManager(vllm_config),
        block_size=block_size,
        hash_block_size=block_size,
        num_queues=num_queues,
        queue_quanta=queue_quanta,
        beta=beta,
        max_proactive_preemptions_per_step=max_proactive_preemptions_per_step,
    )


def make_request(
    req_id: str,
    num_tokens: int = 4,
    max_tokens: int = 16,
    arrival_time: float | None = None,
    block_size: int = 16,
    resumable: bool = False,
) -> Request:
    """Build a decode-only ``Request`` (MLFQ is program-agnostic)."""
    global _none_hash_initialized
    if not _none_hash_initialized:
        init_none_hash(sha256)
        _none_hash_initialized = True
    block_hasher = get_request_block_hasher(block_size, sha256)

    sampling_params = SamplingParams(ignore_eos=True, max_tokens=max_tokens)
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
        resumable=resumable,
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
    scheduler: Scheduler, req_id: str, token: int = 100, max_steps: int = 400
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
# Self-configuration + sourced constants (POLICY_REFERENCE §4, D5)
# --------------------------------------------------------------------------- #


def test_self_configures_priority_and_exposes_d5_constants():
    """__init__ flips ordering to PRIORITY and exposes the tunable D5 constants."""
    scheduler = create_mlfq_scheduler()

    assert scheduler.policy == SchedulingPolicy.PRIORITY
    assert isinstance(scheduler.waiting, PriorityRequestQueue)
    assert isinstance(scheduler.skipped_waiting, PriorityRequestQueue)

    # D5 / FastServe geometric defaults (POLICY_REFERENCE.md §4).
    assert scheduler.num_queues == 7
    assert scheduler.queue_quanta == (1, 2, 4, 8, 16, 32, 64)
    assert scheduler.beta == 8.0
    assert len(scheduler.queue_quanta) == scheduler.num_queues


def test_constants_are_tunable_for_ablation():
    """K / quanta / beta are constructor params (a beta/K ablation is possible)."""
    scheduler = create_mlfq_scheduler(num_queues=3, queue_quanta=(2, 4, 8), beta=4.0)
    assert scheduler.num_queues == 3
    assert scheduler.queue_quanta == (2, 4, 8)
    assert scheduler.beta == 4.0


def test_mismatched_quanta_length_is_rejected():
    """A quanta tuple whose length != num_queues is a construction error."""
    with pytest.raises(ValueError):
        create_mlfq_scheduler(num_queues=7, queue_quanta=(1, 2, 4))


# --------------------------------------------------------------------------- #
# Gate: every new call starts in the top queue Q1 (priority 0)
# --------------------------------------------------------------------------- #


def test_new_call_starts_in_top_queue():
    """A newly-arrived call is placed in Q1 with the top queue's quantum."""
    scheduler = create_mlfq_scheduler()
    call = make_request("c1", arrival_time=1.0)
    scheduler.add_request(call)

    assert call.priority == 0
    state = scheduler._call_state["c1"]
    assert state.queue_index == 0
    assert state.quantum_remaining == 1  # queue_quanta[0]
    assert state.wait_window == 0
    assert state.service_window == 0


def test_second_new_call_also_starts_in_top_queue():
    """MLFQ is program-agnostic: prior calls do not bias a new call's queue."""
    scheduler = create_mlfq_scheduler(max_num_seqs=1)
    first = make_request("first", max_tokens=3, arrival_time=1.0)
    scheduler.add_request(first)
    _run_to_completion(scheduler, "first")

    second = make_request("second", max_tokens=3, arrival_time=2.0)
    scheduler.add_request(second)
    assert second.priority == 0  # unaffected by the earlier, now-finished call


def test_readd_of_tracked_call_preserves_mlfq_state():
    """A streaming re-add of a tracked call does not reset its queue to Q1.

    A streaming continuation re-enters ``add_request`` for an id already in
    ``self.requests``; the guard must skip state re-initialisation so a
    mid-flight, demoted call keeps its queue instead of jumping back to Q1.
    """
    scheduler = create_mlfq_scheduler(max_num_seqs=1)
    call = make_request("stream", max_tokens=200, arrival_time=1.0, resumable=True)
    scheduler.add_request(call)
    # Demote the tracked call so a spurious reset would be observable.
    scheduler._call_state["stream"].queue_index = 4
    scheduler._call_state["stream"].quantum_remaining = 3
    call.priority = 4

    continuation = make_request(
        "stream", max_tokens=200, arrival_time=1.0, resumable=True
    )
    scheduler.add_request(continuation)

    assert scheduler._call_state["stream"].queue_index == 4  # preserved, not reset
    assert scheduler._call_state["stream"].quantum_remaining == 3
    assert call.priority == 4


# --------------------------------------------------------------------------- #
# Gate: demotion one level per queue quantum (priority increments, quantum resets)
# --------------------------------------------------------------------------- #


def test_demotes_one_level_after_each_queue_quantum():
    """After queue i's quantum decode steps, the call demotes to queue i+1."""
    scheduler = create_mlfq_scheduler(max_num_seqs=1)
    call = make_request("d1", num_tokens=4, max_tokens=200, arrival_time=1.0)
    scheduler.add_request(call)
    assert call.priority == 0

    # Queue 0 quantum == 1: a single decode step demotes to queue 1.
    _step(scheduler)
    assert call.priority == 1
    assert scheduler._call_state["d1"].quantum_remaining == 2  # queue_quanta[1]
    assert scheduler._call_state["d1"].service_window == 1

    # Queue 1 quantum == 2: two decode steps demote to queue 2.
    _step(scheduler)
    assert call.priority == 1
    _step(scheduler)
    assert call.priority == 2
    assert scheduler._call_state["d1"].quantum_remaining == 4  # queue_quanta[2]

    # Queue 2 quantum == 4: four decode steps demote to queue 3.
    for _ in range(4):
        _step(scheduler)
    assert call.priority == 3
    assert scheduler._call_state["d1"].quantum_remaining == 8  # queue_quanta[3]
    assert scheduler._call_state["d1"].service_window == 7  # 1 + 2 + 4


def test_demotion_saturates_at_bottom_queue():
    """Demotion never moves a call past the last queue (index K-1)."""
    scheduler = create_mlfq_scheduler(max_num_seqs=1)
    call = make_request("s1", num_tokens=4, max_tokens=300, arrival_time=1.0)
    scheduler.add_request(call)

    # 1 + 2 + 4 + 8 + 16 + 32 == 63 decode steps reach the bottom queue (index 6).
    for _ in range(63):
        _step(scheduler)
    assert call.priority == 6
    assert scheduler._call_state["s1"].quantum_remaining == 64  # queue_quanta[6]

    # Exhausting the bottom quantum keeps the call in the last queue.
    for _ in range(64):
        _step(scheduler)
    assert call.priority == 6
    assert scheduler._call_state["s1"].quantum_remaining == 64


def test_prefill_chunks_do_not_consume_quantum():
    """Non-final prefill chunks accrue no service and never demote."""
    scheduler = create_mlfq_scheduler(
        max_num_seqs=1,
        long_prefill_token_threshold=8,  # cap prefill at 8 tokens/step
        max_model_len=256,
    )
    call = make_request("cp", num_tokens=20, max_tokens=3, arrival_time=1.0)
    scheduler.add_request(call)

    prefill_chunk_steps = 0
    while True:
        _step(scheduler)
        state = scheduler._call_state["cp"]
        if scheduler.requests["cp"].is_prefill_chunk:
            prefill_chunk_steps += 1
            assert state.service_window == 0
            assert call.priority == 0  # still in the top queue
            assert state.quantum_remaining == 1
        else:
            break
    assert prefill_chunk_steps >= 1, "prompt was not chunked across steps"
    # The step that finishes prefill (first decode) consumes the Q0 quantum.
    assert scheduler._call_state["cp"].service_window == 1
    assert call.priority == 1


# --------------------------------------------------------------------------- #
# Gate: anti-starvation promotion (W / max(1, T) >= beta) resets windows to Q1
# --------------------------------------------------------------------------- #


def test_starving_waiting_call_is_promoted_to_top_queue():
    """A waiting call crossing the W/T ratio is promoted to Q1, windows reset.

    The promotion bookkeeping is invoked directly so the threshold boundary
    (below vs at ``beta``) is asserted deterministically, isolated from the
    admission loop.
    """
    scheduler = create_mlfq_scheduler(max_num_seqs=1, beta=8.0)
    call = make_request("L", arrival_time=1.0)
    scheduler.add_request(call)

    # Simulate that L already ran, demoted to queue 2, and accrued 3 service.
    state = scheduler._call_state["L"]
    state.queue_index = 2
    state.service_window = 3
    state.wait_window = 0
    call.priority = 2

    # W / max(1, T) = 23 / 3 = 7.67 < 8: not yet promoted.
    for _ in range(23):
        scheduler._accrue_wait_and_promote()
    assert call.priority == 2
    assert state.wait_window == 23

    # W / max(1, T) = 24 / 3 = 8.0 >= 8: promoted to Q1, windows reset.
    scheduler._accrue_wait_and_promote()
    assert call.priority == 0
    assert state.queue_index == 0
    assert state.wait_window == 0
    assert state.service_window == 0
    assert state.quantum_remaining == scheduler.queue_quanta[0]


def test_top_queue_call_is_not_repromoted():
    """A call already in Q1 accrues wait but is never re-promoted (no reset)."""
    scheduler = create_mlfq_scheduler()
    call = make_request("q0", arrival_time=1.0)
    scheduler.add_request(call)
    state = scheduler._call_state["q0"]

    for _ in range(100):
        scheduler._accrue_wait_and_promote()

    assert call.priority == 0
    assert state.queue_index == 0
    assert state.wait_window == 100  # accrued, never reset while already top-queue


def test_promotion_reorders_the_waiting_heap():
    """After promotion the waiting heap yields the promoted call first."""
    scheduler = create_mlfq_scheduler(max_num_seqs=1)
    # A blocker occupies the single running slot so both others stay waiting.
    blocker = make_request("blk", max_tokens=500, arrival_time=0.0)
    scheduler.add_request(blocker)
    scheduler.schedule()  # admit the blocker
    assert [r.request_id for r in scheduler.running] == ["blk"]

    low = make_request("low", arrival_time=1.0)
    high_later = make_request("late", arrival_time=2.0)
    scheduler.add_request(low)
    scheduler.add_request(high_later)
    # Make `low` a demoted, starving waiting call.
    state = scheduler._call_state["low"]
    state.queue_index = 3
    state.service_window = 1
    low.priority = 3

    # Drive promotion of `low` (W/max(1,1) >= 8 after 8 passes).
    for _ in range(8):
        scheduler._accrue_wait_and_promote()
    assert low.priority == 0
    # Heap order now reflects the new priority: `low` (Q1) outranks `late` (Q1,
    # later arrival) despite having been enqueued at Q3.
    assert scheduler.waiting.peek_request().request_id == "low"


# --------------------------------------------------------------------------- #
# Gate: proactive preemption of the worst running call when the batch is full
# --------------------------------------------------------------------------- #


def test_proactive_preemption_admits_higher_priority_waiting_call():
    """A fresh Q1 call preempts the worst running call when the batch is full."""
    scheduler = create_mlfq_scheduler(max_num_seqs=1)
    low = make_request("low", num_tokens=4, max_tokens=200, arrival_time=1.0)
    scheduler.add_request(low)
    for _ in range(7):  # demote the running call to queue 3
        _step(scheduler)
    assert low.priority == 3
    assert low.status == RequestStatus.RUNNING
    assert [r.request_id for r in scheduler.running] == ["low"]

    high = make_request("high", num_tokens=4, max_tokens=5, arrival_time=8.0)
    scheduler.add_request(high)
    assert high.priority == 0

    before = scheduler.proactive_preemption_count
    output = scheduler.schedule()

    assert scheduler.requests["low"].status == RequestStatus.PREEMPTED
    assert any(r.req_id == "high" for r in output.scheduled_new_reqs)
    assert [r.request_id for r in scheduler.running] == ["high"]
    assert scheduler.proactive_preemption_count == before + 1


def test_no_proactive_preemption_when_disabled():
    """With the bound at 0, the base v1 loop never swaps a running call out.

    This is the contrast that proves proactive preemption is *added* behaviour:
    stock v1 only preempts reactively on KV pressure, never to admit a
    higher-priority waiting call.
    """
    scheduler = create_mlfq_scheduler(
        max_num_seqs=1, max_proactive_preemptions_per_step=0
    )
    low = make_request("low", num_tokens=4, max_tokens=200, arrival_time=1.0)
    scheduler.add_request(low)
    for _ in range(7):
        _step(scheduler)
    high = make_request("high", num_tokens=4, max_tokens=5, arrival_time=8.0)
    scheduler.add_request(high)

    scheduler.schedule()

    assert scheduler.requests["low"].status == RequestStatus.RUNNING
    assert scheduler.requests["high"].status == RequestStatus.WAITING
    assert scheduler.proactive_preemption_count == 0


def test_equal_priority_waiting_call_does_not_preempt():
    """Proactive preemption needs a STRICT outrank; an equal queue never swaps."""
    scheduler = create_mlfq_scheduler(max_num_seqs=1)
    running = make_request("run", num_tokens=4, max_tokens=200, arrival_time=1.0)
    scheduler.add_request(running)
    _step(scheduler)  # admit + first decode step demotes it to queue 1
    assert running.priority == 1
    assert running.status == RequestStatus.RUNNING

    peer = make_request("peer", num_tokens=4, max_tokens=200, arrival_time=2.0)
    scheduler.add_request(peer)
    # Put the waiting peer in the SAME queue as the running call (not better),
    # so outranks(1, 1) is False.
    scheduler._call_state["peer"].queue_index = 1
    peer.priority = 1

    before = scheduler.proactive_preemption_count
    scheduler.schedule()

    assert scheduler.requests["run"].status == RequestStatus.RUNNING
    assert scheduler.requests["peer"].status == RequestStatus.WAITING
    assert scheduler.proactive_preemption_count == before


# --------------------------------------------------------------------------- #
# Gate: no starvation — a long low-priority call completes under sustained load
# --------------------------------------------------------------------------- #


def test_no_starvation_long_call_completes_under_sustained_load():
    """Under a stream of fresh Q1 calls the long call still completes (bounded)."""
    scheduler = create_mlfq_scheduler(max_num_seqs=1)
    long_call = make_request("long", num_tokens=4, max_tokens=6, arrival_time=0.0)
    scheduler.add_request(long_call)
    for _ in range(3):  # let it demote so fresh calls can preempt it
        _step(scheduler)

    completed_step = None
    for i in range(1, 3000):
        if "long" not in scheduler.requests:
            completed_step = i
            break
        fresh = make_request(f"f{i}", num_tokens=4, max_tokens=1, arrival_time=float(i))
        scheduler.add_request(fresh)
        _step(scheduler)

    assert completed_step is not None, "long call starved (never completed)"
    assert long_call.num_output_tokens == 6
    # It genuinely competed: it was proactively preempted at least once.
    assert long_call.num_preemptions >= 1


# --------------------------------------------------------------------------- #
# Gate: proactive preemption is bounded (no thrash) in steady state
# --------------------------------------------------------------------------- #


def test_proactive_preemption_is_bounded_no_thrash():
    """Once queues converge, proactive preemption stops (no unbounded thrash).

    Anti-starvation is disabled (huge beta) to isolate the proactive-preemption
    bound: with a fixed call set and no new arrivals the queues climb to the
    bottom and preemption reaches a fixpoint.
    """
    scheduler = create_mlfq_scheduler(max_num_seqs=2, beta=1e18)
    for i in range(4):
        scheduler.add_request(
            make_request(f"c{i}", num_tokens=4, max_tokens=1000, arrival_time=float(i))
        )

    for _ in range(400):
        _step(scheduler)

    alive = [f"c{i}" for i in range(4) if f"c{i}" in scheduler.requests]
    assert alive, "expected the long calls to still be running"
    assert all(scheduler.requests[r].priority == 6 for r in alive)

    converged = scheduler.proactive_preemption_count
    assert converged > 0, "expected proactive preemption during the queue climb"

    # Steady state: further steps trigger no new proactive preemptions.
    for _ in range(200):
        _step(scheduler)
    assert scheduler.proactive_preemption_count == converged


# --------------------------------------------------------------------------- #
# Gate: per-call state is released on completion AND on external abort
# --------------------------------------------------------------------------- #


def test_call_state_released_on_normal_completion():
    """Finishing a call frees its per-call MLFQ state (no leak)."""
    scheduler = create_mlfq_scheduler(max_num_seqs=1)
    call = make_request("done", num_tokens=4, max_tokens=3, arrival_time=1.0)
    scheduler.add_request(call)
    assert "done" in scheduler._call_state
    _run_to_completion(scheduler, "done")
    assert "done" not in scheduler._call_state


def test_call_state_released_on_abort():
    """An externally-aborted call frees its per-call MLFQ state (no leak).

    Aborts arrive via ``finish_requests`` (client disconnect / timeout), which
    bypasses ``update_from_output``; without a dedicated hook the per-call state
    would leak permanently. Mirrors Phase 1's abort regression test.
    """
    scheduler = create_mlfq_scheduler(max_num_seqs=1)
    call = make_request("ab", num_tokens=4, max_tokens=50, arrival_time=1.0)
    scheduler.add_request(call)
    for _ in range(3):
        _step(scheduler)
    assert scheduler._call_state["ab"].service_window == 3
    assert call.status == RequestStatus.RUNNING

    aborted = scheduler.finish_requests("ab", RequestStatus.FINISHED_ABORTED)
    assert ("ab", call.client_index) in aborted
    assert "ab" not in scheduler._call_state
    assert "ab" not in scheduler.requests


def test_finish_unknown_request_is_a_noop():
    """Aborting an unknown id yields no work and leaves tracked state intact."""
    scheduler = create_mlfq_scheduler(max_num_seqs=1)
    call = make_request("keep", num_tokens=4, max_tokens=50, arrival_time=1.0)
    scheduler.add_request(call)

    aborted = scheduler.finish_requests("nonexistent", RequestStatus.FINISHED_ABORTED)

    assert aborted == []
    assert "keep" in scheduler._call_state
