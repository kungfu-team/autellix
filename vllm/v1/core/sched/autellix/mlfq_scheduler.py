# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Program-agnostic FastServe-style multi-level feedback queue (MLFQ) scheduler.

MLFQ is the paper's call-level baseline (Autellix §4.2.2): it ignores program
identity entirely and schedules by each call's own accrued service. Every new
call enters the top queue Q1; it is demoted one level whenever it exhausts its
queue's decode-step quantum; a call that waits too long relative to its service
is promoted back to Q1 (anti-starvation); and when the running batch is full a
higher-priority waiting call proactively preempts the worst running call. This is
the FastServe scheme the paper builds on, realised on the vLLM v1 engine by
composing the pure-Python Phase-0 core (:class:`MlfqBinner`) with vLLM's native
priority queue and recompute-based preemption.

Mapping (see ``notes/autellix_scheduling/POLICY_REFERENCE.md`` §3-5):

* "Queue index" maps directly to ``request.priority`` (Q1 == priority 0 ==
  highest; a larger priority is a lower queue). The native
  ``PriorityRequestQueue`` then orders waiting calls by
  ``(priority, arrival_time, request_id)``, and the base scheduler's reactive
  KV-pressure victim ``max(running, key=(priority, arrival_time))`` is already
  the worst (most-demoted) running call.
* Service is proxied by decode steps: each scheduled decode step adds one unit
  (D4). No cumulative :class:`AttainedServiceTracker` is needed here because the
  per-call service window is the anti-starvation ``T`` and is reset on promotion.
* All state is strictly **per call** (there is no process table); it is released
  on both normal completion and external abort so nothing leaks.

Constants (POLICY_REFERENCE.md §4, decision D5): ``K = 7`` queues, per-queue
decode-step quanta ``(1, 2, 4, 8, 16, 32, 64)`` (the paper's cited FastServe
geometric ×2 scheme), and anti-starvation ratio ``beta = 8.0``. They are exposed
as constructor parameters so a ``beta`` / ``K`` ablation is possible.

Proactive preemption + thrash bound: v1 never preempts a running call to admit a
higher-priority waiting one (it only preempts reactively on KV OOM), so MLFQ adds
this in a bookkeeping pass run *before* delegating to ``super().schedule()``. To
avoid preempt/recompute thrash it preempts at most
``max_proactive_preemptions_per_step`` calls per step (default 1, matching the
paper's "preempt the worst running call"), and only ever the current worst
running call while a waiting call *strictly* outranks it — so once the queues
converge (no waiting call outranks any running call) preemption stops.
"""

import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from vllm.v1.core.sched.autellix.mlfq import MlfqBinner
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreOutputs
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus

# D5 defaults (POLICY_REFERENCE.md §4): K=7 queues with the paper's cited
# FastServe geometric ×2 decode-step quanta, and anti-starvation ratio beta=8.0.
_NUM_QUEUES = 7
_QUEUE_QUANTA: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)
_BETA = 8.0

# Preempt at most one running call per scheduling step (the paper preempts "the
# worst running call"); this bounds recompute cost and prevents thrash.
_MAX_PROACTIVE_PREEMPTIONS_PER_STEP = 1


@dataclass
class _MlfqCallState:
    """Mutable per-call MLFQ bookkeeping, keyed by ``request_id``.

    Attributes:
        queue_index: The call's current queue; equals ``request.priority`` (queue
            0 is Q1, the highest priority).
        quantum_remaining: Decode steps left before the call is demoted a level.
        wait_window: Decode steps the call has waited (``W_call``); reset to 0 on
            anti-starvation promotion.
        service_window: Decode steps the call has been served (``T_call``); reset
            to 0 on anti-starvation promotion.
    """

    queue_index: int
    quantum_remaining: int
    wait_window: int
    service_window: int


class MLFQScheduler(Scheduler):
    """FastServe-style preemptive MLFQ scheduler (program-agnostic)."""

    def __init__(
        self,
        *args: Any,
        num_queues: int = _NUM_QUEUES,
        queue_quanta: Sequence[int] = _QUEUE_QUANTA,
        beta: float = _BETA,
        max_proactive_preemptions_per_step: int = _MAX_PROACTIVE_PREEMPTIONS_PER_STEP,
        **kwargs: Any,
    ) -> None:
        """Build the scheduler and self-configure priority ordering.

        Calls the base initialiser, then switches the scheduling policy to
        ``PRIORITY`` and rebuilds the (empty) waiting queues, so deploying with
        only ``--scheduler-cls ...MLFQScheduler`` yields the right ordering. The
        MLFQ constants default to the D5 / FastServe values and are accepted as
        parameters so a ``beta`` / ``K`` ablation is possible.

        Args:
            num_queues: Number of feedback queues ``K``.
            queue_quanta: Per-queue decode-step quantum; must have length
                ``num_queues``.
            beta: Anti-starvation wait-to-service ratio threshold.
            max_proactive_preemptions_per_step: Upper bound on proactive
                preemptions per ``schedule()`` call (0 disables it).

        Raises:
            ValueError: If ``len(queue_quanta) != num_queues``.
        """
        if len(queue_quanta) != num_queues:
            raise ValueError(
                f"queue_quanta must have length num_queues = {num_queues}, "
                f"got {len(queue_quanta)}"
            )
        super().__init__(*args, **kwargs)

        # Self-configure priority ordering. Only `policy`, `waiting`, and
        # `skipped_waiting` depend on the scheduling policy; both queues are
        # empty at construction, so rebuilding them loses no state.
        self.policy = SchedulingPolicy.PRIORITY
        self.waiting = create_request_queue(self.policy)
        self.skipped_waiting = create_request_queue(self.policy)

        self.num_queues = num_queues
        self.queue_quanta = tuple(queue_quanta)
        self.beta = beta
        self.max_proactive_preemptions_per_step = max_proactive_preemptions_per_step
        # Only demote / anti_starvation / outranks are used; MLFQ never bins by
        # service (every new call starts in Q1), so the binner's thresholds are
        # irrelevant here.
        self.binner = MlfqBinner(num_queues=num_queues)
        self._call_state: dict[str, _MlfqCallState] = {}
        self.proactive_preemption_count = 0

    def add_request(self, request: Request) -> None:
        """Place every newly-arrived call in the top queue Q1 (priority 0).

        The priority is assigned before the base enqueues (and heapifies) the
        request, so the call enters the waiting queue at Q1.
        """
        if request.request_id not in self.requests:
            request.priority = 0
            self._call_state[request.request_id] = _MlfqCallState(
                queue_index=0,
                quantum_remaining=self.queue_quanta[0],
                wait_window=0,
                service_window=0,
            )
        super().add_request(request)

    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
        """Run MLFQ bookkeeping, then delegate to the base scheduling loop.

        The bookkeeping pass (anti-starvation promotion + proactive preemption)
        mutates ``request.priority`` and the waiting heap so that the base loop,
        which is unchanged, admits and evicts calls in MLFQ order.
        """
        now = time.monotonic()
        self._accrue_wait_and_promote()
        self._proactively_preempt(now)
        return super().schedule(throttle_prefills)

    def _accrue_wait_and_promote(self) -> None:
        """Accrue wait for waiting calls and promote the starving ones to Q1.

        Every call currently waiting (in either the main or the skipped queue)
        accrues one unit of wait. A call outside Q1 whose ``W / max(1, T)`` has
        reached ``beta`` is promoted to Q1 with its windows reset, and the queue
        is re-heapified so the base loop sees the new ordering.
        """
        for queue in (self.waiting, self.skipped_waiting):
            promoted: list[Request] = []
            for request in list(queue):
                # Every queued call was registered by add_request, so its state
                # is present; a missing entry is a real invariant violation.
                state = self._call_state[request.request_id]
                state.wait_window += 1
                if state.queue_index != 0 and self.binner.anti_starvation(
                    state.wait_window, state.service_window, self.beta
                ):
                    self._promote_to_top(request, state)
                    promoted.append(request)
            if promoted:
                # Priorities changed in place; re-insert to restore heap order.
                queue.remove_requests(promoted)
                for request in promoted:
                    queue.add_request(request)

    def _promote_to_top(self, request: Request, state: _MlfqCallState) -> None:
        """Promote a starving call back to Q1 and reset its call-level windows."""
        state.queue_index = 0
        state.quantum_remaining = self.queue_quanta[0]
        state.wait_window = 0
        state.service_window = 0
        request.priority = 0

    def _proactively_preempt(self, now: float) -> None:
        """Preempt the worst running call(s) so better waiting calls can run.

        Only fires when the batch is full. Each iteration preempts the current
        worst running call while the best waiting call *strictly* outranks it,
        up to ``max_proactive_preemptions_per_step`` times. Freed calls return to
        the waiting queue (recompute-based, KV blocks freed) so the base loop can
        admit the better waiting calls into the vacated slots. The strict-outrank
        guard makes preemption stop once the queues converge, and the per-step
        cap bounds recompute so it cannot thrash.
        """
        if len(self.running) < self.max_num_running_reqs:
            return
        freed = 0
        while (
            freed < self.max_proactive_preemptions_per_step
            and self.waiting
            and self.running
        ):
            best_waiting = self.waiting.peek_request()
            worst_running = max(
                self.running, key=lambda r: (r.priority, r.arrival_time)
            )
            if not self.binner.outranks(best_waiting.priority, worst_running.priority):
                break
            self.running.remove(worst_running)
            self._preempt_request(worst_running, now)
            self.proactive_preemption_count += 1
            freed += 1

    def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
        """Charge one decode step of quantum/service and demote on exhaustion.

        A running call whose quantum is exhausted is demoted one level and its
        priority updated in place; this is correct even for a call already
        running (unlike PLAS, an MLFQ call's queue is not fixed for its
        lifetime). Non-final prefill chunks are not charged.
        """
        super()._update_after_schedule(scheduler_output)
        for req_id in scheduler_output.num_scheduled_tokens:
            request = self.requests.get(req_id)
            if request is None or request.is_prefill_chunk:
                continue
            # A scheduled call is always registered (see add_request).
            state = self._call_state[req_id]
            state.quantum_remaining -= 1
            state.service_window += 1
            if state.quantum_remaining <= 0:
                state.queue_index = self.binner.demote(state.queue_index)
                request.priority = state.queue_index
                state.quantum_remaining = self.queue_quanta[state.queue_index]

    def _release_call_state(self, req_ids: Iterable[str]) -> None:
        """Drop each finished call's per-call state.

        Idempotent per call (``pop`` with a default), so the two completion
        paths -- ``update_from_output`` for normal stops and ``finish_requests``
        for aborts -- never error on an already-released id.
        """
        for req_id in req_ids:
            self._call_state.pop(req_id, None)

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs]:
        """Release per-call state for normally-stopped calls.

        Uses the base's per-step ``finished_req_ids`` (populated by
        ``_free_request`` and reset at the end of each ``schedule``'s
        ``_update_after_schedule``) as the finished-call signal.
        """
        engine_core_outputs = super().update_from_output(
            scheduler_output, model_runner_output
        )
        if self.finished_req_ids:
            self._release_call_state(self.finished_req_ids)
        return engine_core_outputs

    def finish_requests(
        self,
        request_ids: str | Iterable[str] | None,
        finished_status: RequestStatus,
    ) -> list[tuple[str, int]]:
        """Release per-call state for externally-aborted calls before they drop.

        Aborts (client disconnect / timeout) enter here rather than through
        ``update_from_output``, and the next ``schedule`` clears
        ``finished_req_ids`` before that release would run. Releasing here
        prevents the per-call state from leaking permanently (mirrors the Phase 1
        abort fix).

        Args:
            request_ids: Ids to finish, or ``None`` to finish all.
            finished_status: The finished status to apply.

        Returns:
            The ``(request_id, client_index)`` pairs actually finished by this
            call, as returned by the base scheduler.
        """
        aborted = super().finish_requests(request_ids, finished_status)
        if aborted:
            self._release_call_state(req_id for req_id, _ in aborted)
        return aborted
