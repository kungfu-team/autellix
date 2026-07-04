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
composing the shared quantum/promotion machinery (:class:`QuantumMlfqMixin`)
with vLLM's native priority queue and recompute-based preemption.

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
* All state is strictly **per call** (there is no process table); the
  anti-starvation ratio uses the call-level windows alone (``W_c / max(1,
  T_c)``), the paper's deliberately "naive" FastServe variant -- the mixin's
  ``_program_windows`` default of ``(0, 0)``. State is released on both normal
  completion and external abort so nothing leaks.

Constants (POLICY_REFERENCE.md §4, decision D5): ``K = 7`` queues, per-queue
decode-step quanta ``(1, 2, 4, 8, 16, 32, 64)`` (the paper's cited FastServe
geometric ×2 scheme), and anti-starvation ratio ``beta = 8.0``. They are exposed
as constructor parameters so a ``beta`` / ``K`` ablation is possible.

Proactive preemption + thrash bound: v1 never preempts a running call to admit a
higher-priority waiting one (it only preempts reactively on KV OOM), so the
mixin adds this in a bookkeeping pass run *before* delegating to
``super().schedule()``. To avoid preempt/recompute thrash it preempts at most
``max_proactive_preemptions_per_step`` calls per step (default 1, matching the
paper's "preempt the worst running call"), and only ever the current worst
running call while a waiting call *strictly* outranks it — so once the queues
converge (no waiting call outranks any running call) preemption stops.
"""

from collections.abc import Iterable, Sequence
from typing import Any

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.autellix.mlfq import MlfqBinner
from vllm.v1.core.sched.autellix.policy_core import QuantumMlfqMixin
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue
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


class MLFQScheduler(QuantumMlfqMixin, AsyncScheduler):
    """FastServe-style preemptive MLFQ scheduler (program-agnostic).

    Subclasses ``AsyncScheduler`` (not ``Scheduler``) so vLLM keeps async
    scheduling enabled; the mixin's overrides compose with the async placeholder
    bookkeeping (``_update_after_schedule`` calls ``super()`` first, then charges
    quantum/service on the final decode steps), and in lockstep (sync) operation
    the placeholder churn nets to zero so behaviour is unchanged.
    """

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

        Calls the base initialiser (``AsyncScheduler`` -> ``Scheduler``, which
        also sets up the async placeholder bookkeeping), then switches the
        scheduling policy to ``PRIORITY`` and rebuilds the (empty) waiting queues,
        so deploying with only ``--scheduler-cls ...MLFQScheduler`` yields the
        right ordering. The MLFQ constants default to the D5 / FastServe values
        and are accepted as parameters so a ``beta`` / ``K`` ablation is possible.

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

        # Only demote / anti_starvation / outranks are used; MLFQ never bins by
        # service (every new call starts in Q1), so the binner's thresholds are
        # irrelevant here.
        self._init_policy_core(
            num_queues=num_queues,
            queue_quanta=tuple(queue_quanta),
            beta=beta,
            max_proactive_preemptions_per_step=max_proactive_preemptions_per_step,
            binner=MlfqBinner(num_queues=num_queues),
        )

    def add_request(self, request: Request) -> None:
        """Place every newly-arrived call in the top queue Q1 (priority 0).

        The priority is assigned before the base enqueues (and heapifies) the
        request, so the call enters the waiting queue at Q1.
        """
        if request.request_id not in self.requests:
            self._register_call_state(request, queue_index=0)
        super().add_request(request)

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
