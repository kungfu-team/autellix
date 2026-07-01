# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Program-aware Least-Attained-Service (PLAS) scheduler.

PLAS groups a program's many LLM calls (keyed by ``program_id``) and schedules
least-served programs first, so short interactive programs are not starved
behind long-running ones. It is the minimal faithful realisation of the paper's
PLAS policy (§4.2, Eq. 1) on the vLLM v1 engine: it composes the pure-Python
Phase-0 policy core with vLLM's native priority queue and recompute-based
preemption.

Mapping (see ``notes/autellix_scheduling/POLICY_REFERENCE.md`` §5):

* On arrival a call inherits its program's cumulative attained service as its
  start priority, discretised into one of the MLFQ bins. The bin is written to
  ``request.priority`` and is **stable for the call's lifetime** -- this
  scheduler performs no demotion or anti-starvation promotion (that is Phase 2's
  ``MLFQScheduler``).
* Attained service is proxied by decode steps: each scheduled decode step adds
  one unit (§2, D4).
* On completion a call's accrued service is folded into its program with the
  PLAS **sum** rule, and idle programs are garbage collected by TTL.

Smaller ``request.priority`` means higher priority, and the native priority
queue orders by ``(priority, arrival_time, request_id)``, so least-served
programs are admitted first and the most-served program's call is the natural
KV-pressure preemption victim (``max(running, key=(priority, arrival_time))``).

It subclasses ``AsyncScheduler`` (not ``Scheduler``) so vLLM keeps async
scheduling enabled -- a plain ``Scheduler`` subclass triggers the documented
"async scheduling being disabled" degradation (``config/scheduler.py``). The
policy overrides compose cleanly with async: ``_update_after_schedule`` calls
``super()`` first (AsyncScheduler manages ``num_output_placeholders`` and sets
``is_prefill_chunk``) then accrues service on scheduled decode steps, and the
base async over-schedule guard keeps that accrual exactly one unit per output
token. In lockstep (sync) operation the placeholder churn nets to zero, so the
scheduler still behaves identically. Fairness across policy lines sharing this
base is handled later (Phase 6/7).
"""

import time
from collections.abc import Iterable
from typing import Any

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.autellix.attained_service import AttainedServiceTracker
from vllm.v1.core.sched.autellix.mlfq import MlfqBinner
from vllm.v1.core.sched.autellix.process_table import ProcessTable
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue
from vllm.v1.engine import EngineCoreOutputs
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus

# D5 defaults (POLICY_REFERENCE.md §4): K=7 queues with the boundary set
# (0, 2, 4, 8, 16, 32, 64, inf); pass the thresholds explicitly since the
# geometric default is not the D5 boundary set.
_NUM_QUEUES = 7
_THRESHOLDS = [2.0, 4.0, 8.0, 16.0, 32.0, 64.0]

# Seconds an idle program (no in-flight calls, no recent completion) may live
# before its state is evicted. Agentic programs issue bursts of calls separated
# by tool-execution / think-time gaps; a generous TTL keeps a program's accrued
# service across those gaps (so PLAS still deprioritises it on its next call)
# while bounding memory for one-shot programs.
_PROGRAM_TTL_S = 600.0


class PLASScheduler(AsyncScheduler):
    """Least-attained-service scheduler that is aware of agentic programs."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Build the scheduler and self-configure priority ordering.

        Calls the base initialiser (``AsyncScheduler`` -> ``Scheduler``, which
        also sets up the async placeholder bookkeeping), then switches the
        scheduling policy to ``PRIORITY`` and rebuilds the waiting queues so that
        deploying with only ``--scheduler-cls ...PLASScheduler`` yields the right
        ordering, and instantiates the Phase-0 policy core (process table,
        attained-service tracker, MLFQ binner).
        """
        super().__init__(*args, **kwargs)

        # Self-configure priority ordering. Only `policy`, `waiting`, and
        # `skipped_waiting` depend on the scheduling policy; both queues are
        # empty at construction, so rebuilding them loses no state.
        self.policy = SchedulingPolicy.PRIORITY
        self.waiting = create_request_queue(self.policy)
        self.skipped_waiting = create_request_queue(self.policy)

        self.process_table = ProcessTable(ttl=_PROGRAM_TTL_S)
        self.attained_service = AttainedServiceTracker()
        self.binner = MlfqBinner(num_queues=_NUM_QUEUES, thresholds=_THRESHOLDS)
        # request_id -> program_id for the call's lifetime.
        self._req_to_pid: dict[str, str] = {}

    def _program_id(self, request: Request) -> str:
        """Return the request's ``program_id``, or its id as a lone program.

        Falls back to the request id when ``extra_args`` is missing or carries
        no ``program_id``, so a request without program metadata is treated as a
        single-call program (graceful, per-request ~FCFS behaviour).
        """
        sampling_params = request.sampling_params
        extra_args = sampling_params.extra_args if sampling_params else None
        if extra_args is not None:
            program_id = extra_args.get("program_id")
            if program_id is not None:
                return str(program_id)
        return request.request_id

    def add_request(self, request: Request) -> None:
        """Bin a newly-arrived call by its program's service, then enqueue.

        The priority is assigned before the base enqueues (and heapifies) the
        request, so the call enters the waiting queue at its LAS bin.
        """
        if request.request_id not in self.requests:
            program_id = self._program_id(request)
            state = self.process_table.get_or_create(program_id)
            request.priority = self.binner.bin(state.service)
            self.process_table.register_call(
                program_id, request.request_id, request.arrival_time
            )
            self._req_to_pid[request.request_id] = program_id
        super().add_request(request)

    def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
        """Accrue one unit of attained service per scheduled decode step.

        ``super()`` runs first: ``AsyncScheduler`` reserves output placeholders
        and delegates to ``Scheduler``, which advances computed tokens and sets
        ``is_prefill_chunk``. We then accrue on the final (non-prefill-chunk)
        steps. The base async over-schedule guard means the number of such steps
        stays exactly ``max_tokens``, so accrual is unchanged by run-ahead.
        """
        super()._update_after_schedule(scheduler_output)
        for req_id in scheduler_output.num_scheduled_tokens:
            request = self.requests.get(req_id)
            # Skip non-final prefill chunks; the base sets `is_prefill_chunk`
            # from the just-advanced computed-token count.
            if request is not None and not request.is_prefill_chunk:
                self.attained_service.record_step(req_id)

    def _fold_completed_calls(self, req_ids: Iterable[str], now: float) -> None:
        """Fold each finished call's attained service into its program.

        Applies the PLAS sum rule, deregisters the call, and garbage collects
        idle programs. Idempotent per call: once a call's ``req_id -> pid`` entry
        is popped it is never folded again, so the two completion paths
        (``update_from_output`` for normal stops, ``finish_requests`` for
        aborts) never double-count.

        Args:
            req_ids: The finished calls' request ids.
            now: The current time, used for completion time and GC.
        """
        for req_id in req_ids:
            program_id = self._req_to_pid.pop(req_id, None)
            service = self.attained_service.pop(req_id)
            if program_id is None:
                continue
            self.process_table.add_service(program_id, service)
            self.process_table.complete_call(program_id, req_id, now)
        self.process_table.gc(now)

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs]:
        """Fold normally-stopped calls' service into their programs (sum rule).

        Uses the base's per-step ``finished_req_ids`` (populated by
        ``_free_request`` and reset at the end of each ``schedule`` in
        ``_update_after_schedule``) as the finished-call signal.
        """
        engine_core_outputs = super().update_from_output(
            scheduler_output, model_runner_output
        )
        if self.finished_req_ids:
            self._fold_completed_calls(self.finished_req_ids, time.time())
        return engine_core_outputs

    def finish_requests(
        self,
        request_ids: str | Iterable[str] | None,
        finished_status: RequestStatus,
    ) -> list[tuple[str, int]]:
        """Fold externally-aborted calls' service before they are dropped.

        Aborts (client disconnect / timeout) enter here rather than through
        ``update_from_output``, and the next ``schedule`` clears
        ``finished_req_ids`` before that fold would run. Folding here prevents
        the call's program state, ``req_id -> pid`` entry, and attained-service
        entry from leaking permanently (which would also un-GC the program and
        drop service that biases the program's later calls).

        Args:
            request_ids: Ids to finish, or ``None`` to finish all.
            finished_status: The finished status to apply.

        Returns:
            The ``(request_id, client_index)`` pairs actually finished by this
            call, as returned by the base scheduler.
        """
        aborted = super().finish_requests(request_ids, finished_status)
        if aborted:
            self._fold_completed_calls((req_id for req_id, _ in aborted), time.time())
        return aborted
