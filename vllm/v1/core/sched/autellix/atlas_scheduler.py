# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Program-aware critical-path Least-Attained-Service (ATLAS) scheduler.

ATLAS is the multi-threaded generalisation of PLAS (paper §4.2, Eq. 2 +
Algorithm 1 line 3). Like PLAS it groups a program's many LLM calls (keyed by
``program_id``) and schedules least-served programs first, but the program's
scalar is its **maximum critical path** rather than the cumulative sum of call
service. This keeps a program's *parallel* threads on an equal footing: their
concurrent calls all inherit the same scalar, share a priority bin, and so a
straggler thread does not starve its siblings.

It differs from ``PLASScheduler`` in exactly two places
(``notes/autellix_scheduling/POLICY_REFERENCE.md`` §2):

* the program scalar consulted on arrival (and as ``T_p`` in the
  anti-starvation ratio) is ``ProgramState.max_critical_path`` (not
  ``.service``); and
* each call **snapshots** that scalar at arrival as its *start priority* and,
  on completion, folds it back with the max rule
  ``max_critical_path = max(max_critical_path, start_priority + call_service)``
  (not the PLAS sum ``service += call_service``).

Key faithfulness note: ATLAS needs **no explicit parent/DAG tracking**. The
single scalar plus start-priority inheritance approximates the program's
critical path, because concurrent calls naturally snapshot the same scalar at
arrival and the max rule then records only the longest thread. ``thread_id`` is
therefore purely optional observability metadata -- it is captured when present
but never influences a scheduling decision. With no concurrency (sequential
calls) every call inherits the full scalar left by its predecessor, the max rule
always increases, and ATLAS degrades exactly to PLAS's cumulative sum.

Mapping to the vLLM v1 engine mirrors PLAS (see its module docstring and
POLICY_REFERENCE.md §6a): the arrival bin of the critical-path scalar is
written to ``request.priority`` (an int queue level, 0 = top), and Algorithm
1's per-call mechanics (via :class:`QuantumMlfqMixin`) layer on top -- quantum
demotion one level at a time, program-level anti-starvation promotion at
``(W_p + W_c) / max(1, T_p + T_c) >= beta`` with only the call-level windows
reset, and bounded proactive preemption at a full batch. Attained service is
proxied by decode steps (one unit per scheduled decode step, §2 D4); on
completion a call folds its critical path with the max rule and its residual
wait window into the program's ``W_p``.
"""

import time
from collections.abc import Iterable, Sequence
from typing import Any

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.autellix.attained_service import AttainedServiceTracker
from vllm.v1.core.sched.autellix.mlfq import MlfqBinner
from vllm.v1.core.sched.autellix.policy_core import QuantumMlfqMixin
from vllm.v1.core.sched.autellix.process_table import ProcessTable
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue
from vllm.v1.engine import EngineCoreOutputs
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus

# D5 defaults (POLICY_REFERENCE.md §4): K=7 queues with the boundary set
# (0, 2, 4, 8, 16, 32, 64, inf) for arrival binning; pass the thresholds
# explicitly since the geometric default is not the D5 boundary set. The
# per-queue decode-step quanta and beta follow the FastServe scheme.
_NUM_QUEUES = 7
_THRESHOLDS = [2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
_QUEUE_QUANTA: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)
_BETA = 8.0
_MAX_PROACTIVE_PREEMPTIONS_PER_STEP = 1

# Seconds an idle program (no in-flight calls, no recent completion) may live
# before its state is evicted. Agentic programs issue bursts of calls separated
# by tool-execution / think-time gaps; a generous TTL keeps a program's
# critical-path scalar across those gaps (so ATLAS still deprioritises it on its
# next call) while bounding memory for one-shot programs.
_PROGRAM_TTL_S = 600.0


class ATLASScheduler(QuantumMlfqMixin, AsyncScheduler):
    """Critical-path LAS scheduler for multi-threaded agentic programs.

    Subclasses ``AsyncScheduler`` (not ``Scheduler``) so vLLM keeps async
    scheduling enabled; the accrual/fold overrides compose with the async
    placeholder bookkeeping exactly as in ``PLASScheduler`` (the max rule folds
    the same per-call service, which the base async over-schedule guard keeps at
    one unit per output token), and sync operation is unchanged.
    """

    def __init__(
        self,
        *args: Any,
        num_queues: int = _NUM_QUEUES,
        thresholds: Sequence[float] = tuple(_THRESHOLDS),
        queue_quanta: Sequence[int] = _QUEUE_QUANTA,
        beta: float = _BETA,
        max_proactive_preemptions_per_step: int = _MAX_PROACTIVE_PREEMPTIONS_PER_STEP,
        **kwargs: Any,
    ) -> None:
        """Build the scheduler and self-configure priority ordering.

        Calls the base initialiser (``AsyncScheduler`` -> ``Scheduler``, which
        also sets up the async placeholder bookkeeping), then switches the
        scheduling policy to ``PRIORITY`` and rebuilds the waiting queues so that
        deploying with only ``--scheduler-cls ...ATLASScheduler`` yields the right
        ordering, and instantiates the policy core (process table,
        attained-service tracker, MLFQ binner + quantum machinery) plus the
        per-call maps. The constants default to the D5 / FastServe values and
        are accepted as parameters so a ``beta`` / ``K`` ablation is possible.

        Args:
            num_queues: Number of priority levels ``K``.
            thresholds: Cumulative service thresholds for arrival binning; must
                have length ``num_queues - 1``.
            queue_quanta: Per-level decode-step quantum; must have length
                ``num_queues``.
            beta: Anti-starvation wait-to-service ratio threshold.
            max_proactive_preemptions_per_step: Upper bound on proactive
                preemptions per ``schedule()`` call (0 disables it).

        Raises:
            ValueError: If the thresholds or quanta lengths do not match
                ``num_queues``.
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
        self._init_policy_core(
            num_queues=num_queues,
            queue_quanta=tuple(queue_quanta),
            beta=beta,
            max_proactive_preemptions_per_step=max_proactive_preemptions_per_step,
            binner=MlfqBinner(num_queues=num_queues, thresholds=list(thresholds)),
        )
        # request_id -> program_id for the call's lifetime.
        self._req_to_pid: dict[str, str] = {}
        # request_id -> the program scalar snapshotted at the call's arrival,
        # the `start_priority` folded back with the max rule on completion.
        self._req_to_start_scalar: dict[str, float] = {}
        # request_id -> thread_id, optional observability metadata only.
        self._req_to_thread_id: dict[str, str] = {}

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

    def _thread_id(self, request: Request) -> str | None:
        """Return the request's optional ``thread_id`` metadata, else ``None``.

        Purely observability: the scheduling decision never consults it (see the
        module docstring), so a missing thread id changes nothing.
        """
        sampling_params = request.sampling_params
        extra_args = sampling_params.extra_args if sampling_params else None
        if extra_args is not None:
            thread_id = extra_args.get("thread_id")
            if thread_id is not None:
                return str(thread_id)
        return None

    def add_request(self, request: Request) -> None:
        """Bin a newly-arrived call by its program's critical path, then enqueue.

        The arrival level is the bin of the program's current
        ``max_critical_path``, snapshotted as the call's start priority for the
        completion-time max rule. It (and its quantum) is assigned before the
        base enqueues (and heapifies) the request, so the call enters the
        waiting queue at its LAS bin.
        """
        if request.request_id not in self.requests:
            program_id = self._program_id(request)
            state = self.process_table.get_or_create(program_id)
            start_scalar = state.max_critical_path
            self._register_call_state(request, self.binner.bin(start_scalar))
            self.process_table.register_call(
                program_id, request.request_id, request.arrival_time
            )
            self._req_to_pid[request.request_id] = program_id
            self._req_to_start_scalar[request.request_id] = start_scalar
            thread_id = self._thread_id(request)
            if thread_id is not None:
                self._req_to_thread_id[request.request_id] = thread_id
        super().add_request(request)

    def _program_windows(self, request: Request) -> tuple[float, float]:
        """Return the program's ``(W_p, T_p)`` for the starvation ratio.

        ``W_p`` is the wait folded from the program's completed calls and
        ``T_p`` is the ATLAS max-rule critical path, so promotion measures
        **program-level** starvation (paper §4.2.2: a call-level-only ratio
        "reduces Autellix to naive MLFQ").
        """
        program_id = self._req_to_pid.get(request.request_id)
        state = self.process_table.get(program_id) if program_id else None
        if state is None:
            return (0.0, 0.0)
        return (state.total_wait, state.max_critical_path)

    def _on_service_step(self, req_id: str) -> None:
        """Accrue one unit of attained service per charged decode step.

        Called by the mixin's ``_update_after_schedule`` for exactly the steps
        that also charge quantum (final, non-prefill-chunk steps), so the
        max-rule fold and the quantum/starvation windows stay in lockstep.
        """
        self.attained_service.record_step(req_id)

    def _fold_completed_calls(self, req_ids: Iterable[str], now: float) -> None:
        """Fold each finished call's critical path into its program (max rule).

        Applies ``max_critical_path = max(prev, start_priority + call_service)``,
        folds the call's residual wait window into the program's ``W_p``
        (residual because promotion resets ``W_c``), deregisters the call, and
        garbage collects idle programs. Idempotent per call: once a call's
        ``req_id -> pid`` entry is popped it is never folded again, so the two
        completion paths (``update_from_output`` for normal stops,
        ``finish_requests`` for aborts) never double-count. The
        attained-service, call-state, start-scalar, and thread-id entries are
        popped unconditionally so no per-call state leaks even for the fallback
        path.

        Args:
            req_ids: The finished calls' request ids.
            now: The current time, used for completion time and GC.
        """
        for req_id in req_ids:
            program_id = self._req_to_pid.pop(req_id, None)
            call_service = self.attained_service.pop(req_id)
            start_scalar = self._req_to_start_scalar.pop(req_id, 0.0)
            call_state = self._call_state.pop(req_id, None)
            self._req_to_thread_id.pop(req_id, None)
            if program_id is None:
                continue
            self.process_table.update_critical_path(
                program_id, start_scalar, call_service
            )
            if call_state is not None:
                self.process_table.add_wait(program_id, call_state.wait_window)
            self.process_table.complete_call(program_id, req_id, now)
        self.process_table.gc(now)

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs]:
        """Fold normally-stopped calls' critical path into their programs.

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
        """Fold externally-aborted calls' critical path before they are dropped.

        Aborts (client disconnect / timeout) enter here rather than through
        ``update_from_output``, and the next ``schedule`` clears
        ``finished_req_ids`` before that fold would run. Folding here prevents
        the call's program state, ``req_id -> pid`` entry, start-scalar
        snapshot, and attained-service entry from leaking permanently (which
        would also un-GC the program and drop critical path that biases the
        program's later calls).

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
