# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared quantum / demotion / promotion machinery for the Autellix schedulers.

Algorithm 1 of the paper runs the same per-call mechanics for every policy
(MLFQ baseline, PLAS, ATLAS): each call holds a discretized priority level with
a per-level decode-step quantum; exhausting the quantum demotes the call one
level; a waiting call accrues wait, and once its wait-to-service ratio reaches
``beta`` it is promoted back to the top level with its *call-level* windows
reset; and when the batch is full a strictly-better waiting call proactively
preempts the worst running call (bounded per step, recompute-based).

The policies differ only in the *windows* fed to the anti-starvation ratio:

* MLFQ (FastServe baseline) uses the call-level windows alone
  (``W_c / max(1, T_c)``) -- the paper's deliberately "naive" variant.
* PLAS / ATLAS use **program-level** starvation (paper §4.2.2):
  ``(W_p + W_c) / max(1, T_p + T_c)``, where ``W_p``/``T_p`` come from the
  process table (PLAS: service sum; ATLAS: critical-path max). Only the
  call-level windows reset on promotion, so a program's calls promote together
  and the program cannot be re-starved by its own promotion.

Priority encoding (shared by all three schedulers): ``request.priority`` is the
call's current queue level, an int in ``[0, K-1]`` with 0 the top queue, so
``Request.__lt__``'s int-priority semantics are untouched. The native
``PriorityRequestQueue`` orders by ``(priority, arrival_time, request_id)`` --
level primary, FCFS within a level. Program attained service influences only
the *arrival* level (PLAS/ATLAS bin the program scalar); demotion and promotion
then move the level in place, exactly as Algorithm 1 moves calls between
queues.

This module hosts the mechanics as a mixin over ``AsyncScheduler`` subclasses;
the schedulers keep their own arrival binning, program folding, and state
release. It relies on base-scheduler attributes (``waiting``,
``skipped_waiting``, ``running``, ``requests``, ``max_num_running_reqs``,
``_preempt_request``) and must precede ``AsyncScheduler`` in the MRO.
"""

import time
from collections.abc import Iterable
from dataclasses import dataclass

from vllm.v1.core.sched.autellix.mlfq import MlfqBinner
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import Request


@dataclass
class CallQueueState:
    """Mutable per-call queue bookkeeping, keyed by ``request_id``.

    Attributes:
        queue_index: The call's current queue; equals ``request.priority``
            (queue 0 is Q1, the highest priority).
        quantum_remaining: Decode steps left before the call is demoted a
            level.
        wait_window: Decode steps the call has waited (``W_c``); reset to 0 on
            anti-starvation promotion.
        service_window: Decode steps the call has been served (``T_c``); reset
            to 0 on anti-starvation promotion.
    """

    queue_index: int
    quantum_remaining: int
    wait_window: int
    service_window: int


class QuantumMlfqMixin:
    """Per-call quantum demotion, beta promotion, and proactive preemption.

    Mix into an ``AsyncScheduler`` subclass *before* the base class. The host
    scheduler must call :meth:`_init_policy_core` from its ``__init__`` and
    :meth:`_register_call_state` for every genuinely new call, and should
    release state via :meth:`_release_call_state` (or pop entries itself) on
    completion/abort. Hooks:

    * :meth:`_program_windows` supplies the program-level ``(W_p, T_p)`` added
      to the call windows in the anti-starvation ratio (default ``(0, 0)`` --
      call-level-only, the MLFQ baseline).
    * :meth:`_on_service_step` is called once per charged decode step so
      program-aware hosts can accrue attained service without a second pass.
    """

    def _init_policy_core(
        self,
        num_queues: int,
        queue_quanta: tuple[int, ...],
        beta: float,
        max_proactive_preemptions_per_step: int,
        binner: MlfqBinner,
    ) -> None:
        """Install the shared constants and per-call state map.

        Args:
            num_queues: Number of feedback queues ``K``.
            queue_quanta: Per-queue decode-step quantum; must have length
                ``num_queues``.
            beta: Anti-starvation wait-to-service ratio threshold.
            max_proactive_preemptions_per_step: Upper bound on proactive
                preemptions per ``schedule()`` call (0 disables it).
            binner: The binner providing demote / anti-starvation / outranks.

        Raises:
            ValueError: If ``len(queue_quanta) != num_queues``.
        """
        if len(queue_quanta) != num_queues:
            raise ValueError(
                f"queue_quanta must have length num_queues = {num_queues}, "
                f"got {len(queue_quanta)}"
            )
        self.num_queues = num_queues
        self.queue_quanta = tuple(queue_quanta)
        self.beta = beta
        self.max_proactive_preemptions_per_step = max_proactive_preemptions_per_step
        self.binner = binner
        self._call_state: dict[str, CallQueueState] = {}
        self.proactive_preemption_count = 0

    def _register_call_state(self, request: Request, queue_index: int) -> None:
        """Enter a new call at ``queue_index`` with that queue's full quantum."""
        request.priority = queue_index
        self._call_state[request.request_id] = CallQueueState(
            queue_index=queue_index,
            quantum_remaining=self.queue_quanta[queue_index],
            wait_window=0,
            service_window=0,
        )

    def _program_windows(self, request: Request) -> tuple[float, float]:
        """Return the program-level ``(W_p, T_p)`` for the starvation ratio.

        The default is call-level-only behaviour (the MLFQ / FastServe
        baseline); program-aware schedulers override this with process-table
        windows.
        """
        return (0.0, 0.0)

    def _on_service_step(self, req_id: str) -> None:
        """Hook called once per charged decode step (default: no-op)."""

    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
        """Run the policy bookkeeping, then delegate to the base loop.

        The bookkeeping pass (anti-starvation promotion + proactive preemption)
        mutates ``request.priority`` and the waiting heap so that the base
        loop, which is unchanged, admits and evicts calls in policy order.

        Proactive preemptions happen before the base loop, so the base only
        reports its own KV-pressure preemptions in ``preempted_req_ids``; the
        proactive victims are merged in afterwards. The v2 model runner frees a
        request's persistent-batch slot only for ids in ``finished_req_ids`` or
        ``preempted_req_ids``, and a proactive preemption always happens at a
        full batch whose vacated slot is immediately re-admitted -- so an
        unreported victim overflows the worker's ``max_num_reqs`` slots
        ("No free indices"). The worker frees slots before adding new/resumed
        requests, so reporting a victim resumed in this same step is safe.
        """
        now = time.monotonic()
        self._accrue_wait_and_promote()
        preempted_req_ids = self._proactively_preempt(now)
        scheduler_output = super().schedule(throttle_prefills)  # type: ignore[misc]
        if preempted_req_ids:
            if scheduler_output.preempted_req_ids is None:
                scheduler_output.preempted_req_ids = preempted_req_ids
            else:
                scheduler_output.preempted_req_ids |= preempted_req_ids
        return scheduler_output

    def _accrue_wait_and_promote(self) -> None:
        """Accrue wait for waiting calls and promote the starving ones to Q1.

        Every call currently waiting (in either the main or the skipped queue)
        accrues one unit of wait ``W_c``. A call outside Q1 whose
        ``(W_p + W_c) / max(1, T_p + T_c)`` has reached ``beta`` (program
        windows from :meth:`_program_windows`) is promoted to Q1 with its
        call-level windows reset, and the queue is re-heapified so the base
        loop sees the new ordering.
        """
        for queue in (self.waiting, self.skipped_waiting):
            promoted: list[Request] = []
            for request in list(queue):
                # Every queued call was registered by add_request, so its state
                # is present; a missing entry is a real invariant violation.
                state = self._call_state[request.request_id]
                state.wait_window += 1
                if state.queue_index == 0:
                    continue
                program_wait, program_service = self._program_windows(request)
                if self.binner.anti_starvation(
                    program_wait + state.wait_window,
                    program_service + state.service_window,
                    self.beta,
                ):
                    self._promote_to_top(request, state)
                    promoted.append(request)
            if promoted:
                # Priorities changed in place; re-insert to restore heap order.
                queue.remove_requests(promoted)
                for request in promoted:
                    queue.add_request(request)

    def _promote_to_top(self, request: Request, state: CallQueueState) -> None:
        """Promote a starving call to Q1, resetting only its call-level windows.

        The program-level windows (``W_p``, ``T_p``) are deliberately left
        untouched (paper §4.2.2): resetting them would immediately re-starve
        the program's other calls.
        """
        state.queue_index = 0
        state.quantum_remaining = self.queue_quanta[0]
        state.wait_window = 0
        state.service_window = 0
        request.priority = 0

    def _proactively_preempt(self, now: float) -> set[str]:
        """Preempt the worst running call(s) so better waiting calls can run.

        Only fires when the batch is full. Each iteration preempts the current
        worst running call while the best waiting call *strictly* outranks it,
        up to ``max_proactive_preemptions_per_step`` times. Freed calls return
        to the waiting queue (recompute-based, KV blocks freed) so the base
        loop can admit the better waiting calls into the vacated slots. The
        strict-outrank guard makes preemption stop once the queues converge,
        and the per-step cap bounds recompute so it cannot thrash.

        Returns:
            The preempted request ids, to be merged into the step's
            ``SchedulerOutput.preempted_req_ids`` (see :meth:`schedule`).
        """
        preempted_req_ids: set[str] = set()
        if len(self.running) < self.max_num_running_reqs:
            return preempted_req_ids
        while (
            len(preempted_req_ids) < self.max_proactive_preemptions_per_step
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
            preempted_req_ids.add(worst_running.request_id)
        return preempted_req_ids

    def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
        """Charge one decode step of quantum/service and demote on exhaustion.

        ``super()`` (``AsyncScheduler`` -> ``Scheduler``) runs first to reserve
        output placeholders and set ``is_prefill_chunk``. A running call whose
        quantum is exhausted is then demoted one level and its priority updated
        in place. Non-final prefill chunks are not charged, and the base async
        over-schedule guard keeps the charge at one unit per output token under
        run-ahead. Each charged step is also forwarded to
        :meth:`_on_service_step` so program-aware hosts accrue attained service
        in the same pass.
        """
        super()._update_after_schedule(scheduler_output)  # type: ignore[misc]
        for req_id in scheduler_output.num_scheduled_tokens:
            request = self.requests.get(req_id)
            if request is None or request.is_prefill_chunk:
                continue
            # A scheduled call is always registered (see add_request).
            state = self._call_state[req_id]
            state.quantum_remaining -= 1
            state.service_window += 1
            self._on_service_step(req_id)
            if state.quantum_remaining <= 0:
                state.queue_index = self.binner.demote(state.queue_index)
                request.priority = state.queue_index
                state.quantum_remaining = self.queue_quanta[state.queue_index]

    def _release_call_state(self, req_ids: Iterable[str]) -> None:
        """Drop each finished call's per-call state.

        Idempotent per call (``pop`` with a default), so the two completion
        paths -- ``update_from_output`` for normal stops and
        ``finish_requests`` for aborts -- never error on an already-released
        id.
        """
        for req_id in req_ids:
            self._call_state.pop(req_id, None)
