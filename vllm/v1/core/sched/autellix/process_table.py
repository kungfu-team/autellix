# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-program scheduling state for the Autellix policies.

A *program* is one multi-step agentic workflow invocation that issues many LLM
calls. The scheduler groups calls by ``program_id`` and accumulates the
program's attained service and critical-path scalars here so that program-aware
policies (PLAS, ATLAS) and anti-starvation logic can consult a single source of
truth. The module is pure Python with no vLLM/torch dependency.
"""

from dataclasses import dataclass, field


@dataclass
class ProgramState:
    """Accumulated scheduling state for a single program.

    Attributes:
        program_id: Stable identifier shared by every call of the program.
        service: Cumulative attained service summed over the program's completed
            calls (the PLAS sum rule). Also acts as ``T`` in the anti-starvation
            ratio ``W / T``.
        max_critical_path: The ATLAS scalar, i.e. the maximum over the program's
            calls of ``start_priority + call_service``. It never decreases.
        total_wait: Accumulated waiting time across the program's calls (``W`` in
            anti-starvation).
        last_arrival: Arrival timestamp of the most recently registered call.
        last_completion: Completion timestamp of the most recently completed
            call; drives time-to-live garbage collection.
        active_call_ids: Identifiers of the program's currently in-flight calls.
    """

    program_id: str
    service: float = 0.0
    max_critical_path: float = 0.0
    total_wait: float = 0.0
    last_arrival: float = 0.0
    last_completion: float = 0.0
    active_call_ids: set[str] = field(default_factory=set)


class ProcessTable:
    """A time-to-live map from ``program_id`` to :class:`ProgramState`.

    Mutating methods create the program entry on first reference so callers never
    have to pre-register a program; :meth:`get` alone returns ``None`` for an
    unknown program so callers can fall back gracefully.
    """

    def __init__(self, ttl: float) -> None:
        """Initialize the table.

        Args:
            ttl: Seconds a program may remain idle (no active calls) after its
                last completion before :meth:`gc` may evict it.
        """
        self._ttl = ttl
        self._states: dict[str, ProgramState] = {}

    def get(self, program_id: str) -> ProgramState | None:
        """Return the program's state, or ``None`` if it is unknown."""
        return self._states.get(program_id)

    def get_or_create(self, program_id: str) -> ProgramState:
        """Return the program's state, creating an empty entry if needed."""
        state = self._states.get(program_id)
        if state is None:
            state = ProgramState(program_id=program_id)
            self._states[program_id] = state
        return state

    def add_service(self, program_id: str, amount: float) -> None:
        """Add attained service to a program (the PLAS sum rule).

        Service must be monotonic non-decreasing, so negative amounts are
        ignored rather than applied.

        Args:
            program_id: The program to credit.
            amount: Non-negative service to add; negative values are a no-op.
        """
        if amount < 0:
            return
        self.get_or_create(program_id).service += amount

    def update_critical_path(
        self, program_id: str, start_priority: float, call_service: float
    ) -> None:
        """Fold a call's critical path into the program scalar (the max rule).

        Sets ``max_critical_path = max(max_critical_path, start_priority +
        call_service)``, so the scalar never decreases.

        Args:
            program_id: The program to update.
            start_priority: The call's start priority (inherited program scalar).
            call_service: The service the call attained.
        """
        state = self.get_or_create(program_id)
        state.max_critical_path = max(
            state.max_critical_path, start_priority + call_service
        )

    def add_wait(self, program_id: str, amount: float) -> None:
        """Accumulate waiting time (``W``) for a program."""
        self.get_or_create(program_id).total_wait += amount

    def register_call(self, program_id: str, call_id: str, arrival_time: float) -> None:
        """Record that a call of the program has arrived and is in flight.

        Args:
            program_id: The owning program.
            call_id: The arriving call's identifier.
            arrival_time: The call's arrival timestamp.
        """
        state = self.get_or_create(program_id)
        state.active_call_ids.add(call_id)
        state.last_arrival = arrival_time

    def complete_call(
        self, program_id: str, call_id: str, completion_time: float
    ) -> None:
        """Record that a call of the program has completed.

        Safe to call more than once, or with a ``call_id`` that is not currently
        active: the call id is simply discarded if present.

        Args:
            program_id: The owning program.
            call_id: The completing call's identifier.
            completion_time: The call's completion timestamp.
        """
        state = self.get_or_create(program_id)
        state.active_call_ids.discard(call_id)
        state.last_completion = completion_time

    def gc(self, now: float) -> list[str]:
        """Evict idle programs whose time-to-live has elapsed.

        A program is evicted only if it has no active calls *and*
        ``last_completion + ttl <= now``. Programs with in-flight calls are never
        evicted.

        Args:
            now: The current time.

        Returns:
            The ids of the evicted programs, in insertion order.
        """
        evicted = [
            program_id
            for program_id, state in self._states.items()
            if not state.active_call_ids and state.last_completion + self._ttl <= now
        ]
        for program_id in evicted:
            del self._states[program_id]
        return evicted
