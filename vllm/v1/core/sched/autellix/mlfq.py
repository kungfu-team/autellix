# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Multi-level feedback queue mechanics for the Autellix policies.

The binner maps an attained-service value to a priority queue index, where queue
``0`` is the highest priority (least served). Queues have geometrically growing
quanta, so a call must attain progressively more service before it demotes to
the next queue. The module also provides the demotion, anti-starvation, and
proactive-preemption comparisons shared by the schedulers. It is pure Python
with no vLLM/torch dependency.
"""

import bisect


class MlfqBinner:
    """Assigns attained service to one of ``num_queues`` priority bins.

    Bin ``0`` is the highest priority (least served) and bin ``num_queues - 1``
    is the lowest (most served, unbounded above). The bin of a service value is
    the number of cumulative thresholds it has crossed, so binning is a pure,
    deterministic, monotonic non-decreasing function of service.
    """

    def __init__(
        self,
        num_queues: int,
        first_quantum: float = 1.0,
        growth_ratio: float = 2.0,
        thresholds: list[float] | None = None,
    ) -> None:
        """Build a binner from geometric quanta or explicit thresholds.

        Args:
            num_queues: Number of priority queues ``K`` (``>= 1``).
            first_quantum: Width of the first queue's quantum (``> 0``). Ignored
                when ``thresholds`` is given.
            growth_ratio: Geometric growth ratio of successive quanta (``>= 1``).
                Ignored when ``thresholds`` is given.
            thresholds: Explicit cumulative thresholds, an alternative to the
                geometric construction. Must have length ``num_queues - 1`` and
                be strictly increasing and positive.

        Raises:
            ValueError: If ``num_queues < 1``; if ``thresholds`` is given with
                the wrong length, a non-positive value, or values that are not
                strictly increasing; or, for the geometric path, if
                ``growth_ratio < 1`` or ``first_quantum <= 0``.
        """
        if num_queues < 1:
            raise ValueError(f"num_queues must be >= 1, got {num_queues}")

        if thresholds is not None:
            if len(thresholds) != num_queues - 1:
                raise ValueError(
                    f"thresholds must have length num_queues - 1 = "
                    f"{num_queues - 1}, got {len(thresholds)}"
                )
            if any(threshold <= 0 for threshold in thresholds):
                raise ValueError(f"thresholds must be positive, got {thresholds}")
            if any(hi <= lo for lo, hi in zip(thresholds, thresholds[1:])):
                raise ValueError(
                    f"thresholds must be strictly increasing, got {thresholds}"
                )
            self._thresholds = list(thresholds)
        else:
            if growth_ratio < 1:
                raise ValueError(f"growth_ratio must be >= 1, got {growth_ratio}")
            if first_quantum <= 0:
                raise ValueError(f"first_quantum must be > 0, got {first_quantum}")
            self._thresholds = []
            cumulative = 0.0
            for i in range(num_queues - 1):
                cumulative += first_quantum * growth_ratio**i
                self._thresholds.append(cumulative)

        self._num_queues = num_queues

    def bin(self, service: float) -> int:
        """Return the priority bin for an attained-service value.

        The bin is the number of thresholds ``<= service``, capped at
        ``num_queues - 1``. Thus ``[0, t_0) -> 0``, ``[t_0, t_1) -> 1``, ...,
        ``[t_{K-2}, inf) -> K - 1``; each threshold falls in the higher bin.

        Args:
            service: The attained service to classify.

        Returns:
            The priority bin index in ``[0, num_queues - 1]``.
        """
        return min(bisect.bisect_right(self._thresholds, service), self._num_queues - 1)

    def demote(self, current_bin: int) -> int:
        """Return the next-lower priority bin, saturating at the last queue."""
        return min(current_bin + 1, self._num_queues - 1)

    def anti_starvation(
        self, total_wait: float, total_service: float, beta: float
    ) -> bool:
        """Return whether a starving entity should be promoted to the top queue.

        An entity is starving when it has waited without being served, or when
        its wait-to-service ratio has reached ``beta``.

        Args:
            total_wait: Accumulated waiting time ``W``.
            total_service: Accumulated attained service ``T``.
            beta: Wait-to-service ratio threshold that triggers promotion.

        Returns:
            ``True`` iff ``total_service <= 0 and total_wait > 0`` (waited but
            never served), or ``total_service > 0 and total_wait / total_service
            >= beta``.
        """
        if total_service <= 0:
            return total_wait > 0
        return total_wait / total_service >= beta

    def outranks(self, waiting_bin: int, running_bin: int) -> bool:
        """Return whether a waiting call sits in a strictly better queue.

        Used to decide proactive preemption: a waiting call that outranks a
        running call is a candidate to preempt it.
        """
        return waiting_bin < running_bin
