"""Constraint simulator filters for UDPReceiver.packet_filter seam.

Filters operate at chunk (UDP packet) level, applied immediately on receive
before any parsing. Returning None signals a simulated drop; returning bytes
(possibly identical) passes the packet through, optionally after a delay.

Filters are stateful classes so they can track their own statistics, since
dropped packets are not counted in UDPReceiver.stats (the receiver short-
circuits with `continue` when the filter returns None).
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Callable, List, Optional


@dataclass
class FilterStats:
    seen: int = 0
    dropped: int = 0
    delayed: int = 0
    total_delay_ms: float = 0.0

    @property
    def drop_rate(self) -> float:
        return self.dropped / self.seen if self.seen > 0 else 0.0

    @property
    def avg_delay_ms(self) -> float:
        return self.total_delay_ms / self.delayed if self.delayed > 0 else 0.0


class RandomDropFilter:
    """Drop each packet independently with probability p."""

    def __init__(self, p: float, seed: Optional[int] = None) -> None:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"drop probability must be in [0, 1], got {p}")
        self.p = p
        self._rng = random.Random(seed)
        self.stats = FilterStats()

    def set_drop_prob(self, new_p: float) -> None:
        """Update drop probability at runtime. Used by dashboard."""
        if not 0.0 <= new_p <= 1.0:
            raise ValueError(f"drop_prob must be in [0, 1], got {new_p}")
        self.p = new_p

    def __call__(self, buf: bytes) -> Optional[bytes]:
        self.stats.seen += 1
        if self._rng.random() < self.p:
            self.stats.dropped += 1
            return None
        return buf

class DelayJitterFilter:
    """Delay each packet by a uniformly random duration in [min_ms, max_ms]."""

    def __init__(
        self,
        min_ms: float,
        max_ms: float,
        seed: Optional[int] = None,
    ) -> None:
        if min_ms < 0 or max_ms < min_ms:
            raise ValueError(f"invalid delay range: [{min_ms}, {max_ms}]")
        self.min_ms = min_ms
        self.max_ms = max_ms
        self._rng = random.Random(seed)
        self.stats = FilterStats()

    def __call__(self, buf: bytes) -> bytes:
        self.stats.seen += 1
        if self.max_ms > 0:
            delay_ms = self._rng.uniform(self.min_ms, self.max_ms)
            time.sleep(delay_ms / 1000.0)
            self.stats.delayed += 1
            self.stats.total_delay_ms += delay_ms
        return buf


class ChainedFilter:
    """Apply filters in order. Short-circuit to None on first drop."""

    def __init__(self, *filters: Callable[[bytes], Optional[bytes]]) -> None:
        if not filters:
            raise ValueError("ChainedFilter requires at least one filter")
        self.filters = filters

    def __call__(self, buf: bytes) -> Optional[bytes]:
        out: Optional[bytes] = buf
        for f in self.filters:
            out = f(out)
            if out is None:
                return None
        return out


def build_filter(
    *,
    drop_prob: float = 0.0,
    delay_min_ms: float = 0.0,
    delay_max_ms: float = 0.0,
    seed: Optional[int] = None,
) -> Optional[Callable[[bytes], Optional[bytes]]]:
    """Build a composite filter from CLI args. Returns None if no constraints active.

    Drop filter is placed first so that dropped packets skip the sleep cost.
    """
    filters: List[Callable[[bytes], Optional[bytes]]] = []
    if drop_prob > 0.0:
        filters.append(RandomDropFilter(drop_prob, seed=seed))
    if delay_max_ms > 0.0:
        filters.append(DelayJitterFilter(delay_min_ms, delay_max_ms, seed=seed))
    if not filters:
        return None
    if len(filters) == 1:
        return filters[0]
    return ChainedFilter(*filters)