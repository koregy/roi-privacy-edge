"""Frame reordering buffer.

UDPReceiver emits ReceivedFrame events in completion order (whichever
frame fills its expected_patches first), not in frame_id order. This
buffer reorders them by frame_id with bounded latency.

Two emit conditions:
  (a) Buffer size threshold: when >= max_size frames are queued, emit
      the one with smallest frame_id.
  (b) Per-frame timeout: a frame in the buffer for more than max_wait_s
      is emitted regardless of out-of-order siblings still in flight.

The buffer is order-preserving across consecutive frame_ids when traffic
is dense. Under heavy loss it degrades gracefully to lossy reordering.
"""
from __future__ import annotations

import heapq
import time
from dataclasses import dataclass, field
from typing import List, Optional

from src.server.transport import ReceivedFrame


@dataclass
class ReorderStats:
    frames_in: int = 0
    frames_out: int = 0
    emitted_by_size: int = 0
    emitted_by_timeout: int = 0
    emitted_by_flush: int = 0
    out_of_order_at_emit: int = 0   # frame emitted with frame_id < last emitted


class ReorderBuffer:
    """Min-heap of (frame_id, arrival_time, frame) tuples."""

    def __init__(self, max_size: int = 5, max_wait_s: float = 0.5) -> None:
        if max_size < 1:
            raise ValueError(f"max_size must be >= 1, got {max_size}")
        if max_wait_s <= 0:
            raise ValueError(f"max_wait_s must be > 0, got {max_wait_s}")
        self.max_size = max_size
        self.max_wait_s = max_wait_s
        self._heap: List[tuple[int, float, int, ReceivedFrame]] = []
        self._tiebreak = 0   # ensures heap stability when frame_ids equal
        self._last_emitted_id: int = -1
        self.stats = ReorderStats()

    def push(self, rf: ReceivedFrame) -> List[ReceivedFrame]:
        """Queue a frame. Returns any frames now eligible for emit."""
        now = time.perf_counter()
        heapq.heappush(self._heap, (rf.frame_id, now, self._tiebreak, rf))
        self._tiebreak += 1
        self.stats.frames_in += 1

        emitted: List[ReceivedFrame] = []
        # Emit while either size or timeout condition is met.
        while self._heap:
            frame_id, arrived_at, _, frame = self._heap[0]
            by_size = len(self._heap) >= self.max_size
            by_timeout = (now - arrived_at) >= self.max_wait_s
            if not by_size and not by_timeout:
                break
            heapq.heappop(self._heap)
            if by_size:
                self.stats.emitted_by_size += 1
            else:
                self.stats.emitted_by_timeout += 1
            self._emit_track(frame)
            emitted.append(frame)
        return emitted

    def flush(self) -> List[ReceivedFrame]:
        """Drain everything in order (called at shutdown)."""
        emitted: List[ReceivedFrame] = []
        while self._heap:
            _, _, _, frame = heapq.heappop(self._heap)
            self.stats.emitted_by_flush += 1
            self._emit_track(frame)
            emitted.append(frame)
        return emitted

    def _emit_track(self, frame: ReceivedFrame) -> None:
        self.stats.frames_out += 1
        if frame.frame_id < self._last_emitted_id:
            self.stats.out_of_order_at_emit += 1
        else:
            self._last_emitted_id = frame.frame_id