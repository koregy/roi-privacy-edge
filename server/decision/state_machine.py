"""Confidence-weighted 3-state decision logic.

Observes the patch reception ratio over a sliding window and classifies
the system into Normal / Warning / Emergency. Transitions are protected
by a hysteresis counter (require N consecutive frames at the new state
before switching) to avoid chattering near threshold boundaries.

The ratio is weighted by YOLOv8 detection confidence, so missing a
high-confidence patch penalizes the ratio more than missing a low-
confidence one. This is a deterministic filtering operation in the
sense of RQ4 (thresholding + confidence-weighted filtering).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Deque, Optional

from server.transport import ReceivedFrame


class DecisionState(Enum):
    NORMAL = "NORMAL"
    WARNING = "WARNING"
    EMERGENCY = "EMERGENCY"


@dataclass
class StateStats:
    frames_normal: int = 0
    frames_warning: int = 0
    frames_emergency: int = 0
    transitions: int = 0


class DecisionStateMachine:
    """Sliding-window confidence-weighted state machine with hysteresis."""

    def __init__(
        self,
        window_size: int = 10,
        warn_threshold: float = 0.80,
        emerg_threshold: float = 0.50,
        hysteresis_frames: int = 2,
    ) -> None:
        if not 0.0 < emerg_threshold < warn_threshold <= 1.0:
            raise ValueError(
                f"thresholds must satisfy 0 < emerg ({emerg_threshold}) "
                f"< warn ({warn_threshold}) <= 1"
            )
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")
        if hysteresis_frames < 1:
            raise ValueError(f"hysteresis_frames must be >= 1, got {hysteresis_frames}")

        self.window_size = window_size
        self.warn_threshold = warn_threshold
        self.emerg_threshold = emerg_threshold
        self.hysteresis = hysteresis_frames
        self._history: Deque[float] = deque(maxlen=window_size)
        self._state = DecisionState.NORMAL
        self._pending_state: Optional[DecisionState] = None
        self._pending_count = 0
        self.stats = StateStats()

    @property
    def state(self) -> DecisionState:
        return self._state

    @property
    def current_ratio(self) -> float:
        """Average reception ratio over the current window."""
        if not self._history:
            return 1.0
        return sum(self._history) / len(self._history)

    def update(self, rf: ReceivedFrame) -> DecisionState:
        """Process one frame; update history, decide state with hysteresis."""
        # Compute confidence-weighted reception ratio for this frame.
        if not rf.patches:
            # No detections in this frame — no concern, treat as 1.0.
            ratio = 1.0
        else:
            weighted_complete = sum(
                p.confidence for p in rf.patches if p.complete
            )
            weighted_total = sum(p.confidence for p in rf.patches)
            ratio = weighted_complete / weighted_total if weighted_total > 0 else 1.0
        self._history.append(ratio)

        avg_ratio = self.current_ratio

        # Determine target state from thresholds.
        if avg_ratio >= self.warn_threshold:
            target = DecisionState.NORMAL
        elif avg_ratio >= self.emerg_threshold:
            target = DecisionState.WARNING
        else:
            target = DecisionState.EMERGENCY

        # Hysteresis: require N consecutive frames at target before switching.
        if target == self._state:
            # No change requested — reset pending.
            self._pending_state = None
            self._pending_count = 0
        elif target == self._pending_state:
            # Same pending change accumulating.
            self._pending_count += 1
            if self._pending_count >= self.hysteresis:
                self._state = target
                self._pending_state = None
                self._pending_count = 0
                self.stats.transitions += 1
        else:
            # New pending direction; restart counter.
            self._pending_state = target
            self._pending_count = 1

        # Count current state.
        if self._state == DecisionState.NORMAL:
            self.stats.frames_normal += 1
        elif self._state == DecisionState.WARNING:
            self.stats.frames_warning += 1
        else:
            self.stats.frames_emergency += 1

        return self._state