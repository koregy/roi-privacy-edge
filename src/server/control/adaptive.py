"""PI-style adaptive JPEG quality controller.

Reads the current patch reception ratio (from the decision state machine)
and adjusts a target JPEG quality value via proportional + integral
feedback. Anti-windup clamps the integral term so transient saturation
doesn't lock the controller in a degraded state after recovery.

This is a textbook PI controller specialized for a single SISO loop
(scalar input ratio, scalar output quality). In the language of RQ4,
this is a filtering operation acting on a control variable: the
controller is a stateful filter mapping observed reception quality to
a recommended encoder setting.

Note: this module computes the target quality only. Delivering it to
the edge encoder is the responsibility of the feedback channel (Day 4)
or an open-loop fallback that re-uses the same computation at the edge.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class AdaptiveStats:
    updates: int = 0
    quality_min_seen: int = 95
    quality_max_seen: int = 30
    quality_history: List[int] = field(default_factory=list)


class PIDAdaptiveController:
    """PI controller: error = target_ratio - current_ratio, output = quality.

    Sign convention: positive error means we're losing patches (current
    below target). Controller responds by *lowering* quality, which
    reduces patch byte size, which reduces UDP chunk count per patch,
    which reduces patch-level loss probability under a fixed per-chunk
    drop rate.
    """

    def __init__(
        self,
        kp: float = 30.0,
        ki: float = 5.0,
        target_ratio: float = 0.85,
        min_quality: int = 30,
        max_quality: int = 95,
        initial_quality: int = 75,
        integral_clamp: float = 5.0,
    ) -> None:
        if not 0.0 < target_ratio <= 1.0:
            raise ValueError(f"target_ratio must be in (0, 1], got {target_ratio}")
        if not 0 < min_quality < max_quality <= 100:
            raise ValueError(
                f"need 0 < min_q ({min_quality}) < max_q ({max_quality}) <= 100"
            )
        if not min_quality <= initial_quality <= max_quality:
            raise ValueError(
                f"initial_quality {initial_quality} not in [{min_quality}, {max_quality}]"
            )

        self.kp = kp
        self.ki = ki
        self.target_ratio = target_ratio
        self.min_q = min_quality
        self.max_q = max_quality
        self.integral_clamp = integral_clamp

        self._quality: int = initial_quality
        self._integral: float = 0.0
        self.stats = AdaptiveStats()
        self.stats.quality_history.append(self._quality)

    @property
    def quality(self) -> int:
        return self._quality

    def update(self, current_ratio: float) -> int:
        """One control step. Returns the new target quality."""
        error = self.target_ratio - current_ratio
        self._integral += error
        # Anti-windup: prevent integral from growing unboundedly when output saturates.
        self._integral = max(
            -self.integral_clamp,
            min(self.integral_clamp, self._integral),
        )
        # Negative coefficient: positive error -> lower quality.
        delta = -(self.kp * error + self.ki * self._integral)
        new_q = self._quality + delta
        self._quality = int(max(self.min_q, min(self.max_q, new_q)))

        # Stats
        self.stats.updates += 1
        self.stats.quality_min_seen = min(self.stats.quality_min_seen, self._quality)
        self.stats.quality_max_seen = max(self.stats.quality_max_seen, self._quality)
        self.stats.quality_history.append(self._quality)

        return self._quality