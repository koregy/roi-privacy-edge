"""Constant-velocity Kalman filter for bbox center tracking.

State: [cx, cy, vx, vy] where (cx, cy) is bbox center, (vx, vy) is
velocity in pixels per frame (assuming dt=1 frame).

Measurement: [cx, cy] from observed bbox.

Used by recovery layer to predict object position during patch loss,
enabling image translation instead of static ZoH placement.

Reset behavior: when frames_since_correct exceeds a threshold, the
filter is treated as stale (caller should probably not use prediction).
This avoids extrapolating too far into the future from a single bad
observation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np


# Constant-velocity transition matrix (dt = 1 frame).
_F = np.array([
    [1.0, 0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0, 1.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
])

# Observation matrix: measure (cx, cy) only.
_H = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
])

# Process noise: allows position drift and modest velocity changes.
_Q = np.diag([1.0, 1.0, 4.0, 4.0])

# Measurement noise: bbox center accurate to ~3 pixels stdev.
_R = np.diag([9.0, 9.0])


@dataclass
class KalmanState:
    """Per-track filter state."""
    x: np.ndarray = field(default_factory=lambda: np.zeros(4))
    P: np.ndarray = field(default_factory=lambda: np.eye(4) * 100.0)
    initialized: bool = False
    frames_since_correct: int = 0


def kf_predict(state: KalmanState) -> None:
    """Advance state by one frame (in-place)."""
    if not state.initialized:
        return
    state.x = _F @ state.x
    state.P = _F @ state.P @ _F.T + _Q
    state.frames_since_correct += 1


def kf_correct(state: KalmanState, cx: float, cy: float) -> None:
    """Update state with a new (cx, cy) observation (in-place)."""
    if not state.initialized:
        state.x = np.array([cx, cy, 0.0, 0.0])
        state.P = np.diag([9.0, 9.0, 100.0, 100.0])
        state.initialized = True
        state.frames_since_correct = 0
        return

    z = np.array([cx, cy])
    y = z - _H @ state.x
    S = _H @ state.P @ _H.T + _R
    K = state.P @ _H.T @ np.linalg.inv(S)
    state.x = state.x + K @ y
    state.P = (np.eye(4) - K @ _H) @ state.P
    state.frames_since_correct = 0


def kf_position(state: KalmanState) -> Tuple[float, float]:
    """Current estimated (cx, cy)."""
    if not state.initialized:
        return 0.0, 0.0
    return float(state.x[0]), float(state.x[1])


def bbox_center(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
    """Convert (x1, y1, x2, y2) to center (cx, cy)."""
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0

def kf_predicted_bbox(state: KalmanState, last_bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    """Predict where last_bbox will be in the current frame.

    Returns last_bbox translated by Kalman-predicted velocity.
    If filter is not initialized, returns last_bbox unchanged.
    """
    if not state.initialized:
        return last_bbox
    # Old center (when last_bbox was observed)
    x1, y1, x2, y2 = last_bbox
    old_cx = (x1 + x2) / 2.0
    old_cy = (y1 + y2) / 2.0
    # Predicted current center
    new_cx, new_cy = kf_position(state)
    dx = int(round(new_cx - old_cx))
    dy = int(round(new_cy - old_cy))
    return (x1 + dx, y1 + dy, x2 + dx, y2 + dy)