"""Zero-order Hold recovery layer.

Replaces incomplete or undecodable patches with the last known complete
patch from the same track. Tracks are maintained by IoUTracker.

The replaced patch keeps the CURRENT frame's bbox (so the visualization
shows where the object is now) but uses the PRIOR patch's data and
expanded_bbox (so the JPEG decodes and pastes correctly into the canvas
at the position where that data was originally captured).

This is the simplest possible recovery — no interpolation, no motion
compensation. Suitable when frame-to-frame object motion is small
relative to bbox size. ZoH artefacts (slight position lag, object
appearing frozen) are intentional and visible in the demo.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from server.transport import ReceivedFrame, ReceivedPatch
from server.recovery.tracker import IoUTracker
from server.recovery.kalman import kf_position, bbox_center

@dataclass
class RecoveryStats:
    frames_seen: int = 0
    patches_seen: int = 0
    patches_recovered: int = 0
    patches_failed_recovery: int = 0   # incomplete & no usable track cache


class RecoveryLayer:
    """Wraps IoUTracker + ZoH replacement policy."""

    def __init__(self, tracker: IoUTracker) -> None:
        self.tracker = tracker
        self.stats = RecoveryStats()

    def enhance(self, rf: ReceivedFrame) -> ReceivedFrame:
        """Return a new ReceivedFrame with incomplete patches replaced
        from track cache when possible.

        Original `rf` is not mutated. New patches list replaces originals;
        complete patches pass through by reference (cheap).
        """
        self.stats.frames_seen += 1

        # ReceivedFrame.patches is a List[ReceivedPatch] (sorted by det_id).
        det_to_track = self.tracker.update(rf.patches)

        new_patches: list[ReceivedPatch] = []
        for p in rf.patches:
            self.stats.patches_seen += 1

            if p.complete:
                # Pass through. Tracker already cached data in update().
                new_patches.append(p)
                continue

            # Patch is incomplete — try recovery.
            track_id = det_to_track.get(p.det_id)
            track = self.tracker.tracks.get(track_id) if track_id is not None else None

            if track is None or not track.last_data:
                # No matching track or matched track has no cached complete data yet.
                self.stats.patches_failed_recovery += 1
                new_patches.append(p)
                continue

            # Recover with motion compensation.
            # 1. Use Kalman prediction to estimate where the object is NOW.
            # 2. Translate the prior expanded_bbox so the cached image is
            #    pasted at the predicted current position.
            # The bbox itself is left unchanged (current detection's bbox),
            # so visualization shows the actual detection location.
            cx_pred, cy_pred = kf_position(track.kalman)
            # Old expanded_bbox center (when this data was captured).
            old_ex = track.last_expanded_bbox
            if old_ex is not None and track.kalman.initialized:
                old_cx, old_cy = bbox_center(old_ex)
                dx = int(round(cx_pred - old_cx))
                dy = int(round(cy_pred - old_cy))
                # Translate the expanded_bbox by (dx, dy).
                x1, y1, x2, y2 = old_ex
                new_expanded = (x1 + dx, y1 + dy, x2 + dx, y2 + dy)
            else:
                # Fall back to static ZoH if no Kalman state.
                new_expanded = track.last_expanded_bbox

            recovered = dataclasses.replace(
                p,
                data=track.last_data,
                expanded_bbox=new_expanded,
                complete=True,
                recovered=True,
            )
            new_patches.append(recovered)
            self.stats.patches_recovered += 1

        # Build new frame with replaced patch list. Other fields shared.
        return dataclasses.replace(rf, patches=new_patches)