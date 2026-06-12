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

from src.server.transport import ReceivedFrame, ReceivedPatch
from src.server.recovery.tracker import IoUTracker
from src.server.recovery.kalman import kf_position, bbox_center

@dataclass
class RecoveryStats:
    frames_seen: int = 0
    patches_seen: int = 0
    patches_recovered: int = 0
    patches_failed_recovery: int = 0   # incomplete & no usable track cache
    patches_predicted: int = 0   # virtual patches from predict-only mode

class RecoveryLayer:
    """Wraps IoUTracker + ZoH replacement policy."""

    def __init__(
        self, 
        tracker: IoUTracker,
        kalman_enabled: bool = True,
        predict_enabled: bool = True,
    ) -> None:
        self.tracker = tracker
        self.enabled = True               # bypass switch
        self.kalman_enabled = kalman_enabled
        self.predict_enabled = predict_enabled
        self.stats = RecoveryStats()

    def enhance(self, rf: ReceivedFrame) -> ReceivedFrame:
        # bypass entire recovery if disabled
        if not self.enabled:
            return rf
        
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
                # new_patches.append(p)
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
            if (self.kalman_enabled
                    and old_ex is not None
                    and track.kalman.initialized):
                cx_pred, cy_pred = kf_position(track.kalman)
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

        # ===== Predict-only mode =====
        # For tracks that have NO detection in this frame, generate
        # virtual patches at Kalman-predicted positions using cached
        # patch data. This preserves continuity for briefly-missing
        # objects, at the cost of growing position uncertainty.
        if self.predict_enabled:
            detected_track_ids = set(det_to_track.values())
            for tid, track in self.tracker.tracks.items():
                if tid in detected_track_ids:
                    continue  # detected in this frame
                if not track.last_data:
                    continue  # no cache yet
                if not track.kalman.initialized:
                    continue
                if track.kalman.frames_since_correct > 5:
                    continue  # too stale
                if track.last_expanded_bbox is None:
                    continue

                # Kalman-predicted position
                cx_pred, cy_pred = kf_position(track.kalman)
                old_cx, old_cy = bbox_center(track.last_expanded_bbox)
                dx = int(round(cx_pred - old_cx))
                dy = int(round(cy_pred - old_cy))

                # Translate both bbox and expanded_bbox
                ex1, ey1, ex2, ey2 = track.last_expanded_bbox
                new_expanded = (ex1 + dx, ey1 + dy, ex2 + dx, ey2 + dy)
                bx1, by1, bx2, by2 = track.last_bbox
                new_bbox = (bx1 + dx, by1 + dy, bx2 + dx, by2 + dy)

                # Update track.last_bbox so next frame's IoU matching
                # uses the predicted position (option Y — virtual patch
                # influences track state).
                track.last_bbox = new_bbox

                # Virtual patch (negative det_id to avoid collision).
                virtual_patch = ReceivedPatch(
                    frame_id=rf.frame_id,
                    det_id=-(tid + 1),
                    quality=track.last_quality,
                    bbox=new_bbox,
                    expanded_bbox=new_expanded,
                    confidence=0.0,
                    data=track.last_data,
                    complete=True,
                    chunks_received=1,
                    chunks_expected=1,
                    recovered=True,
                )
                new_patches.append(virtual_patch)
                self.stats.patches_predicted += 1

        # Build new frame with replaced patch list. Other fields shared.
        return dataclasses.replace(rf, patches=new_patches)