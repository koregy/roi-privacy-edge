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

            # Recover: replace data + expanded_bbox with prior values.
            # Keep CURRENT bbox (where the object is now) for visualization.
            recovered = dataclasses.replace(
                p,
                data=track.last_data,
                expanded_bbox=track.last_expanded_bbox,
                complete=True,
                recovered=True,
            )
            new_patches.append(recovered)
            self.stats.patches_recovered += 1

        # Build new frame with replaced patch list. Other fields shared.
        return dataclasses.replace(rf, patches=new_patches)