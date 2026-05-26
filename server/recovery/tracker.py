"""IoU-based greedy multi-object tracker for recovery layer.

Matches per-frame detections (ReceivedPatch.bbox) to persistent tracks
across frames. Used by RecoveryLayer to identify "same object" across
time so that Zero-order Hold can fill missing patches from the last
known good capture of that object.

Design choices
--------------
- Greedy IoU matching: at small object counts (1-10) this is indistinguishable
  from Hungarian and is easier to debug.
- Tracks age out after max_age frames without a match.
- Tracks are keyed by an integer track_id, monotonically increasing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from server.transport import ReceivedFrame, ReceivedPatch


Bbox = Tuple[int, int, int, int]


def iou(a: Bbox, b: Bbox) -> float:
    """Intersection-over-union of two (x1, y1, x2, y2) boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    a_area = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    b_area = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = a_area + b_area - inter
    if union <= 0:
        return 0.0
    return inter / union


@dataclass
class Track:
    """Per-object state held across frames.

    `last_*` fields hold the most recent COMPLETE patch's data, used to
    fill in for this object when a future frame's patch is incomplete.
    """
    track_id: int
    last_bbox: Bbox
    last_expanded_bbox: Optional[Bbox]
    last_data: bytes
    last_quality: int
    last_frame_id: int
    age: int = 0   # frames since last match


@dataclass
class TrackerStats:
    next_track_id: int = 0
    matches: int = 0
    new_tracks: int = 0
    expired_tracks: int = 0


class IoUTracker:
    """Greedy IoU tracker keyed on patch.bbox."""

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_age: int = 10,
    ) -> None:
        if not 0.0 < iou_threshold <= 1.0:
            raise ValueError(f"iou_threshold must be in (0, 1], got {iou_threshold}")
        if max_age < 1:
            raise ValueError(f"max_age must be >= 1, got {max_age}")
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self._tracks: Dict[int, Track] = {}
        self.stats = TrackerStats()

    @property
    def tracks(self) -> Dict[int, Track]:
        return self._tracks

    def update(self, patches: List[ReceivedPatch]) -> Dict[int, int]:
        """Match incoming patches to existing tracks.

        Returns
        -------
        dict mapping patch.det_id -> track_id for every patch in `patches`.
        Unmatched patches receive a new track_id.

        Side effect: aging + expiring tracks, caching complete-patch data
        into the matched track.
        """
        # Age all existing tracks by one; we'll reset to 0 on match below.
        for t in self._tracks.values():
            t.age += 1

        # Build pairwise IoU between current patches and existing tracks.
        # Greedy: pick the highest IoU pair, assign, repeat.
        candidates: List[Tuple[float, int, int]] = []  # (iou, det_id, track_id)
        for p in patches:
            for tid, t in self._tracks.items():
                score = iou(p.bbox, t.last_bbox)
                if score >= self.iou_threshold:
                    candidates.append((score, p.det_id, tid))
        candidates.sort(reverse=True)

        det_to_track: Dict[int, int] = {}
        used_tracks: set[int] = set()
        used_dets: set[int] = set()
        for score, det_id, tid in candidates:
            if det_id in used_dets or tid in used_tracks:
                continue
            det_to_track[det_id] = tid
            used_dets.add(det_id)
            used_tracks.add(tid)
            self.stats.matches += 1

        # Patches without a match get new tracks.
        for p in patches:
            if p.det_id in det_to_track:
                continue
            new_id = self.stats.next_track_id
            self.stats.next_track_id += 1
            self.stats.new_tracks += 1
            self._tracks[new_id] = Track(
                track_id=new_id,
                last_bbox=p.bbox,
                last_expanded_bbox=p.expanded_bbox,
                last_data=p.data if p.complete else b"",
                last_quality=p.quality,
                last_frame_id=p.frame_id,
                age=0,
            )
            det_to_track[p.det_id] = new_id

        # Update matched tracks: bbox always; data only if complete.
        for det_id, tid in det_to_track.items():
            patch = next(p for p in patches if p.det_id == det_id)
            t = self._tracks[tid]
            t.last_bbox = patch.bbox
            t.last_frame_id = patch.frame_id
            t.age = 0
            if patch.complete:
                t.last_data = patch.data
                t.last_expanded_bbox = patch.expanded_bbox
                t.last_quality = patch.quality

        # Expire stale tracks.
        expired = [tid for tid, t in self._tracks.items() if t.age > self.max_age]
        for tid in expired:
            del self._tracks[tid]
            self.stats.expired_tracks += 1

        return det_to_track