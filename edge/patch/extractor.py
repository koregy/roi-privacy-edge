"""Patch extractor: turn person detections into croppable image patches.

Given an original frame and a list of Detections from YOLOv8TRT,
produces a list of Patch objects ready for JPEG encoding and transport.

Design notes
------------
- Only person (COCO class 0) detections are kept; other classes are
  ignored at this stage (we may revisit if we ever add e.g. vehicle ROI).
- Each bbox is expanded by BBOX_MARGIN on every side before cropping,
  then clipped to frame bounds. The margin gives downstream stitching
  some slack against bbox jitter.
- Patches smaller than MIN_PATCH_SIZE on either dimension after clipping
  are dropped — they cost packet overhead without giving recovery much.
- The returned Patch carries enough metadata (frame_id, det_id, original
  bbox in frame coords, expanded bbox actually used) for the server side
  to reconstruct where each crop belongs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from common.config import (
    BBOX_MARGIN,
    MIN_PATCH_SIZE,
    PERSON_CLASS_ID,
)
from edge.detector.yolov8_trt import Detection


@dataclass
class Patch:
    """A cropped region of interest ready to be JPEG-encoded.

    Attributes
    ----------
    frame_id : int
        Monotonic frame counter from the capture loop. The transport layer
        bundles this into packet headers so the server can reassemble.
    det_id : int
        Index of this patch within its source frame (0..N-1 for N persons).
    image : np.ndarray
        The cropped BGR pixel data, shape (H, W, 3), uint8. This is what
        gets handed to the JPEG encoder.
    original_bbox : tuple[int, int, int, int]
        Original detection bbox (x1, y1, x2, y2) in full-frame coordinates,
        BEFORE margin expansion. The server uses this to place the patch.
    expanded_bbox : tuple[int, int, int, int]
        Bbox actually used for cropping (x1, y1, x2, y2), AFTER margin
        expansion and clipping to frame bounds. Equals `image`'s placement.
    conf : float
        Detector confidence, forwarded for downstream prioritization.
    """

    frame_id: int
    det_id: int
    image: np.ndarray
    original_bbox: tuple[int, int, int, int]
    expanded_bbox: tuple[int, int, int, int]
    conf: float

    @property
    def shape(self) -> tuple[int, int]:
        """(height, width) of the crop, for quick logging."""
        return self.image.shape[0], self.image.shape[1]


def _expand_and_clip(
    bbox: tuple[int, int, int, int],
    frame_w: int,
    frame_h: int,
    margin: float,
) -> tuple[int, int, int, int]:
    """Expand bbox by `margin` fraction on each side, then clip to frame.

    Margin is computed relative to the bbox dimensions so a small bbox
    gets a proportionally small margin (we don't want a 30x30 bbox to
    suddenly grow by 50 pixels on each side).
    """
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    dx = int(round(w * margin))
    dy = int(round(h * margin))

    x1e = max(0, x1 - dx)
    y1e = max(0, y1 - dy)
    x2e = min(frame_w, x2 + dx)
    y2e = min(frame_h, y2 + dy)

    return x1e, y1e, x2e, y2e


def extract_patches(
    frame: np.ndarray,
    detections: Sequence[Detection],
    frame_id: int,
    margin: float = BBOX_MARGIN,
    min_size: int = MIN_PATCH_SIZE,
    person_class_id: int = PERSON_CLASS_ID,
) -> List[Patch]:
    """Extract person patches from a frame.

    Parameters
    ----------
    frame : np.ndarray
        Source image, BGR, shape (H, W, 3), uint8 — the same frame that
        was fed to the detector.
    detections : sequence of Detection
        Raw detector output. Non-person detections are filtered out here.
    frame_id : int
        Frame counter to stamp on every emitted patch.
    margin, min_size, person_class_id : see common.config

    Returns
    -------
    list of Patch
        Empty list if no qualifying person detections.

    Notes
    -----
    The crop is a numpy view into `frame` (no copy). If the caller mutates
    `frame` afterward, the patch data changes too. The JPEG encoder reads
    the data immediately, so in the normal pipeline this is fine; if you
    queue patches across frames, call `.copy()` on `Patch.image` first.
    """
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(
            f"Expected BGR frame of shape (H, W, 3), got {frame.shape}"
        )
    frame_h, frame_w = frame.shape[:2]

    patches: List[Patch] = []
    det_id = 0
    for det in detections:
        if det.class_id != person_class_id:
            continue

        orig = (int(det.x1), int(det.y1), int(det.x2), int(det.y2))
        x1e, y1e, x2e, y2e = _expand_and_clip(orig, frame_w, frame_h, margin)

        # Skip degenerate / too-small crops after clipping.
        if (x2e - x1e) < min_size or (y2e - y1e) < min_size:
            continue

        crop = frame[y1e:y2e, x1e:x2e]  # view, not copy
        patches.append(
            Patch(
                frame_id=frame_id,
                det_id=det_id,
                image=crop,
                original_bbox=orig,
                expanded_bbox=(x1e, y1e, x2e, y2e),
                conf=float(det.confidence),
            )
        )
        det_id += 1

    return patches
