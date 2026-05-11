"""Naive ROI stitcher: paste received patches onto a neutral background.

This is the Week 1 baseline. It does NOT recover lost patches — those
areas remain blank background. The recovery layer (Week 2) builds on
top of this by filling missing patches from prior frames.

Behaviour
---------
- Background colour is configurable (default mid-grey 128). Choosing
  grey rather than black/white avoids biasing PSNR/SSIM when we later
  compare stitched ↔ original.
- A patch is pasted at its `expanded_bbox` (the same rectangle the
  edge crop was taken from). This means margin-extended areas overwrite
  background, giving a thin halo around each person.
- Incomplete patches: cv2.imdecode either succeeds with whatever bytes
  arrived (truncated JPEG often decodes the top portion) or fails. On
  failure we skip the patch — the area stays background.
- For visualization, we optionally draw a thin rectangle at original
  bbox so the viewer can tell complete vs incomplete frames apart.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from server.transport import ReceivedFrame, ReceivedPatch


# Background grey — middle of [0,255], neutral for downstream metrics.
DEFAULT_BG_VALUE = 128

# Default fallback size when FRAME_HEADER was lost and patches don't
# tell us the source resolution. We won't paste anything outside this,
# but at least the function won't crash. Real deployments should always
# have headers; this is a safety net.
DEFAULT_FALLBACK_SIZE = (720, 1280)  # (H, W)


@dataclass
class StitchResult:
    """Output of the stitcher for one frame."""

    frame_id: int
    image: np.ndarray              # BGR, (H, W, 3), uint8
    n_pasted: int                  # successfully decoded + pasted patches
    n_skipped_decode: int          # bytes present but cv2.imdecode failed
    n_skipped_incomplete: int      # patches marked incomplete (we still try)


def stitch_frame(
    frame: ReceivedFrame,
    *,
    bg_value: int = DEFAULT_BG_VALUE,
    draw_bbox: bool = False,
    fallback_size: tuple[int, int] = DEFAULT_FALLBACK_SIZE,
) -> StitchResult:
    """Render a ReceivedFrame onto a neutral background.

    Parameters
    ----------
    frame : ReceivedFrame
    bg_value : 0..255
        Fill colour for the background canvas.
    draw_bbox : bool
        If True, draw a thin green rectangle at each patch's original
        bbox. Helpful for debug visualizations; off by default.
    fallback_size :
        Used only if the frame metadata is missing (header lost AND no
        patch has a usable bbox). Practically a no-op for normal traffic.
    """
    h = frame.frame_h
    w = frame.frame_w
    if h <= 0 or w <= 0:
        # Header lost AND no usable size info — best effort.
        h, w = fallback_size

    canvas = np.full((h, w, 3), bg_value, dtype=np.uint8)

    n_pasted = 0
    n_skipped_decode = 0
    n_skipped_incomplete = 0

    for p in frame.patches:
        if not p.complete:
            n_skipped_incomplete += 1
            # Try anyway — partial JPEG often decodes top rows OK.
        img = _decode_jpeg(p.data)
        if img is None:
            n_skipped_decode += 1
            continue
        _paste(canvas, img, p)
        n_pasted += 1
        if draw_bbox:
            x1, y1, x2, y2 = p.bbox
            colour = (0, 255, 0) if p.complete else (0, 165, 255)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, 2)

    return StitchResult(
        frame_id=frame.frame_id,
        image=canvas,
        n_pasted=n_pasted,
        n_skipped_decode=n_skipped_decode,
        n_skipped_incomplete=n_skipped_incomplete,
    )


def _decode_jpeg(data: bytes) -> Optional[np.ndarray]:
    if not data:
        return None
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def _paste(canvas: np.ndarray, patch_img: np.ndarray, p: ReceivedPatch) -> None:
    """Paste patch_img onto canvas using p.bbox as the original location.

    The patch image's actual size may differ from bbox (margin was added
    on the edge side via expanded_bbox, but we only carry the original
    bbox over the wire to save header space). We crop the patch back to
    bbox size — this is a Day-6 simplification; Week 2 will pass
    expanded_bbox over the wire too and paste with the margin.
    """
    H, W = canvas.shape[:2]
    x1, y1, x2, y2 = p.bbox
    x1c, y1c = max(0, x1), max(0, y1)
    x2c, y2c = min(W, x2), min(H, y2)
    target_w = x2c - x1c
    target_h = y2c - y1c
    if target_w <= 0 or target_h <= 0:
        return

    # Resize the patch image to fit the bbox. Patch was cropped with
    # margin, so its actual size is slightly larger than bbox; resizing
    # rather than centre-cropping preserves the subject.
    src_h, src_w = patch_img.shape[:2]
    if (src_h, src_w) != (target_h, target_w):
        patch_img = cv2.resize(
            patch_img, (target_w, target_h), interpolation=cv2.INTER_LINEAR
        )

    canvas[y1c:y2c, x1c:x2c] = patch_img
