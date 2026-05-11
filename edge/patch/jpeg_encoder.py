"""JPEG encoder for ROI patches.

Wraps cv2.imencode with the project's conventions:
- a small dataclass (EncodedPatch) that pairs the bytes with the metadata
  the transport layer needs (frame_id, det_id, bbox, quality used);
- a class-based encoder (PatchJPEGEncoder) that holds quality as state so
  the adaptive controller can later mutate it per-frame without changing
  call sites in the capture loop;
- a one-shot functional API (encode_patch) for tests and scripts.

Design notes
------------
We expose `quality` as a per-call override AND as encoder state because
that's exactly the seam the adaptive-quality contribution sits on:
- Week 1-2: fixed quality everywhere, set once at startup.
- Week 4-5: server feedback (loss rate, RTT) drives an adaptive policy
  that calls `encoder.set_quality(new_q)` between frames.
- Week 6+: per-patch quality (close subject = higher q, far = lower q)
  via the per-call override.

By baking that seam in now, we avoid touching every call site later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import cv2
import numpy as np

from common.config import DEFAULT_JPEG_QUALITY, MAX_PATCH_BYTES
from edge.patch.extractor import Patch


@dataclass
class EncodedPatch:
    """A JPEG-compressed patch + the metadata to reassemble it server-side.

    Attributes
    ----------
    frame_id, det_id : int
        Copied through from the source Patch.
    data : bytes
        Raw JPEG bytes (starts with FFD8FF... ends with FFD9).
    original_bbox, expanded_bbox : tuple[int, int, int, int]
        Source rectangles in frame coordinates. expanded_bbox tells the
        server where to paste the decoded image; original_bbox is the
        detector's actual answer, useful for tracking / metrics.
    quality : int
        JPEG quality value actually used (1-100). Stored so the receiver
        and the offline analysis know what the encoder picked.
    conf : float
        Detector confidence carried through for prioritization.
    """

    frame_id: int
    det_id: int
    data: bytes
    original_bbox: tuple[int, int, int, int]
    expanded_bbox: tuple[int, int, int, int]
    quality: int
    conf: float

    @property
    def size_bytes(self) -> int:
        return len(self.data)


def encode_patch(
    patch: Patch,
    quality: int = DEFAULT_JPEG_QUALITY,
) -> EncodedPatch:
    """Encode a single Patch to JPEG.

    Parameters
    ----------
    patch : Patch
    quality : int, 1-100
        Higher = better fidelity, larger bytes. cv2 default is 95.

    Raises
    ------
    ValueError
        If quality is out of range or cv2.imencode reports failure.
    """
    if not 1 <= quality <= 100:
        raise ValueError(f"JPEG quality must be in [1, 100], got {quality}")

    params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    ok, buf = cv2.imencode(".jpg", patch.image, params)
    if not ok:
        raise ValueError(
            f"cv2.imencode failed for patch frame={patch.frame_id} "
            f"det={patch.det_id} shape={patch.image.shape}"
        )

    return EncodedPatch(
        frame_id=patch.frame_id,
        det_id=patch.det_id,
        data=buf.tobytes(),
        original_bbox=patch.original_bbox,
        expanded_bbox=patch.expanded_bbox,
        quality=quality,
        conf=patch.conf,
    )


class PatchJPEGEncoder:
    """Stateful encoder. The seam where adaptive quality control plugs in.

    Hold one instance for the lifetime of the capture loop; mutate quality
    via set_quality() when the policy decides to. Per-call `quality=`
    overrides take precedence over the instance setting, which lets a
    per-patch policy coexist with a global default.

    Tracks lightweight stats (encode count, bytes out, oversized count)
    that the dashboard / logging can read without hooking the encode path.
    """

    def __init__(self, default_quality: int = DEFAULT_JPEG_QUALITY) -> None:
        if not 1 <= default_quality <= 100:
            raise ValueError(
                f"default_quality must be in [1, 100], got {default_quality}"
            )
        self._quality = int(default_quality)

        # Stats (cleared by reset_stats()).
        self.encoded_count = 0
        self.total_bytes_out = 0
        self.oversized_count = 0  # patches > MAX_PATCH_BYTES after encode

    @property
    def quality(self) -> int:
        return self._quality

    def set_quality(self, quality: int) -> None:
        """Update the default quality. Adaptive controller calls this."""
        if not 1 <= quality <= 100:
            raise ValueError(f"quality must be in [1, 100], got {quality}")
        self._quality = int(quality)

    def encode(
        self,
        patch: Patch,
        quality: Optional[int] = None,
    ) -> EncodedPatch:
        """Encode one patch. `quality` overrides instance default if given."""
        q = self._quality if quality is None else quality
        enc = encode_patch(patch, quality=q)

        self.encoded_count += 1
        self.total_bytes_out += enc.size_bytes
        if enc.size_bytes > MAX_PATCH_BYTES:
            self.oversized_count += 1

        return enc

    def encode_many(
        self,
        patches: Sequence[Patch],
        quality: Optional[int] = None,
    ) -> List[EncodedPatch]:
        """Encode a whole frame's worth of patches in one call."""
        return [self.encode(p, quality=quality) for p in patches]

    def reset_stats(self) -> None:
        self.encoded_count = 0
        self.total_bytes_out = 0
        self.oversized_count = 0

    @property
    def avg_bytes(self) -> float:
        if self.encoded_count == 0:
            return 0.0
        return self.total_bytes_out / self.encoded_count
