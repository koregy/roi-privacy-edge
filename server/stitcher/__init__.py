"""Server-side stitching: turn received patches into a full frame."""

from server.stitcher.naive import (
    DEFAULT_BG_VALUE,
    StitchResult,
    stitch_frame,
)

__all__ = ["stitch_frame", "StitchResult", "DEFAULT_BG_VALUE"]
