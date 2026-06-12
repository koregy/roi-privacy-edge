"""Shared configuration for edge and server nodes."""

# =============================================================================
# Patch extraction & JPEG encoding (Day 3-4)
# =============================================================================

# COCO class id for "person" — YOLOv8 was trained on COCO so id=0
PERSON_CLASS_ID = 0

# Skip patches smaller than this (in pixels, on either dimension).
# Tiny crops compress poorly per-byte and tracker/recovery side struggles
# with sub-32px patches anyway.
MIN_PATCH_SIZE = 32

# Expand bbox by this fraction on each side before cropping.
# Helps the server-side stitcher tolerate bbox jitter between frames,
# and gives recovery (zero-order hold) some context around the subject.
# 0.10 = 10% margin on each side (so a 100x200 bbox becomes ~120x240).
BBOX_MARGIN = 0.10

# Default JPEG quality (1-100). 75 is a common "looks fine to humans"
# baseline; we will sweep this in the adaptive-quality experiment later.
DEFAULT_JPEG_QUALITY = 75

# Hard cap on encoded patch size (bytes). Patches larger than this after
# JPEG encoding will trigger a warning — they probably mean the bbox is
# huge (close subject) and we should consider downscaling before encode.
# 64KB matches a comfortable UDP payload budget after headers.
MAX_PATCH_BYTES = 64 * 1024
