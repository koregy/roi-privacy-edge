"""Sanity check for the Day 3-4 patch pipeline.

Runs the full edge-node prefix on a test image:
    frame -> YOLOv8TRT -> extract_patches -> PatchJPEGEncoder

Outputs:
    results/patches/frame{id}_det{n}_q{Q}.jpg  -- the actual encoded patches
    results/patch_sizes.csv                    -- size vs quality sweep
    stdout                                     -- per-stage timing summary

Run from project root:
    python scripts/test_patch_pipeline.py
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import cv2
import numpy as np

from common.config import DEFAULT_JPEG_QUALITY
from edge.detector.yolov8_trt import YOLOv8TRT
from edge.patch import PatchJPEGEncoder, extract_patches


# ---------- Paths ----------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENGINE_PATH = PROJECT_ROOT / "engines" / "yolov8n_fp16.engine"
TEST_IMAGE = PROJECT_ROOT / "data" / "test_images" / "persons.jpg"
RESULTS_DIR = PROJECT_ROOT / "results"
PATCHES_DIR = RESULTS_DIR / "patches"
CSV_PATH = RESULTS_DIR / "patch_sizes.csv"


def main() -> None:
    PATCHES_DIR.mkdir(parents=True, exist_ok=True)

    # ---------- Load image ----------
    if not TEST_IMAGE.exists():
        raise FileNotFoundError(f"Test image not found: {TEST_IMAGE}")
    frame = cv2.imread(str(TEST_IMAGE))
    if frame is None:
        raise RuntimeError(f"cv2.imread returned None for {TEST_IMAGE}")
    print(f"[load]    frame shape: {frame.shape}  ({TEST_IMAGE.name})")

    # ---------- Detect ----------
    print(f"[engine]  loading {ENGINE_PATH.name}")
    detector = YOLOv8TRT(str(ENGINE_PATH))

    t0 = time.perf_counter()
    detections = detector.detect(frame)
    t_det = (time.perf_counter() - t0) * 1000.0
    print(f"[detect]  {len(detections)} detections in {t_det:.2f} ms")
    for i, d in enumerate(detections):
        print(
            f"          [{i}] class={d.class_id} conf={d.confidence:.3f} "
            f"bbox=({d.x1},{d.y1},{d.x2},{d.y2})"
        )

    # ---------- Extract patches ----------
    t0 = time.perf_counter()
    patches = extract_patches(frame, detections, frame_id=0)
    t_ext = (time.perf_counter() - t0) * 1000.0
    print(f"[extract] {len(patches)} person patches in {t_ext:.3f} ms")
    for p in patches:
        h, w = p.shape
        print(
            f"          det={p.det_id} crop={w}x{h} "
            f"orig_bbox={p.original_bbox} expanded={p.expanded_bbox} "
            f"conf={p.conf:.3f}"
        )

    if not patches:
        print("[warn]    no person patches extracted; stopping here.")
        return

    # ---------- Encode at default quality + save to disk ----------
    encoder = PatchJPEGEncoder(default_quality=DEFAULT_JPEG_QUALITY)
    t0 = time.perf_counter()
    encoded = encoder.encode_many(patches)
    t_enc = (time.perf_counter() - t0) * 1000.0
    print(
        f"[encode]  q={DEFAULT_JPEG_QUALITY}  "
        f"{len(encoded)} patches in {t_enc:.3f} ms  "
        f"total {encoder.total_bytes_out} B  "
        f"avg {encoder.avg_bytes:.0f} B/patch"
    )

    for enc in encoded:
        out = PATCHES_DIR / (
            f"frame{enc.frame_id}_det{enc.det_id}_q{enc.quality}.jpg"
        )
        out.write_bytes(enc.data)
        print(f"          wrote {out.relative_to(PROJECT_ROOT)}  ({enc.size_bytes} B)")

    # ---------- Quality sweep ----------
    # Re-encode all patches at multiple qualities for the bytes/quality
    # tradeoff curve. This is the raw material for Week 4's adaptive policy.
    sweep_q = [30, 50, 60, 70, 75, 80, 85, 90, 95]
    print(f"[sweep]   re-encoding at qualities {sweep_q}")

    with open(CSV_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["det_id", "crop_w", "crop_h", "quality", "size_bytes"])
        for p in patches:
            ph, pw = p.shape
            for q in sweep_q:
                enc = encoder.encode(p, quality=q)
                w.writerow([p.det_id, pw, ph, q, enc.size_bytes])
    print(f"[sweep]   wrote {CSV_PATH.relative_to(PROJECT_ROOT)}")

    # ---------- Quick per-quality summary (avg across patches) ----------
    print()
    print(f"{'quality':>8} {'avg bytes':>12} {'kB/frame':>10}")
    for q in sweep_q:
        sizes = []
        for p in patches:
            sizes.append(encoder.encode(p, quality=q).size_bytes)
        avg = float(np.mean(sizes))
        per_frame_kb = sum(sizes) / 1024.0
        print(f"{q:>8d} {avg:>12.0f} {per_frame_kb:>10.2f}")


if __name__ == "__main__":
    main()
