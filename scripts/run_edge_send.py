"""Edge-side sender — run on Jetson.

One-shot: read persons.jpg, detect, extract patches, encode JPEG,
send chunks to <server_host>:<server_port>, print stats, exit.

Run on the Jetson:
    PYTHONPATH=. python scripts/run_edge_send.py --server 192.168.35.153

The companion receiver (run_server_recv.py) must be running on the
server before you start this — UDP has no connection establishment,
so packets sent before the receiver binds are simply lost.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from common.config import DEFAULT_JPEG_QUALITY
from edge.detector.yolov8_trt import YOLOv8TRT
from edge.patch import PatchJPEGEncoder, extract_patches
from edge.transport import UDPSender

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENGINE = PROJECT_ROOT / "engines" / "yolov8n_fp16.engine"
DEFAULT_IMAGE = PROJECT_ROOT / "data" / "test_images" / "persons.jpg"


def main() -> None:
    ap = argparse.ArgumentParser(description="Edge-side patch sender (one-shot).")
    ap.add_argument("--server", required=True, help="Server host (IP or hostname)")
    ap.add_argument("--port", type=int, default=9999, help="Server UDP port (default 9999)")
    ap.add_argument("--image", default=str(DEFAULT_IMAGE), help="Input image path")
    ap.add_argument("--engine", default=str(DEFAULT_ENGINE), help="TensorRT engine path")
    ap.add_argument("--frame-id", type=int, default=1, help="frame_id to stamp on packets")
    ap.add_argument(
        "--quality", type=int, default=DEFAULT_JPEG_QUALITY,
        help=f"JPEG quality (1-100, default {DEFAULT_JPEG_QUALITY})",
    )
    ap.add_argument(
        "--sleep-us", type=float, default=0.0,
        help="Per-chunk sleep in microseconds (pacing). Default 0 = no pacing.",
    )
    ap.add_argument(
        "--repeat", type=int, default=1,
        help="Send this many frames (frame_id increments). Default 1.",
    )
    ap.add_argument(
        "--interval-ms", type=float, default=33.3,
        help="Sleep between frames when --repeat > 1 (default 33.3 = ~30 FPS).",
    )
    args = ap.parse_args()

    # ---------- Load image + detect once ----------
    frame = cv2.imread(args.image)
    if frame is None:
        raise SystemExit(f"cv2.imread failed: {args.image}")
    print(f"[setup]   frame {frame.shape}  → {args.server}:{args.port}")

    detector = YOLOv8TRT(args.engine)
    detections = detector.detect(frame)
    print(f"[detect]  {len(detections)} person(s)")

    encoder = PatchJPEGEncoder(default_quality=args.quality)

    with UDPSender(
        args.server,
        args.port,
        per_chunk_sleep_us=args.sleep_us,
    ) as tx:
        for i in range(args.repeat):
            fid = args.frame_id + i
            patches = extract_patches(frame, detections, frame_id=fid)
            encoded = encoder.encode_many(patches)

            t0 = time.perf_counter()
            chunks = tx.send_frame(encoded)
            dt_ms = (time.perf_counter() - t0) * 1000.0

            total_bytes = sum(e.size_bytes for e in encoded)
            print(
                f"[send {i+1:>3}/{args.repeat}] frame={fid} "
                f"patches={len(encoded)} chunks={chunks} "
                f"jpeg={total_bytes} B wire={tx.stats.bytes_sent} B "
                f"send_time={dt_ms:.2f} ms"
            )
            # Reset per-frame view of wire bytes so each line is per-frame.
            wire_before = tx.stats.bytes_sent

            if i + 1 < args.repeat:
                time.sleep(max(0.0, args.interval_ms / 1000.0))

        print(
            f"[total]   pkts={tx.stats.packets_sent} "
            f"bytes={tx.stats.bytes_sent} "
            f"errors={tx.stats.send_errors}"
        )


if __name__ == "__main__":
    main()
