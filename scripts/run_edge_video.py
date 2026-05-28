"""Edge-side video capture loop — runs on Jetson.

Reads frames from a MOT17 image sequence (or any folder of JPG/PNG
frames, or a video file), runs the full edge pipeline on each frame,
and sends the patches to the server.

MOT17 format
------------
Each sequence is a folder of consecutive numbered jpgs:
    MOT17-04/img1/000001.jpg
    MOT17-04/img1/000002.jpg
    ...

We auto-detect: if --source is a folder, glob *.jpg and sort. If it's a
.mp4/.avi, use cv2.VideoCapture.

Run on the Jetson:
    PYTHONPATH=. python -u scripts/run_edge_video.py \\
        --server 192.168.35.153 \\
        --source data/mot17/train/MOT17-04-FRCNN/img1 \\
        --max-frames 100 \\
        --target-fps 30
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Iterator
from typing import Optional

import cv2
import numpy as np

from common.config import DEFAULT_JPEG_QUALITY
from edge.detector.yolov8_trt import YOLOv8TRT
from edge.patch import PatchJPEGEncoder, extract_patches
from edge.transport import UDPSender
from edge.control import FeedbackReceiver

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENGINE = PROJECT_ROOT / "engines" / "yolov8n_fp16.engine"


def iter_frames(source: Path) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (frame_index, BGR frame) from a folder or video file."""
    if source.is_dir():
        jpgs = sorted(
            list(source.glob("*.jpg")) + list(source.glob("*.png")),
        )
        if not jpgs:
            raise SystemExit(f"No frames found in {source}")
        for i, p in enumerate(jpgs):
            img = cv2.imread(str(p))
            if img is None:
                print(f"[warn]  cv2.imread failed: {p}", file=sys.stderr)
                continue
            yield i, img
    else:
        # Camera index ("0", "1", ...) -> int for cv2.VideoCapture
        # /dev/videoN device path -> str
        # mp4/avi path -> str
        src_str = str(source)
        if src_str.isdigit():
            cap_src = int(src_str)
            is_camera = True
        elif src_str.startswith("/dev/video"):
            cap_src = src_str
            is_camera = True
        else:
            cap_src = src_str
            is_camera = False

        cap = cv2.VideoCapture(cap_src)
        if not cap.isOpened():
            raise SystemExit(f"cv2.VideoCapture failed: {source}")

        # Configure camera resolution (no-op for file input)
        if is_camera:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            # Confirm what was actually applied
            w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            fps = cap.get(cv2.CAP_PROP_FPS)
            print(
                f"[cam]     opened {source} -> {int(w)}x{int(h)} @ {fps:.1f}fps",
                flush=True,
            )

        i = 0
        try:
            while True:
                ok, img = cap.read()
                if not ok:
                    break
                yield i, img
                i += 1
        finally:
            cap.release()

def main() -> None:
    ap = argparse.ArgumentParser(description="Edge-side video sender.")
    ap.add_argument("--server", required=True, help="Server host")
    ap.add_argument("--port", type=int, default=9999)
    ap.add_argument("--source", required=True, help="Folder of frames or video file")
    ap.add_argument("--engine", default=str(DEFAULT_ENGINE))
    ap.add_argument("--quality", type=int, default=DEFAULT_JPEG_QUALITY)
    ap.add_argument(
        "--header-redundancy", type=int, default=1,
        help="Send each FRAME_HEADER N times (default 1 = no redundancy, "
             "3 recommended for high-loss demos to mitigate frame-level "
             "cascade from single-packet header loss).",
    )
    ap.add_argument(
        "--max-frames", type=int, default=0,
        help="Stop after N frames (0 = process all).",
    )
    ap.add_argument(
        "--target-fps", type=float, default=30.0,
        help="Throttle send rate. 0 = as fast as possible.",
    )
    ap.add_argument(
        "--sleep-us", type=float, default=0.0,
        help="Per-chunk sleep microseconds (pacing).",
    )
    ap.add_argument(
        "--log-every", type=int, default=10,
        help="Print per-frame stats every N frames.",
    )
    ap.add_argument(
        "--enable-feedback", action="store_true",
        help="Enable feedback receiver and dynamic quality control.",
    )
    ap.add_argument(
        "--feedback-bind", default="0.0.0.0",
        help="Bind address for feedback receiver (default 0.0.0.0).",
    )
    ap.add_argument(
        "--feedback-port", type=int, default=9001,
        help="UDP port to listen for feedback (default 9001).",
    )
    args = ap.parse_args()

    print(f"[setup]   engine={args.engine}")
    print(f"[setup]   source={args.source} → {args.server}:{args.port}")
    print(f"[setup]   target_fps={args.target_fps} quality={args.quality}")

    detector = YOLOv8TRT(args.engine)
    encoder = PatchJPEGEncoder(default_quality=args.quality)

    feedback_receiver: Optional[FeedbackReceiver] = None
    if args.enable_feedback:
        feedback_receiver = FeedbackReceiver(
            args.feedback_bind, args.feedback_port,
        )
        print(
            f"[fb-rx]   listening on {args.feedback_bind}:{args.feedback_port}",
            flush=True,
        )

    target_interval = 1.0 / args.target_fps if args.target_fps > 0 else 0.0

    n_frames = 0
    n_patches_total = 0
    detect_ms_sum = 0.0
    encode_ms_sum = 0.0
    send_ms_sum = 0.0
    loop_t0 = time.perf_counter()
    next_tick = loop_t0

    with UDPSender(
        args.server, args.port, per_chunk_sleep_us=args.sleep_us,
    ) as tx:
        for fid, frame in iter_frames(Path(args.source)):
            if args.max_frames and n_frames >= args.max_frames:
                break

            # ----- Pace to target FPS -----
            if target_interval > 0:
                now = time.perf_counter()
                if now < next_tick:
                    time.sleep(next_tick - now)
                next_tick += target_interval

            h, w = frame.shape[:2]

            # ----- Detect -----
            t0 = time.perf_counter()
            detections = detector.detect(frame)
            detect_ms = (time.perf_counter() - t0) * 1000.0

            # ----- Extract + encode -----
            t0 = time.perf_counter()
            patches = extract_patches(frame, detections, frame_id=fid)

            # Poll for quality feedback from server (closed-loop adaptive).
            if feedback_receiver is not None:
                new_q = feedback_receiver.poll()
                if new_q is not None:
                    encoder.set_quality(new_q)
                    print(
                        f"[fb-rx]   frame={fid} quality changed to {new_q}",
                        flush=True,
                    )

            encoded = encoder.encode_many(patches)
            encode_ms = (time.perf_counter() - t0) * 1000.0

            # ----- Send -----
            t0 = time.perf_counter()
            tx.send_frame(
                frame_id=fid, encoded=encoded,
                frame_w=w, frame_h=h, header_redundancy=args.header_redundancy
            )
            send_ms = (time.perf_counter() - t0) * 1000.0

            n_frames += 1
            n_patches_total += len(encoded)
            detect_ms_sum += detect_ms
            encode_ms_sum += encode_ms
            send_ms_sum += send_ms

            if args.log_every and (n_frames % args.log_every == 0):
                jpg_bytes = sum(e.size_bytes for e in encoded)
                print(
                    f"[frame {fid:>4}] persons={len(encoded)} "
                    f"det={detect_ms:.1f}ms enc={encode_ms:.1f}ms "
                    f"send={send_ms:.1f}ms jpg={jpg_bytes/1024:.1f}KB"
                )

    elapsed = time.perf_counter() - loop_t0
    print()
    print(f"[done]    {n_frames} frames in {elapsed:.2f} s "
          f"({n_frames/max(elapsed,1e-6):.1f} FPS)")
    print(f"          patches_sent={n_patches_total} "
          f"avg_detect={detect_ms_sum/max(n_frames,1):.2f}ms "
          f"avg_encode={encode_ms_sum/max(n_frames,1):.2f}ms "
          f"avg_send={send_ms_sum/max(n_frames,1):.2f}ms")
    print(f"          pkts={tx.stats.packets_sent} "
          f"bytes={tx.stats.bytes_sent} "
          f"errors={tx.stats.send_errors}")

    if feedback_receiver is not None:
        fs = feedback_receiver.stats
        print()
        print(
            f"[fb-rx]   packets={fs.packets_received} bytes={fs.bytes_received} "
            f"changes={fs.quality_changes} bad_hdr={fs.bad_header} "
            f"bad_pay={fs.bad_payload} stale={fs.stale_frame_id}"
        )
        feedback_receiver.close()

if __name__ == "__main__":
    main()
