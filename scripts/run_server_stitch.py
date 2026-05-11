"""Server-side receiver + stitcher — runs on the laptop.

Listens on UDP, reassembles frames, stitches received patches onto a
neutral background, and writes the stitched frames to an mp4 (or PNG
sequence) for visual inspection.

Run on the laptop:
    PYTHONPATH=. python -u scripts/run_server_stitch.py \\
        --output results/stitched.mp4 \\
        --fps 30 \\
        --idle-timeout 5
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path
from typing import Optional

import cv2

from server.stitcher import StitchResult, stitch_frame
from server.transport import ReceivedFrame, ReceivedPatch, UDPReceiver

PROJECT_ROOT = Path(__file__).resolve().parent.parent


_running = True


def _on_sigint(signum, frame):  # noqa: ARG001
    global _running
    _running = False
    print("\n[ctrl-c]  shutting down...", flush=True)


class VideoWriterLazy:
    """Lazy mp4 writer — opens on first frame so we know the size."""

    def __init__(self, path: Path, fps: float) -> None:
        self.path = path
        self.fps = fps
        self._writer: Optional[cv2.VideoWriter] = None
        self._size: Optional[tuple[int, int]] = None
        self._n = 0

    def write(self, img) -> None:
        h, w = img.shape[:2]
        if self._writer is None:
            self._size = (w, h)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(
                str(self.path), fourcc, self.fps, (w, h)
            )
            if not self._writer.isOpened():
                raise RuntimeError(f"cv2.VideoWriter failed for {self.path}")
        elif (w, h) != self._size:
            # Frame size changed mid-stream; mp4 can't handle that.
            # Just skip resized frames with a warning.
            print(
                f"[warn]    frame size changed {self._size} -> ({w},{h}), skipping",
                file=sys.stderr, flush=True,
            )
            return
        self._writer.write(img)
        self._n += 1

    @property
    def frames_written(self) -> int:
        return self._n

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None


def main() -> None:
    ap = argparse.ArgumentParser(description="Server receiver + naive stitcher.")
    ap.add_argument("--bind", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=9999)
    ap.add_argument(
        "--output", default=str(PROJECT_ROOT / "results" / "stitched.mp4"),
        help="Output mp4 path (parent dirs created automatically).",
    )
    ap.add_argument(
        "--png-dir", default="",
        help="If set, also save each stitched frame as PNG in this directory.",
    )
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument(
        "--draw-bbox", action="store_true",
        help="Draw bbox rectangles over the stitched image (green=complete, orange=partial).",
    )
    ap.add_argument(
        "--idle-timeout", type=float, default=0.0,
        help="Exit after this many seconds with no new frames. 0 = never (Ctrl+C only).",
    )
    ap.add_argument("--ttl-ms", type=float, default=500.0)
    args = ap.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    png_dir = Path(args.png_dir) if args.png_dir else None
    if png_dir is not None:
        png_dir.mkdir(parents=True, exist_ok=True)

    signal.signal(signal.SIGINT, _on_sigint)

    print(f"[bind]    {args.bind}:{args.port}  frame_ttl={args.ttl_ms}ms")
    print(f"[out]     {out_path}", flush=True)
    if png_dir:
        print(f"[png]     {png_dir}", flush=True)
    print("[ready]   listening — press Ctrl+C to stop", flush=True)

    writer = VideoWriterLazy(out_path, args.fps)

    # Stats accumulators.
    last_status_t = time.perf_counter()
    last_frame_t = time.perf_counter()
    frames_done = 0
    patches_complete = 0
    patches_incomplete = 0

    with UDPReceiver(
        args.bind, args.port,
        patch_ttl_s=args.ttl_ms / 1000.0 * 0.4,  # patch TTL < frame TTL
        frame_ttl_s=args.ttl_ms / 1000.0,
    ) as rx:
        while _running:
            for event in rx.poll(timeout_s=0.05):
                if isinstance(event, ReceivedPatch):
                    if event.complete:
                        patches_complete += 1
                    else:
                        patches_incomplete += 1
                elif isinstance(event, ReceivedFrame):
                    _handle_frame(event, writer, png_dir, args.draw_bbox)
                    frames_done += 1
                    last_frame_t = time.perf_counter()

            # Idle timeout: exit if nothing's arrived in a while.
            now = time.perf_counter()
            if args.idle_timeout > 0 and frames_done > 0:
                if (now - last_frame_t) > args.idle_timeout:
                    print(
                        f"[idle]    no frame for {args.idle_timeout:.1f}s, exiting",
                        flush=True,
                    )
                    break

            # Heartbeat.
            if now - last_status_t > 5.0:
                s = rx.stats
                print(
                    f"[alive]   frames={frames_done} "
                    f"patches_c/i={patches_complete}/{patches_incomplete} "
                    f"pkts={s.packets_received} "
                    f"fhdr={s.frame_headers_received} "
                    f"frames_partial={s.frames_partial_ttl} "
                    f"orphan={s.orphan_patches}",
                    flush=True,
                )
                last_status_t = now

        # Flush at shutdown.
        for event in rx.flush():
            if isinstance(event, ReceivedFrame):
                _handle_frame(event, writer, png_dir, args.draw_bbox)
                frames_done += 1

    writer.close()

    s = rx.stats
    print()
    print(f"[final]   frames_stitched={frames_done} "
          f"frames_complete={s.frames_complete} "
          f"frames_partial_ttl={s.frames_partial_ttl}")
    print(f"          patches_complete={patches_complete} "
          f"patches_incomplete={patches_incomplete}")
    print(f"          pkts={s.packets_received} fhdr={s.frame_headers_received} "
          f"orphan={s.orphan_patches} "
          f"bad_hdr={s.packets_dropped_bad_header} "
          f"bad_pay={s.packets_dropped_bad_payload} "
          f"dup={s.duplicate_chunks}")
    print(f"          mp4 frames written: {writer.frames_written}")
    sys.exit(0)


def _handle_frame(
    rf: ReceivedFrame,
    writer: VideoWriterLazy,
    png_dir: Optional[Path],
    draw_bbox: bool,
) -> None:
    res: StitchResult = stitch_frame(rf, draw_bbox=draw_bbox)
    writer.write(res.image)
    tag = "complete" if rf.complete else "PARTIAL "
    print(
        f"[stitch] frame={rf.frame_id:>5} {tag} "
        f"patches={rf.n_complete_patches}/{rf.expected_patches} "
        f"pasted={res.n_pasted} "
        f"size={rf.frame_w}x{rf.frame_h}",
        flush=True,
    )
    if png_dir is not None:
        out = png_dir / f"frame_{rf.frame_id:06d}.png"
        cv2.imwrite(str(out), res.image)


if __name__ == "__main__":
    main()
