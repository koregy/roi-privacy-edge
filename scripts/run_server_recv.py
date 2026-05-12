"""Server-side receiver — run on the laptop.

Binds a UDP port and prints every patch as it arrives. Saves each
complete patch to disk and validates by decoding with cv2.imread.
Runs until Ctrl+C.

Run on the laptop:
    PYTHONPATH=. python scripts/run_server_recv.py

Then on the Jetson, run scripts/run_edge_send.py to send.

You should see one '✓ complete' line per detected person on the
Jetson side, with byte counts that match what the sender printed.
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from server.transport import UDPReceiver

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = PROJECT_ROOT / "results" / "udp_recv"


_running = True


def _on_sigint(signum, frame):  # noqa: ARG001
    global _running
    _running = False
    print("\n[ctrl-c]  shutting down...")


def main() -> None:
    ap = argparse.ArgumentParser(description="Server-side patch receiver.")
    ap.add_argument("--bind", default="0.0.0.0", help="Bind host (default 0.0.0.0 = all interfaces)")
    ap.add_argument("--port", type=int, default=9999, help="UDP port to listen on (default 9999)")
    ap.add_argument(
        "--out-dir", default=str(DEFAULT_OUT_DIR),
        help="Where to save received JPEG patches",
    )
    ap.add_argument(
        "--ttl-ms", type=float, default=200.0,
        help="Patch TTL in ms — partial patches are flushed after this. Default 200.",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    signal.signal(signal.SIGINT, _on_sigint)

    print(f"[bind]    {args.bind}:{args.port}  ttl={args.ttl_ms} ms")
    print(f"[out]     {out_dir}")
    print("[ready]   listening — press Ctrl+C to stop")
    print()

    last_status_t = time.perf_counter()
    patches_seen = 0

    with UDPReceiver(
        args.bind, args.port, patch_ttl_s=args.ttl_ms / 1000.0
    ) as rx:
        while _running:
            for rp in rx.poll(timeout_s=0.05):
                patches_seen += 1
                tag = "✓ complete  " if rp.complete else "✗ incomplete"
                print(
                    f"[recv]    {tag} frame={rp.frame_id} det={rp.det_id} "
                    f"chunks={rp.chunks_received}/{rp.chunks_expected} "
                    f"bytes={len(rp.data)} q={rp.quality} "
                    f"bbox={rp.bbox} conf={rp.confidence:.3f}"
                )

                # Save bytes to disk regardless of completeness — partial
                # ones are useful for the future progressive-decode work.
                suffix = "" if rp.complete else ".partial"
                out = out_dir / (
                    f"frame{rp.frame_id:06d}_det{rp.det_id}_q{rp.quality}.jpg{suffix}"
                )
                out.write_bytes(rp.data)

                # Validate complete patches by decoding.
                if rp.complete:
                    # Decode from in-memory bytes (don't depend on disk write
                    # having flushed yet).
                    arr = np.frombuffer(rp.data, dtype=np.uint8)
                    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if img is None:
                        print(f"          ⚠ cv2.imdecode failed for {out.name}")
                    else:
                        h, w = img.shape[:2]
                        print(f"          decoded {w}x{h} → {out.relative_to(PROJECT_ROOT)}")

            # Heartbeat every 5 s so user knows it's alive.
            now = time.perf_counter()
            if now - last_status_t > 5.0:
                s = rx.stats
                print(
                    f"[alive]   patches={patches_seen} "
                    f"pkts_rcv={s.packets_received} "
                    f"complete={s.patches_complete} "
                    f"incomplete_ttl={s.patches_incomplete_ttl} "
                    f"bad_hdr={s.packets_dropped_bad_header} "
                    f"bad_pay={s.packets_dropped_bad_payload} "
                    f"dup={s.duplicate_chunks}"
                )
                last_status_t = now

    # Final flush + summary.
    s = rx.stats
    print()
    print(
        f"[final]   patches_seen={patches_seen} "
        f"pkts_rcv={s.packets_received} "
        f"complete={s.patches_complete} "
        f"incomplete_ttl={s.patches_incomplete_ttl} "
        f"bad_hdr={s.packets_dropped_bad_header} "
        f"bad_pay={s.packets_dropped_bad_payload} "
        f"dup={s.duplicate_chunks}"
    )
    sys.exit(0)


if __name__ == "__main__":
    main()