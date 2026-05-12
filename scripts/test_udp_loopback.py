"""End-to-end UDP loopback sanity check (single process, 127.0.0.1).

Day 6: tests FRAME_HEADER plumbing (ReceivedFrame event w/ expected_patches
and frame size).

Week 2 prep: also verifies expanded_bbox round-trip. expanded_bbox is
carried as an 8-byte prefix on chunk 0 of each patch (see common/packet.py).
The original bbox stays in the packet header; expanded comes back as a
new field on ReceivedPatch.
"""

from __future__ import annotations

import socket
import time
from pathlib import Path

import cv2

from common.config import DEFAULT_JPEG_QUALITY
from edge.detector.yolov8_trt import YOLOv8TRT
from edge.patch import PatchJPEGEncoder, extract_patches
from edge.transport import UDPSender
from server.transport import ReceivedFrame, ReceivedPatch, UDPReceiver

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENGINE_PATH = PROJECT_ROOT / "engines" / "yolov8n_fp16.engine"
TEST_IMAGE = PROJECT_ROOT / "data" / "test_images" / "persons.jpg"
OUT_DIR = PROJECT_ROOT / "results" / "udp_loopback"

HOST = "127.0.0.1"


def _pick_free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind((HOST, 0))
    port = s.getsockname()[1]
    s.close()
    return port


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    frame = cv2.imread(str(TEST_IMAGE))
    if frame is None:
        raise RuntimeError(f"cv2.imread failed: {TEST_IMAGE}")
    h, w = frame.shape[:2]
    print(f"[setup]   frame {frame.shape}  loading {ENGINE_PATH.name}")

    detector = YOLOv8TRT(str(ENGINE_PATH))
    detections = detector.detect(frame)
    patches = extract_patches(frame, detections, frame_id=42)

    encoder = PatchJPEGEncoder(default_quality=DEFAULT_JPEG_QUALITY)
    encoded = encoder.encode_many(patches)
    print(
        f"[encode]  {len(encoded)} patches, "
        f"sizes={[e.size_bytes for e in encoded]} B"
    )

    # Keep the full EncodedPatch as truth so we can check bbox/expanded_bbox
    # round-trip in addition to JPEG bytes.
    truth = {(e.frame_id, e.det_id): e for e in encoded}

    port = _pick_free_port()
    print(f"[net]     binding receiver on {HOST}:{port}")

    received_patches: dict[tuple[int, int], ReceivedPatch] = {}
    received_frames: list[ReceivedFrame] = []
    completed_count = 0
    incomplete_count = 0

    with UDPReceiver(HOST, port, frame_ttl_s=2.0, patch_ttl_s=1.0) as rx, \
         UDPSender(HOST, port) as tx:

        t0 = time.perf_counter()
        chunks_sent = tx.send_frame(
            frame_id=42, encoded=encoded, frame_w=w, frame_h=h,
        )
        send_ms = (time.perf_counter() - t0) * 1000.0
        print(
            f"[send]    {chunks_sent} pkts in {send_ms:.2f} ms "
            f"({tx.stats.bytes_sent} wire bytes, "
            f"frames_sent={tx.stats.frames_sent})"
        )

        deadline = time.perf_counter() + 1.0
        while not received_frames and time.perf_counter() < deadline:
            for event in rx.poll(timeout_s=0.02):
                if isinstance(event, ReceivedPatch):
                    key = (event.frame_id, event.det_id)
                    if event.complete:
                        received_patches[key] = event
                        completed_count += 1
                        print(
                            f"          ✓ patch complete frame={event.frame_id} "
                            f"det={event.det_id} bytes={len(event.data)}"
                        )
                    else:
                        incomplete_count += 1
                elif isinstance(event, ReceivedFrame):
                    received_frames.append(event)
                    print(
                        f"          ✓ FRAME complete frame={event.frame_id} "
                        f"patches={event.n_complete_patches}/{event.expected_patches} "
                        f"hdr_seen={event.header_seen} "
                        f"size={event.frame_w}x{event.frame_h}"
                    )

        s = rx.stats
        print(
            f"[stats]   pkts={s.packets_received} "
            f"fhdr={s.frame_headers_received} "
            f"bad_hdr={s.packets_dropped_bad_header} "
            f"bad_pay={s.packets_dropped_bad_payload} "
            f"dup={s.duplicate_chunks}"
        )

    print()
    print("[verify]  byte-for-byte JPEG comparison")
    all_ok = True
    for key in sorted(truth.keys()):
        want = truth[key]
        got = received_patches.get(key)
        if got is None:
            print(f"          ✗ {key}: MISSING")
            all_ok = False
        elif got.data != want.data:
            print(f"          ✗ {key}: MISMATCH ({len(want.data)} vs {len(got.data)})")
            all_ok = False
        else:
            print(f"          ✓ {key}: {len(got.data)} B identical")

    print()
    print("[verify]  FRAME_HEADER plumbing")
    if not received_frames:
        print("          ✗ no ReceivedFrame event")
        all_ok = False
    else:
        rf = received_frames[0]
        ok_block = True
        if rf.expected_patches != len(encoded):
            print(
                f"          ✗ expected_patches mismatch: "
                f"got {rf.expected_patches}, want {len(encoded)}"
            )
            ok_block = False
        if (rf.frame_w, rf.frame_h) != (w, h):
            print(
                f"          ✗ frame size mismatch: "
                f"got {rf.frame_w}x{rf.frame_h}, want {w}x{h}"
            )
            ok_block = False
        if not rf.header_seen:
            print("          ✗ header_seen=False")
            ok_block = False
        if not rf.complete:
            print("          ✗ complete=False")
            ok_block = False
        if ok_block:
            print(
                f"          ✓ frame=42 expected={rf.expected_patches} "
                f"size={rf.frame_w}x{rf.frame_h} complete=True"
            )
        all_ok = all_ok and ok_block

    print()
    print("[verify]  bbox round-trip (original in header, expanded in chunk 0 prefix)")
    bbox_ok = True
    for key in sorted(truth.keys()):
        want = truth[key]
        got = received_patches.get(key)
        if got is None:
            # already counted as MISSING above; skip without double-counting
            continue
        if got.bbox != want.original_bbox:
            print(
                f"          ✗ {key}: original bbox mismatch "
                f"got={got.bbox} want={want.original_bbox}"
            )
            bbox_ok = False
            continue
        if got.expanded_bbox is None:
            print(f"          ✗ {key}: expanded_bbox is None (chunk 0 lost?)")
            bbox_ok = False
            continue
        if got.expanded_bbox != want.expanded_bbox:
            print(
                f"          ✗ {key}: expanded_bbox mismatch "
                f"got={got.expanded_bbox} want={want.expanded_bbox}"
            )
            bbox_ok = False
            continue
        print(
            f"          ✓ {key}: original={got.bbox} expanded={got.expanded_bbox}"
        )
    all_ok = all_ok and bbox_ok

    print()
    print(f"[result]  {'PASS ✓' if all_ok else 'FAIL ✗'}")


if __name__ == "__main__":
    main()