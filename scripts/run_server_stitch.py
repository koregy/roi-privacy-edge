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
import json
import queue
from pathlib import Path
from typing import Optional

import cv2

from server.stitcher import StitchResult, stitch_frame
from server.transport import (
    ReceivedFrame, 
    ReceivedPatch, 
    UDPReceiver,
    build_filter,
)
from server.transport.reorder import ReorderBuffer
from server.recovery import IoUTracker, RecoveryLayer
from server.decision import DecisionStateMachine, DecisionState
from server.control import PIDAdaptiveController
from server.control.feedback_sender import FeedbackSender

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
    ap.add_argument(
        "--drop-prob", type=float, default=0.0,
        help="Random chunk drop probability in [0, 1]. 0 = no drop.",
    )
    ap.add_argument(
        "--delay-min-ms", type=float, default=0.0,
        help="Minimum per-chunk delay in ms (used with --delay-max-ms).",
    )
    ap.add_argument(
        "--delay-max-ms", type=float, default=0.0,
        help="Maximum per-chunk delay in ms. 0 = no delay.",
    )
    ap.add_argument(
        "--sim-seed", type=int, default=None,
        help="RNG seed for reproducible drop/delay across runs.",
    )
    ap.add_argument(
        "--recovery", action="store_true",
        help="Enable Zero-order Hold recovery via IoU tracking.",
    )
    ap.add_argument(
        "--recovery-iou", type=float, default=0.3,
        help="IoU threshold for tracker (default 0.3).",
    )
    ap.add_argument(
        "--recovery-max-age", type=int, default=10,
        help="Max frames a track lives without a match (default 10).",
    )
    ap.add_argument(
        "--reorder", action="store_true",
        help="Reorder ReceivedFrame events by frame_id before stitching.",
    )
    ap.add_argument(
        "--reorder-size", type=int, default=5,
        help="Reorder buffer size in frames (default 5, adds ~5/fps seconds latency).",
    )
    ap.add_argument(
        "--reorder-wait-ms", type=float, default=500.0,
        help="Max time a frame waits in reorder buffer (default 500ms).",
    )
    ap.add_argument(
        "--decision", action="store_true",
        help="Enable confidence-weighted decision state machine.",
    )
    ap.add_argument(
        "--decision-window", type=int, default=10,
        help="Sliding window size in frames (default 10).",
    )
    ap.add_argument(
        "--decision-warn", type=float, default=0.80,
        help="Normal->Warning threshold on patch reception ratio (default 0.80).",
    )
    ap.add_argument(
        "--decision-emerg", type=float, default=0.50,
        help="Warning->Emergency threshold on patch reception ratio (default 0.50).",
    )
    ap.add_argument(
        "--adaptive", action="store_true",
        help="Enable PI adaptive quality controller (requires --decision).",
    )
    ap.add_argument(
        "--adaptive-target", type=float, default=0.85,
        help="Target reception ratio for adaptive controller (default 0.85).",
    )
    ap.add_argument(
        "--adaptive-initial-q", type=int, default=75,
        help="Initial JPEG quality for adaptive controller (default 75).",
    )
    ap.add_argument(
        "--adaptive-min-q", type=int, default=30,
        help="Minimum JPEG quality controller can output (default 30). "
             "Higher values prevent patch from becoming single-chunk "
             "and increase recoverability at cost of bandwidth.",
    )
    ap.add_argument(
        "--adaptive-max-q", type=int, default=95,
        help="Maximum JPEG quality controller can output (default 95).",
    )
    ap.add_argument(
        "--feedback", action="store_true",
        help="Enable closed-loop feedback: send quality changes to edge "
             "(requires --adaptive).",
    )
    ap.add_argument(
        "--feedback-edge-host", default="",
        help="Edge IP to send feedback packets to (required with --feedback).",
    )
    ap.add_argument(
        "--feedback-port", type=int, default=9001,
        help="UDP port on edge that listens for feedback (default 9001).",
    )
    ap.add_argument(
        "--feedback-redundancy", type=int, default=2,
        help="How many times to send each feedback packet (default 2).",
    )
    ap.add_argument(
        "--no-kalman", action="store_true",
        help="Disable Kalman in recovery"
    )
    ap.add_argument(
        "--no-predict", action="store_true",
        help="Disable predict-only in recovery"
    )
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

    packet_filter = build_filter(
        drop_prob=args.drop_prob,
        delay_min_ms=args.delay_min_ms,
        delay_max_ms=args.delay_max_ms,
        seed=args.sim_seed,
    )

    recovery: Optional[RecoveryLayer] = None
    if args.recovery:
        recovery = RecoveryLayer(
            IoUTracker(
                iou_threshold=args.recovery_iou,
                max_age=args.recovery_max_age,
            ),
            kalman_enabled=not args.no_kalman,
            predict_enabled=not args.no_predict,
        )
    
    reorder: Optional[ReorderBuffer] = None
    if args.reorder:
        reorder = ReorderBuffer(
            max_size=args.reorder_size,
            max_wait_s=args.reorder_wait_ms / 1000.0,
        )
    
    state_machine: Optional[DecisionStateMachine] = None
    if args.decision:
        state_machine = DecisionStateMachine(
            window_size=args.decision_window,
            warn_threshold=args.decision_warn,
            emerg_threshold=args.decision_emerg,
        )

    adaptive: Optional[PIDAdaptiveController] = None
    if args.adaptive:
        if state_machine is None:
            raise SystemExit("--adaptive requires --decision")
        adaptive = PIDAdaptiveController(
            target_ratio=args.adaptive_target,
            initial_quality=args.adaptive_initial_q,
            min_quality=args.adaptive_min_q,
            max_quality=args.adaptive_max_q,
        )

    feedback_sender: Optional[FeedbackSender] = None
    if args.feedback:
        if adaptive is None:
            raise SystemExit("--feedback requires --adaptive")
        if not args.feedback_edge_host:
            raise SystemExit("--feedback requires --feedback-edge-host")
        feedback_sender = FeedbackSender(
            edge_host=args.feedback_edge_host,
            edge_port=args.feedback_port,
            redundancy=args.feedback_redundancy,
        )

    current_state = DecisionState.NORMAL
    last_sent_quality: Optional[int] = None

    with UDPReceiver(
        args.bind, args.port,
        patch_ttl_s=args.ttl_ms / 1000.0 * 0.4,
        frame_ttl_s=args.ttl_ms / 1000.0,
        packet_filter=packet_filter,
    ) as rx:
        while _running:
            for event in rx.poll(timeout_s=0.05):
                if isinstance(event, ReceivedPatch):
                    if event.complete:
                        patches_complete += 1
                    else:
                        patches_incomplete += 1
                elif isinstance(event, ReceivedFrame):
                    # Reorder first (if enabled), then recovery+stitch each
                    # emitted frame in order.
                    if reorder is not None:
                        emitted = reorder.push(event)
                    else:
                        emitted = [event]
                    for ev in emitted:
                        # Apply dashboard control (every 10 frames)
                        if frames_done % 10 == 0:
                            _apply_control(recovery=recovery, packet_filter=packet_filter)
                        
                        current_state, new_q = _process_frame(
                            ev,
                            state_machine=state_machine, adaptive=adaptive,
                            recovery=recovery, writer=writer, png_dir=png_dir,
                            draw_bbox=args.draw_bbox, prev_state=current_state,
                        )
                        if feedback_sender is not None and new_q is not None and new_q != last_sent_quality:
                            feedback_sender.send_quality(
                                frame_id=ev.frame_id, target_quality=new_q,
                            )
                            last_sent_quality = new_q
                        frames_done += 1

                        # Dump stats (every 5 frames)
                        if frames_done % 5 == 0:
                            _dump_stats(
                                frames_done=frames_done,
                                current_state=current_state,
                                state_machine=state_machine,
                                adaptive=adaptive,
                                recovery=recovery,
                                packet_filter=packet_filter,
                            )
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
                if reorder is not None:
                    emitted = reorder.push(event)
                else:
                    emitted = [event]
                for ev in emitted:
                    # Apply dashboard control (every 10 frames)
                    if frames_done % 10 == 0:
                        _apply_control(recovery=recovery, packet_filter=packet_filter)
                    
                    current_state, new_q = _process_frame(
                        ev,
                        state_machine=state_machine, adaptive=adaptive,
                        recovery=recovery, writer=writer, png_dir=png_dir,
                        draw_bbox=args.draw_bbox, prev_state=current_state,
                    )
                    if feedback_sender is not None and new_q is not None and new_q != last_sent_quality:
                        feedback_sender.send_quality(
                            frame_id=ev.frame_id, target_quality=new_q,
                        )
                        last_sent_quality = new_q
                    frames_done += 1

                    # Dump stats (every 5 frames)
                    if frames_done % 5 == 0:
                        _dump_stats(
                            frames_done=frames_done,
                            current_state=current_state,
                            state_machine=state_machine,
                            adaptive=adaptive,
                            recovery=recovery,
                            packet_filter=packet_filter,
                        )

        # Drain reorder buffer.
        if reorder is not None:
            for ev in reorder.flush():
                # Apply dashboard control (every 10 frames)
                if frames_done % 10 == 0:
                    _apply_control(recovery=recovery, packet_filter=packet_filter)
                
                current_state, new_q = _process_frame(
                    ev,
                    state_machine=state_machine, adaptive=adaptive,
                    recovery=recovery, writer=writer, png_dir=png_dir,
                    draw_bbox=args.draw_bbox, prev_state=current_state,
                )
                if feedback_sender is not None and new_q is not None and new_q != last_sent_quality:
                    feedback_sender.send_quality(
                        frame_id=ev.frame_id, target_quality=new_q,
                    )
                    last_sent_quality = new_q
                frames_done += 1

                # Dump stats (every 5 frames)
                if frames_done % 5 == 0:
                    _dump_stats(
                        frames_done=frames_done,
                        current_state=current_state,
                        state_machine=state_machine,
                        adaptive=adaptive,
                        recovery=recovery,
                        packet_filter=packet_filter,
                    )

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
    
    # Constraint simulator stats (only when active).
    if packet_filter is not None:
        from server.transport.constraint_sim import (
            ChainedFilter, RandomDropFilter, DelayJitterFilter,
        )
        filters_to_report = (
            packet_filter.filters
            if isinstance(packet_filter, ChainedFilter)
            else (packet_filter,)
        )
        print()
        for f in filters_to_report:
            if isinstance(f, RandomDropFilter):
                print(
                    f"[sim]     RandomDropFilter p={f.p:.2f} "
                    f"seen={f.stats.seen} dropped={f.stats.dropped} "
                    f"drop_rate={f.stats.drop_rate:.3f}"
                )
            elif isinstance(f, DelayJitterFilter):
                print(
                    f"[sim]     DelayJitterFilter [{f.min_ms:.1f}, {f.max_ms:.1f}]ms "
                    f"seen={f.stats.seen} delayed={f.stats.delayed} "
                    f"avg_delay={f.stats.avg_delay_ms:.2f}ms"
                )

    if recovery is not None:
        rs = recovery.stats
        ts = recovery.tracker.stats
        print()
        print(
            f"[recov]   frames={rs.frames_seen} patches_seen={rs.patches_seen} "
            f"recovered={rs.patches_recovered} failed={rs.patches_failed_recovery}"
            f"predicted={rs.patches_predicted}"
        )
        print(
            f"          tracks: total_created={ts.next_track_id} matches={ts.matches} "
            f"new={ts.new_tracks} expired={ts.expired_tracks}"
        )

    if reorder is not None:
        rb = reorder.stats
        print()
        print(
            f"[order]   in={rb.frames_in} out={rb.frames_out} "
            f"by_size={rb.emitted_by_size} by_timeout={rb.emitted_by_timeout} "
            f"by_flush={rb.emitted_by_flush} "
            f"ooo_at_emit={rb.out_of_order_at_emit}"
        )

    if state_machine is not None:
        s = state_machine.stats
        total = s.frames_normal + s.frames_warning + s.frames_emergency
        print()
        print(
            f"[state]   total={total} "
            f"normal={s.frames_normal} ({100*s.frames_normal/total:.1f}%) "
            f"warning={s.frames_warning} ({100*s.frames_warning/total:.1f}%) "
            f"emergency={s.frames_emergency} ({100*s.frames_emergency/total:.1f}%) "
            f"transitions={s.transitions}"
        )

    if adaptive is not None:
        s = adaptive.stats
        avg_q = sum(s.quality_history) / len(s.quality_history) if s.quality_history else 0
        print(
            f"[ctrl]    updates={s.updates} "
            f"q_range=[{s.quality_min_seen}, {s.quality_max_seen}] "
            f"q_avg={avg_q:.1f} q_final={adaptive.quality}"
        )

    if feedback_sender is not None:
        fs = feedback_sender.stats
        print(
            f"[fb-tx]   sends={fs.sends} packets={fs.packets_sent} "
            f"bytes={fs.bytes_sent} errors={fs.send_errors}"
        )
    
    if feedback_sender is not None:
        feedback_sender.close()

    sys.exit(0)

_opencv_window_initialized = False

def _handle_frame(
    rf: ReceivedFrame,
    writer: VideoWriterLazy,
    png_dir: Optional[Path],
    draw_bbox: bool,
) -> None:
    global _opencv_window_initialized

    res: StitchResult = stitch_frame(rf, draw_bbox=draw_bbox)
    writer.write(res.image)

    # Live OpenCV display
    if not _opencv_window_initialized:
        cv2.namedWindow("ROI Live View", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("ROI Live View", 540, 960)
        _opencv_window_initialized = True
    cv2.imshow("ROI Live View", res.image)
    cv2.waitKey(1)

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

def _process_frame(
    ev,
    *,
    state_machine,
    adaptive,
    recovery,
    writer,
    png_dir,
    draw_bbox,
    prev_state,
):
    """One frame: decision -> adaptive -> recovery -> stitch -> writer.

    Returns (new_state, current_quality) for logging.
    """
    state = prev_state
    quality = None
    if state_machine is not None:
        state = state_machine.update(ev)
        if adaptive is not None:
            quality = adaptive.update(state_machine.current_ratio)
        if state != prev_state:
            print(
                f"[state]   frame={ev.frame_id} {prev_state.value} -> {state.value} "
                f"(ratio={state_machine.current_ratio:.3f}"
                + (f", q={quality}" if quality is not None else "")
                + ")",
                flush=True,
            )
    if recovery is not None:
        ev = recovery.enhance(ev)
    _handle_frame(ev, writer, png_dir, draw_bbox)
    return state, quality    

# ===== Dashboard Stats / Control I/O =====

STATS_PATH = "/tmp/roi_stats.json"
CONTROL_PATH = "/tmp/roi_control.json"


def _dump_stats(
    *,
    frames_done: int,
    current_state,
    state_machine,
    adaptive,
    recovery,
    packet_filter,
) -> None:
    """Write current system stats to STATS_PATH for dashboard polling."""
    # Find current drop probability inside packet_filter
    from server.transport.constraint_sim import (
        ChainedFilter, RandomDropFilter,
    )
    drop_prob = 0.0
    if packet_filter is not None:
        filters = (
            packet_filter.filters
            if isinstance(packet_filter, ChainedFilter)
            else (packet_filter,)
        )
        for f in filters:
            if isinstance(f, RandomDropFilter):
                drop_prob = f.p
                break

    stats = {
        "timestamp": time.time(),
        "frames_done": frames_done,
        "state": current_state.value if current_state else "UNKNOWN",
        "current_ratio": (
            state_machine.current_ratio if state_machine else 1.0
        ),
        "quality": adaptive.quality if adaptive else None,
        "tracks_count": (
            len(recovery.tracker.tracks) if recovery else 0
        ),
        "patches_recovered": (
            recovery.stats.patches_recovered if recovery else 0
        ),
        "patches_predicted": (
            recovery.stats.patches_predicted if recovery else 0
        ),
        "patches_failed": (
            recovery.stats.patches_failed_recovery if recovery else 0
        ),
        "drop_prob_current": drop_prob,
        "recovery_enabled": (
            recovery.enabled if recovery else False
        ),
        "kalman_enabled": (
            recovery.kalman_enabled if recovery else False
        ),
        "predict_enabled": (
            recovery.predict_enabled if recovery else False
        ),
    }
    try:
        with open(STATS_PATH, "w") as f:
            json.dump(stats, f)
    except OSError:
        pass  # non-fatal


def _apply_control(*, recovery, packet_filter) -> None:
    """Read dashboard control inputs from CONTROL_PATH and apply."""
    try:
        with open(CONTROL_PATH) as f:
            ctrl = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return

    # Recovery flags
    if recovery is not None:
        if "recovery_enabled" in ctrl:
            recovery.enabled = bool(ctrl["recovery_enabled"])
        if "kalman_enabled" in ctrl:
            recovery.kalman_enabled = bool(ctrl["kalman_enabled"])
        if "predict_enabled" in ctrl:
            recovery.predict_enabled = bool(ctrl["predict_enabled"])

    # Drop probability
    if packet_filter is not None and "drop_prob" in ctrl:
        from server.transport.constraint_sim import (
            ChainedFilter, RandomDropFilter,
        )
        filters = (
            packet_filter.filters
            if isinstance(packet_filter, ChainedFilter)
            else (packet_filter,)
        )
        for f in filters:
            if isinstance(f, RandomDropFilter):
                f.set_drop_prob(float(ctrl["drop_prob"]))
                break

if __name__ == "__main__":
    main()
