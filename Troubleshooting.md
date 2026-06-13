# Troubleshooting

Major bugs and design problems encountered during development, with root causes, fixes, and verification. Test labels (A–O) refer to the controlled experiment runs in `results/`; all loss experiments use a seeded constraint simulator (`--sim-seed 42`) for reproducibility.

---

## 1. Duplicate frame emission with header redundancy (empty-frame bug)

**Symptom.** After introducing FRAME_HEADER redundancy, a 100-frame run with zero packet drop produced a 166-frame output video. Frames with no detections were written multiple times.

**Root cause.** The receiver treated every arriving FRAME_HEADER packet as a new frame. With redundancy N, a frame whose `expected_patches = 0` becomes "complete" the instant its header arrives — so each of the N redundant headers triggered a separate emit of the same `frame_id`. Frames with patches were partially protected (patch attachment kept them in the assembly table), but empty frames slipped through every time.

**Fix.** Added an `_emitted_frame_ids: set[int]` to `src/server/transport/udp_receiver.py` with guards at all four emission paths (`handle_frame_header`, `attach_to_frame`, `maybe_emit_frame`, `sweep_expired_frames`), making frame emission idempotent regardless of how many duplicate headers arrive.

**Verification.** Test J (drop 0%, redundancy 3): output went from 166 frames to exactly 100/100.

**Lesson.** Any redundancy mechanism on the sender requires an explicit deduplication contract on the receiver. The two were designed in the same session and the bug still shipped — because the "obvious" dedup (patch reassembly) covered the common case and silently missed the edge case (zero-patch frames).

---

## 2. FRAME_HEADER loss cascading into whole-frame loss

**Symptom.** At 30% simulated packet drop, only 58/100 frames appeared in the output video — far worse than 30% loss should cause. Frames whose patches had all arrived intact were still missing.

**Root cause.** The frame header is a single 32-byte UDP packet carrying `n_patches` and frame dimensions. Without it the receiver cannot finalize the frame, so one lost packet cascaded into discarding the entire frame and every patch that did arrive. At 30% drop, header survival is only 70% — matching the observed 58/100 (`fhdr=58/100` in receiver stats).

**Fix.** Transmit each FRAME_HEADER N times (`--header-redundancy`, edge side), combined with the receiver dedup from issue #1. Total-loss probability drops to pᴺ: at 30% drop and N=3, 0.3³ ≈ 2.7%.

**Verification.** Test I (drop 30%, redundancy 3): 97/100 frames written, matching the theoretical 97.3. Bandwidth cost is negligible — 200 extra packets per 100 frames ≈ 6.4 KB. The live demo uses N=5, which keeps headers alive even at 50% drop (0.5⁵ ≈ 3%).

---

## 3. Out-of-order frame emission breaking track identity

**Symptom.** Under packet drop, the tracker fragmented a single person into 7 different track IDs (Test F), causing recovery patches to flicker and lose continuity. Receiver stats also showed frames emitted out of order (`ooo_at_emit=4`).

**Root cause.** Two interacting problems. (a) UDP gives no ordering guarantee, and the receiver emitted frames as soon as they completed — a late frame could be emitted after its successor. (b) The IoU tracker matches consecutive frames; an out-of-order frame presents bounding boxes that don't overlap the previous frame's predicted positions, so matching fails and a new track ID is allocated. Each ID switch resets the recovery state (ZoH cache, Kalman history).

**Fix.** Added a min-heap reorder buffer (`src/server/transport/reorder.py`) between receiver and tracker, with bounded latency: frames are released in `frame_id` order, but a size threshold and a per-frame timeout (defaults: 5 frames / 500 ms) cap the added delay so one permanently lost frame can't stall the pipeline.

**Verification.** Same conditions, reorder enabled (Test H): track count went from 7 to 1. The fix to tracking came from transport-layer ordering, not tracker tuning — the IoU threshold never changed.

---

## 4. Zero-order Hold misalignment on moving people ("ghosting")

**Symptom.** ZoH recovery worked — lost patches were filled from the track's last complete image — but for a walking person the recovered patch was pasted at a visibly stale position, producing a ghost that lagged behind, then snapped forward when a real detection arrived.

**Root cause.** ZoH reuses both the pixels *and the position* of the last complete patch. Pixels going stale for a few frames is acceptable; position going stale is not, because the person has moved.

**Fix.** Added a 4-state constant-velocity Kalman filter per track (`src/server/recovery/kalman.py`). The tracker runs predict/correct on every frame; on recovery, ZoH translates the cached patch to the Kalman-predicted bounding box instead of the stale one (`src/server/recovery/zoh.py`). The predicted box also feeds the IoU matching itself, improving association for fast movers. As a final layer, when detection fails entirely, predict-only mode emits virtual patches at predicted positions.

**Verification.** On the single-person test clip the difference is immediately visible: the recovered patch slides smoothly with the person instead of teleporting. In the demo, recovered/predicted patches are drawn with distinct boxes precisely so this mechanism is visible.

---

## 5. State-machine frame count mismatch

**Symptom.** After wiring up the decision state machine, its total processed-frame count was consistently 4 less than `frames_stitched` — the two counters should be identical.

**Root cause.** Frames reach the stitcher through three call paths: the main receive poll, the receiver's end-of-stream `flush()`, and the reorder buffer's `flush()`. Frame processing had been unified into a `_process_frame` helper (which updates the state machine and PI controller), but the two flush paths still called the old `_handle_frame` directly, bypassing the decision logic. The 4 missing frames were exactly the frames drained by the flushes at shutdown.

**Fix.** Routed both flush paths through `_process_frame` in `scripts/run_server_stitch.py`.

**Lesson.** When refactoring into a helper, grep for every call site of the old path — the rare paths (shutdown, flush, error handling) are precisely the ones that escape testing.

---

## 6. PI controller saturation (integral windup)

**Symptom.** Under sustained heavy loss the adaptive controller's quality command pinned at the minimum, and after the network recovered it stayed low far longer than expected before climbing back.

**Root cause.** Classic integral windup: while the actuator is saturated at `min_q`, the integral term keeps accumulating error. When conditions improve, the controller must first "unwind" the accumulated integral before its output re-enters the valid range.

**Fix.** Anti-windup with an integral clamp (±5) in `src/server/control/adaptive.py` (PI controller: kp=30, ki=5, target reception ratio 0.85). A sign-convention bug surfaced during this work as well — positive error (losing patches) must *lower* quality — and is now documented in the module.

**Verification.** Sweep over drop {0, 15, 30, 50}%: quality adapts monotonically (avg q ≈ 94.5 at 0–15% drop, ≈ 52 at 30–50%), and recovery back to q=95 after the network clears is prompt. State transitions stay at 6–10 per run thanks to hysteresis in the state machine — no oscillation.

---

## 7. Closed-loop quality made the demo *look* worse

**Symptom.** With the full closed loop active, quantitative behavior was correct — but the video visibly "breathed": patch sharpness oscillated as the PID adjusted JPEG quality, which reads as instability to a human observer even though reception ratio was being optimized.

**Root cause.** Not a bug — a genuine control/perception tradeoff. The controller optimizes patch arrival probability; the human eye penalizes quality *variance* more than moderately low constant quality.

**Resolution.** Split the demo into two configurations: a fixed-quality run for visual continuity (showing the recovery stack), and a separate closed-loop run for the quantitative adaptation story (quality range and state transitions on the dashboard charts). Future work: smooth the PID output (EWMA) or rate-limit quality changes.

**Lesson.** "Working correctly" and "demonstrating well" are different objectives; instrumenting both and presenting them separately is more honest than tuning one configuration to fake both.

---

## 8. Delay-only simulation produced no measurable loss

**Symptom.** The delay/jitter filter (20–50 ms) showed zero effect on any metric — every patch and frame survived.

**Root cause.** The receiver's reassembly TTLs (patch 200 ms, frame 500 ms) comfortably absorb sub-50 ms delays. Delay only becomes loss when it exceeds TTL.

**Resolution.** Documented that delay experiments need the 100–200 ms range to interact with the TTLs, and kept packet drop as the primary stressor for evaluation. A related subtlety from the chained-filter tests: filter order matters — placing drop before delay means dropped packets skip the delay sleep, measurably changing filter-level statistics (`seen=188` vs `219` between the two filters on the same run).

---

## 9. `ReceivedFrame.patches` — List vs Dict confusion

**Symptom.** First version of the recovery layer crashed with `AttributeError` when iterating patches.

**Root cause.** The receiver uses two containers with the same name and different types: the internal assembly table holds `Dict[int, ReceivedPatch]` (keyed by `det_id`), while the public `ReceivedFrame.patches` exposes a det_id-sorted `List[ReceivedPatch]`. The recovery code assumed the Dict form.

**Fix.** Corrected to the List interface; the type distinction is now documented at both definition sites.

---

## 10. Chunk-0 loss and the expanded-bbox prefix (design decision)

Not a bug, but a wire-format decision made after hitting its consequences. Each patch carries two boxes: the detector's `original_bbox` (in every chunk's 32-byte header, so it survives any chunk loss) and the margin-`expanded_bbox` used for cropping, which rides as an 8-byte prefix on chunk 0 only. The deliberate asymmetry: if chunk 0 is lost, the JPEG SOI marker is lost with it and the patch is undecodable anyway — so replicating the expanded box across chunks would protect data that can never be used. The stitcher falls back to `original_bbox` when the prefix is missing.

---

## Known limitations (open)

- **Small/distant objects flicker** at the detector level; tracker tuning cannot fix what YOLO doesn't see. Predict-only mode masks short gaps but is not a substitute for detection.
- **Crowded scenes over-count tracks** — a 16-person clip produced 21 tracks (5 ID switches). The demo configuration mitigates this with a lower IoU matching threshold (0.15 vs the 0.3 default) and a longer track max-age (30 frames vs 10), trading some ID-merge risk for continuity.
- **SIGINT shutdown** occasionally skips the final stats line; no reliable reproduction yet.
- **`cv2.imwrite` return value unchecked** in the PNG-dump path — disk-full or bad-path errors fail silently.
- **TensorRT wrapper uses `assert` for CUDA error checks** — disabled under `python -O`; should be `raise RuntimeError`.