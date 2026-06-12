# ROI Privacy-Preserving Edge Streaming

**A resilient, privacy-aware surveillance pipeline for unreliable networks.**

Bandwidth-adaptive, privacy-preserving distributed inference on NVIDIA Jetson Orin Nano.

**Demo video:** [YouTube link](https://youtu.be/B3gTI9zGksQ)

---

## 1. Motivation

Surveillance systems face two conflicting demands:

- **Privacy** — they need to monitor people, but transmitting full video frames exposes everything in the scene, including the background and bystander context that the operator never needed.
- **Reliability** — they need to work in real time, but real networks are unreliable: packets drop, latency spikes, bandwidth fluctuates.

This project addresses both at once. Instead of streaming entire frames, the edge device detects **regions of interest (people)** on-device and transmits **only the cropped patches**. Background pixels never leave the camera. On top of that, a multi-layer recovery stack maintains continuous tracking even under severe packet loss, and a closed-loop controller adapts compression quality to network conditions in real time.

**Privacy by design. Adaptive by necessity. Resilient by engineering.**

## 2. System Overview

The system runs on two logically separated nodes connected by an unreliable UDP channel.

```
┌──────────────── EDGE (Jetson Orin Nano) ────────────────┐
│ Camera → YOLOv8n (TensorRT FP16) → Patch crop           │
│        → Adaptive JPEG encode → UDP sender              │
│          (frame header sent 5× for redundancy)          │
└──────────────────────────┬──────────────────────────────┘
                  patches  │  ▲ feedback (target quality)
                           ▼  │
              [ UDP channel: simulated drop / delay ]
                           │  ▲
┌──────────────── SERVER (Laptop) ──────────┴─────────────┐
│ Receiver + reorder buffer                               │
│   → Recovery (Zero-order Hold + Kalman + Predict-only)  │
│   → Stitcher (reassemble patches on neutral background) │
│   → Decision machine (Normal / Warning / Emergency)     │
│   → PID quality controller → feedback to edge           │
│   → Streamlit dashboard + live stitched view            │
└──────────────────────────────────────────────────────────┘
```

- **Edge node** — runs YOLOv8n person detection with a TensorRT FP16 engine. For each detection it crops the bounding-box patch, JPEG-encodes it at the currently commanded quality, and ships it as UDP chunks. Everything outside the bounding boxes is discarded *at the edge* — it never reaches the network.
- **Server node** — reassembles patches into a synthetic frame on a neutral grey canvas, recovers missing patches, renders the live view, and outputs the security state.
- **Channel** — UDP with a configurable constraint simulator (packet drop / delay), representing real-world degraded networks.

## 3. Core Logic

### 3.1 ROI extraction and adaptive encoding (edge)

1. Each frame is run through a YOLOv8n FP16 TensorRT engine (person class).
2. Each detection's bounding box is cropped into an independent patch.
3. Patches are JPEG-encoded at a quality level commanded by the server's PID controller (range 30–95).
4. Each patch is split into UDP chunks (≤ 1368 B payload per packet, below the MTU to avoid IP fragmentation) using a custom 32-byte wire header carrying frame ID, patch ID, chunk index, and bounding-box metadata.

### 3.2 Four layers of recovery (server)

Network degradation causes patch loss. Four mechanisms keep tracking alive:

| # | Mechanism | What it does |
|---|---|---|
| 1 | **Frame header redundancy** | Each frame header is transmitted 5×. Even at 50% packet drop, total-header-loss probability is ≈ 3% — frames are almost never lost outright. |
| 2 | **Zero-order Hold (ZoH)** | When a patch fails to arrive completely, the previous frame's patch image is reused. The person doesn't disappear from view. |
| 3 | **Kalman motion compensation** | A per-object Kalman filter estimates each person's velocity and translates the ZoH-recovered patch to the predicted position, eliminating misalignment from stale crops. |
| 4 | **Predict-only mode** | When detection itself fails (e.g., heavy loss prevents any patch from arriving), the Kalman filter generates virtual patches at expected positions. Tracking continues even when YOLO can't see. |

In the live view, **green boxes** mark original complete detections, **cyan boxes** mark patches recovered or predicted by the system.

### 3.3 Closed-loop adaptation and decision

- **PID quality controller** — monitors the patch reception ratio and adjusts the target JPEG quality (30–95) on the edge via a feedback channel. Under heavy loss, quality drops to shrink patches and raise per-patch arrival probability; when the network recovers, quality climbs back.
- **3-state decision machine** — outputs **Normal / Warning / Emergency** from reception statistics. Note the deliberate design choice: even when ZoH + Kalman preserve *visual* continuity, the state machine still reports Emergency if detection confidence is low. The system alarms honestly rather than masking degraded sensing — in a real security system this distinction matters.

## 4. Dashboard

A Streamlit dashboard provides live control and observability:

- **Sidebar controls** — network drop level (Clean 0% / Heavy 30% / Severe 50%) and system mode (no protection / full recovery).
- **Main panel** — current security state, real-time metrics (reception ratio, JPEG quality, recovered / predicted patch counters), and adaptation charts.
- **Comparison tab** — side-by-side replay of the same video under three configurations (baseline / 30% drop without recovery / 30% drop with full system).
- A separate window renders the live stitched video output.

## 5. Results

### 5.1 Baseline pipeline performance

Measured on a 1080×810 test image (4 person detections) and a 768×432 @ 12 FPS test clip, same-subnet Wi-Fi:

| Stage | Measurement |
|---|---|
| FP16 engine size | 8.8 MB |
| GPU compute (detector) | 4.33 ms |
| End-to-end detect | 27.6 ms |
| Patch extraction | 0.07 ms |
| JPEG encode (4 patches, q=75) | 4.27 ms |
| Bytes per frame @ q=75 | 84 KB (4 persons) |
| UDP transport per frame | ~1.8 ms (64 chunks) |
| Loopback byte-equality | 100% (4 patches, 64 chunks) |
| Clean Wi-Fi delivery | 100/100 frames complete, 0 loss |
| Pipeline FPS | 12.1 FPS (matched source native rate) |

The clean Wi-Fi baseline shows zero natural packet loss, so all recovery-layer evaluation injects artificial loss through the constraint simulator.

### 5.2 Recovery under 30% packet loss

Same video, same network conditions, three configurations:

| Configuration | Outcome |
|---|---|
| Baseline (0% drop) | 2,168 patches complete, no loss |
| 30% drop, **no recovery** | 1,180 patches lost — people visibly disappear from the stream |
| 30% drop, **full system** | 1,135 patches recovered via ZoH + Kalman, plus 192 virtual patches generated in predict-only mode — tracking continuity preserved |

Even at 50% packet drop, the PID controller pins quality at its minimum to maximize arrival probability, and Kalman prediction + ZoH keep tracking alive. When the network is restored, quality climbs back to 95 and the state returns to Normal.

See `results/` for the full quality-vs-bytes sweep and evaluation outputs.

## 6. Repository Layout

| Folder | Role |
|---|---|
| `src/edge/` | Runs on Jetson Orin Nano |
| `src/edge/detector/` | YOLOv8n TensorRT inference wrapper |
| `src/edge/patch/` | Bounding-box crop + adaptive JPEG encoding |
| `src/edge/transport/` | UDP sender, packet builder, chunking, header redundancy |
| `src/server/` | Runs on the laptop |
| `src/server/transport/` | UDP receiver + constraint simulator (drop / delay) |
| `src/server/recovery/` | IoU tracker, Zero-order Hold, Kalman filter, predict-only mode |
| `src/server/stitcher/` | ROI stitching (paste patches onto neutral background) |
| `src/server/decision/` | Normal / Warning / Emergency state machine + PID controller |
| `src/common/` | Packet schemas and shared config |
| `dashboard/` | Streamlit web GUI (controls, metrics, comparison tab) |
| `data/` | Sample test images and demo clips |
| `scripts/` | Build, run, and test scripts |
| `models/` | Source models (`.pt`, `.onnx`) |
| `engines/` | TensorRT `.engine` files (built on Jetson, gitignored) |
| `results/` | Evaluation outputs, plots, tables |
| `docs/` | Proposal, diagrams, report |

## 7. Quick Start

### 7.1 Build the FP16 engine (on Jetson, one-time)

```bash
bash scripts/build_engine_fp16.sh
```

Output: `engines/yolov8n_fp16.engine` (build log in `logs/`).

### 7.2 Single-image sanity check (Jetson)

```bash
PYTHONPATH=. python -u scripts/test_patch_pipeline.py
```

Runs detect → extract → encode on a test image, writes patch JPEGs to `results/patches/` and a quality sweep CSV to `results/patch_sizes.csv`.

### 7.3 Loopback UDP test (Jetson, single process)

```bash
PYTHONPATH=. python -u scripts/test_udp_loopback.py
```

Verifies the wire format and chunk reassembly byte-for-byte over 127.0.0.1.

### 7.4 End-to-end ROI streaming (two machines)

On the **laptop** (server), start the receiver + stitcher:

```bash
PYTHONPATH=. python -u scripts/run_server_stitch.py \
    --output results/stitched.mp4 \
    --fps 12 \
    --idle-timeout 5
```

On the **Jetson** (edge), start the capture loop:

```bash
PYTHONPATH=. python -u scripts/run_edge_video.py \
    --server <laptop_ip> \
    --source data/videos/person-bicycle-car-detection.mp4 \
    --target-fps 12 \
    --max-frames 100
```

The laptop writes `results/stitched.mp4` — persons only, on a neutral grey canvas.

### 7.5 Live dashboard

On the laptop:

```bash
PYTHONPATH=. streamlit run dashboard/app.py
```

Use the sidebar to switch network drop levels (0% / 30% / 50%) and toggle the recovery system on or off; the comparison tab replays the same clip under all three configurations.

## 8. Hardware / Software

- **Edge:** NVIDIA Jetson Orin Nano 8GB (JetPack 6.x, TensorRT, CUDA)
- **Server:** laptop with Python 3.10+, OpenCV, Streamlit
- **Communication:** UDP sockets, custom 32-byte wire header, application-level chunking (≤ 1368 B payload per packet to avoid IP fragmentation), 5× frame-header redundancy

## 9. Documentation

- [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md) — major bugs encountered and how they were solved
- `docs/` — proposal, architecture diagrams, final report