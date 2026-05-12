# RoI-Based Privacy-Preserving Edge Streaming Pipeline

Bandwidth-adaptive, privacy-aware distributed inference on Jetson Orin Nano.

## Concept

The edge node detects regions of interest (people) on-device and transmits only
the cropped patches over UDP. Background pixels never leave the camera. The
server reassembles, recovers missing patches via per-object zero-order hold,
and stitches a usable view onto a neutral background.

## Architecture

```
Camera → YOLOv8n (FP16) → Patch crop → Adaptive JPEG → UDP chunks
                                                          │
                                                          ▼
                                          [drop / delay / noise simulator]
                                                          │
                                                          ▼
                              Reassembly → Recovery → Stitching → Decision
                                                          │
                                                          ▼
                                                Streamlit dashboard
              (feedback: target JPEG quality, priority) ◀ ─ ─ ─ ─ ─ ─ ─
```

## Layout

| Folder | Role |
|---|---|
| `edge/`           | Runs on Jetson Orin Nano |
| `edge/detector/`  | YOLOv8n TensorRT inference wrapper |
| `edge/patch/`     | bbox crop + adaptive JPEG encoding |
| `edge/transport/` | UDP sender, packet builder, chunking |
| `server/`         | Runs on laptop (incl. Streamlit dashboard) |
| `server/transport/` | UDP receiver + constraint simulator (drop/delay/noise) |
| `server/stitcher/`  | Naive ROI stitching (paste patches onto neutral bg) |
| `server/recovery/`  | IoU tracker, zero-order hold (planned, Week 2) |
| `server/decision/`  | Normal / Warning / Emergency logic (planned, Week 5) |
| `server/dashboard/` | Streamlit app (planned, Week 6) |
| `common/`         | Packet schemas and shared config |
| `scripts/`        | Build, run, and test scripts |
| `models/`         | Source models (`.pt`, `.onnx`) |
| `engines/`        | TensorRT `.engine` files (built on Jetson, gitignored) |
| `data/`           | Test images and demo clips (gitignored) |
| `logs/`           | Build / runtime logs (gitignored) |
| `results/`        | Evaluation outputs, plots, tables |
| `docs/`           | Proposal, diagrams, report |

## Status

**Week 1 — Edge baseline + ROI streaming end-to-end** ✅
- [x] Project scaffold
- [x] YOLOv8n PyTorch → ONNX export
- [x] YOLOv8n FP16 TensorRT engine build (Jetson)
- [x] Edge: patch extractor + JPEG encoder (quality sweep)
- [x] Edge: UDP transport with FRAME_HEADER + chunking
- [x] Server: reassembly + naive stitching
- [x] End-to-end clip verified visually faithful

**Week 2 — Baselines + recovery layer** (next)
- [ ] Full-frame JPEG / H.264 baselines for comparison
- [ ] Server: IoU tracker + zero-order hold recovery
- [ ] Server: constraint simulator (drop / delay / noise via `tc netem`)

**Weeks 3-6** (planned per proposal)
- [ ] Adaptive quality controller (server → edge feedback)
- [ ] Per-patch quality policy
- [ ] Server: Streamlit dashboard
- [ ] Evaluation (bandwidth, mAP under stress, decision correctness)
- [ ] (Stretch) Dynamic-batched ResNet18 classifier

## Results so far

Measurements on persons.jpg (1080×810, 4 person detections) and
person-bicycle-car-detection.mp4 (768×432, 12 FPS), same-subnet Wi-Fi.

| Stage | Measurement |
|---|---|
| FP16 engine size            | 8.8 MB |
| GPU compute (detector)      | 4.33 ms |
| End-to-end detect            | 27.6 ms |
| Patch extract                | 0.07 ms |
| JPEG encode (4 patches q=75) | 4.27 ms |
| Bytes per frame @ q=75       | 84 KB (4 persons) |
| UDP transport per frame      | ~1.8 ms (64 chunks) |
| Loopback byte-equality       | 100% (4 patches, 64 chunks) |
| Wi-Fi delivery (Day 6)       | 100/100 frames complete, 0 loss |
| Pipeline FPS (cap'd)         | 12.1 FPS (matched source native) |

The clean Wi-Fi baseline shows zero natural packet loss, so the
Week 4-5 recovery-layer evaluation will inject artificial loss with
`tc netem` (1% / 5% / 10% / 20% loss rates).

See `results/baselines.md` for the full quality-vs-bytes sweep table
and per-day measurements.

## Quick start

### Build the FP16 engine (on Jetson, one-time)

```bash
bash scripts/build_engine_fp16.sh
```
Output: `engines/yolov8n_fp16.engine`, build log at `logs/build_fp16.log`.

### Single-image sanity check (Jetson)

```bash
PYTHONPATH=. python -u scripts/test_patch_pipeline.py
```
Runs detect → extract → encode on persons.jpg, writes 4 JPEGs to
`results/patches/` and a quality sweep CSV to `results/patch_sizes.csv`.

### Loopback UDP test (Jetson, one process)

```bash
PYTHONPATH=. python -u scripts/test_udp_loopback.py
```
Verifies wire format and chunk reassembly byte-for-byte on 127.0.0.1.

### End-to-end ROI streaming (two machines)

On the **laptop** (server), receiver + stitcher:
```bash
PYTHONPATH=. python -u scripts/run_server_stitch.py \
    --output results/stitched.mp4 \
    --fps 12 \
    --idle-timeout 5
```

On the **Jetson** (edge), capture loop:
```bash
PYTHONPATH=. python -u scripts/run_edge_video.py \
    --server <laptop_ip> \
    --source data/videos/person-bicycle-car-detection.mp4 \
    --target-fps 12 \
    --max-frames 100
```

The laptop writes `results/stitched.mp4`: persons-only on a neutral grey
canvas, the visual proof that ROI streaming works end-to-end.

## Hardware / Software

- Edge: NVIDIA Jetson Orin Nano (JetPack 6.x, TensorRT, CUDA)
- Server: laptop with Python 3.10+, Streamlit (Week 6), OpenCV
- Communication: UDP sockets, custom 32-byte wire header,
  application-level chunking (≤1368 B payload per packet to avoid
  IP fragmentation)