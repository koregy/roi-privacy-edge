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
| `server/recovery/`  | IoU tracker, zero-order hold, stitching |
| `server/decision/`  | Normal / Warning / Emergency logic |
| `server/dashboard/` | Streamlit app |
| `common/`         | Packet schemas and shared config |
| `scripts/`        | Build and utility scripts |
| `models/`         | Source models (`.onnx`) |
| `engines/`        | TensorRT `.engine` files (built on Jetson, gitignored) |
| `data/`           | MOT17 clips, demo footage (gitignored) |
| `logs/`           | Build / runtime logs (gitignored) |
| `results/`        | Evaluation outputs, plots, tables |
| `docs/`           | Proposal, diagrams, report |

## Status

- [x] Project scaffold
- [x] YOLOv8n PyTorch → ONNX export
- [ ] YOLOv8n FP16 TensorRT engine build (Jetson)
- [ ] Edge: patch extractor + adaptive JPEG encoder
- [ ] Edge: UDP transport with chunking
- [ ] Server: reassembly + stitching
- [ ] Server: IoU tracker + zero-order hold recovery
- [ ] Server: constraint simulator (drop / delay / noise)
- [ ] Server: Streamlit dashboard
- [ ] Evaluation (bandwidth, mAP under stress, decision correctness)
- [ ] (Stretch) Dynamic-batched ResNet18 classifier

## Build the FP16 engine (on Jetson)

```bash
bash scripts/build_engine_fp16.sh
```

Output: `engines/yolov8n_fp16.engine`, build log at `logs/build_fp16.log`.

## Hardware / Software

- Edge: NVIDIA Jetson Orin Nano (JetPack 6.x, TensorRT, CUDA)
- Server: laptop with Python 3.10+, Streamlit, OpenCV
- Communication: UDP sockets