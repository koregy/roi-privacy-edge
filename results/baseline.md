cd ~/roi-privacy-edge
cat > results/baselines.md << 'EOF'
# YOLOv8n TensorRT Engine Baselines

Measured on Jetson Orin Nano (MAXN power mode), input 1×3×640×640.

| Engine | GPU Compute (mean) | Throughput | Engine size | Built |
|---|---|---|---|---|
| FP32  | 11.97 ms | ~78 qps  | 14 MB  | 2026-05-09 |
| FP16  |  4.33 ms | 230 qps  | 8.8 MB | 2026-05-10 |

FP16 speedup over FP32: **2.76× GPU compute, 2.95× throughput**.

## End-to-end Python pipeline (FP16)

Measured on test image (810×1080) with `scripts/test_inference.py`:

| Stage | Latency (mean) |
|---|---|
| Engine GPU compute (trtexec) | 4.33 ms |
| Python end-to-end (preprocess + infer + NMS) | 27.62 ms |
| Effective FPS | 36.2 |

Overhead breakdown (rough): preprocess ~6ms, H2D ~1ms, GPU ~4.5ms,
D2H ~1ms, postprocess+NMS ~15ms. Postprocessing dominates.

## Source

- ONNX: `models/yolov8n.onnx` (opset 12, simplify=True, imgsz=640)
- Build: `bash scripts/build_engine_fp16.sh`
- Logs: `logs/build_fp16.log`

## Notes

- FP32 baseline is kept for the project report only; the runtime system uses FP16.
- INT8 was originally planned but dropped in proposal v2 (focus shifted from
  precision dispatch to RoI-based privacy streaming).
