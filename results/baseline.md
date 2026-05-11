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

## Patch encoding (FP16 detector → 4 persons, q sweep)

Measured on persons.jpg (1080×810) with `scripts/test_patch_pipeline.py`:

| JPEG quality | avg bytes/patch | KB/frame (4 persons) |
|---|---|---|
| 30  | 11,262 |  44.0 |
| 50  | 15,074 |  58.9 |
| 60  | 16,934 |  66.2 |
| 70  | 19,791 |  77.3 |
| 75  | 21,494 |  84.0 |
| 80  | 24,041 |  93.9 |
| 85  | 27,440 | 107.2 |
| 90  | 33,319 | 130.2 |
| 95  | 45,416 | 177.4 |

Knee around q=85-90 — above q=90, +36% bytes for +5q (visually
imperceptible). Default `DEFAULT_JPEG_QUALITY=75` set as bytes/quality
sweet spot. Full-frame 1080p JPEG baseline for comparison: ~200-400 KB,
so ROI approach already yields ~3-4× reduction even at 4 persons.
