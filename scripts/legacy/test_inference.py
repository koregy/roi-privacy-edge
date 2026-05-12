"""Sanity check: load the FP16 engine, detect on persons.jpg, save annotated output."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2

# Add project root to path so 'edge.*' imports work
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from edge.detector.yolov8_trt import YOLOv8TRT  # noqa: E402


def main() -> None:
    engine_path = ROOT / "engines" / "yolov8n_fp16.engine"
    image_path = ROOT / "data" / "test_images" / "persons.jpg"
    output_path = ROOT / "results" / "test_inference.jpg"

    output_path.parent.mkdir(exist_ok=True)

    print(f"Loading engine: {engine_path}")
    detector = YOLOv8TRT(str(engine_path))
    print(f"  Input  tensor: {detector.input_name} shape={detector.input_shape}")
    print(f"  Output tensor: {detector.output_name} shape={detector.output_shape}")

    print(f"\nReading image: {image_path}")
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read {image_path}")
    h, w = image.shape[:2]
    print(f"  Image size: {w}x{h}")

    # Warmup
    print("\nWarmup (5 iterations)...")
    for _ in range(5):
        detector.detect(image)

    # Timed run
    print("Timed run (50 iterations)...")
    t0 = time.perf_counter()
    detections = []
    for _ in range(50):
        detections = detector.detect(image)
    elapsed = (time.perf_counter() - t0) / 50 * 1000  # ms per call
    print(f"  Mean end-to-end latency: {elapsed:.2f} ms ({1000 / elapsed:.1f} FPS)")

    # Report
    print(f"\nDetections (person only): {len(detections)}")
    for i, d in enumerate(detections):
        print(f"  [{i}] bbox=({d.x1},{d.y1})-({d.x2},{d.y2}) "
              f"conf={d.confidence:.3f} class={d.class_id}")

    # Visualize
    annotated = image.copy()
    for d in detections:
        cv2.rectangle(annotated, (d.x1, d.y1), (d.x2, d.y2), (0, 255, 0), 2)
        label = f"person {d.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(annotated, (d.x1, d.y1 - th - 6), (d.x1 + tw, d.y1), (0, 255, 0), -1)
        cv2.putText(annotated, label, (d.x1, d.y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    cv2.imwrite(str(output_path), annotated)
    print(f"\nAnnotated image saved: {output_path}")


if __name__ == "__main__":
    main()
