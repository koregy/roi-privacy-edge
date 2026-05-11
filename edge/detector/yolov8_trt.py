"""YOLOv8 TensorRT inference wrapper.

Loads a FP16 .engine built from yolov8n.onnx, runs inference on a single image,
and returns post-processed detections (xyxy bboxes + confidence + class id).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import cv2
import numpy as np
import tensorrt as trt
from cuda.bindings import runtime as cudart


@dataclass
class Detection:
    """One post-processed detection in original image coordinates."""
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_id: int


class YOLOv8TRT:
    """TensorRT inference for YOLOv8n.

    The ONNX export from ultralytics produces a single output tensor of shape
    (1, 84, 8400) where 84 = 4 bbox coords + 80 class scores, and 8400 is the
    number of anchor positions for input 640x640.
    """

    def __init__(
        self,
        engine_path: str | Path,
        input_size: int = 640,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.5,
    ) -> None:
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        # --- Load engine ---
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to load engine: {engine_path}")
        self.context = self.engine.create_execution_context()

        # --- Inspect IO tensors (TensorRT 10 API) ---
        self.input_name = None
        self.output_name = None
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.input_name = name
                self.input_shape = self.engine.get_tensor_shape(name)
            else:
                self.output_name = name
                self.output_shape = self.engine.get_tensor_shape(name)

        # --- Allocate device buffers ---
        # Input:  (1, 3, 640, 640) float16  (we'll feed fp32 → TRT casts to fp16)
        # Output: (1, 84, 8400)    float32
        self.input_nbytes = int(np.prod(self.input_shape) * np.dtype(np.float32).itemsize)
        self.output_nbytes = int(np.prod(self.output_shape) * np.dtype(np.float32).itemsize)

        err, self.d_input = cudart.cudaMalloc(self.input_nbytes)
        assert err == cudart.cudaError_t.cudaSuccess
        err, self.d_output = cudart.cudaMalloc(self.output_nbytes)
        assert err == cudart.cudaError_t.cudaSuccess

        err, self.stream = cudart.cudaStreamCreate()
        assert err == cudart.cudaError_t.cudaSuccess

        # Pre-allocated host output buffer
        self.h_output = np.empty(self.output_shape, dtype=np.float32)

        # Bind tensor addresses
        self.context.set_tensor_address(self.input_name, int(self.d_input))
        self.context.set_tensor_address(self.output_name, int(self.d_output))

    # --- Pre-process ---
    def _preprocess(self, image_bgr: np.ndarray) -> tuple[np.ndarray, float, int, int]:
        """Letterbox resize to (input_size, input_size), keep aspect ratio.

        Returns:
            blob: (1, 3, H, W) float32 normalized to [0,1], RGB.
            scale: scaling factor applied to original.
            pad_x, pad_y: padding added to (left, top).
        """
        h0, w0 = image_bgr.shape[:2]
        scale = self.input_size / max(h0, w0)
        new_w, new_h = int(round(w0 * scale)), int(round(h0 * scale))
        resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        pad_x = (self.input_size - new_w) // 2
        pad_y = (self.input_size - new_h) // 2
        canvas = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

        # BGR → RGB, HWC → CHW, normalize, add batch dim
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        blob = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
        blob = np.ascontiguousarray(blob[None, ...])  # (1,3,H,W)
        return blob, scale, pad_x, pad_y

    # --- Inference ---
    def _infer(self, blob: np.ndarray) -> np.ndarray:
        """Run TensorRT inference and return raw output (1, 84, 8400)."""
        # H2D
        err, = cudart.cudaMemcpyAsync(
            self.d_input, blob.ctypes.data, blob.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self.stream,
        )
        assert err == cudart.cudaError_t.cudaSuccess

        # Execute
        ok = self.context.execute_async_v3(stream_handle=self.stream)
        assert ok, "TensorRT execute_async_v3 failed"

        # D2H
        err, = cudart.cudaMemcpyAsync(
            self.h_output.ctypes.data, self.d_output, self.output_nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, self.stream,
        )
        assert err == cudart.cudaError_t.cudaSuccess

        err, = cudart.cudaStreamSynchronize(self.stream)
        assert err == cudart.cudaError_t.cudaSuccess

        return self.h_output

    # --- Post-process ---
    def _postprocess(
        self,
        raw: np.ndarray,
        scale: float,
        pad_x: int,
        pad_y: int,
        orig_h: int,
        orig_w: int,
        person_only: bool = True,
    ) -> List[Detection]:
        """Decode raw output, filter, NMS, transform to original coords."""
        # raw: (1, 84, 8400) → transpose to (8400, 84)
        preds = raw[0].T  # (8400, 84)

        # Split: first 4 cols are xywh (center), remaining 80 are class scores
        boxes_xywh = preds[:, :4]
        scores = preds[:, 4:]  # (8400, 80)

        # Per-anchor best class
        class_ids = np.argmax(scores, axis=1)
        confs = scores[np.arange(len(scores)), class_ids]

        # Confidence filter
        mask = confs >= self.conf_threshold
        if person_only:
            mask &= (class_ids == 0)  # COCO class 0 = person

        boxes_xywh = boxes_xywh[mask]
        confs = confs[mask]
        class_ids = class_ids[mask]

        if len(boxes_xywh) == 0:
            return []

        # xywh (center) → xyxy
        cx, cy, w, h = boxes_xywh.T
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

        # NMS (OpenCV expects xywh in absolute coords)
        boxes_for_nms = np.stack([x1, y1, w, h], axis=1).astype(np.float32)
        keep = cv2.dnn.NMSBoxes(
            boxes_for_nms.tolist(),
            confs.astype(np.float32).tolist(),
            self.conf_threshold,
            self.iou_threshold,
        )
        if len(keep) == 0:
            return []
        keep = np.array(keep).flatten()

        boxes_xyxy = boxes_xyxy[keep]
        confs = confs[keep]
        class_ids = class_ids[keep]

        # Undo letterbox: subtract pad, divide by scale
        boxes_xyxy[:, [0, 2]] = (boxes_xyxy[:, [0, 2]] - pad_x) / scale
        boxes_xyxy[:, [1, 3]] = (boxes_xyxy[:, [1, 3]] - pad_y) / scale

        # Clip to image bounds
        boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, orig_w - 1)
        boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, orig_h - 1)

        detections: List[Detection] = []
        for (x1, y1, x2, y2), c, cls in zip(boxes_xyxy, confs, class_ids):
            detections.append(Detection(
                x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2),
                confidence=float(c), class_id=int(cls),
            ))
        return detections

    # --- Public API ---
    def detect(self, image_bgr: np.ndarray, person_only: bool = True) -> List[Detection]:
        """Run end-to-end detection on a BGR image (OpenCV format)."""
        h0, w0 = image_bgr.shape[:2]
        blob, scale, pad_x, pad_y = self._preprocess(image_bgr)
        raw = self._infer(blob)
        return self._postprocess(raw, scale, pad_x, pad_y, h0, w0, person_only)

    def __del__(self) -> None:
        # Best-effort cleanup
        if hasattr(self, "d_input"):
            cudart.cudaFree(self.d_input)
        if hasattr(self, "d_output"):
            cudart.cudaFree(self.d_output)
        if hasattr(self, "stream"):
            cudart.cudaStreamDestroy(self.stream)
