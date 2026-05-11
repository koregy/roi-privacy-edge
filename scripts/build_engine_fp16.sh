#!/bin/bash
# Build a TensorRT FP16 engine for YOLOv8n on Jetson.
#
# Run from anywhere — the script cd's to the project root first.
#   bash scripts/build_engine_fp16.sh

set -e

# Move to project root regardless of where the script is invoked from.
cd "$(dirname "$0")/.."

ONNX_PATH="models/yolov8n.onnx"
ENGINE_PATH="engines/yolov8n_fp16.engine"
LOG_PATH="logs/build_fp16.log"
TRTEXEC="/usr/src/tensorrt/bin/trtexec"

# Sanity checks
if [ ! -f "$ONNX_PATH" ]; then
    echo "ERROR: ONNX model not found at $ONNX_PATH"
    exit 1
fi

if [ ! -x "$TRTEXEC" ]; then
    echo "ERROR: trtexec not found at $TRTEXEC"
    echo "       Adjust TRTEXEC path in this script if your JetPack puts it elsewhere."
    exit 1
fi

mkdir -p engines logs

echo "=== Building FP16 TensorRT engine ==="
echo "Input:  $ONNX_PATH"
echo "Output: $ENGINE_PATH"
echo "Log:    $LOG_PATH"
echo ""

"$TRTEXEC" \
    --onnx="$ONNX_PATH" \
    --saveEngine="$ENGINE_PATH" \
    --fp16 \
    --memPoolSize=workspace:2048M \
    --verbose 2>&1 | tee "$LOG_PATH"

echo ""
echo "=== Build complete ==="
ls -lh "$ENGINE_PATH"
