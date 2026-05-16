#!/bin/bash

if [ -z "$GROOT_CHECKPOINT_PATH" ]; then
    echo "Error: GROOT_CHECKPOINT_PATH environment variable is not set" >&2
    exit 1
fi

NUM_ACTION_STEPS="${NUM_ACTION_STEPS:-32}"

echo "Starting GR00T HTTP server with INPAINTING mode (client-driven RTC)"
echo "  Action horizon:     $NUM_ACTION_STEPS"
echo "  Prefix-clamp params (current_index, estimated_delay) come from the"
echo "  ROS client per-request in the HTTP body 'rtc' block. No server-side"
echo "  --num-prefix-steps / --n-action-execute-steps anymore."

# Action-chunk capture for offline RTC-vs-no-RTC plot comparison.
# Override via env: CAPTURE_CHUNKS_PATH=/some/path.npz MAX_CAPTURES=N bash ...
CAPTURE_CHUNKS_PATH="${CAPTURE_CHUNKS_PATH:-/tmp/chunks_rtc.npz}"
MAX_CAPTURES="${MAX_CAPTURES:-5}"

python scripts/inference_service.py \
    --server \
    --http-server \
    --host 10.111.83.67 \
    --port 8000 \
    --model-path "$GROOT_CHECKPOINT_PATH" \
    --num-action-steps "$NUM_ACTION_STEPS" \
    --use-inpainting \
    --capture-chunks-path "$CAPTURE_CHUNKS_PATH" \
    --max-captures "$MAX_CAPTURES"
