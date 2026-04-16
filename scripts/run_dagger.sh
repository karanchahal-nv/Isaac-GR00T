#!/bin/bash
# Run the DAgger loop (wait for interventions → convert → retrain → restart HTTP inference).
#
# Typical: source ./scripts/set_dagger_env_vars.sh && ./scripts/run_dagger.sh
# Requires GR00T on PYTHONPATH and isaac_manipulator_finetuning for `python -m dagger.dagger_loop`.
#
# Required env:
#   ISAAC_MANIPULATOR_FINETUNING_ROOT  — path to isaac_manipulator_finetuning (parent of dagger/)
#   DAGGER_PRETRAIN_DATASET_PATH       — LeRobot pretrain dataset directory
#   DAGGER_ROSBAG_INBOX                — where iteration_N folders + rosbags land
#   DAGGER_OUTPUT_ROOT                 — checkpoints + converted LeRobot datasets
#
# Base checkpoint (one of):
#   GROOT_CHECKPOINT_PATH or DAGGER_BASE_MODEL_PATH — initial HF id or local checkpoint dir
#
# Optional env:
#   GROOT_ROOT              — Isaac-GR00T root (default: parent of this script)
#   DAGGER_INFERENCE_SCRIPT — path to inference_service.py (default: $GROOT_ROOT/scripts/inference_service.py)
#   DAGGER_INFERENCE_HOST   — bind address for subprocess HTTP server (default: 0.0.0.0)
#   DAGGER_INFERENCE_PORT   — (default: 8000)
#   DAGGER_HTTP_READY_PORT  — if set and > 0, enable POST /dagger/ready on the training machine
#   DAGGER_HTTP_READY_HOST  — bind for ready API (default: 0.0.0.0)
#
# Optional ALT OOD on inference (same as run_inference.sh): set ALT_CHECKPOINT, ALT_LOOKUP_TABLE,
# and ISAAC_MANIPULATOR_FINETUNING_ROOT; then extend dagger_loop / inference launch to pass those
# flags (not wired in the subprocess launcher yet).
#
# Extra CLI args are forwarded to dagger_loop, e.g.:
#   ./scripts/run_dagger.sh --num-iterations 5 --max-steps-per-iter 2000

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GROOT_ROOT="${GROOT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"

if [ -z "${ISAAC_MANIPULATOR_FINETUNING_ROOT:-}" ]; then
    echo "Error: ISAAC_MANIPULATOR_FINETUNING_ROOT is not set" >&2
    exit 1
fi
if [ -z "${DAGGER_PRETRAIN_DATASET_PATH:-}" ]; then
    echo "Error: DAGGER_PRETRAIN_DATASET_PATH is not set" >&2
    exit 1
fi
if [ -z "${DAGGER_ROSBAG_INBOX:-}" ]; then
    echo "Error: DAGGER_ROSBAG_INBOX is not set" >&2
    exit 1
fi
if [ -z "${DAGGER_OUTPUT_ROOT:-}" ]; then
    echo "Error: DAGGER_OUTPUT_ROOT is not set" >&2
    exit 1
fi

BASE_MODEL="${DAGGER_BASE_MODEL_PATH:-${GROOT_CHECKPOINT_PATH:-}}"
if [ -z "$BASE_MODEL" ]; then
    echo "Error: set GROOT_CHECKPOINT_PATH or DAGGER_BASE_MODEL_PATH" >&2
    exit 1
fi

export PYTHONPATH="${GROOT_ROOT}:${ISAAC_MANIPULATOR_FINETUNING_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

INFERENCE_SCRIPT="${DAGGER_INFERENCE_SCRIPT:-$GROOT_ROOT/scripts/inference_service.py}"
INFERENCE_HOST="${DAGGER_INFERENCE_HOST:-0.0.0.0}"
INFERENCE_PORT="${DAGGER_INFERENCE_PORT:-8000}"

READY_ARGS=()
if [ -n "${DAGGER_HTTP_READY_PORT:-}" ] && [ "${DAGGER_HTTP_READY_PORT}" -gt 0 ] 2>/dev/null; then
    READY_ARGS+=(--http-ready-port "$DAGGER_HTTP_READY_PORT")
    READY_ARGS+=(--http-ready-host "${DAGGER_HTTP_READY_HOST:-0.0.0.0}")
fi

cd "$ISAAC_MANIPULATOR_FINETUNING_ROOT"

exec python -m dagger.dagger_loop \
    --pretrain-dataset-path "$DAGGER_PRETRAIN_DATASET_PATH" \
    --rosbag-inbox "$DAGGER_ROSBAG_INBOX" \
    --output-root "$DAGGER_OUTPUT_ROOT" \
    --base-model-path "$BASE_MODEL" \
    --inference-script "$INFERENCE_SCRIPT" \
    --inference-host "$INFERENCE_HOST" \
    --inference-port "$INFERENCE_PORT" \
    "${READY_ARGS[@]}" \
    "$@"
