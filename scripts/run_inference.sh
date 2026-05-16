#!/bin/bash
set -euo pipefail

# Repo root (so the script works from any cwd).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.." || exit 1

# Optional: rebuild ALT lookup table from LeRobot data before starting inference.
#   ./scripts/run_inference.sh --rebuild-alt-lookup
# or: REBUILD_ALT_LOOKUP=1 ./scripts/run_inference.sh
# Requires: ALT_CHECKPOINT, ALT_DATASET_PATH (LeRobot root), ALT_LOOKUP_TABLE (output .pkl path).
# Uses isaac_manipulator_finetuning on PYTHONPATH (export ISAAC_MANIPULATOR_FINETUNING_ROOT=...).
# Optional: ALT_DATASET_CAMERA_KEYS (comma-separated) -> --camera-keys k1 k2 ... for build_lookup_table.
# Optional: ALT_BUILD_DEVICE (e.g. cuda|cpu), ALT_ACTION_CHUNK_SIZE (default 16 in build_lookup_table).
#
# Shortcut: source ./set_alt_params.sh then run (same as exporting ALT_* yourself):
#   ./scripts/run_inference.sh --alt
REBUILD_ALT_LOOKUP_TABLE=0
USE_ALT=0
for _arg in "$@"; do
  case "$_arg" in
    --rebuild-alt-lookup) REBUILD_ALT_LOOKUP_TABLE=1 ;;
    --alt) USE_ALT=1 ;;
  esac
done
if [ "${REBUILD_ALT_LOOKUP:-0}" = 1 ]; then
  REBUILD_ALT_LOOKUP_TABLE=1
fi

if [ "$USE_ALT" = 1 ]; then
  _alt_params="$SCRIPT_DIR/../set_alt_params.sh"
  if [ ! -f "$_alt_params" ]; then
    echo "Error: --alt expects ${_alt_params}" >&2
    exit 1
  fi
  # shellcheck source=/dev/null
  source "$_alt_params"
  : "${ALT_CHECKPOINT:?set ALT_CHECKPOINT in set_alt_params.sh}"
  : "${ALT_LOOKUP_TABLE:?set ALT_LOOKUP_TABLE in set_alt_params.sh}"
  : "${ALT_DATASET_PATH:?set ALT_DATASET_PATH in set_alt_params.sh}"
  : "${VIZ_PORT:?set VIZ_PORT in set_alt_params.sh}"
  : "${ALT_DATASET_CAMERA_KEYS:?set ALT_DATASET_CAMERA_KEYS in set_alt_params.sh}"
  : "${ISAAC_MANIPULATOR_FINETUNING_ROOT:?set ISAAC_MANIPULATOR_FINETUNING_ROOT in set_alt_params.sh}"
fi

# Prefer conda env ``gr00t`` (torch, pyarrow, GR00T deps). Override: ``PYTHON=/path/to/python bash ...``
if [ -z "${PYTHON:-}" ]; then
  for _py in \
    "${HOME}/miniconda3/envs/gr00t/bin/python" \
    "${HOME}/anaconda3/envs/gr00t/bin/python" \
    "${HOME}/mambaforge/envs/gr00t/bin/python"; do
    if [ -x "$_py" ]; then PYTHON="$_py"; break; fi
  done
  PYTHON="${PYTHON:-python3}"
fi

: "${GROOT_CHECKPOINT_PATH:?set GROOT_CHECKPOINT_PATH (GR00T checkpoint dir) before running}"

# Optional ALT OOD score on each HTTP /act response (ood_score = NN cosine, higher => more in-distribution).
# Requires isaac_manipulator_finetuning on PYTHONPATH or pass --alt-finetuning-root via ALT_ARGS below.
#   export ISAAC_MANIPULATOR_FINETUNING_ROOT=/path/to/isaac_manipulator_finetuning
#   export ALT_CHECKPOINT=/path/to/encoder_final.pt
#   export ALT_LOOKUP_TABLE=/path/to/lookup_table.pkl
#
# Optional localhost dashboard (SSE + JPEG thumbnails; does not bloat /act JSON):
#   export ALT_DATASET_PATH=/path/to/lerobot_dataset_root   # same root used for build_lookup_table.py
#   export VIZ_PORT=8765
#   export VIZ_HOST=127.0.0.1
# If lookup_table.pkl lacks or truncates model_config.camera_keys, set LeRobot video keys as
# **one comma-separated string** (passed as a single CLI arg; avoids tyro dropping earlier flags):
#   export ALT_DATASET_CAMERA_KEYS="observation.images.camera_1_color_image_raw_compressed,observation.images.camera_2_color_image_raw_compressed"
ALT_ARGS=()
if [ -n "${ALT_CHECKPOINT:-}" ] && [ -n "${ALT_LOOKUP_TABLE:-}" ]; then
    ALT_ARGS+=(--alt-checkpoint "$ALT_CHECKPOINT" --alt-lookup-table "$ALT_LOOKUP_TABLE")
    [ -n "${ISAAC_MANIPULATOR_FINETUNING_ROOT:-}" ] && ALT_ARGS+=(--alt-finetuning-root "$ISAAC_MANIPULATOR_FINETUNING_ROOT")
fi
if [ -n "${ALT_DATASET_PATH:-}" ] && [ -n "${VIZ_PORT:-}" ]; then
    ALT_ARGS+=(--alt-dataset-path "$ALT_DATASET_PATH" --viz-port "$VIZ_PORT")
    [ -n "${VIZ_HOST:-}" ] && ALT_ARGS+=(--viz-host "$VIZ_HOST")
    if [ -n "${ALT_DATASET_CAMERA_KEYS:-}" ]; then
        ALT_ARGS+=(--alt-dataset-camera-keys "${ALT_DATASET_CAMERA_KEYS}")
    fi
fi

if [ "$REBUILD_ALT_LOOKUP_TABLE" = 1 ]; then
    if [ -z "${ALT_CHECKPOINT:-}" ] || [ -z "${ALT_DATASET_PATH:-}" ] || [ -z "${ALT_LOOKUP_TABLE:-}" ]; then
        echo "Error: --rebuild-alt-lookup (or REBUILD_ALT_LOOKUP=1) requires ALT_CHECKPOINT, ALT_DATASET_PATH, and ALT_LOOKUP_TABLE" >&2
        exit 1
    fi
    if [ ! -d "$ALT_DATASET_PATH" ]; then
        echo "Error: ALT_DATASET_PATH is not a directory: $ALT_DATASET_PATH" >&2
        exit 1
    fi
    if [ -z "${ISAAC_MANIPULATOR_FINETUNING_ROOT:-}" ]; then
        echo "Error: rebuild needs ISAAC_MANIPULATOR_FINETUNING_ROOT so \`python -m alt.build_lookup_table\` can import the alt package" >&2
        exit 1
    fi
    export PYTHONPATH="$ISAAC_MANIPULATOR_FINETUNING_ROOT${PYTHONPATH:+:$PYTHONPATH}"
    echo "Rebuilding ALT lookup table from ALT_DATASET_PATH -> ALT_LOOKUP_TABLE ..."
    _build_cmd=(
        "$PYTHON" -m alt.build_lookup_table
        --checkpoint "$ALT_CHECKPOINT"
        --dataset-path "$ALT_DATASET_PATH"
        --output "$ALT_LOOKUP_TABLE"
    )
    if [ -n "${ALT_ACTION_CHUNK_SIZE:-}" ]; then
        _build_cmd+=(--action-chunk-size "$ALT_ACTION_CHUNK_SIZE")
    fi
    if [ -n "${ALT_BUILD_DEVICE:-}" ]; then
        _build_cmd+=(--device "$ALT_BUILD_DEVICE")
    fi
    if [ -n "${ALT_DATASET_CAMERA_KEYS:-}" ]; then
        # Tyro expects one --camera-keys followed by all values (not repeated flags).
        IFS=',' read -r -a _cam_keys <<< "$ALT_DATASET_CAMERA_KEYS"
        _cam_trimmed=()
        for _k in "${_cam_keys[@]}"; do
            _k="${_k#"${_k%%[![:space:]]*}"}"
            _k="${_k%"${_k##*[![:space:]]}"}"
            [ -n "$_k" ] && _cam_trimmed+=("$_k")
        done
        if [ "${#_cam_trimmed[@]}" -gt 0 ]; then
            _build_cmd+=(--camera-keys "${_cam_trimmed[@]}")
        fi
    fi
    "${_build_cmd[@]}"
fi

# Action-chunk capture for offline RTC-vs-no-RTC plot comparison.
# Override via env: CAPTURE_CHUNKS_PATH=/some/path.npz MAX_CAPTURES=N bash ...
CAPTURE_CHUNKS_PATH="${CAPTURE_CHUNKS_PATH:-/tmp/chunks_no_rtc.npz}"
MAX_CAPTURES="${MAX_CAPTURES:-2}"

python scripts/inference_service.py --server --http-server --host 10.111.83.67 --port 8000 --model-path "${GROOT_CHECKPOINT_PATH}" --num-action-steps 32 --capture-chunks-path "$CAPTURE_CHUNKS_PATH" --max-captures "$MAX_CAPTURES" "${ALT_ARGS[@]}"
