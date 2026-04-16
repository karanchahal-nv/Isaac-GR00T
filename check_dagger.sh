#!/usr/bin/env bash
# Local GR00T v3 dual-camera finetune (pretrain + intervention LeRobot roots).
set -euo pipefail

ISAAC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ISAAC_ROOT}/isaac_manipulator_finetuning"

# Prefer conda env `gr00t` (torch, gr00t, tyro).
if command -v conda &>/dev/null; then
  eval "$(conda shell.bash hook)"
elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "${HOME}/anaconda3/etc/profile.d/conda.sh"
else
  echo "ERROR: conda not found. Install conda or run: conda activate gr00t" >&2
  exit 1
fi
conda activate gr00t

PRE="${ISAAC_ROOT}/lerobot_dataset/cable_pickup_v1_tester_tray_one_end_inserted_trimmed_full_1_0_lerobot/cable_pickup_v1_tester_tray_one_end_inserted_trimmed_full_1_0_lerobot_dataset"
INT="${ISAAC_ROOT}/lerobot_dataset/dagger_interventions_rollout_1_lerobot/inference_intervention_lerobot_dataset"
OUT="${ISAAC_ROOT}/groot_models/local_mixture_1epoch"

python finetune_groot_v3_dual_camera.py \
  --dataset-path "$PRE" \
  --intervention-dataset-path "$INT" \
  --base-model-path nvidia/GR00T-N1.5-3B \
  --video-backend torchcodec \
  --output-dir "$OUT" \
  --report-to tensorboard \
  --num-action-steps 32 \
  --max-steps 1082 \
  --save-steps 500 \
  --batch-size 16 \
  --max-action-dim 7
