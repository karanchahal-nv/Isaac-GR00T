# --- Required by run_dagger.sh (defaults assume monorepo checkout) ---
export ISAAC_MANIPULATOR_FINETUNING_ROOT=~/workspaces/isaac/isaac_manipulator_finetuning
# Osmo unpacks an outer folder; LeRobot root is the inner *_dataset directory (has meta/modality.json).
# If you train against a trimmed_full_1_0 LeRobot release, point this at that dataset’s inner *_dataset path.
export DAGGER_PRETRAIN_DATASET_PATH=/home/karan/workspaces/isaac/lerobot_dataset/cable_pickup_v1_tester_tray_one_end_inserted_trimmed_full_0_3_lerobot/cable_pickup_v1_tester_tray_one_end_inserted_trimmed_full_0_3_lerobot_dataset/
export DAGGER_ROSBAG_INBOX=~/workspaces/isaac/dagger_rosbag_inbox
# Fresh root per base policy so iter_* checkpoints/interventions do not collide with prior DAgger runs.
export DAGGER_OUTPUT_ROOT=~/workspaces/isaac/dagger_output_gr00t_trimmed_full_1_0
# Base policy for this DAgger line (HF-style folder with checkpoint-* step dir).
export GROOT_CHECKPOINT_PATH=~/workspaces/isaac/groot_models/gr00t_policy_cable_pickup_v1_trimmed_full_1_0/finetuned_gr00t_v3_dual_camera/checkpoint-10000/
export DAGGER_BASE_MODEL_PATH=~/workspaces/isaac/groot_models/gr00t_policy_cable_pickup_v1_trimmed_full_1_0/finetuned_gr00t_v3_dual_camera/checkpoint-10000/

# --- Inference subprocess (same host:port the robot uses for /act) ---
export DAGGER_INFERENCE_SCRIPT=~/workspaces/isaac/groot/Isaac-GR00T/scripts/inference_service.py
export DAGGER_INFERENCE_HOST=10.111.83.67
export DAGGER_INFERENCE_PORT=8000

# --- HTTP trigger: POST /dagger/ready JSON {"iteration": N} (separate from /act) ---
export DAGGER_HTTP_READY_HOST=10.111.83.67
export DAGGER_HTTP_READY_PORT=8765

# --- Optional: match run_inference.sh ALT OOD (dagger_loop subprocess does not pass these yet) ---
export ALT_CHECKPOINT=~/workspaces/isaac/alt_models/alt_policy_cable_pickup/alt_policy_output/encoder_final.pt
export ALT_LOOKUP_TABLE=~/workspaces/isaac/alt_models/alt_policy_cable_pickup/alt_policy_output/lookup_table.pkl

mkdir -p "$DAGGER_ROSBAG_INBOX" "$DAGGER_OUTPUT_ROOT"

echo "ISAAC_MANIPULATOR_FINETUNING_ROOT=$ISAAC_MANIPULATOR_FINETUNING_ROOT"
echo "DAGGER_PRETRAIN_DATASET_PATH=$DAGGER_PRETRAIN_DATASET_PATH"
echo "DAGGER_ROSBAG_INBOX=$DAGGER_ROSBAG_INBOX"
echo "DAGGER_OUTPUT_ROOT=$DAGGER_OUTPUT_ROOT"
echo "DAGGER_BASE_MODEL_PATH=$DAGGER_BASE_MODEL_PATH"
echo "GROOT_CHECKPOINT_PATH=${GROOT_CHECKPOINT_PATH}"
echo "DAGGER_INFERENCE_HOST=$DAGGER_INFERENCE_HOST"
echo "DAGGER_INFERENCE_PORT=$DAGGER_INFERENCE_PORT"
echo "DAGGER_HTTP_READY_HOST=$DAGGER_HTTP_READY_HOST"
echo "DAGGER_HTTP_READY_PORT=$DAGGER_HTTP_READY_PORT"
