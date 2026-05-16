# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
GR00T Inference Service

This script provides both ZMQ and HTTP server/client implementations for deploying GR00T models.
The HTTP server exposes a REST API for easy integration with web applications and other services.

1. Default is zmq server.

Run server: python scripts/inference_service.py --server
Run client: python scripts/inference_service.py --client

2. Run as Http Server:

Dependencies for `http_server` mode:
    => Server (runs GR00T model on GPU): `pip install uvicorn fastapi json-numpy`
    => Client: `pip install requests json-numpy`

HTTP Server Usage:
    python scripts/inference_service.py --server --http-server --port 8000

ALT OOD score (HTTP only): add ``--alt-checkpoint`` and ``--alt-lookup-table``; each ``/act``
response then includes ``ood_score`` (nearest-neighbor cosine vs the lookup table; higher is
more in-distribution) and ``is_ood`` when ``--alt-ood-threshold`` is set. Set
``ISAAC_MANIPULATOR_FINETUNING_ROOT`` or ``--alt-finetuning-root`` so ``import alt.policy`` works.

Localhost dashboard: ``--viz-port`` (e.g. 8765) with ``--alt-dataset-path`` to the LeRobot root used
to build the lookup table. Opens a second HTTP app on ``--viz-host`` (default ``127.0.0.1``) with
SSE + thumbnails (no large JSON on ``/act``). Requires ALT checkpoint + lookup table.

HTTP Client Usage (assuming a server running on 0.0.0.0:8000):
    python scripts/inference_service.py --client --http-server --host 0.0.0.0 --port 8000

You can use bore to forward the port to your client: `159.223.171.199` is bore.pub.
    bore local 8000 --to 159.223.171.199

3. TensorRT Support:

For accelerated inference using TensorRT, first build the TensorRT engines using the deployment scripts,
then run the server with the --use-tensorrt flag:

TensorRT Server Usage:
    python scripts/inference_service.py --server --use-tensorrt --trt-engine-path gr00t_engine

TensorRT HTTP Server Usage:
    python scripts/inference_service.py --server --http-server --use-tensorrt --trt-engine-path gr00t_engine --port 8000

Note: TensorRT engines must be built before running with --use-tensorrt flag.
See deployment_scripts/README.md for instructions on building TensorRT engines.
"""

import math
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Literal, Optional

from gr00t.eval.inference_viz_state import (
    InferenceVizState,
    numpy_action_to_rms_per_dim,
    observation_video_to_jpeg,
    action_t0_joint_values_degrees,
    per_joint_mean_delta_vs_alt_degrees,
)

import numpy as np
import tyro

from gr00t.data.embodiment_tags import EMBODIMENT_TAG_MAPPING
from gr00t.eval.robot import RobotInferenceClient, RobotInferenceServer
from gr00t.experiment.data_config import load_data_config
from gr00t.data.dataset import ModalityConfig
from gr00t.model.policy import Gr00tPolicy, Gr00tInpaintingPolicy


@dataclass
class ArgsConfig:
    """Command line arguments for the inference service."""

    model_path: str = None
    """Path to the model checkpoint directory."""

    embodiment_tag: Literal[tuple(EMBODIMENT_TAG_MAPPING.keys())] = "new_embodiment"
    """The embodiment tag for the model."""

    data_config: str = "fourier_gr1_arms_waist"
    """
    The name of the data config to use, e.g. so100, fourier_gr1_arms_only, unitree_g1, etc.

    Or a path to a custom data config file. e.g. "module:ClassName" format.
    See gr00t/experiment/data_config.py for more details.
    """

    port: int = 5555
    """The port number for the server."""

    host: str = "localhost"
    """The host address for the server."""

    server: bool = False
    """Whether to run the server."""

    client: bool = False
    """Whether to run the client."""

    denoising_steps: int = 4
    """The number of denoising steps to use."""

    api_token: str = None
    """API token for authentication. If not provided, authentication is disabled."""

    http_server: bool = False
    """Whether to run it as HTTP server. Default is ZMQ server."""

    use_tensorrt: bool = False
    """Whether to use TensorRT for inference. Requires TensorRT engines to be built."""

    trt_engine_path: str = "gr00t_engine"
    """Path to the TensorRT engine directory. Only used when use_tensorrt is True."""

    vit_dtype: Literal["fp16", "fp8"] = "fp8"
    """ViT model dtype (fp16, fp8). Only used when use_tensorrt is True."""

    llm_dtype: Literal["fp16", "nvfp4", "fp8"] = "nvfp4"
    """LLM model dtype (fp16, nvfp4, fp8). Only used when use_tensorrt is True."""

    dit_dtype: Literal["fp16", "fp8"] = "fp8"
    """DiT model dtype (fp16, fp8). Only used when use_tensorrt is True."""

    num_action_steps: int = 1
    """Number of action steps to use."""

    use_inpainting: bool = False
    """Whether to use the stateful Gr00tInpaintingPolicy for client-driven RTC.
    When enabled, the policy caches its returned chunks and the client drives
    the prefix-clamp parameters per-request via the ``rtc`` block in the
    HTTP body (``current_action_sequence_index``, ``estimated_delay_ticks``).
    The first request omits ``rtc`` and falls through to vanilla denoise."""

    inpainting_mode: Literal["hybrid", "it_rtc"] = "hybrid"
    """Phase-2 inpainting algorithm. 'hybrid' (default) does hard-clamp [0,d) +
    soft-guidance [d,K). 'it_rtc' is pure PI IT-RTC (no overwriting, in-distribution
    inputs, constraint enforced entirely via gradient guidance with W[0:d]=1.0)."""

    max_guidance_weight_override: Optional[float] = None
    """If set, overrides the max_guidance_weight value sent by the ROS client.
    Used for sweeping guidance strength without changing the ROS payload."""

    capture_chunks_path: Optional[str] = None
    """If set, save the first ``max_captures`` predicted action chunks to this
    .npz path. Used for offline RTC-vs-no-RTC comparison plots."""

    max_captures: int = 2
    """Number of action chunks to capture before writing the .npz."""

    alt_checkpoint: Optional[str] = None
    """If set with ``alt_lookup_table``, run ALT encoder + lookup each ``/act`` and add scores."""

    alt_lookup_table: Optional[str] = None
    """Path to ``lookup_table.pkl`` from ``build_lookup_table``."""

    alt_finetuning_root: Optional[str] = None
    """Directory containing the ``alt`` package (``isaac_manipulator_finetuning``). If unset, uses env ``ISAAC_MANIPULATOR_FINETUNING_ROOT`` or an existing ``import alt``."""

    alt_video_keys: tuple[str, ...] = (
        "video.camera_1_color_image_raw_compressed",
        "video.camera_2_color_image_raw_compressed",
    )
    """GR00T observation video keys in **ALT training camera order**"""

    alt_state_key: str = "state.joint_state_position"
    """GR00T observation key for robot state (must match ALT training ``state_dim``)."""

    alt_ood_threshold: Optional[float] = 0.75
    """ALT cosine threshold; ``None`` disables ``is_ood`` (``ood_score`` still returned)."""

    alt_device: str = "cuda"
    """Device for ALT encoder."""

    alt_dataset_path: Optional[str] = None
    """LeRobot dataset root (same as ``build_lookup_table``). Required when ``viz_port > 0``."""

    alt_dataset_camera_keys: tuple[str, ...] = ()
    """LeRobot ``meta/info.json`` video feature names (sorted, same as ALT training).

    Prefer **one** comma-separated value (avoids tyro only keeping the last repeated flag), e.g.
    ``--alt-dataset-camera-keys 'observation.images.camera_1_...,observation.images.camera_2_...'``.
    You may still repeat the flag; each value can also contain commas.
    Required when ``lookup_table.pkl`` has no or wrong ``model_config["camera_keys"]``.
    """

    viz_port: int = 0
    """If > 0, serve the ALT/VLA dashboard on this port (default host ``127.0.0.1`` only)."""

    viz_host: str = "127.0.0.1"
    """Bind address for the dashboard (use ``127.0.0.1`` for local-only)."""

    viz_top_k: int = 3
    """Number of ALT neighbors to show."""

    viz_thumb_max_side: int = 320
    """Max width/height for dashboard JPEG thumbnails."""

    viz_jpeg_quality: int = 80
    """JPEG quality (1–100) for dashboard images."""

    viz_action_units: Literal["radians", "degrees"] = "radians"
    """Interpretation of action components for RMS bars: ``radians`` multiplies by 180/π for display."""

#####################################################################################


def _prepend_sys_path(root: str) -> None:
    p = str(Path(root).resolve())
    if p not in sys.path:
        sys.path.insert(0, p)


def _ensure_alt_importable(alt_finetuning_root: Optional[str]) -> None:
    if alt_finetuning_root:
        _prepend_sys_path(alt_finetuning_root)
    env_root = os.environ.get("ISAAC_MANIPULATOR_FINETUNING_ROOT", "").strip()
    if env_root:
        _prepend_sys_path(env_root)


def _flatten_alt_dataset_camera_keys(keys: tuple[str, ...]) -> list[str]:
    """Split comma-separated entries; de-duplicate; sort (matches ALT training / LeRobot)."""
    flat: list[str] = []
    for k in keys:
        for part in k.split(","):
            p = part.strip()
            if p:
                flat.append(p)
    return sorted(set(flat))


def build_alt_http_response_extras(
    *,
    alt_checkpoint: str,
    alt_lookup_table: str,
    alt_finetuning_root: Optional[str],
    alt_video_keys: tuple[str, ...],
    alt_state_key: str,
    alt_ood_threshold: Optional[float],
    alt_device: str,
) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """Return a callable for ``HTTPInferenceServer(response_extras=...)``."""

    _ensure_alt_importable(alt_finetuning_root)
    try:
        from alt.policy import ALTPolicy
    except ImportError as e:
        raise ImportError(
            "ALT extras require the ``alt`` package (``isaac_manipulator_finetuning``). "
            "Set --alt-finetuning-root or ISAAC_MANIPULATOR_FINETUNING_ROOT, or install "
            "the package so ``import alt.policy`` works.",
        ) from e

    import torch

    alt_policy = ALTPolicy(
        alt_checkpoint,
        alt_lookup_table,
        ood_threshold=alt_ood_threshold,
        device=alt_device,
    )

    def response_extras(obs: Dict[str, Any]) -> Dict[str, Any]:
        images = []
        for vk in alt_video_keys:
            if vk not in obs:
                raise KeyError(
                    f"ALT video key {vk!r} missing from observation; "
                    f"have keys: {sorted(obs.keys())}",
                )
            arr = np.asarray(obs[vk])
            if arr.ndim == 4 and arr.shape[0] == 1:
                arr = arr[0]
            t = torch.from_numpy(arr)
            if t.ndim == 3 and t.shape[-1] == 3:
                t = t.permute(2, 0, 1).contiguous()
            images.append(t)

        if alt_state_key not in obs:
            raise KeyError(
                f"ALT state key {alt_state_key!r} missing from observation",
            )
        st = np.asarray(obs[alt_state_key], dtype=np.float32)
        if st.ndim == 2 and st.shape[0] == 1:
            st = st[0]
        st_t = torch.from_numpy(st.reshape(-1))

        out_alt = alt_policy.get_action(images, st_t)
        # Nearest-neighbor cosine: higher => more similar to the retrieval database.
        extra: Dict[str, Any] = {
            "ood_score": float(out_alt["cosine_similarity"]),
        }
        if alt_ood_threshold is not None:
            extra["is_ood"] = bool(out_alt["is_ood"])
        return extra

    return response_extras


def build_alt_viz_post_act(
    *,
    alt_checkpoint: str,
    alt_lookup_table: str,
    alt_dataset_path: str,
    alt_finetuning_root: Optional[str],
    alt_video_keys: tuple[str, ...],
    alt_dataset_camera_keys: tuple[str, ...],
    alt_state_key: str,
    alt_ood_threshold: Optional[float],
    alt_device: str,
    viz_state: InferenceVizState,
    viz_top_k: int,
    viz_thumb_max_side: int,
    viz_jpeg_quality: int,
    viz_action_units: Literal["radians", "degrees"],
) -> Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]]:
    """Single ALT forward + viz snapshot per ``/act``; returns HTTP extras (``ood_score``, ``is_ood``)."""

    _ensure_alt_importable(alt_finetuning_root)
    try:
        from alt.lerobot_frame_cache import LeRobotFrameCache
        from alt.policy import ALTPolicy
    except ImportError as e:
        raise ImportError(
            "ALT viz requires ``alt`` from ``isaac_manipulator_finetuning`` "
            "(``--alt-finetuning-root`` or ``ISAAC_MANIPULATOR_FINETUNING_ROOT``).",
        ) from e

    import torch

    alt_policy = ALTPolicy(
        alt_checkpoint,
        alt_lookup_table,
        ood_threshold=alt_ood_threshold,
        device=alt_device,
    )
    if alt_dataset_camera_keys:
        cam_keys = _flatten_alt_dataset_camera_keys(alt_dataset_camera_keys)
    else:
        raw = alt_policy.lookup_model_config.get("camera_keys")
        if not raw or not isinstance(raw, list):
            raise ValueError(
                "lookup_table.pkl has no ``model_config['camera_keys']`` (common with older "
                "pickles). Fix one of: (1) Re-run ``python -m alt.build_lookup_table`` with the "
                "current script so the table embeds checkpoint config; or (2) Pass "
                "``--alt-dataset-camera-keys`` as a **comma-separated** list (recommended), e.g. "
                "``observation.images.camera_1_color_image_raw_compressed,"
                "observation.images.camera_2_color_image_raw_compressed``.",
            )
        cam_keys = sorted(set(raw))
    n_model_cams = int(alt_policy.model.num_cameras)
    if len(cam_keys) != len(alt_video_keys):
        raise ValueError(
            f"ALT dataset camera_keys ({len(cam_keys)}): {cam_keys!r} must match "
            f"--alt-video-keys ({len(alt_video_keys)}): {list(alt_video_keys)!r}. "
            f"Encoder was trained with num_cameras={n_model_cams}. "
            "If you exported ALT_DATASET_CAMERA_KEYS, also set ALT_DATASET_PATH and VIZ_PORT "
            "in the same shell so run_inference.sh forwards --alt-dataset-camera-keys. "
            "If ALT used a single camera, retrain/rebuild with both views or pass one "
            "--alt-video-keys that matches the checkpoint.",
        )
    if len(cam_keys) != n_model_cams:
        raise ValueError(
            f"Dataset camera key count ({len(cam_keys)}) != encoder num_cameras ({n_model_cams}). "
            f"camera_keys={cam_keys!r}. Rebuild lookup_table.pkl or fix --alt-dataset-camera-keys.",
        )

    frame_cache = LeRobotFrameCache(alt_dataset_path, cam_keys)
    scale = 1.0 if viz_action_units == "degrees" else (180.0 / math.pi)
    rms_suffix = (
        "deg (from rad)" if viz_action_units == "radians" else "same units as action"
    )

    def post_act(obs: Dict[str, Any], action: Dict[str, Any]) -> Dict[str, Any]:
        images = []
        for vk in alt_video_keys:
            if vk not in obs:
                raise KeyError(
                    f"ALT video key {vk!r} missing from observation; "
                    f"have keys: {sorted(obs.keys())}",
                )
            arr = np.asarray(obs[vk])
            if arr.ndim == 4 and arr.shape[0] == 1:
                arr = arr[0]
            t = torch.from_numpy(arr)
            if t.ndim == 3 and t.shape[-1] == 3:
                t = t.permute(2, 0, 1).contiguous()
            images.append(t)

        if alt_state_key not in obs:
            raise KeyError(
                f"ALT state key {alt_state_key!r} missing from observation",
            )
        st = np.asarray(obs[alt_state_key], dtype=np.float32)
        if st.ndim == 2 and st.shape[0] == 1:
            st = st[0]
        st_t = torch.from_numpy(st.reshape(-1))

        neighbors = alt_policy.top_neighbors(images, st_t, k=viz_top_k)
        if not neighbors:
            return {}

        cos = float(neighbors[0]["cosine_similarity"])
        is_ood = False
        if alt_ood_threshold is not None:
            is_ood = cos < alt_ood_threshold

        joint_rows = numpy_action_to_rms_per_dim(action, scale=scale)

        rank1_idx = int(neighbors[0]["table_index"])
        alt_chunk = np.asarray(alt_policy.actions[rank1_idx], dtype=np.float64)
        diff_rows, diff_info, err_grid_deg, err_col_labels = per_joint_mean_delta_vs_alt_degrees(
            action,
            alt_chunk,
            n_joints=7,
            n_steps=16,
        )
        t0_g, t0_alt = action_t0_joint_values_degrees(
            action,
            alt_chunk,
            n_joints=7,
            scale=scale,
        )

        jpeg_query: dict[int, bytes] = {}
        for ci, vk in enumerate(alt_video_keys):
            arr = np.asarray(obs[vk])
            if arr.ndim == 4 and arr.shape[0] == 1:
                arr = arr[0]
            jpeg_query[ci] = observation_video_to_jpeg(
                arr,
                max_side=viz_thumb_max_side,
                quality=viz_jpeg_quality,
            )

        jpeg_match: dict[tuple[int, int], bytes] = {}
        for ni, nb in enumerate(neighbors):
            for ci, ck in enumerate(cam_keys):
                jpeg_match[(ni, ci)] = frame_cache.read_camera_jpeg(
                    nb["episode_index"],
                    nb["frame_index"],
                    ck,
                    max_side=viz_thumb_max_side,
                    quality=viz_jpeg_quality,
                )

        neighbors_meta = [
            {
                "table_index": int(n["table_index"]),
                "episode_index": int(n["episode_index"]),
                "frame_index": int(n["frame_index"]),
                "cosine_similarity": float(n["cosine_similarity"]),
                "distance": float(n["distance"]),
            }
            for n in neighbors
        ]

        meta: Dict[str, Any] = {
            "ood_score": cos,
            "is_ood": bool(is_ood) if alt_ood_threshold is not None else None,
            "alt_neighbors": neighbors_meta,
            "alt_video_keys": list(alt_video_keys),
            "dataset_camera_keys": list(cam_keys),
            "joint_action_rms": joint_rows,
            "joint_rms_description": rms_suffix,
            "joint_alt_vs_groot_diff_rms": diff_rows,
            "joint_compare_info": diff_info,
            "joint_diff_description": (
                "mean over first ≤16 steps of (GR00T − ALT rank-1), per joint; "
                "then radians → degrees"
            ),
            "err_time_joint_deg": err_grid_deg,
            "err_joint_col_labels": err_col_labels,
            "groot_action_t0_deg": t0_g,
            "alt_lookup_action_t0_deg": t0_alt,
            "action_t0_description": rms_suffix + "; row index 0 of predicted / retrieved chunk",
        }
        viz_state.publish(meta, jpeg_query, jpeg_match)

        http_extra: Dict[str, Any] = {"ood_score": cos}
        if alt_ood_threshold is not None:
            http_extra["is_ood"] = bool(is_ood)
        return http_extra

    return post_act


def _start_viz_server(viz_host: str, viz_port: int, viz_state: InferenceVizState) -> None:
    import uvicorn

    from gr00t.eval.inference_viz_app import create_viz_app

    app = create_viz_app(viz_state)
    cfg = uvicorn.Config(app, host=viz_host, port=viz_port, log_level="warning")
    server = uvicorn.Server(cfg)
    server.run()


def _example_zmq_client_call(obs: dict, host: str, port: int, api_token: str):
    """
    Example ZMQ client call to the server.
    """
    # Original ZMQ client mode
    # Create a policy wrapper
    policy_client = RobotInferenceClient(host=host, port=port, api_token=api_token)

    print("Available modality config available:")
    modality_configs = policy_client.get_modality_config()
    print(modality_configs.keys())

    time_start = time.time()
    action = policy_client.get_action(obs)
    print(f"Total time taken to get action from server: {time.time() - time_start} seconds")
    return action


def _example_http_client_call(obs: dict, host: str, port: int, api_token: str):
    """
    Example HTTP client call to the server.
    """
    import json_numpy

    json_numpy.patch()
    import requests

    # Send request to HTTP server
    print("Testing HTTP server...")

    time_start = time.time()
    response = requests.post(f"http://{host}:{port}/act", json={"observation": obs})
    print(f"Total time taken to get action from HTTP server: {time.time() - time_start} seconds")

    if response.status_code == 200:
        action = response.json()
        return action
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return {}


def main(args: ArgsConfig):
    if args.server:
        # Create a policy
        # The `Gr00tPolicy` class is being used to create a policy object that encapsulates
        # the model path, transform name, embodiment tag, and denoising steps for the robot
        # inference system. This policy object is then utilized in the server mode to start
        # the Robot Inference Server for making predictions based on the specified model and
        # configuration.

        # we will use an existing data config to create the modality config and transform
        # if a new data config is specified, this expect user to
        # construct your own modality config and transform
        # see gr00t/utils/data.py for more details
        # data_config = load_data_config(args.data_config)
        # modality_config = data_config.modality_config()
        # modality_transform = data_config.transform()
        
        # 1.1 modality configs and transforms
        # data_config_cls = load_data_config(config.data_config)
        # modality_configs = data_config_cls.modality_config()


        # video_keys = ["video.camera_1_color_image_raw_compressed", "video.camera_2_color_image_raw_compressed"]
        video_keys = ["video.front_stereo_camera_left_image_raw_compressed"]
        video_modality = ModalityConfig(
            delta_indices=[0],
            modality_keys=video_keys,
        )

        state_modality = ModalityConfig(
            delta_indices=[0],
            modality_keys=['state.joint_state_position',
                        # 'state.joint_state_velocity',
                        # 'state.insertion_position',
                        # 'state.insertion_rotation'
                        ],
        )

        action_modality = ModalityConfig(
            delta_indices=list(range(args.num_action_steps)),
            modality_keys=["action.joint_position"],
        )

        language_modality = ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.task_description"],
        )

        modality_config = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }



        # transforms = data_config_cls.transform()
        
        from gr00t.data.transform.base import ComposedModalityTransform
        from gr00t.data.transform import VideoToTensor, VideoCrop, VideoResize, VideoColorJitter, VideoToNumpy
        from gr00t.data.transform.state_action import StateActionToTensor, StateActionTransform
        from gr00t.data.transform.concat import ConcatTransform
        from gr00t.model.transforms import GR00TTransform


        # select the transforms you want to apply to the data
        modality_transform = ComposedModalityTransform(
            transforms=[
                # video transforms
                VideoToTensor(apply_to=video_modality.modality_keys, backend="torchvision"),
                VideoCrop(apply_to=video_modality.modality_keys, scale=0.95, backend="torchvision"),
                VideoResize(apply_to=video_modality.modality_keys, height=224, width=224,
                            interpolation="linear", backend="torchvision" ),
                VideoColorJitter(apply_to=video_modality.modality_keys, brightness=0.3,
                                contrast=0.4, saturation=0.5, hue=0.08, backend="torchvision"),
                VideoToNumpy(apply_to=video_modality.modality_keys),

                # state transforms
                StateActionToTensor(apply_to=state_modality.modality_keys),
                StateActionTransform(apply_to=state_modality.modality_keys, normalization_modes={
                    "state.joint_state_position": "min_max",
                    # "state.joint_state_velocity": "min_max",
                    # "state.insertion_position": "min_max",
                    # "state.insertion_rotation": "min_max",
                }),

                # action transforms
                StateActionToTensor(apply_to=action_modality.modality_keys),
                StateActionTransform(apply_to=action_modality.modality_keys, normalization_modes={
                    "action.joint_position": "min_max",
                }),

                # ConcatTransform
                ConcatTransform(
                    video_concat_order=video_modality.modality_keys,
                    state_concat_order=state_modality.modality_keys,
                    action_concat_order=action_modality.modality_keys,
                ),
                # model-specific transform
                GR00TTransform(
                    state_horizon=len(state_modality.delta_indices),
                    action_horizon=args.num_action_steps,
                    max_state_dim=64,
                    max_action_dim=6,
                ),
            ]
        )

        if args.use_inpainting:
            print(f"Using INPAINTING policy: client-driven RTC (mode={args.inpainting_mode})")
            policy = Gr00tInpaintingPolicy(
                model_path=args.model_path,
                modality_config=modality_config,
                modality_transform=modality_transform,
                embodiment_tag=args.embodiment_tag,
                denoising_steps=args.denoising_steps,
                capture_chunks_path=args.capture_chunks_path,
                max_captures=args.max_captures,
                inpainting_mode=args.inpainting_mode,
                max_guidance_weight_override=args.max_guidance_weight_override,
            )
        else:
            policy = Gr00tPolicy(
                model_path=args.model_path,
                modality_config=modality_config,
                modality_transform=modality_transform,
                embodiment_tag=args.embodiment_tag,
                denoising_steps=args.denoising_steps,
                capture_chunks_path=args.capture_chunks_path,
                max_captures=args.max_captures,
            )

        # Setup TensorRT if requested
        if args.use_tensorrt:
            print(f"Setting up TensorRT engines from: {args.trt_engine_path}")
            print(f"  ViT dtype: {args.vit_dtype}")
            print(f"  LLM dtype: {args.llm_dtype}")
            print(f"  DiT dtype: {args.dit_dtype}")
            from deployment_scripts.trt_model_forward import setup_tensorrt_engines

            setup_tensorrt_engines(
                policy, args.trt_engine_path, args.vit_dtype, args.llm_dtype, args.dit_dtype
            )
            print("TensorRT engines loaded successfully!")

        # Start the server
        if args.http_server:
            from gr00t.eval.http_server import HTTPInferenceServer  # noqa: F401

            response_extras = None
            post_act = None
            if args.viz_port > 0:
                if not args.alt_checkpoint or not args.alt_lookup_table:
                    raise ValueError(
                        "--viz-port requires --alt-checkpoint and --alt-lookup-table.",
                    )
                if not args.alt_dataset_path:
                    raise ValueError(
                        "--alt-dataset-path is required when --viz-port is set "
                        "(LeRobot root used to build the lookup table).",
                    )
                viz_state = InferenceVizState()
                post_act = build_alt_viz_post_act(
                    alt_checkpoint=args.alt_checkpoint,
                    alt_lookup_table=args.alt_lookup_table,
                    alt_dataset_path=args.alt_dataset_path,
                    alt_finetuning_root=args.alt_finetuning_root,
                    alt_video_keys=args.alt_video_keys,
                    alt_dataset_camera_keys=args.alt_dataset_camera_keys,
                    alt_state_key=args.alt_state_key,
                    alt_ood_threshold=args.alt_ood_threshold,
                    alt_device=args.alt_device,
                    viz_state=viz_state,
                    viz_top_k=args.viz_top_k,
                    viz_thumb_max_side=args.viz_thumb_max_side,
                    viz_jpeg_quality=args.viz_jpeg_quality,
                    viz_action_units=args.viz_action_units,
                )
                vt = threading.Thread(
                    target=_start_viz_server,
                    args=(args.viz_host, args.viz_port, viz_state),
                    daemon=True,
                )
                vt.start()
                time.sleep(0.15)
                print(
                    f"ALT viz dashboard: http://{args.viz_host}:{args.viz_port}/ "
                    f"(SSE + JPEG thumbnails; inference stays on {args.host}:{args.port}).",
                )
            elif args.alt_checkpoint or args.alt_lookup_table:
                if not args.alt_checkpoint or not args.alt_lookup_table:
                    raise ValueError(
                        "ALT HTTP extras require both --alt-checkpoint and --alt-lookup-table.",
                    )
                response_extras = build_alt_http_response_extras(
                    alt_checkpoint=args.alt_checkpoint,
                    alt_lookup_table=args.alt_lookup_table,
                    alt_finetuning_root=args.alt_finetuning_root,
                    alt_video_keys=args.alt_video_keys,
                    alt_state_key=args.alt_state_key,
                    alt_ood_threshold=args.alt_ood_threshold,
                    alt_device=args.alt_device,
                )
                print(
                    "ALT OOD scoring enabled: ood_score = NN cosine similarity "
                    "(higher => closer to lookup table).",
                )

            server = HTTPInferenceServer(
                policy,
                port=args.port,
                host=args.host,
                api_token=args.api_token,
                response_extras=response_extras,
                post_act=post_act,
            )
            server.run()
        else:
            server = RobotInferenceServer(policy, host='10.111.83.67', port=args.port, api_token=args.api_token)
            server.run()

    # Here is mainly a testing code
    elif args.client:
        # In this mode, we will send a random observation to the server and get an action back
        # This is useful for testing the server and client connection

        # Making prediction...
        # - obs: video.ego_view: (1, 256, 256, 3)
        # - obs: state.left_arm: (1, 7)
        # - obs: state.right_arm: (1, 7)
        # - obs: state.left_hand: (1, 6)
        # - obs: state.right_hand: (1, 6)
        # - obs: state.waist: (1, 3)

        # - action: action.left_arm: (16, 7)
        # - action: action.right_arm: (16, 7)
        # - action: action.left_hand: (16, 6)
        # - action: action.right_hand: (16, 6)
        # - action: action.waist: (16, 3)
        obs = {
            "video.camera_1_color_image_raw_compressed": np.random.randint(0, 256, (1, 480, 640, 3), dtype=np.uint8),
            "video.camera_2_color_image_raw_compressed": np.random.randint(0, 256, (1, 480, 640, 3), dtype=np.uint8),
            "front_stereo_camera_left_image_raw_compressed": np.random.randint(0, 256, (1, 480, 640, 3), dtype=np.uint8),
            "state.joint_state_position": np.random.rand(1, 6),
            # "state.right_arm": np.random.rand(1, 7),
            # "state.left_hand": np.random.rand(1, 6),
            # "state.right_hand": np.random.rand(1, 6),
            # "state.waist": np.random.rand(1, 3),
            "annotation.human.action.task_description": ["do your thing!"],
        }

        if args.http_server:
            action = _example_http_client_call(obs, args.host, args.port, args.api_token)
        else:
            for i in range(100):
                action = _example_zmq_client_call(obs, args.host, args.port, args.api_token)

        for key, value in action.items():
            if hasattr(value, "shape"):
                print(f"{key}: shape {getattr(value, 'shape', '')}")
            else:
                print(f"{key}: {value}")
    else:
        raise ValueError("Please specify either --server or --client")


if __name__ == "__main__":
    config = tyro.cli(ArgsConfig)
    main(config)
