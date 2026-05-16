#!/usr/bin/env python3
"""
GR00T HTTP Server Module

This module provides HTTP server functionality for GR00T model inference.
It exposes a REST API for easy integration with web applications and other services.

Dependencies:
    => Server: `pip install uvicorn fastapi json-numpy`
    => Client: `pip install requests json-numpy`
"""

import json
import logging
import traceback
from typing import Any, Callable, Dict, Optional

import base64
import numpy as np
import json_numpy
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from gr00t.model.policy import Gr00tPolicy, Gr00tInpaintingPolicy

# Patch json to handle numpy arrays
json_numpy.patch()



def decode_numpy_from_base64(obj):
    """Decode numpy array from C++ base64 format."""
    if isinstance(obj, dict) and "__ndarray__" in obj:
        raw_bytes = base64.b64decode(obj["__ndarray__"])
        dtype = np.dtype(obj.get("dtype", "uint8"))
        shape = tuple(obj.get("shape", []))
        return np.frombuffer(raw_bytes, dtype=dtype).reshape(shape)
    return obj


def log_video_observation_shape(key: str, value: Any) -> None:
    """Log video payload shape before GR00T transforms run."""
    video = np.asarray(value)
    resolution = tuple(video.shape[-3:-1][::-1]) if video.ndim >= 3 else None
    print(
        f"Decoded video observation {key}: "
        f"shape={video.shape} size={video.size} ndim={video.ndim} "
        f"dtype={video.dtype} inferred_resolution={resolution}",
        flush=True,
    )


def flatten_nested_observation(obs: Dict[str, Any]) -> Dict[str, Any]:
    """Accept C++ InferenceHttpNode's grouped observation payload."""
    if not any(key in obs for key in ("video", "state", "language")):
        return obs

    flattened = dict(obs)
    video_group = flattened.pop("video", {})
    for key, value in video_group.items():
        full_key = f"video.{key}"
        flattened[full_key] = decode_numpy_from_base64(value)
        log_video_observation_shape(full_key, flattened[full_key])

    state_group = flattened.pop("state", {})
    for key, value in state_group.items():
        flattened[f"state.{key}"] = np.asarray(value)

    language_group = flattened.pop("language", {})
    for key, value in language_group.items():
        flattened[key] = value

    return flattened


class HTTPInferenceServer:
    def __init__(
        self,
        policy: Gr00tPolicy,
        port: int,
        host: str = "0.0.0.0",
        api_token: Optional[str] = None,
        response_extras: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        post_act: Optional[Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]]] = None,
    ):
        """
        A simple HTTP server for GR00T models; exposes `/act` to predict an action for a given observation.
            => Takes in observation dict with numpy arrays
            => Returns action dict with numpy arrays

        If the policy is a ``Gr00tInpaintingPolicy``, a ``POST /reset`` endpoint
        is also registered to clear the action buffer between episodes.

        Args:
            response_extras: Optional callable ``(observation_after_decode) -> dict`` with
                JSON-serializable values (float, int, bool, str, lists). Merged into the
                ``/act`` response after action keys (e.g. ALT ``ood_score``).
            post_act: If set, called as ``(observation, action_dict) -> dict`` **after**
                ``get_action``. Return value is merged into the response (same as
                ``response_extras``). When ``post_act`` is set, ``response_extras`` is
                ignored (use one path to avoid duplicate ALT work).
        """
        self.policy = policy
        self.port = port
        self.host = host
        self.api_token = api_token
        self.response_extras = response_extras
        self.post_act = post_act
        self.app = FastAPI(title="GR00T Inference Server", version="1.0.0")

        # Register endpoints
        self.app.post("/act")(self.predict_action)
        self.app.get("/health")(self.health_check)
        self.app.post("/reset")(self.reset_action_buffer)

    def predict_action(self, payload: Dict[str, Any]) -> JSONResponse:
        """Predict action from observation."""
        try:
            # Handle double-encoded payloads (for compatibility)
            if "encoded" in payload:
                assert len(payload.keys()) == 1, "Only uses encoded payload!"
                payload = json.loads(payload["encoded"])

            # Validate required fields
            if "observation" not in payload:
                raise HTTPException(
                    status_code=400, detail="Missing 'observation' field in payload"
                )

            obs = flatten_nested_observation(payload["observation"])

            for key in list(obs.keys()):
                if key.startswith("video."):
                    obs[key] = decode_numpy_from_base64(obs[key])
                    log_video_observation_shape(key, obs[key])
            # print(obs["video.camera"])

            # Stateful client-driven RTC: if the request body has an "rtc"
            # block AND the policy is an inpainting policy, stage the RTC
            # info so the policy uses the right prefix this call.
            rtc = payload.get("rtc")
            if rtc is not None and isinstance(self.policy, Gr00tInpaintingPolicy):
                self.policy.stage_external_rtc(
                    current_index=int(rtc["current_action_sequence_index"]),
                    estimated_delay=int(rtc["estimated_delay_ticks"]),
                    prefix_attention_horizon=(
                        int(rtc["prefix_attention_horizon"])
                        if "prefix_attention_horizon" in rtc else None
                    ),
                    prefix_attention_schedule=(
                        str(rtc["prefix_attention_schedule"])
                        if "prefix_attention_schedule" in rtc else None
                    ),
                    max_guidance_weight=(
                        float(rtc["max_guidance_weight"])
                        if "max_guidance_weight" in rtc else None
                    ),
                    last_actual_delay_ticks=(
                        int(rtc["last_actual_delay_ticks"])
                        if "last_actual_delay_ticks" in rtc else None
                    ),
                )

            # Run inference
            action = self.policy.get_action(obs)

            # Return action as JSON with numpy arrays
            out: Dict[str, Any] = {k: v.tolist() for k, v in action.items()}
            if self.post_act is not None:
                extra = self.post_act(obs, action)
                if extra:
                    out.update(extra)
            elif self.response_extras is not None:
                extra = self.response_extras(obs)
                if extra:
                    out.update(extra)
            if "ood_score" in out:
                msg = f"ood_score={out['ood_score']}"
                if "is_ood" in out:
                    msg += f" is_ood={out['is_ood']}"
                print(msg)
            return JSONResponse(content=out)

        except Exception as e:
            logging.error(traceback.format_exc())
            logging.warning(
                "Your request threw an error; make sure your request complies with the expected format:\n"
                "{'observation': dict} where observation contains the required modalities.\n"
                "Example observation keys: video.ego_view, state.left_arm, state.right_arm, etc."
            )
            raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

    def reset_action_buffer(self) -> Dict[str, str]:
        """Reset the inpainting action buffer (call between episodes).

        Only has an effect when the policy is a ``Gr00tInpaintingPolicy``.
        Safe to call on a regular ``Gr00tPolicy`` -- it simply returns OK.
        """
        if hasattr(self.policy, "reset_action_buffer"):
            self.policy.reset_action_buffer()
            print("Action buffer reset (inpainting state cleared)")
            return {"status": "reset", "inpainting": True}
        return {"status": "reset", "inpainting": False}

    def health_check(self) -> Dict[str, Any]:
        """Health check endpoint."""
        is_inpainting = isinstance(self.policy, Gr00tInpaintingPolicy)
        return {
            "status": "healthy",
            "model": "GR00T",
            "inpainting": is_inpainting,
            "response_extras": self.response_extras is not None,
            "post_act": self.post_act is not None,
        }

    def run(self) -> None:
        """Start the HTTP server."""
        is_inpainting = isinstance(self.policy, Gr00tInpaintingPolicy)
        print(f"Starting GR00T HTTP server on {self.host}:{self.port}")
        print(f"Inpainting mode: {is_inpainting}")
        print("Available endpoints:")
        print("  POST /act   - Get action prediction from observation")
        print("  POST /reset - Reset inpainting action buffer (between episodes)")
        print("  GET  /health - Health check")
        uvicorn.run(self.app, host=self.host, port=self.port)


def create_http_server(
    policy: Gr00tPolicy,
    port: int,
    host: str = "0.0.0.0",
    api_token: Optional[str] = None,
    response_extras: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    post_act: Optional[Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]]] = None,
) -> HTTPInferenceServer:
    """Factory function to create an HTTP inference server."""
    return HTTPInferenceServer(policy, port, host, api_token, response_extras, post_act)
