# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Thread-safe latest inference snapshot + JPEG blobs for the localhost ALT/VLA dashboard."""

from __future__ import annotations

import copy
import math
import threading
from typing import Any

import numpy as np
import torch


class InferenceVizState:
    """Holds the latest ``/act`` visualization payload (updated from the HTTP worker thread)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._seq = 0
        self._meta: dict[str, Any] | None = None
        self._jpeg_query: dict[int, bytes] = {}
        self._jpeg_match: dict[tuple[int, int], bytes] = {}

    def publish(
        self,
        meta: dict[str, Any],
        jpeg_query: dict[int, bytes],
        jpeg_match: dict[tuple[int, int], bytes],
    ) -> int:
        """Replace snapshot; return monotonic sequence id for cache-busting URLs."""
        with self._lock:
            self._seq += 1
            self._meta = copy.deepcopy(meta)
            self._jpeg_query = dict(jpeg_query)
            self._jpeg_match = dict(jpeg_match)
            return self._seq

    def get_seq(self) -> int:
        with self._lock:
            return self._seq

    def get_meta_copy(self) -> dict[str, Any] | None:
        with self._lock:
            if self._meta is None:
                return None
            return copy.deepcopy(self._meta)

    def get_jpeg_query(self, cam_index: int) -> bytes | None:
        with self._lock:
            return self._jpeg_query.get(cam_index)

    def get_jpeg_match(self, neighbor_index: int, cam_index: int) -> bytes | None:
        with self._lock:
            return self._jpeg_match.get((neighbor_index, cam_index))


def numpy_action_to_rms_per_dim(
    action_dict: dict[str, Any],
    *,
    scale: float,
) -> list[dict[str, Any]]:
    """For each 2D float array ``(T, D)``, RMS over time per column, times ``scale``.

    ``scale`` is typically ``180/pi`` (radians to degrees) or ``1.0`` if already degrees.
    """
    rows: list[dict[str, Any]] = []
    for key, val in sorted(action_dict.items()):
        if key.startswith("annotation."):
            continue
        arr = val
        if hasattr(arr, "detach"):
            arr = arr.detach().cpu().numpy()
        else:
            arr = np.asarray(arr)
        if arr.dtype.kind not in "fiu" or arr.ndim != 2:
            continue
        arr = arr.astype(np.float64, copy=False)
        t, d = arr.shape
        if t < 1 or d < 1:
            continue
        for j in range(d):
            col = arr[:, j]
            rms = float(math.sqrt(float(np.mean(col * col))) * scale)
            rows.append(
                {
                    "label": f"{key}[{j}]",
                    "rms": rms,
                },
            )
    return rows


def flatten_action_dict_to_matrix(
    action_dict: dict[str, Any],
) -> tuple[np.ndarray, list[str]]:
    """Stack all 2D float action blocks (sorted keys) into ``(T, D)`` and per-column labels."""
    blocks: list[np.ndarray] = []
    labels: list[str] = []
    for key in sorted(action_dict.keys()):
        if key.startswith("annotation."):
            continue
        val = action_dict[key]
        if hasattr(val, "detach"):
            val = val.detach().cpu().numpy()
        else:
            val = np.asarray(val)
        if val.dtype.kind not in "fiu" or val.ndim != 2:
            continue
        blocks.append(val.astype(np.float64, copy=False))
        for j in range(blocks[-1].shape[1]):
            labels.append(f"{key}[{j}]")
    if not blocks:
        return np.zeros((0, 0), dtype=np.float64), []
    t_min = min(b.shape[0] for b in blocks)
    cols = [b[:t_min] for b in blocks]
    mat = np.concatenate(cols, axis=1)
    return mat, labels


def per_joint_mean_delta_vs_alt_degrees(
    action_dict: dict[str, Any],
    alt_chunk: np.ndarray,
    *,
    n_joints: int = 7,
    n_steps: int = 16,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[list[float]], list[str]]:
    """ALT vs GR00T deltas in degrees.

    - Yellow bars: per joint ``mean_t (GR00T[t,i] - ALT[t,i])`` (radians → degrees).
    - ``err_time_joint_deg``: signed ``err[t][i]`` in degrees, shape ``(T_use, D_use)`` rows=t,
      columns=joint index (JSON: list of rows).
    """
    groot, labels = flatten_action_dict_to_matrix(action_dict)
    alt = np.asarray(alt_chunk, dtype=np.float64)
    if groot.size == 0 or alt.size == 0:
        return [], {"ok": False, "reason": "empty_action_or_alt"}, [], []
    ta, da = alt.shape
    tg, dg = groot.shape
    t_use = min(n_steps, ta, tg)
    d_use = min(n_joints, da, dg)
    diff = groot[:t_use, :d_use] - alt[:t_use, :d_use]
    diff_deg = diff * (180.0 / math.pi)
    mean_rad = np.mean(diff, axis=0)
    mean_deg = mean_rad * (180.0 / math.pi)
    info: dict[str, Any] = {
        "ok": True,
        "T_compare": int(t_use),
        "D_compare": int(d_use),
        "T_alt": int(ta),
        "T_gr00t": int(tg),
        "D_alt": int(da),
        "D_gr00t": int(dg),
        "n_joints_cap": int(n_joints),
        "n_steps_cap": int(n_steps),
    }
    rows: list[dict[str, Any]] = []
    for j in range(d_use):
        lab = labels[j] if j < len(labels) else f"joint[{j}]"
        rows.append({"label": lab, "rms": float(mean_deg[j])})
    grid: list[list[float]] = diff_deg.astype(np.float64).tolist()
    col_labels = [labels[j] if j < len(labels) else f"joint[{j}]" for j in range(d_use)]
    return rows, info, grid, col_labels


def action_t0_joint_values_degrees(
    action_dict: dict[str, Any],
    alt_chunk: np.ndarray,
    *,
    n_joints: int = 7,
    scale: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Per-joint values at chunk index ``t=0``: GR00T vs lookup-stored parquet ``action``.

    ``alt_chunk`` is a row from ``lookup_table.pkl`` ``actions`` (raw training
    targets). ``scale`` is ``180/π`` if policy actions are radians, else ``1``.
    """
    groot, labels = flatten_action_dict_to_matrix(action_dict)
    alt = np.asarray(alt_chunk, dtype=np.float64)
    if groot.size == 0 or alt.size == 0 or groot.shape[0] < 1 or alt.shape[0] < 1:
        return [], []
    d_use = min(n_joints, groot.shape[1], alt.shape[1])
    g0 = groot[0, :d_use] * scale
    a0 = alt[0, :d_use] * scale
    rows_g: list[dict[str, Any]] = []
    rows_a: list[dict[str, Any]] = []
    for j in range(d_use):
        lab = labels[j] if j < len(labels) else f"joint[{j}]"
        rows_g.append({"label": lab, "rms": float(g0[j])})
        rows_a.append({"label": lab, "rms": float(a0[j])})
    return rows_g, rows_a


def observation_video_to_jpeg(
    arr: np.ndarray,
    *,
    max_side: int = 320,
    quality: int = 80,
) -> bytes:
    """Numpy video frame -> JPEG bytes (resize, CHW uint8)."""
    import torch.nn.functional as F
    from torchvision.io import encode_jpeg

    x = np.asarray(arr)
    if x.ndim == 4 and x.shape[0] == 1:
        x = x[0]
    if x.ndim != 3:
        raise ValueError(f"expected HWC or 1HWC uint8 image, got shape {x.shape}")
    if x.shape[-1] == 3:
        t = torch.from_numpy(x).permute(2, 0, 1)
    elif x.shape[0] == 3:
        t = torch.from_numpy(x)
    else:
        raise ValueError(f"expected last or first dim RGB, got shape {x.shape}")
    if t.dtype != torch.uint8:
        t = t.float()
        if t.max() <= 1.0:
            t = t * 255.0
        t = t.clamp(0, 255).to(torch.uint8)
    c, h, w = t.shape
    m = max(h, w)
    if m > max_side and max_side > 0:
        scale = max_side / float(m)
        nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
        xf = t.float() / 255.0
        xf = F.interpolate(
            xf.unsqueeze(0),
            size=(nh, nw),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        t = (xf * 255.0).clamp(0, 255).to(torch.uint8)
    enc = encode_jpeg(t.cpu(), quality=quality)
    if isinstance(enc, torch.Tensor):
        return bytes(enc.cpu().numpy().tobytes())
    return bytes(enc)
