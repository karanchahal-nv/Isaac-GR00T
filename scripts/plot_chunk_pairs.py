"""Plot consecutive chunk pairs from a single RTC capture as a 4-row x 6-col grid.

Each row is one (chunk_i, chunk_{i+1}) pair; each column is one joint. Auto-detects
the time-offset c by finding the longest exact prefix match between chunks, so the
overlap region is drawn at the correct x-position. The hard-clamped region is
shaded so you can see what the model copied vs. what it freely denoised.

Usage:
    python scripts/plot_chunk_pairs.py \\
        --chunks /tmp/chunks_rtc.npz \\
        --output /tmp/chunk_pairs_grid.png
"""

import argparse
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np


def detect_offset(prev: np.ndarray, curr: np.ndarray, eps: float = 1e-6) -> Tuple[int, int]:
    """Find (c, d) such that curr[0:d] == prev[c+1:c+1+d] is the longest exact match.

    Returns (-1, 0) if no exact match is found at any c.
    """
    H = prev.shape[0]
    best_c, best_d = -1, 0
    for c in range(H):
        max_d = 0
        for d in range(1, H - c):
            if np.abs(curr[:d] - prev[c + 1 : c + 1 + d]).max() < eps:
                max_d = d
            else:
                break
        if max_d > best_d:
            best_c, best_d = c, max_d
    return best_c, best_d


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--chunks", type=Path, default=Path("/tmp/chunks_rtc.npz"),
                    help="Path to the .npz with shape (N, T, D) under action.* keys")
    ap.add_argument("--output", type=Path, default=Path("/tmp/chunk_pairs_grid.png"))
    ap.add_argument("--key", type=str, default=None,
                    help="Action key to plot. Auto-pick if omitted.")
    ap.add_argument("--joint-names", type=str, default=None,
                    help="Comma-separated names for joint axes (optional)")
    args = ap.parse_args()

    data = np.load(args.chunks)
    if args.key is None:
        action_keys = [k for k in data.files if k.startswith("action")]
        preferred = [k for k in action_keys if "joint" in k or "position" in k]
        args.key = (preferred or action_keys)[0]
    chunks = data[args.key]
    if chunks.ndim != 3:
        raise ValueError(f"{args.chunks}: '{args.key}' has shape {chunks.shape}; expected (N, T, D)")

    N, T, D = chunks.shape
    print(f"Loaded {N} chunks of shape ({T}, {D}) from key '{args.key}'")
    if N < 2:
        raise ValueError(f"need at least 2 chunks, got {N}")

    # Optional per-call RTC metadata recorded by Gr00tInpaintingPolicy.
    # Each is an array of length N indexed by chunk index. -1 = unset.
    d_actual_per_call = data["rtc.last_actual_delay_ticks"] if "rtc.last_actual_delay_ticks" in data.files else None
    if d_actual_per_call is not None:
        print(f"  RTC metadata present: rtc.last_actual_delay_ticks = {d_actual_per_call.tolist()}")

    n_pairs = N - 1
    joint_names = (
        [s.strip() for s in args.joint_names.split(",")] if args.joint_names else
        [f"joint {i}" for i in range(D)]
    )

    fig, axes = plt.subplots(
        nrows=n_pairs, ncols=D,
        figsize=(3.2 * D, 2.6 * n_pairs),
        squeeze=False,
    )
    fig.suptitle(f"Consecutive chunk pairs (from {args.chunks.name})", y=0.995)

    for pair_idx in range(n_pairs):
        prev_chunk = chunks[pair_idx]
        curr_chunk = chunks[pair_idx + 1]
        c, d = detect_offset(prev_chunk, curr_chunk)
        offset = c + 1 if c >= 0 else 0   # x-offset for curr_chunk

        for j in range(D):
            ax = axes[pair_idx, j]
            x_prev = np.arange(T)
            x_curr = np.arange(T) + offset

            # Shade the hard-clamped region
            if c >= 0 and d > 0:
                ax.axvspan(
                    offset, offset + d,
                    color="lightyellow", alpha=0.7,
                    label="clamped" if (pair_idx == 0 and j == D - 1) else None,
                )

            ax.plot(x_prev, prev_chunk[:, j],
                    color="tab:blue", linewidth=1.4, marker="o", markersize=2.5,
                    label=f"chunk {pair_idx}" if j == D - 1 else None)
            ax.plot(x_curr, curr_chunk[:, j],
                    color="tab:orange", linewidth=1.4, marker="s", markersize=2.5,
                    label=f"chunk {pair_idx + 1}" if j == D - 1 else None)

            # Black dotted line: end of server hard-clamp region (= offset + d_est).
            # This is where the model regained freedom in the server's view.
            if c >= 0 and d > 0:
                ax.axvline(
                    offset + d,
                    color="black", linestyle=":", linewidth=1.0, alpha=0.7,
                    label="end of clamp (offset + d_est)"
                    if (pair_idx == 0 and j == D - 1) else None,
                )

            # Red dashed line: ACTUAL robot transition tick (offset + d_actual).
            # The client measures d_actual when a chunk ARRIVES, then reports
            # it on the NEXT request. So d_actual of chunks[pair_idx+1] is at
            # rtc.last_actual_delay_ticks[pair_idx+2], not [pair_idx+1].
            # For the last pair there is no follow-up request, so d_actual is
            # unknown and we skip the line.
            d_actual_idx = pair_idx + 2
            if d_actual_per_call is not None and d_actual_idx < len(d_actual_per_call):
                d_actual = int(d_actual_per_call[d_actual_idx])
                if d_actual >= 0 and c >= 0:
                    ax.axvline(
                        offset + d_actual,
                        color="tab:red", linestyle="--", linewidth=1.2, alpha=0.9,
                        label="actual transition (offset + d_actual)"
                        if (pair_idx == 0 and j == D - 1) else None,
                    )

            if pair_idx == 0:
                ax.set_title(joint_names[j] if j < len(joint_names) else f"joint {j}")
            if j == 0:
                ax.set_ylabel(f"pair {pair_idx}->{pair_idx + 1}\nc={c}, d={d}",
                              fontsize=9)
            if pair_idx == n_pairs - 1:
                ax.set_xlabel("global tick")
            ax.tick_params(labelsize=8)

        # Per-row legend on the rightmost axis
        axes[pair_idx, D - 1].legend(loc="upper right", fontsize=7, framealpha=0.9)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=130)
    print(f"wrote: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
