"""Plot action chunks captured by inference_service.py for RTC vs no-RTC comparison.

Expected input: two .npz files produced by ``Gr00tPolicy._flush_captured_chunks``.
Each .npz contains one array per ``action.*`` key, shape ``(num_captures, T, action_dim)``.

Typical usage after running both inference servers and sending 2 requests each:

    python scripts/plot_action_chunks.py \\
        --no-rtc /tmp/chunks_no_rtc.npz \\
        --rtc    /tmp/chunks_rtc.npz \\
        --output /tmp/action_chunk_comparison.png \\
        --execute-steps 8

The ``--execute-steps`` value should match ``n_action_execute_steps`` from the
inpainting config. It controls the temporal offset between chunk 1 and chunk 2
on the x-axis so you can visually inspect the seam where the two chunks meet:

  - With RTC: chunk 2's prefix should match chunk 1's tail.
  - Without RTC: discontinuity is expected at the seam.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _load_npz(path: Path) -> Tuple[str, np.ndarray]:
    """Load a chunk-capture .npz and return (action_key, array of shape (N, T, D))."""
    if not path.exists():
        raise FileNotFoundError(f"capture file not found: {path}")
    data = np.load(path)
    # Prefer joint-position action key; fall back to first action.* key.
    action_keys = [k for k in data.files if k.startswith("action")]
    if not action_keys:
        raise ValueError(f"{path}: no action.* keys found; got {list(data.files)}")
    preferred = [k for k in action_keys if "joint" in k or "position" in k]
    key = preferred[0] if preferred else action_keys[0]
    arr = data[key]
    if arr.ndim != 3:
        raise ValueError(
            f"{path}: key '{key}' has shape {arr.shape}; expected (N, T, D)"
        )
    return key, arr


def _plot_config(
    ax_row: List[plt.Axes],
    chunks: np.ndarray,
    label_prefix: str,
    color_cycle: List[str],
    execute_steps: int,
) -> None:
    """Draw one config's chunks onto a row of joint subplots.

    ``chunks`` has shape (N, T, D). Each chunk is plotted with its own time
    offset: chunk k starts at t = k * execute_steps so consecutive chunks
    visually align where they overlap in real time.
    """
    n_chunks, T, D = chunks.shape
    for joint_idx in range(D):
        ax = ax_row[joint_idx]
        for k in range(n_chunks):
            x = np.arange(T) + k * execute_steps
            ax.plot(
                x,
                chunks[k, :, joint_idx],
                color=color_cycle[k % len(color_cycle)],
                linewidth=1.6,
                marker="o",
                markersize=3,
                label=f"{label_prefix} chunk {k + 1}",
            )
        # Visually mark where each chunk *starts* (helps see the seam).
        for k in range(1, n_chunks):
            ax.axvline(
                k * execute_steps,
                color="grey",
                linestyle=":",
                linewidth=0.8,
                alpha=0.6,
            )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--no-rtc", type=Path, required=True, help="path to no-RTC chunk .npz")
    p.add_argument("--rtc", type=Path, required=True, help="path to RTC chunk .npz")
    p.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/action_chunk_comparison.png"),
        help="output PNG path",
    )
    p.add_argument(
        "--execute-steps",
        type=int,
        default=8,
        help="n_action_execute_steps; controls x-offset between consecutive chunks",
    )
    p.add_argument("--joint-names", type=str, default=None,
                   help="comma-separated joint names for subplot titles (optional)")
    args = p.parse_args()

    key_a, chunks_no_rtc = _load_npz(args.no_rtc)
    key_b, chunks_rtc = _load_npz(args.rtc)
    print(f"no-RTC: key='{key_a}' shape={chunks_no_rtc.shape}")
    print(f"RTC   : key='{key_b}' shape={chunks_rtc.shape}")

    if chunks_no_rtc.shape != chunks_rtc.shape:
        print(
            f"WARNING: chunk shapes differ "
            f"({chunks_no_rtc.shape} vs {chunks_rtc.shape}); plotting anyway."
        )

    D = chunks_no_rtc.shape[-1]
    joint_names = (
        [s.strip() for s in args.joint_names.split(",")] if args.joint_names else
        [f"joint {i}" for i in range(D)]
    )
    if len(joint_names) < D:
        joint_names += [f"joint {i}" for i in range(len(joint_names), D)]

    # Two rows: top = no-RTC, bottom = RTC. D columns = joints.
    fig, axes = plt.subplots(
        nrows=2,
        ncols=D,
        figsize=(3.2 * D, 6.0),
        sharex=True,
        squeeze=False,
    )
    fig.suptitle("Action-chunk comparison: no-RTC (top) vs RTC inpainting (bottom)")

    _plot_config(
        axes[0].tolist(),
        chunks_no_rtc,
        label_prefix="no-RTC",
        color_cycle=["tab:blue", "tab:orange", "tab:green", "tab:red"],
        execute_steps=args.execute_steps,
    )
    _plot_config(
        axes[1].tolist(),
        chunks_rtc,
        label_prefix="RTC",
        color_cycle=["tab:purple", "tab:brown", "tab:pink", "tab:olive"],
        execute_steps=args.execute_steps,
    )

    for col, name in enumerate(joint_names[:D]):
        axes[0, col].set_title(name)
        axes[1, col].set_xlabel("time step (global)")
    axes[0, 0].set_ylabel("action (no RTC)")
    axes[1, 0].set_ylabel("action (RTC)")

    # One legend per row.
    axes[0, -1].legend(loc="best", fontsize=8)
    axes[1, -1].legend(loc="best", fontsize=8)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=130)
    print(f"wrote: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
