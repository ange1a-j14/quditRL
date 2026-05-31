"""Training metrics logging and curve plotting.

RunLogger writes a CSV to output/<run_name>.csv as training progresses so
partial runs are readable. plot_run reads that CSV and saves a PNG alongside it.

Usage (standalone):
    from metrics import plot_run
    plot_run("output/ppo_actor-critic_d4_haar_h256_seed0.csv")
"""

from __future__ import annotations

import csv
import os
from typing import Optional

import numpy as np


OUTPUT_DIR = "output"

_FIELDS = ["iter", "timestep", "episodes", "train_fidelity", "train_pulses", "eval_fidelity", "eval_pulses"]


class RunLogger:
    """Append-mode CSV logger for one training run.

    Parameters
    ----------
    run_name:
        Used to derive the output path: ``output/<run_name>.csv``.
    output_dir:
        Directory to write the CSV into (created if absent).
    """

    def __init__(self, run_name: str, output_dir: str = OUTPUT_DIR) -> None:
        os.makedirs(output_dir, exist_ok=True)
        self.path = os.path.join(output_dir, f"{run_name}.csv")
        self._file = open(self.path, "w", newline="", encoding="utf-8", buffering=1)  # line-buffered
        self._writer = csv.DictWriter(self._file, fieldnames=_FIELDS)
        self._writer.writeheader()
        print(f"Logging metrics to {self.path}")

    def log(
        self,
        *,
        iter: int,  # noqa: A002
        timestep: int,
        episodes: int,
        train_fidelity: float,
        train_pulses: float,
        eval_fidelity: Optional[float] = None,
        eval_pulses: Optional[float] = None,
    ) -> None:
        self._writer.writerow({
            "iter": iter,
            "timestep": timestep,
            "episodes": episodes,
            "train_fidelity": f"{train_fidelity:.6f}",
            "train_pulses": f"{train_pulses:.4f}",
            "eval_fidelity": f"{eval_fidelity:.6f}" if eval_fidelity is not None else "",
            "eval_pulses": f"{eval_pulses:.4f}" if eval_pulses is not None else "",
        })

    def close(self) -> None:
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def plot_run(csv_path: str, save: bool = True) -> str:
    """Read a run CSV and save a training-curve PNG next to it.

    Parameters
    ----------
    csv_path:
        Path to a CSV written by RunLogger.
    save:
        If True (default), write the PNG and return its path.

    Returns
    -------
    Path to the saved PNG.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError as e:
        raise ImportError("matplotlib is required for plotting: pip install matplotlib") from e

    iters, timesteps = [], []
    train_fid, train_pulses = [], []
    eval_ts, eval_fid, eval_pulses = [], [], []

    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            iters.append(int(row["iter"]))
            timesteps.append(int(row["timestep"]))
            train_fid.append(float(row["train_fidelity"]) if row["train_fidelity"] else float("nan"))
            train_pulses.append(float(row["train_pulses"]) if row["train_pulses"] else float("nan"))
            if row["eval_fidelity"]:
                eval_ts.append(int(row["timestep"]))
                eval_fid.append(float(row["eval_fidelity"]))
                eval_pulses.append(float(row["eval_pulses"]))

    # Convert fidelity -> infidelity (1 - F) for a log-scale "gate error" view.
    # Floor at a small positive value so perfect fidelity (F=1) is plottable on a log axis.
    _floor = 1e-9
    train_infid = [max(1.0 - f, _floor) if not np.isnan(f) else float("nan") for f in train_fid]
    eval_infid = [max(1.0 - f, _floor) for f in eval_fid]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    fig.suptitle(os.path.splitext(os.path.basename(csv_path))[0], fontsize=11)

    # Infidelity panel (log scale): 1 - F, lower is better
    ax1.semilogy(timesteps, train_infid, color="steelblue", linewidth=1, alpha=0.6, label="train")
    ax1.semilogy(timesteps, _smooth(train_infid), color="steelblue", linewidth=2, label="train (smooth)")
    if eval_infid:
        ax1.semilogy(eval_ts, eval_infid, "o--", color="tomato", linewidth=1.5, label="eval")
    ax1.set_ylabel("infidelity (1 − F)")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3, which="both")

    # Pulse count panel
    ax2.plot(timesteps, train_pulses, color="mediumseagreen", linewidth=1, alpha=0.6, label="train")
    ax2.plot(timesteps, _smooth(train_pulses), color="mediumseagreen", linewidth=2, label="train (smooth)")
    if eval_pulses:
        ax2.plot(eval_ts, eval_pulses, "o--", color="darkorange", linewidth=1.5, label="eval")
    ax2.set_ylabel("pulses per episode")
    ax2.set_xlabel("timestep")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{int(x):,}"))

    plt.tight_layout()

    if save:
        out = os.path.splitext(csv_path)[0] + ".png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved plot to {out}")
        return out

    plt.show()
    plt.close(fig)
    return csv_path


def _smooth(values: list[float], window: int = 20) -> list[float]:
    """Simple uniform moving average; handles NaNs by propagating them."""
    out = []
    for i, _ in enumerate(values):
        chunk = [x for x in values[max(0, i - window + 1):i + 1] if not np.isnan(x)]
        out.append(float(np.mean(chunk)) if chunk else float("nan"))
    return out
