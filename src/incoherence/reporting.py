"""Utilities for turning experiment datasets into publication-ready artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Mapping

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, colors
from numpy.random import default_rng


def records_to_dicts(records: Iterable) -> List[dict]:
    return [record.as_dict() for record in records]


def write_effective_horizon_summary(path: Path, records: List[dict]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "spec_name",
                "seed",
                "horizon",
                "num_actions",
                "H_hat",
                "kappa_uniform",
                "kappa_piG",
                "J_uniform",
                "J_piG",
                "J_star",
            ]
        )
        for record in records:
            metrics = record["metrics"]
            writer.writerow(
                [
                    record["spec_name"],
                    record["seed"],
                    record["horizon"],
                    record["num_actions"],
                    "" if record["H_hat"] is None else record["H_hat"],
                    metrics["kappa_uniform"],
                    metrics["kappa_piG"],
                    metrics["J_uniform"],
                    metrics["J_piG"],
                    metrics["J_star"],
                ]
            )


def _truncated_blues() -> colors.Colormap:
    base_cmap = plt.get_cmap("Blues")
    return colors.LinearSegmentedColormap.from_list(
        "BluesTruncated",
        base_cmap(np.linspace(0.2, 1.0, 256)),
    )


def plot_effective_horizon(
    records: List[dict],
    size_metric: Mapping[str, float],
    path_uniform: Path,
    path_cond: Path,
) -> None:
    xs = []
    ys_uniform = []
    ys_cond = []
    sizes = []
    for record in records:
        H_hat = record["H_hat"]
        if H_hat is None:
            continue
        xs.append(float(H_hat))
        ys_uniform.append(float(record["metrics"]["kappa_uniform"]))
        ys_cond.append(float(record["metrics"]["kappa_piG"]))
        sizes.append(size_metric.get(record["spec_name"], 0.0))
    if not xs:
        return

    xs = np.array(xs)
    ys_uniform = np.array(ys_uniform)
    ys_cond = np.array(ys_cond)
    sizes = np.array(sizes)
    norm = colors.Normalize(vmin=sizes.min(), vmax=sizes.max())
    cmap = _truncated_blues()
    color_values = cmap(norm(sizes))

    def _scatter(ax, ys, ylabel, title, out_path):
        ax.scatter(xs, ys, c=color_values, alpha=0.85, edgecolor="none")
        slope, intercept = np.polyfit(xs, ys, 1)
        x_line = np.linspace(xs.min(), xs.max(), 200)
        ax.plot(x_line, slope * x_line + intercept, color="black", linestyle="--", label=f"slope={slope:.3f}")
        ax.set_xlabel("Estimated effective horizon $\\hat{H}$")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize="small")
        mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array(sizes)
        cbar = plt.colorbar(mappable, ax=ax)
        cbar.set_label("MDP size")
        fig = ax.get_figure()
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)

    fig_u, ax_u = plt.subplots(figsize=(6, 4))
    _scatter(
        ax_u,
        ys_uniform,
        r"Incoherence $\kappa_\delta(\pi_0)$",
        r"$\pi_0$ (uniform prior)",
        path_uniform,
    )

    fig_c, ax_c = plt.subplots(figsize=(6, 4))
    _scatter(
        ax_c,
        ys_cond,
        r"Incoherence $\kappa_\delta(\pi_1)$",
        r"$\pi_1 = \mathcal{G}(\pi_0)$",
        path_cond,
    )


def plot_effective_horizon_complexity(
    records: List[dict],
    complexity: Mapping[str, Mapping[str, float]],
    output_path: Path,
) -> None:
    xs = []
    states = []
    actions = []
    slots = []
    for record in records:
        H_hat = record["H_hat"]
        if H_hat is None:
            continue
        comp = complexity.get(record["spec_name"], {})
        xs.append(float(H_hat))
        states.append(comp.get("states", 0.0))
        actions.append(comp.get("actions", 0.0))
        slots.append(comp.get("slots", 0.0))
    if not xs:
        return
    xs = np.array(xs)
    states = np.array(states)
    actions = np.array(actions)
    slots = np.array(slots)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True)
    for ax, ys, title in zip(
        axes,
        (states, actions, slots),
        ("Number of states", "Number of actions", "State-action slots"),
    ):
        ax.scatter(xs, ys, color="C0", alpha=0.7, edgecolor="none")
        slope, intercept = np.polyfit(xs, ys, 1)
        x_line = np.linspace(xs.min(), xs.max(), 200)
        ax.plot(x_line, slope * x_line + intercept, color="black", linestyle="--", linewidth=1.0)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    axes[0].set_xlabel("Estimated effective horizon $\\hat{H}$")
    axes[0].set_ylabel("Count")
    for ax in axes[1:]:
        ax.set_xlabel("Estimated effective horizon $\\hat{H}$")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _bootstrap_ci(xs: np.ndarray, ys: np.ndarray, *, num_samples: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = default_rng(seed)
    n = len(xs)
    if n <= 3 or np.std(xs) == 0 or np.std(ys) == 0:
        corr = float(np.corrcoef(xs, ys)[0, 1]) if n > 1 else float("nan")
        return corr, corr
    boot = []
    for _ in range(num_samples):
        idx = rng.integers(0, n, n)
        sample_x = xs[idx]
        sample_y = ys[idx]
        if np.std(sample_x) == 0 or np.std(sample_y) == 0:
            continue
        boot.append(np.corrcoef(sample_x, sample_y)[0, 1])
    if not boot:
        corr = float(np.corrcoef(xs, ys)[0, 1])
        return corr, corr
    lower, upper = np.percentile(boot, [2.5, 97.5])
    return float(lower), float(upper)


def plot_misalignment(results: List[dict], path: Path) -> None:
    temps = []
    corrs = []
    lower_ci = []
    upper_ci = []
    for entry in results:
        corr = entry["correlation"]
        if corr is None:
            continue
        temp = entry["temperature"]
        temps.append(temp)
        corrs.append(corr)
        instances = entry.get("instances", [])
        misalign = np.array([inst["misalignment"] for inst in instances], dtype=float)
        incoh = np.array([inst["incoherence"] for inst in instances], dtype=float)
        lo, hi = _bootstrap_ci(misalign, incoh)
        lower_ci.append(lo)
        upper_ci.append(hi)
    if not temps:
        return
    temps = np.array(temps)
    corrs = np.array(corrs)
    lower_ci = np.array(lower_ci)
    upper_ci = np.array(upper_ci)
    order = np.argsort(temps)
    temps = temps[order]
    corrs = corrs[order]
    lower_ci = lower_ci[order]
    upper_ci = upper_ci[order]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(temps, corrs, marker="o", color="C0", label="Correlation")
    ax.fill_between(temps, lower_ci, upper_ci, color="C0", alpha=0.2, label="95% CI")
    ax.set_xscale("log")
    ax.set_xlabel("Temperature")
    ax.set_ylabel("Correlation (misalignment vs. incoherence)")
    ax.set_title("Policy misalignment vs. incoherence")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize="small")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
