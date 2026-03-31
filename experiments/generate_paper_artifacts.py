from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, List, Mapping

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, colors

from incoherence.experiments import (
    EnvInstance,
    instances_from_records,
    load_effective_horizon_config,
    load_effective_horizon_dataset,
    load_misalignment_config,
)
from incoherence.correlation_misalignment_incoherence import run_alignment_vs_incoherence_suite
from incoherence.experiment_horizon_monte_carlo import (
    collect_effective_horizon_data,
    save_effective_horizon_data,
    EffectiveHorizonRecord,
)
from incoherence.mdp import create_random_mdp


def _records_to_dicts(records: Iterable[EffectiveHorizonRecord]) -> List[dict]:
    return [record.as_dict() for record in records]


def _write_effective_horizon_summary(path: Path, records: List[dict]) -> None:
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
                    record["H_hat"],
                    metrics["kappa_uniform"],
                    metrics["kappa_piG"],
                    metrics["J_uniform"],
                    metrics["J_piG"],
                    metrics["J_star"],
                ]
            )


def _plot_effective_horizon(
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
    cmap = plt.get_cmap("Blues")
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
        cbar.set_label("State-action slots")
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


def _plot_misalignment(results: List[dict], path: Path) -> None:
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
        n = len(instances)
        if n > 3:
            z = np.arctanh(corr)
            se = 1 / np.sqrt(n - 3)
            delta = 1.96 * se
            lower_ci.append(float(np.tanh(z - delta)))
            upper_ci.append(float(np.tanh(z + delta)))
        else:
            lower_ci.append(corr)
            upper_ci.append(corr)
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


def main() -> None:
    horizon_config = load_effective_horizon_config(Path("configs/effective_horizon.yaml"))

    # Effective horizon study
    size_metric = {}
    for spec in horizon_config.env_specs:
        mdp = create_random_mdp(spec.num_actions, spec.horizon, deterministic_transitions=spec.deterministic, seed=0)
        total_slots = sum(len(actions) for actions in mdp.actions.values())
        size_metric[spec.name] = float(total_slots)

    dataset_path = horizon_config.results_dir / "effective_horizon.json"
    print(dataset_path)
    if dataset_path.exists():
        horizon_dicts = json.loads(dataset_path.read_text())
    else:
        horizon_records = collect_effective_horizon_data(
            env_specs=horizon_config.env_specs,
            settings=horizon_config.settings,
            temperature=horizon_config.temperature,
            global_seed=horizon_config.global_seed,
            verbose=False,
            max_workers=1,
        )
        horizon_dicts = _records_to_dicts(horizon_records)
        save_effective_horizon_data(dataset_path, horizon_records)
    _write_effective_horizon_summary(horizon_config.results_dir / "effective_horizon_summary.csv", horizon_dicts)
    _plot_effective_horizon(
        horizon_dicts,
        size_metric,
        horizon_config.results_dir / "effective_horizon_vs_incoherence_uniform.png",
        horizon_config.results_dir / "effective_horizon_vs_incoherence_pi1.png",
    )

    # Misalignment vs incoherence
    misalignment_config = load_misalignment_config(Path("configs/misalignment.yaml"))
    if misalignment_config.dataset_path != dataset_path and not misalignment_config.dataset_path.exists():
        misalignment_config.dataset_path.parent.mkdir(parents=True, exist_ok=True)
        misalignment_config.dataset_path.write_text(json.dumps(horizon_dicts, indent=2))

    if misalignment_config.dataset_path == dataset_path:
        dataset_records = horizon_dicts
    else:
        dataset_records = load_effective_horizon_dataset(misalignment_config.dataset_path)

    misalignment_path = misalignment_config.results_dir / "misalignment_vs_incoherence.json"
    if misalignment_path.exists():
        misalignment_results = json.loads(misalignment_path.read_text())
    else:
        instances: List[EnvInstance] = instances_from_records(dataset_records, misalignment_config.env_specs)
        misalignment_results = run_alignment_vs_incoherence_suite(
            instances,
            temperatures=misalignment_config.temperatures,
        )
        misalignment_path.write_text(json.dumps(misalignment_results, indent=2))
    _plot_misalignment(misalignment_results, misalignment_config.results_dir / "misalignment_vs_incoherence.png")

    # Table for misalignment summary
    with (misalignment_config.results_dir / "misalignment_summary.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["temperature", "correlation", "num_instances"])
        for entry in misalignment_results:
            writer.writerow([
                entry["temperature"],
                "" if entry["correlation"] is None else entry["correlation"],
                entry["num_instances"],
            ])

    print("Generated paper artifacts under", horizon_config.results_dir)
    if misalignment_config.results_dir != horizon_config.results_dir:
        print("Misalignment artifacts saved under", misalignment_config.results_dir)


if __name__ == "__main__":
    main()
