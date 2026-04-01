from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np

from incoherence.distributions import bernoulli
from incoherence.experiments import (
    StrongReturnConfig,
    generate_instances,
    load_strong_return_config,
)
from incoherence.mdp import compute_J, create_random_mdp, make_uniform_policy
from incoherence.training import iterate_G

MIN_REWARD_PROB = 1e-3

def _clip_reward_support(mdp) -> None:
    for state in mdp.states:
        actions = mdp.rewards[state]
        if not actions:
            continue
        for action, reward_dist in actions.items():
            prob = float(reward_dist.dist[1])
            prob = float(np.clip(prob, MIN_REWARD_PROB, 1.0 - MIN_REWARD_PROB))
            mdp.rewards[state][action] = bernoulli(prob)


def _return_history(mdp, max_iterations: int) -> List[float]:
    policies = iterate_G(mdp, make_uniform_policy(mdp), max_iterations)
    return [float(compute_J(mdp, policy)) for policy in policies]


def _collect_trajectories(specs: Iterable, global_seed: int, max_iterations: int) -> List[dict]:
    trajectories: List[dict] = []
    instances = generate_instances(specs, global_seed)
    for inst in instances:
        mdp = create_random_mdp(
            inst.spec.num_actions,
            inst.spec.horizon,
            deterministic_transitions=inst.spec.deterministic,
            seed=inst.seed,
        )
        _clip_reward_support(mdp)
        trajectories.append(
            {
                "spec_name": inst.spec.name,
                "deterministic": inst.spec.deterministic,
                "seed": inst.seed,
                "return_history": _return_history(mdp, max_iterations),
            }
        )
    return trajectories


def _split_records(records: Iterable[dict]) -> Dict[str, List[np.ndarray]]:
    grouped: Dict[str, List[np.ndarray]] = {"deterministic": [], "stochastic": []}
    for record in records:
        key = "deterministic" if record.get("deterministic") else "stochastic"
        grouped[key].append(np.asarray(record["return_history"], dtype=float))
    return {k: v for k, v in grouped.items() if v}


def _plot_family(trajectories: List[np.ndarray], title: str, cmap_name: str, output_path: Path) -> None:
    if not trajectories:
        return
    steps = np.arange(trajectories[0].shape[0])
    cmap = plt.get_cmap(cmap_name)
    shades = np.linspace(0.35, 0.9, len(trajectories))
    fig, ax = plt.subplots(figsize=(6, 4))
    for shade, series in zip(shades, trajectories):
        ax.plot(
            steps,
            series,
            marker="o",
            linewidth=2,
            color=cmap(shade),
            alpha=0.95,
        )
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"Return $J(\pi_k)$")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def run(config: StrongReturnConfig, output_path: Path) -> Dict[str, Path]:
    deterministic_records = _collect_trajectories(
        config.deterministic_specs, config.global_seed, config.max_iterations
    )
    stochastic_records = _collect_trajectories(
        config.stochastic_specs, config.global_seed, config.max_iterations
    )
    all_records = deterministic_records + stochastic_records
    if not all_records:
        raise SystemExit("No trajectories collected; check configuration.")

    dataset_path = config.results_dir / "strong_return_improvement.json"
    dataset_path.write_text(json.dumps(all_records, indent=2))

    trajectories = _split_records(all_records)
    paths: Dict[str, Path] = {}
    if "deterministic" in trajectories:
        det_path = output_path.with_name("strong_return_improvement_deterministic.png")
        _plot_family(
            trajectories["deterministic"],
            "Strong return improvement (deterministic)",
            "Blues",
            det_path,
        )
        paths["deterministic"] = det_path
    if "stochastic" in trajectories:
        sto_path = output_path.with_name("strong_return_improvement_stochastic.png")
        _plot_family(
            trajectories["stochastic"],
            "Strong return improvement (stochastic)",
            "Oranges",
            sto_path,
        )
        paths["stochastic"] = sto_path
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strong return improvement visualisation")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/strong_return_improvement.yaml"),
        help="Path to the strong return improvement config.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/strong_return_improvement_deterministic.png"),
        help="Base output path for deterministic plot (stochastic plot shares directory).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_strong_return_config(args.config)
    paths = run(config, args.output)
    for kind, path in paths.items():
        print(f"Saved {kind} strong return improvement figure to {path}")


if __name__ == "__main__":
    main()
