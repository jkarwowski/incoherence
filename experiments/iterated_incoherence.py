from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from numpy.random import SeedSequence, default_rng

from incoherence.distributions import bernoulli
from incoherence.experiments import load_iterated_incoherence_config
from incoherence.mdp import create_random_mdp, make_uniform_policy, compute_J
from incoherence.policy import boltzmann_incoherence_causal
from incoherence.training import iterate_G


def _kappa(mdp, policy, temperature: float) -> float:
    return float(boltzmann_incoherence_causal(mdp, policy, temperature))


def _collect_policies(mdp, max_iterations: int) -> List:
    initial = make_uniform_policy(mdp)
    return list(iterate_G(mdp, initial, max_iterations))[1:]


def run(config_path: Path) -> Path:
    config = load_iterated_incoherence_config(config_path)
    results = []
    for spec_index, spec in enumerate(config.env_specs):
        rng = default_rng(SeedSequence([config.global_seed, spec_index]))
        seen_seeds: set[int] = set()
        attempts = 0
        while len([r for r in results if r["spec_name"] == spec.name]) < spec.num_instances and attempts < spec.search_limit:
            seed = int(rng.integers(0, 2**32, dtype=np.uint32))
            attempts += 1
            if seed in seen_seeds:
                continue
            seen_seeds.add(seed)
            mdp = create_random_mdp(
                spec.num_actions,
                spec.horizon,
                deterministic_transitions=spec.deterministic,
                seed=seed,
            )
            if spec.deterministic:
                for state in mdp.states:
                    actions = mdp.rewards[state]
                    if not actions:
                        continue
                    best_action = max(actions.keys(), key=lambda a: actions[a].dist[1])
                    for action, reward_dist in actions.items():
                        deterministic_prob = 1.0 if action == best_action else 0.0
                        mdp.rewards[state][action] = bernoulli(deterministic_prob)
            policies = _collect_policies(mdp, config.max_iterations)
            kappa_history = [_kappa(mdp, policy, config.temperature) for policy in policies]
            returns_history = [float(compute_J(mdp, policy)) for policy in policies]
            results.append(
                {
                    "spec_name": spec.name,
                    "deterministic": spec.deterministic,
                    "seed": seed,
                    "horizon": spec.horizon,
                    "num_actions": spec.num_actions,
                    "kappa_history": kappa_history,
                    "return_history": returns_history,
                }
            )
        if len([r for r in results if r["spec_name"] == spec.name]) < spec.num_instances:
            raise RuntimeError(f"Could not gather {spec.num_instances} instances for spec {spec.name} within search limit")

    output_path = config.results_dir / "iterated_incoherence.json"
    output_path.write_text(json.dumps(results, indent=2))

    _plot_histories(results, config.results_dir, config.max_iterations)
    _plot_corollaries(results, config)

    return output_path


def _plot_histories(results: List[dict], results_dir: Path, max_iterations: int) -> None:
    det_histories = [r for r in results if r["deterministic"]]
    sto_histories = [r for r in results if not r["deterministic"]]

    def plot_group(group: List[dict], title: str, path: Path) -> None:
        if not group:
            return
        iterations = np.arange(max_iterations + 1)
        values = np.array([np.array(r["kappa_history"], dtype=float) for r in group])
        mean = values.mean(axis=0)
        std = values.std(axis=0)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(iterations, mean, marker="o", color="C0", label="mean")
        ax.fill_between(iterations, mean - std, mean + std, color="C0", alpha=0.2, label="±1 std")
        ax.set_xlabel("Iteration")
        ax.set_ylabel(r"Incoherence $\kappa_\delta$")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize="small")
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)

    plot_group(
        det_histories,
        "Deterministic environments",
        results_dir / "iterated_incoherence_deterministic.png",
    )
    plot_group(
        sto_histories,
        "Stochastic environments",
        results_dir / "iterated_incoherence_stochastic.png",
    )


def _plot_corollaries(results: List[dict], config) -> None:
    deterministic_records = [r for r in results if r["deterministic"]]
    if not deterministic_records:
        return
    reference = deterministic_records[0]
    spec = reference["spec_name"]
    seed = reference["seed"]
    mdp = create_random_mdp(
        reference["num_actions"],
        reference["horizon"],
        deterministic_transitions=True,
        seed=seed,
    )
    for state in mdp.states:
        for action, reward_dist in mdp.rewards[state].items():
            prob = reward_dist.dist[1]
            mdp.rewards[state][action] = bernoulli(1.0 if prob >= 0.5 else 0.0)

    policies = _collect_policies(mdp, config.max_iterations)

    deltas = config.deltas or (config.temperature,)
    iterations = np.arange(config.max_iterations + 1)

    fig_delta, ax_delta = plt.subplots(figsize=(6, 4))
    for delta in deltas:
        kappas = [_kappa(mdp, policy, delta) for policy in policies]
        ax_delta.plot(iterations, kappas, marker="o", label=f"δ={delta}")
    ax_delta.set_xlabel("Iteration")
    ax_delta.set_ylabel(r"Incoherence $\kappa_\delta$")
    ax_delta.set_title("Corollary 5.11: κ→0 as iterations increase")
    ax_delta.grid(True, alpha=0.3)
    ax_delta.legend(loc="best", fontsize="small")
    fig_delta.tight_layout()
    fig_delta.savefig(config.results_dir / "corollary_5_11.png")
    plt.close(fig_delta)

    returns = [float(compute_J(mdp, policy)) for policy in policies]
    delta_returns = np.diff(returns)
    scaled = np.arange(1, len(returns)) * delta_returns

    fig_return, ax_return = plt.subplots(figsize=(6, 4))
    ax_return.plot(np.arange(len(returns)), returns, marker="o", label="Return")
    ax_right = ax_return.twinx()
    ax_right.plot(np.arange(1, len(returns)), delta_returns, marker="s", color="C1", label="ΔJ")
    ax_right.plot(np.arange(1, len(returns)), scaled, marker="^", color="C2", label="k·ΔJ")
    ax_return.set_xlabel("Iteration")
    ax_return.set_ylabel("Return J")
    ax_right.set_ylabel("Differences")
    ax_return.set_title("Corollary 5.10: return improvement rate")
    ax_return.grid(True, alpha=0.3)
    ax_return.legend(loc="upper left", fontsize="small")
    ax_right.legend(loc="upper right", fontsize="small")
    fig_return.tight_layout()
    fig_return.savefig(config.results_dir / "corollary_5_10.png")
    plt.close(fig_return)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Iterated incoherence experiment")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/iterated_incoherence.yaml"),
        help="Path to the iterated incoherence config",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = run(args.config)
    print(f"Saved iterated incoherence data to {output_path}")


if __name__ == "__main__":
    main()
