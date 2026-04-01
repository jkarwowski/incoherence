from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from numpy.random import SeedSequence, default_rng

from incoherence.distributions import bernoulli
from incoherence.experiments import load_iterated_incoherence_config
from incoherence.mdp import create_random_mdp, make_uniform_policy, compute_J
from incoherence.policy import boltzmann_incoherence_causal
from incoherence.training import iterate_G


def _kappa(mdp, policy, temperature: float) -> float:
    return float(boltzmann_incoherence_causal(mdp, policy, temperature))


MIN_REWARD_PROB = 1e-3


def collect_policies(mdp, max_iterations: int) -> List:
    initial = make_uniform_policy(mdp)
    return list(iterate_G(mdp, initial, max_iterations))


def clip_reward_support(mdp) -> None:
    """Ensure reward probabilities avoid exact 0/1 to keep KL finite."""
    for state in mdp.states:
        actions = mdp.rewards[state]
        if not actions:
            continue
        for action, reward_dist in actions.items():
            prob = float(reward_dist.dist[1])
            prob = float(np.clip(prob, MIN_REWARD_PROB, 1.0 - MIN_REWARD_PROB))
            mdp.rewards[state][action] = bernoulli(prob)


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
                clip_reward_support(mdp)
            policies = collect_policies(mdp, config.max_iterations)
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
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize="small")
        fig.tight_layout()
        fig.savefig(path, dpi=300)
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
    def _plot_temperature_series(
        mdp,
        policies,
        deltas,
        iterations,
        path: Path,
        cmap_name: str,
        shade_transform,
    ) -> None:
        if not policies:
            return
        cmap = plt.get_cmap(cmap_name)
        vmin = float(min(deltas))
        vmax = float(max(deltas))
        if math.isclose(vmin, vmax):
            vmax = vmin + 1e-6
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        fig, ax = plt.subplots(figsize=(6, 4))
        for delta in sorted(deltas):
            kappas = [_kappa(mdp, policy, delta) for policy in policies]
            shade = float(shade_transform(norm(delta)))
            ax.plot(
                iterations,
                kappas,
                marker="o",
                label=f"δ={delta:g}",
                color=cmap(shade),
                linewidth=2,
            )
        ax.set_xlabel("Iteration")
        ax.set_ylabel(r"Incoherence $\kappa_\delta$")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize="small", title="Temperature")
        fig.tight_layout()
        fig.savefig(path, dpi=300)
        plt.close(fig)

    deterministic_records = [r for r in results if r["deterministic"]]
    stochastic_records = [r for r in results if not r["deterministic"]]
    deltas = config.deltas or (config.temperature,)
    iterations = np.arange(config.max_iterations + 1)

    if deterministic_records:
        reference = deterministic_records[0]
        mdp = create_random_mdp(
            reference["num_actions"],
            reference["horizon"],
            deterministic_transitions=True,
            seed=reference["seed"],
        )
        clip_reward_support(mdp)
        policies = collect_policies(mdp, config.max_iterations)
        _plot_temperature_series(
            mdp,
            policies,
            deltas,
            iterations,
            config.results_dir / "corollary_5_11.png",
            "Blues",
            lambda value: 1.0 - 0.5 * value,
        )

        returns = [float(compute_J(mdp, policy)) for policy in policies]
        delta_returns = np.diff(returns)
        scaled = np.arange(1, len(returns)) * delta_returns

        fig_return, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(6, 6), sharex=True)

        iterations_full = np.arange(len(returns))
        ax_top.plot(iterations_full, returns, marker="o", color="C0", linewidth=2)
        ax_top.set_ylabel("Return $J$")
        ax_top.grid(True, alpha=0.3)

        steps = np.arange(1, len(returns))
        ax_bottom.plot(steps, delta_returns, marker="s", color="C1", linewidth=2, label=r"ΔJ$_k$")
        ax_bottom.plot(
            steps,
            scaled,
            marker="^",
            color="C2",
            linewidth=2,
            linestyle="--",
            label=r"$k·ΔJ_k$",
        )
        ax_bottom.axhline(0.0, color="black", linestyle=":", linewidth=1.0, alpha=0.4)
        ax_bottom.set_xlabel("Iteration $k$")
        ax_bottom.set_ylabel("Improvement")
        ax_bottom.grid(True, alpha=0.3)
        ax_bottom.legend(loc="best", fontsize="small")

        fig_return.tight_layout()
        fig_return.savefig(config.results_dir / "corollary_5_10.png", dpi=300)
        plt.close(fig_return)

    if stochastic_records:
        reference = stochastic_records[0]
        mdp = create_random_mdp(
            reference["num_actions"],
            reference["horizon"],
            deterministic_transitions=False,
            seed=reference["seed"],
        )
        clip_reward_support(mdp)
        policies = collect_policies(mdp, config.max_iterations)
        _plot_temperature_series(
            mdp,
            policies,
            deltas,
            iterations,
            config.results_dir / "corollary_5_11_stochastic.png",
            "Oranges",
            lambda value: 0.2 + 0.8 * value,
        )


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
