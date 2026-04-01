"""Experiment utilities and configuration helpers.

This module consolidates the legacy helpers that previously lived under
``incoherence.experiments.*`` into a single, flat namespace.  It exposes the
configuration dataclasses used by the experiment entry points as well as the
historical interactive helpers relied upon by the test-suite.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
import yaml
from matplotlib import pyplot as plt
from numpy.random import SeedSequence, default_rng

from .mdp import (
    MDP,
    Policy,
    compute_J,
    compute_marginals,
    compute_prob_over_trajectories,
    create_random_mdp,
    make_uniform_policy,
    posterior_cond_R,
    sample_trajectory,
    stochastify,
)
from .metrics import compute_J_causal_entropy, compute_J_entropy
from .policy import (
    boltzmann_incoherence_causal,
    boltzmann_incoherence_flat,
    compute_Vs_Qs,
    converge_boltzmann_coherence,
    iterated_boltzmann_coherence_causal,
    iterated_boltzmann_coherence_causal_belousov,
    iterated_boltzmann_coherence_flat,
    policy_iteration,
    print_Q,
    print_V,
)
from .scenarios import create_counterexample, create_two_cards_game
from .training import fold_posterior_into_reward, increase_temp, retrain_agent
from .utils import print_js, print_policy, print_trajectory, print_trajectory_prob


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"


@dataclass(frozen=True)
class EnvSpec:
    """Specification of an environment family used in experiments."""

    name: str
    horizon: int
    num_actions: int
    num_instances: int
    deterministic: bool = True
    search_limit: int = 2000


@dataclass(frozen=True)
class EffectiveHorizonSettings:
    ks: Sequence[int]
    success_threshold: float
    m_max: int
    n_runs: int


@dataclass(frozen=True)
class EffectiveHorizonConfig:
    env_specs: Sequence[EnvSpec]
    settings: EffectiveHorizonSettings
    temperature: float
    global_seed: int
    results_dir: Path


@dataclass(frozen=True)
class MisalignmentConfig:
    env_specs: Sequence[EnvSpec]
    temperatures: Sequence[float]
    global_seed: int
    results_dir: Path
    dataset_path: Path


@dataclass(frozen=True)
class EnvInstance:
    spec: EnvSpec
    seed: int


@dataclass(frozen=True)
class IteratedIncoherenceConfig:
    env_specs: Sequence[EnvSpec]
    max_iterations: int
    temperature: float
    global_seed: int
    results_dir: Path
    deltas: Sequence[float] | None = None


@dataclass(frozen=True)
class StrongReturnConfig:
    deterministic_specs: Sequence[EnvSpec]
    stochastic_specs: Sequence[EnvSpec]
    max_iterations: int
    temperature: float
    global_seed: int
    results_dir: Path


def _parse_env_specs(raw_specs: Iterable[Mapping]) -> List[EnvSpec]:
    specs: List[EnvSpec] = []
    for raw in raw_specs:
        specs.append(
            EnvSpec(
                name=raw["name"],
                horizon=int(raw["horizon"]),
                num_actions=int(raw["num_actions"]),
                num_instances=int(raw["num_instances"]),
                deterministic=bool(raw.get("deterministic", True)),
                search_limit=int(raw.get("search_limit", 2000)),
            )
        )
    return specs


def _resolve_path(value: str | None, default: Path) -> Path:
    if value is None:
        return default
    path = Path(value)
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return path


def load_effective_horizon_config(path: Path) -> EffectiveHorizonConfig:
    data = yaml.safe_load(path.read_text())
    env_specs = _parse_env_specs(data["env_specs"])
    settings_raw = data["effective_horizon"]
    settings = EffectiveHorizonSettings(
        ks=tuple(int(x) for x in settings_raw.get("ks", [1, 2, 3, 4])),
        success_threshold=float(settings_raw.get("success_threshold", 0.5)),
        m_max=int(settings_raw.get("m_max", 256)),
        n_runs=int(settings_raw.get("n_runs", 500)),
    )
    temperature = float(data.get("temperature", 1.0))
    global_seed = int(data.get("global_seed", 0))
    results_dir = _resolve_path(data.get("results_dir"), DEFAULT_RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)
    return EffectiveHorizonConfig(
        env_specs=env_specs,
        settings=settings,
        temperature=temperature,
        global_seed=global_seed,
        results_dir=results_dir,
    )


def load_misalignment_config(path: Path) -> MisalignmentConfig:
    data = yaml.safe_load(path.read_text())
    env_specs = _parse_env_specs(data["env_specs"])
    temperatures = tuple(float(x) for x in data.get("temperatures", []))
    global_seed = int(data.get("global_seed", 0))
    results_dir = _resolve_path(data.get("results_dir"), DEFAULT_RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = _resolve_path(
        data.get("dataset"), results_dir / "misalignment_instances.json"
    )
    return MisalignmentConfig(
        env_specs=env_specs,
        temperatures=temperatures,
        global_seed=global_seed,
        results_dir=results_dir,
        dataset_path=dataset_path,
    )


def load_effective_horizon_dataset(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"Effective horizon dataset not found at {path}. Run the horizon experiment first."
        )
    return json.loads(path.read_text())


def instances_from_records(records: Iterable[Mapping], env_specs: Sequence[EnvSpec]) -> List[EnvInstance]:
    spec_by_name = {spec.name: spec for spec in env_specs}
    instances: List[EnvInstance] = []
    for record in records:
        spec_name = record["spec_name"]
        if spec_name not in spec_by_name:
            raise KeyError(f"Unknown env spec '{spec_name}' in dataset; update configuration.")
        instances.append(EnvInstance(spec=spec_by_name[spec_name], seed=int(record["seed"])))
    return instances


def generate_instances(env_specs: Sequence[EnvSpec], global_seed: int) -> List[EnvInstance]:
    instances: List[EnvInstance] = []
    for index, spec in enumerate(env_specs):
        rng = default_rng(SeedSequence([global_seed, index]))
        seeds: set[int] = set()
        attempts = 0
        while len(seeds) < spec.num_instances:
            if attempts >= spec.search_limit:
                raise RuntimeError(
                    f"Could not sample {spec.num_instances} instances for {spec.name} within search limit"
                )
            seed = int(rng.integers(0, 2**32, dtype=np.uint32))
            attempts += 1
            if seed in seeds:
                continue
            seeds.add(seed)
            instances.append(EnvInstance(spec=spec, seed=seed))
    return instances


def records_from_instances(instances: Sequence[EnvInstance]) -> List[dict]:
    return [
        {
            "spec_name": inst.spec.name,
            "seed": int(inst.seed),
        }
        for inst in instances
    ]


def load_iterated_incoherence_config(path: Path) -> IteratedIncoherenceConfig:
    data = yaml.safe_load(path.read_text())
    env_specs = _parse_env_specs(data["env_specs"])
    max_iterations = int(data.get("max_iterations", 5))
    temperature = float(data.get("temperature", 1.0))
    global_seed = int(data.get("global_seed", 0))
    results_dir = _resolve_path(data.get("results_dir"), DEFAULT_RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)
    deltas = data.get("deltas")
    if deltas is not None:
        deltas = tuple(float(x) for x in deltas)
    return IteratedIncoherenceConfig(
        env_specs=env_specs,
        max_iterations=max_iterations,
        temperature=temperature,
        global_seed=global_seed,
        results_dir=results_dir,
        deltas=deltas,
    )


def load_strong_return_config(path: Path) -> StrongReturnConfig:
    data = yaml.safe_load(path.read_text())
    deterministic_specs = _parse_env_specs(data.get("deterministic_envs", []))
    stochastic_specs = _parse_env_specs(data.get("stochastic_envs", []))
    max_iterations = int(data.get("max_iterations", 5))
    temperature = float(data.get("temperature", 1.0))
    global_seed = int(data.get("global_seed", 0))
    results_dir = _resolve_path(data.get("results_dir"), DEFAULT_RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)
    return StrongReturnConfig(
        deterministic_specs=deterministic_specs,
        stochastic_specs=stochastic_specs,
        max_iterations=max_iterations,
        temperature=temperature,
        global_seed=global_seed,
        results_dir=results_dir,
    )


TEMP_INCOH: float = 0.8


@dataclass
class Result:
    name: str
    data: dict
    policy: Policy
    mdp: MDP

    @staticmethod
    def new_result(name: str, mdp: MDP, policy: Policy, **data) -> "Result":
        return Result(name=name, data=data, policy=policy, mdp=mdp)

    @cached_property
    def j(self) -> float:
        return compute_J(self.mdp, self.policy)

    @cached_property
    def j_entropy(self) -> float:
        return compute_J_entropy(self.mdp, self.policy)

    @cached_property
    def j_causal_entropy(self) -> float:
        return compute_J_causal_entropy(self.mdp, self.policy)

    def incoherence_flat(self, temp: float) -> float:
        return boltzmann_incoherence_flat(self.mdp, self.policy, temp)

    def incoherence_causal(self, temp: float) -> float:
        return boltzmann_incoherence_causal(self.mdp, self.policy, temp)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Result):
            return NotImplemented
        return self.policy == other.policy and self.mdp == other.mdp


def run_retrain(n: int, mdp: MDP) -> List[Result]:
    policy = make_uniform_policy(mdp)
    results: List[Result] = [Result.new_result(name="retrain", step=0, mdp=mdp, policy=policy)]
    retrained_policy = policy
    for k in range(n):
        retrained_policy = retrain_agent(mdp, retrained_policy)
        results.append(
            Result.new_result(name="retrain", step=k + 1, mdp=mdp, policy=retrained_policy)
        )
    return results


def run_boltzmann(n: int) -> None:
    mdp = create_two_cards_game()
    policy = make_uniform_policy(mdp)
    js = [compute_J(mdp, policy)]
    temps = [1 / 2**k for k in range(n)]
    for temp in temps:
        retrained_policy = converge_boltzmann_coherence(mdp, policy, temperature=temp)
        js.append(compute_J(mdp, retrained_policy))
    print_js(js)


def run_increase_temp(n: int, mdp: MDP, schedule=None) -> List[Result]:
    orig_mdp = mdp
    uniform = make_uniform_policy(mdp)
    results: List[Result] = [Result.new_result(name="temp", temp=np.inf, mdp=mdp, policy=uniform)]
    schedule = (lambda count: range(1, count + 1)) if schedule is None else schedule
    for temp in schedule(n):
        mdp = increase_temp(orig_mdp, temp)
        retrained_policy = retrain_agent(mdp, uniform)
        results.append(
            Result.new_result(name="temp", temp=temp, mdp=orig_mdp, policy=retrained_policy)
        )
    return results


def run_fold_posterior_into_reward_and_policy(n: int, mdp: MDP) -> None:
    orig_mdp = mdp
    policy = make_uniform_policy(mdp)
    js = [compute_J(orig_mdp, policy)]
    for _ in range(n):
        mdp, policy = fold_posterior_into_reward(mdp, mdp, policy)
        js.append(compute_J(orig_mdp, policy))
    print_js(js)


def run_fold_posterior_into_reward(n: int, mdp: MDP) -> List[Result]:
    orig_mdp = mdp
    uniform = make_uniform_policy(mdp)
    results: List[Result] = [Result.new_result(name="fold", step=0, mdp=orig_mdp, policy=uniform)]
    for i in range(n):
        mdp, policy = fold_posterior_into_reward(mdp, mdp, uniform)
        results.append(Result.new_result(name="fold", step=i, mdp=orig_mdp, policy=policy))
    return results


def run_fold_posterior_into_reward_orig_mdp(n: int, mdp: MDP) -> List[Result]:
    orig_mdp = mdp
    uniform = make_uniform_policy(mdp)
    results: List[Result] = [Result.new_result(name="fold_orig", step=0, mdp=orig_mdp, policy=uniform)]
    for i in range(n):
        prob = compute_prob_over_trajectories(mdp, uniform)
        posterior = posterior_cond_R(prob, R=1)
        policy = compute_marginals(orig_mdp, posterior)
        results.append(Result.new_result(name="fold_orig", step=i + 1, mdp=orig_mdp, policy=policy))
        mdp, _ = fold_posterior_into_reward(orig_mdp, mdp, uniform)
    return results


def print_results(results: Dict[str, List[Result]], *, axs=None, temp: float = 0.15, alpha: float = 0.3) -> None:
    colors = {"retrain": "blue", "temp": "orange", "fold_orig": "green"}
    markers = {"retrain": ".", "temp": "+", "fold_orig": "v"}
    filtered = {k: v for k, v in results.items() if k in colors}
    for name, res in filtered.items():
        print(f"J   {name:<10} {[round(float(r.j), 8) for r in res]}")
    print()
    if axs is None:
        _, axs = plt.subplots(2, 3)
    for name, res in filtered.items():
        data = [round(float(r.j), 8) for r in res]
        axs[0, 0].plot(
            data,
            label=name,
            c=colors[name],
            linewidth=2,
            alpha=alpha,
            marker=markers[name],
            markersize=10,
        )
    axs[0, 0].set_title("J")
    for name, res in filtered.items():
        data = [round(float(r.incoherence_flat(temp)), 8) for r in res]
        axs[0, 1].plot(
            data,
            label=name,
            c=colors[name],
            linewidth=2,
            alpha=alpha,
            marker=markers[name],
            markersize=10,
        )
    axs[0, 1].set_title("Incoherence flat")
    for name, res in filtered.items():
        data = [round(float(r.incoherence_causal(temp)), 8) for r in res]
        axs[0, 2].plot(
            data,
            label=name,
            c=colors[name],
            linewidth=2,
            alpha=alpha,
            marker=markers[name],
            markersize=10,
        )
    axs[0, 2].set_title("Incoherence causal")
    for name, res in filtered.items():
        data = [round(float(r.j_entropy), 8) for r in res]
        axs[1, 0].plot(
            data,
            label=name,
            c=colors[name],
            linewidth=2,
            alpha=alpha,
            marker=markers[name],
            markersize=10,
        )
    axs[1, 0].set_title("J entropy")
    for name, res in filtered.items():
        data = [round(float(r.j_causal_entropy), 8) for r in res]
        axs[1, 1].plot(
            data,
            label=name,
            c=colors[name],
            linewidth=2,
            alpha=alpha,
            marker=markers[name],
            markersize=10,
        )
    axs[1, 1].set_title("J causal entropy")


def run_mdp_suite(mdp: MDP, k: int) -> None:
    results: Dict[str, List[Result]] = {}
    results["retrain"] = run_retrain(k, mdp=mdp)
    results["temp"] = run_increase_temp(k, mdp=mdp)
    results["fold_orig"] = run_fold_posterior_into_reward_orig_mdp(k, mdp=mdp)
    results["temp_exp"] = run_increase_temp(k, mdp=mdp, schedule=lambda n: [2 ** i for i in range(0, n)])
    results["fold"] = run_fold_posterior_into_reward(k, mdp=mdp)
    print_results(results)


def run_two_cards_suite(k: int) -> None:
    print("Two cards")
    mdp = create_two_cards_game()
    run_mdp_suite(mdp, k)
    print("Test two cards OK")


def run_two_cards_stoch_suite(k: int) -> None:
    print("Two cards stoch 5")
    mdp = create_two_cards_game()
    mdp = stochastify(mdp)
    run_mdp_suite(mdp, k)


def run_random_det_transitions(k: int, T: int = 3, A: int = 4) -> None:
    print("Random det transition")
    mdp = create_random_mdp(A, T, deterministic_transitions=True)
    run_mdp_suite(mdp, k)


def run_random_stoch_transitions(k: int, T: int = 3, A: int = 4) -> None:
    print("Random stoch trans")
    mdp = create_random_mdp(A, T, deterministic_transitions=True)
    mdp = stochastify(mdp)
    run_mdp_suite(mdp, k)


def run_three_state_counterexample() -> None:
    mdp = create_counterexample()
    mdp = create_random_mdp(2, 1, deterministic_transitions=False)
    k = 2
    results: Dict[str, List[Result]] = {}
    results["retrain"] = run_retrain(k, mdp=mdp)
    results["temp"] = run_increase_temp(k, mdp=mdp)
    print("Retrain")
    print_policy(mdp, results["retrain"][-1].policy)
    print("Temp")
    print_policy(mdp, results["temp"][-1].policy)


def run_incoherence_limit() -> None:
    mdp = create_random_mdp(2, 4, True)
    k = 8
    _, axs = plt.subplots(1, 3, figsize=(12, 4))
    results: Dict[str, List[Result]] = {}
    results["retrain"] = run_retrain(k, mdp=mdp)
    results["temp"] = run_increase_temp(k, mdp=mdp)
    results["fold_orig"] = run_fold_posterior_into_reward_orig_mdp(k, mdp=mdp)
    for temp in [0.6, 0.4, 0.2, 0.1]:
        print_results(results, temp=temp, axs=axs, alpha=temp)


def run_compute_prob_over_trajectories() -> None:
    mdp = create_two_cards_game()
    policy = make_uniform_policy(mdp)
    prob = compute_prob_over_trajectories(mdp, policy)
    print("Initial dist")
    print_trajectory_prob(prob)
    print()
    for i in range(3):
        prob = posterior_cond_R(prob, R=1)
        print("Posterior dist ", i)
        print_trajectory_prob(prob)
        policy = compute_marginals(mdp, prob)
        print_policy(mdp, policy)
        print()


def run_qv_diagnostics() -> None:
    mdp = create_two_cards_game()
    policy = make_uniform_policy(mdp)
    print("---------- Trajectories -----------")
    for _ in range(8):
        print_trajectory(sample_trajectory(mdp, policy))
    V, Q = compute_Vs_Qs(mdp, policy)
    print("---------- Q values -----------")
    print_Q(Q)
    print("---------- V values -----------")
    print_V(V)
    print("B-incoherence flat", boltzmann_incoherence_flat(mdp, policy, temperature=1.0))
    print("B-incoherence caus", boltzmann_incoherence_causal(mdp, policy, temperature=1.0))
    print("J", compute_J(mdp, policy))
    print(
        "Iterated Boltzmann policy flat",
        iterated_boltzmann_coherence_flat(mdp, policy, temperature=1.0, num_iterations=3),
    )
    print(
        "Iterated Boltzmann policy causal",
        iterated_boltzmann_coherence_causal(mdp, policy, temperature=1.0, num_iterations=3),
    )
    print(
        "Iterated Boltzmann policy causal belousov",
        iterated_boltzmann_coherence_causal_belousov(
            mdp, policy, temperature=1.0, num_iterations=3
        ),
    )
    new_policy = policy_iteration(mdp)
    print("---- Optimal policy ----")
    print_policy(mdp, new_policy)
    print("J", compute_J(mdp, new_policy))


def get_j(results: List[Result]) -> np.ndarray:
    return np.array([results[i].j for i in range(len(results))])


__all__ = [
    "EnvSpec",
    "EffectiveHorizonSettings",
    "EffectiveHorizonConfig",
    "MisalignmentConfig",
    "EnvInstance",
    "load_effective_horizon_config",
    "load_misalignment_config",
    "load_effective_horizon_dataset",
    "instances_from_records",
    "generate_instances",
    "records_from_instances",
    "IteratedIncoherenceConfig",
    "load_iterated_incoherence_config",
    "StrongReturnConfig",
    "load_strong_return_config",
    "TEMP_INCOH",
    "Result",
    "run_compute_prob_over_trajectories",
    "run_fold_posterior_into_reward",
    "run_fold_posterior_into_reward_orig_mdp",
    "run_fold_posterior_into_reward_and_policy",
    "run_increase_temp",
    "run_mdp_suite",
    "run_qv_diagnostics",
    "run_random_det_transitions",
    "run_random_stoch_transitions",
    "run_retrain",
    "run_two_cards_suite",
    "run_two_cards_stoch_suite",
    "run_three_state_counterexample",
    "run_incoherence_limit",
    "run_boltzmann",
    "print_results",
    "get_j",
]
