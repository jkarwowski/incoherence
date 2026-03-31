from __future__ import annotations

import json
import math
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np  # type: ignore
from numpy.random import Generator, SeedSequence, default_rng
from tqdm import tqdm

from incoherence.experiments import (
    EffectiveHorizonSettings,
    EnvSpec,
    load_effective_horizon_config,
)
from incoherence.mdp import (
    MDP,
    Policy,
    Action,
    State,
    compute_J,
    create_random_mdp,
    make_uniform_policy,
)
from incoherence.policy import boltzmann_incoherence_causal
from incoherence.training import retrain_agent


# -----------------------------------------------------------
# 1. Optimal Q* via finite-horizon dynamic programming
# -----------------------------------------------------------

def optimal_Q_V(
    mdp: MDP,
) -> Tuple[Dict[int, Dict[State, Dict[Action, float]]], Dict[int, Dict[State, float]]]:
    """Compute optimal Q* and V* for a finite-horizon MDP."""
    decision_T = max(mdp.state_time[s] for s in mdp.states if mdp.actions[s]) + 1

    V: Dict[int, Dict[State, float]] = {
        t: {s: 0.0 for s in mdp.states} for t in range(decision_T + 1)
    }
    Q: Dict[int, Dict[State, Dict[Action, float]]] = {
        t: {s: {} for s in mdp.states} for t in range(decision_T)
    }

    for t in reversed(range(decision_T)):
        for s in mdp.states:
            if mdp.state_time[s] != t:
                continue
            acts = mdp.actions[s]
            if not acts:
                continue
            q_s: Dict[Action, float] = {}
            for a in acts:
                r = mdp.rewards[s][a].expectation()
                trans = mdp.transitions[s][a]
                exp_next = sum(p * V[t + 1].get(s2, 0.0) for s2, p in trans.dist.items())
                q_s[a] = float(r + exp_next)
            Q[t][s] = q_s
            V[t][s] = max(q_s.values())
    return Q, V


# -----------------------------------------------------------
# 2. GORP simulation (deterministic dynamics)
# -----------------------------------------------------------

def deterministic_next_state(mdp: MDP, s: State, a: Action) -> State:
    trans = mdp.transitions[s][a].dist
    positive = [s2 for s2, p in trans.items() if p > 1e-9]
    if not positive:
        return s
    if len(positive) > 1:
        raise ValueError("Non-deterministic transitions not supported here.")
    return positive[0]


def sequences_from(mdp: MDP, start_state: State, k: int) -> List[List[Action]]:
    decision_T = max(mdp.state_time[s] for s in mdp.states if mdp.actions[s]) + 1
    start_t = mdp.state_time[start_state]
    seqs: List[List[Action]] = []

    def rec(s: State, t: int, depth: int, prefix: List[Action]) -> None:
        if depth >= 1:
            seqs.append(prefix.copy())
        if depth == k or t >= decision_T or not mdp.actions[s]:
            return
        for a in mdp.actions[s]:
            s2 = deterministic_next_state(mdp, s, a)
            t2 = mdp.state_time[s2]
            prefix.append(a)
            rec(s2, t2, depth + 1, prefix)
            prefix.pop()

    rec(start_state, start_t, 0, [])
    return seqs


def simulate_gorp_once(
    mdp: MDP,
    k: int,
    m: int,
    Q_star: Dict[int, Dict[State, Dict[Action, float]]],
    rng: random.Random,
    np_rng: Generator,
) -> bool:
    """Simulate a single GORP-like run."""
    decision_T = max(mdp.state_time[s] for s in mdp.states if mdp.actions[s]) + 1

    def pi_expl(t: int, s: State) -> Action:
        return rng.choice(mdp.actions[s])

    pi: Dict[Tuple[int, State], Action] = {}

    def follow_prefix() -> Tuple[State, int]:
        s = mdp.initial_state
        t = mdp.state_time[s]
        while t < decision_T and mdp.actions[s]:
            key = (t, s)
            if key not in pi:
                break
            a = pi[key]
            s = deterministic_next_state(mdp, s, a)
            t = mdp.state_time[s]
        return s, t

    for i in range(decision_T):
        s_i, t_i = follow_prefix()
        if t_i >= decision_T or not mdp.actions[s_i]:
            break
        assert t_i == i
        seqs = sequences_from(mdp, s_i, k)
        if not seqs:
            break

        best_val_by_a: Dict[Action, float] = {}
        for seq in seqs:
            a0 = seq[0]
            total = 0.0
            for _ in range(m):
                s = s_i
                t = t_i
                for a in seq:
                    prob_r1 = mdp.rewards[s][a].dist[1]
                    r = 1 if np_rng.random() < prob_r1 else 0
                    total += r
                    s = deterministic_next_state(mdp, s, a)
                    t = mdp.state_time[s]
                    if t >= decision_T or not mdp.actions[s]:
                        break
                while t < decision_T and mdp.actions[s]:
                    a = pi_expl(t, s)
                    prob_r1 = mdp.rewards[s][a].dist[1]
                    r = 1 if np_rng.random() < prob_r1 else 0
                    total += r
                    s = deterministic_next_state(mdp, s, a)
                    t = mdp.state_time[s]
            q_hat = total / m
            if a0 not in best_val_by_a or q_hat > best_val_by_a[a0]:
                best_val_by_a[a0] = q_hat

        max_val = max(best_val_by_a.values())
        best_actions = [a for a, v in best_val_by_a.items() if v == max_val]
        chosen = rng.choice(best_actions)
        pi[(t_i, s_i)] = chosen

    s = mdp.initial_state
    t = mdp.state_time[s]
    while t < decision_T and mdp.actions[s]:
        key = (t, s)
        if key not in pi:
            return False
        a = pi[key]
        qs = Q_star[t][s]
        max_q = max(qs.values())
        optimal_actions = [aa for aa, q in qs.items() if abs(q - max_q) < 1e-9]
        if a not in optimal_actions:
            return False
        s = deterministic_next_state(mdp, s, a)
        t = mdp.state_time[s]
    return True


def estimate_success_prob(
    mdp: MDP,
    k: int,
    m: int,
    Q_star,
    n_runs: int,
    rng: Generator,
) -> float:
    successes = 0
    for _ in range(n_runs):
        np_seed = int(rng.integers(0, 2**32, dtype=np.uint32))
        py_seed = int(rng.integers(0, 2**32, dtype=np.uint32))
        np_rng = default_rng(np_seed)
        py_rng = random.Random(py_seed)
        if simulate_gorp_once(mdp, k, m, Q_star, py_rng, np_rng):
            successes += 1
    return successes / n_runs


# -----------------------------------------------------------
# 3. Effective horizon H_hat via adaptive search in m
# -----------------------------------------------------------

def estimate_effective_horizon(
    mdp: MDP,
    *,
    ks: Iterable[int],
    success_threshold: float,
    m_max: int,
    n_runs: int,
    rng: Generator,
):
    Q_star, V_star = optimal_Q_V(mdp)
    A = max(len(acts) for acts in mdp.actions.values() if acts)

    decision_T = max(mdp.state_time[s] for s in mdp.states if mdp.actions[s]) + 1

    results: dict[int, dict] = {}
    for k in ks:
        if k > decision_T:
            continue
        best_m: int | None = None
        probs: dict[int, float] = {}
        m = 1
        j = 0
        while m <= m_max:
            p = estimate_success_prob(
                mdp,
                k,
                m,
                Q_star,
                n_runs=n_runs,
                rng=rng,
            )
            probs[m] = p
            if p >= success_threshold:
                best_m = m
                break
            m *= 2
            j += 1
        if best_m is None:
            H_k = math.inf
        else:
            H_k = k + math.log(best_m, A)
        results[k] = {"best_m": best_m, "H_k": H_k, "probs": probs}

    finite = [v["H_k"] for v in results.values() if math.isfinite(v["H_k"])]
    H_hat = min(finite) if finite else math.inf
    return H_hat, results


# -----------------------------------------------------------
# 4. Incoherence & goal-conditioning helpers
# -----------------------------------------------------------

def naive_goal_conditioned_policy(mdp: MDP, prior: Policy | None = None) -> Policy:
    if prior is None:
        prior = make_uniform_policy(mdp)
    return retrain_agent(mdp, prior)


def compute_incoherence_and_returns(mdp: MDP, temperature: float = 1.0) -> dict:
    uniform = make_uniform_policy(mdp)
    pi_G = naive_goal_conditioned_policy(mdp, uniform)

    kappa_uniform = float(boltzmann_incoherence_causal(mdp, uniform, temperature))
    kappa_piG = float(boltzmann_incoherence_causal(mdp, pi_G, temperature))

    J_uniform = float(compute_J(mdp, uniform))
    J_piG = float(compute_J(mdp, pi_G))

    _, V_star = optimal_Q_V(mdp)
    J_star = float(V_star[0][mdp.initial_state])

    return {
        "kappa_uniform": kappa_uniform,
        "kappa_piG": kappa_piG,
        "J_uniform": J_uniform,
        "J_piG": J_piG,
        "J_star": J_star,
    }


# -----------------------------------------------------------
# 5. High-level orchestration
# -----------------------------------------------------------

@dataclass
class EffectiveHorizonRecord:
    spec_name: str
    seed: int
    horizon: int
    num_actions: int
    H_hat: float
    per_k: dict[int, dict]
    metrics: dict

    def as_dict(self) -> dict:
        payload = {
            "spec_name": self.spec_name,
            "seed": self.seed,
            "horizon": self.horizon,
            "num_actions": self.num_actions,
            "H_hat": self.H_hat if math.isfinite(self.H_hat) else None,
            "per_k": {},
            "metrics": self.metrics,
        }
        for k, info in self.per_k.items():
            payload["per_k"][k] = {
                "best_m": info["best_m"],
                "H_k": info["H_k"] if math.isfinite(info["H_k"]) else None,
                "probs": info["probs"],
            }
        return payload


def _select_instances_for_spec(
    spec: EnvSpec,
    *,
    temperature: float,
    global_seed: int,
    spec_index: int,
    settings: EffectiveHorizonSettings,
) -> List[EffectiveHorizonRecord]:
    records: List[EffectiveHorizonRecord] = []
    candidate_rng = default_rng(SeedSequence([global_seed, spec_index]))
    seen_seeds: set[int] = set()
    attempts = 0
    while len(records) < spec.num_instances and attempts < spec.search_limit:
        seed = int(candidate_rng.integers(0, 2**32, dtype=np.uint32))
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
        env_rng = default_rng(SeedSequence([global_seed, spec_index, seed]))
        H_hat, per_k = estimate_effective_horizon(
            mdp,
            ks=settings.ks,
            success_threshold=settings.success_threshold,
            m_max=settings.m_max,
            n_runs=settings.n_runs,
            rng=env_rng,
        )
        if not math.isfinite(H_hat):
            continue
        metrics = compute_incoherence_and_returns(mdp, temperature=temperature)
        records.append(
            EffectiveHorizonRecord(
                spec_name=spec.name,
                seed=seed,
                horizon=spec.horizon,
                num_actions=spec.num_actions,
                H_hat=H_hat,
                per_k=per_k,
                metrics=metrics,
            )
        )
        if len(records) >= spec.num_instances:
            break
    if len(records) < spec.num_instances:
        raise RuntimeError(
            f"Could not find {spec.num_instances} finite-horizon MDPs for {spec.name} "
            f"within {spec.search_limit} attempts using global seed {global_seed}."
        )
    return records


def collect_effective_horizon_data(
    *,
    env_specs: Iterable[EnvSpec],
    settings: EffectiveHorizonSettings,
    temperature: float,
    global_seed: int,
    verbose: bool = True,
    show_progress: bool = True,
    max_workers: int | None = None,
) -> List[EffectiveHorizonRecord]:
    env_specs = list(env_specs)
    if not env_specs:
        return []

    if max_workers is None:
        max_workers = min(len(env_specs), (os.cpu_count() or 2) - 1)
    max_workers = max(1, max_workers)

    progress = tqdm(total=len(env_specs), desc="Effective horizon", leave=False) if show_progress else None
    results: List[Tuple[int, List[EffectiveHorizonRecord]]] = []

    if max_workers == 1:
        for spec_index, spec in enumerate(env_specs):
            records = _select_instances_for_spec(
                spec,
                temperature=temperature,
                global_seed=global_seed,
                spec_index=spec_index,
                settings=settings,
            )
            results.append((spec_index, records))
            if progress:
                progress.update(1)
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _select_instances_for_spec,
                    spec,
                    temperature=temperature,
                    global_seed=global_seed,
                    spec_index=spec_index,
                    settings=settings,
                ): spec_index
                for spec_index, spec in enumerate(env_specs)
            }
            for future in as_completed(futures):
                spec_index = futures[future]
                records = future.result()
                results.append((spec_index, records))
                if progress:
                    progress.update(1)

    if progress:
        progress.close()
    results.sort(key=lambda item: item[0])
    all_records: List[EffectiveHorizonRecord] = []
    for _, records in results:
        if verbose:
            for record in records:
                print_effective_horizon_summary(record)
        all_records.extend(records)
    return all_records


def print_effective_horizon_summary(record: EffectiveHorizonRecord) -> None:
    print(f"\n=== Environment: {record.spec_name}_seed{record.seed} ===")
    if math.isfinite(record.H_hat):
        print(f"Estimated effective horizon H_hat: {record.H_hat:.3f}")
    else:
        print("Estimated effective horizon H_hat: inf")
    print("Per-k details (k : best_m, H_k, probs(m)):")
    for k in sorted(record.per_k.keys()):
        info = record.per_k[k]
        best_m = info["best_m"]
        H_k = info["H_k"]
        probs = info["probs"]
        H_str = "inf" if not math.isfinite(H_k) else f"{H_k:.3f}"
        print(f"  k={k}: best_m={best_m}, H_k={H_str}, probs={probs}")
    metrics = record.metrics
    print("\nIncoherence & returns (δ=1.0 by default):")
    print(f"  κδ(uniform)     = {metrics['kappa_uniform']:.6f}")
    print(f"  κδ(π^G_1 = G(p)) = {metrics['kappa_piG']:.6f}")
    print(f"  J(uniform)      = {metrics['J_uniform']:.4f}")
    print(f"  J(π^G_1)        = {metrics['J_piG']:.4f}")
    print(f"  J* (optimal)    = {metrics['J_star']:.4f}")
    print(
        "FINAL: env "
        f"{record.spec_name}_seed{record.seed} H_k {record.H_hat:.3f} "
        f"unif-incoh {metrics['kappa_uniform']:.6f} cond-incoh {metrics['kappa_piG']:.6f}"
    )


def save_effective_horizon_data(path: Path, records: List[EffectiveHorizonRecord]) -> None:
    serialisable = [record.as_dict() for record in records]
    path.write_text(json.dumps(serialisable, indent=2))


def main() -> None:
    config_path = Path("configs/effective_horizon.yaml")
    if not config_path.exists():
        raise SystemExit(
            "Effective horizon configuration not found. Please create configs/effective_horizon.yaml."
        )
    config = load_effective_horizon_config(config_path)
    records = collect_effective_horizon_data(
        env_specs=config.env_specs,
        settings=config.settings,
        temperature=config.temperature,
        global_seed=config.global_seed,
        verbose=True,
    )
    output_path = config.results_dir / "effective_horizon.json"
    save_effective_horizon_data(output_path, records)
    print(f"\nSaved {len(records)} records to {output_path}")


if __name__ == "__main__":
    main()
