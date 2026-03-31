from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np  # type: ignore

from incoherence.experiments import (
    EnvInstance,
    instances_from_records,
    load_effective_horizon_dataset,
    load_misalignment_config,
)
from incoherence.mdp import (
    MDP,
    Policy,
    Action,
    State,
    create_random_mdp,
    make_uniform_policy,
    occupancy_measure_states_time,
    compute_J,
)
from incoherence.policy import boltzmann_incoherence_causal
from incoherence.training import retrain_agent


# ------------------------------------------------------
# 1. Tabular Q^pi and Q^* via dynamic programming
# ------------------------------------------------------

def compute_q_v_pi(
    mdp: MDP, policy: Policy
) -> Tuple[Dict[State, Dict[Action, float]], Dict[State, float]]:
    time_to_states = mdp.time_to_state()
    decision_times = [t for t, states in time_to_states.items() if any(mdp.actions[s] for s in states)]
    max_t = max(decision_times) if decision_times else 0

    V: Dict[State, float] = {s: 0.0 for s in mdp.states}
    Q: Dict[State, Dict[Action, float]] = {s: {} for s in mdp.states}

    for t in reversed(range(max_t + 1)):
        for s in time_to_states.get(t, []):
            acts = mdp.actions[s]
            if not acts:
                Q[s] = {}
                V[s] = 0.0
                continue

            q_s: Dict[Action, float] = {}
            for a in acts:
                r = float(mdp.rewards[s][a].expectation())
                p_next = mdp.transitions[s][a]

                def v_next(s2: State) -> float:
                    return V[s2]

                exp_next = float(p_next.expectation(v_next))
                q_s[a] = r + exp_next
            Q[s] = q_s

            def q_of_action(a: Action) -> float:
                return Q[s][a]

            V[s] = float(policy[s].expectation(q_of_action))
    return Q, V


def compute_q_v_star(mdp: MDP) -> Tuple[Dict[State, Dict[Action, float]], Dict[State, float], Policy]:
    time_to_states = mdp.time_to_state()
    decision_times = [t for t, states in time_to_states.items() if any(mdp.actions[s] for s in states)]
    max_t = max(decision_times) if decision_times else 0

    V_star: Dict[State, float] = {s: 0.0 for s in mdp.states}
    Q_star: Dict[State, Dict[Action, float]] = {s: {} for s in mdp.states}

    for t in reversed(range(max_t + 1)):
        for s in time_to_states.get(t, []):
            acts = mdp.actions[s]
            if not acts:
                Q_star[s] = {}
                V_star[s] = 0.0
                continue

            best_q = -float("inf")
            q_s: Dict[Action, float] = {}
            for a in acts:
                r = float(mdp.rewards[s][a].expectation())
                p_next = mdp.transitions[s][a]

                def v_next(s2: State) -> float:
                    return V_star[s2]

                exp_next = float(p_next.expectation(v_next))
                q_val = r + exp_next
                q_s[a] = q_val
                if q_val > best_q:
                    best_q = q_val
            Q_star[s] = q_s
            V_star[s] = best_q

    from incoherence.distributions import P

    pi_star: Policy = {}
    for s in mdp.states:
        acts = mdp.actions[s]
        if not acts:
            continue
        q_s = Q_star[s]
        best_a = max(acts, key=lambda a: q_s[a])
        pi_star[s] = P({a: 1.0 if a == best_a else 0.0 for a in acts})
    return Q_star, V_star, pi_star


# ------------------------------------------------------
# 2. Alignment metric between greedy(Q_rand) and greedy(Q*)
# ------------------------------------------------------

def greedy_action(Q_s: Dict[Action, float], actions: List[Action]) -> Action | None:
    if not actions:
        return None
    return max(actions, key=lambda a: Q_s[a])


def compute_alignment_with_random_q(mdp: MDP) -> float:
    uniform = make_uniform_policy(mdp)
    Q_rand, _ = compute_q_v_pi(mdp, uniform)
    Q_star, _, _ = compute_q_v_star(mdp)

    occ = occupancy_measure_states_time(mdp, uniform)

    alignment = 0.0
    total_mass = 0.0

    for t, p_states in occ.items():
        for s, mass in p_states.dist.items():
            if mass <= 0.0:
                continue
            acts = mdp.actions[s]
            if not acts:
                continue

            a_rand = greedy_action(Q_rand[s], acts)
            a_star = greedy_action(Q_star[s], acts)
            if a_rand is None or a_star is None:
                continue

            match = 1.0 if a_rand == a_star else 0.0
            alignment += mass * match
            total_mass += mass

    if total_mass == 0.0:
        return 0.0
    return alignment / total_mass


# ------------------------------------------------------
# 3. Experiment driver: alignment vs incoherence
# ------------------------------------------------------

def _misalignment_for_instance(inst: EnvInstance, temp: float) -> Tuple[float, float, float, float]:
    mdp = create_random_mdp(
        inst.spec.num_actions,
        inst.spec.horizon,
        deterministic_transitions=inst.spec.deterministic,
        seed=inst.seed,
    )
    uniform = make_uniform_policy(mdp)
    alignment = compute_alignment_with_random_q(mdp)
    naive_policy = retrain_agent(mdp, uniform)
    incoh = float(boltzmann_incoherence_causal(mdp, naive_policy, temperature=temp))
    J_naive = compute_J(mdp, naive_policy)
    _, V_star, _ = compute_q_v_star(mdp)
    J_star = V_star[mdp.initial_state]
    return alignment, incoh, float(J_naive), float(J_star)


def run_alignment_vs_incoherence_suite(
    instances: Iterable[EnvInstance],
    temperatures: Iterable[float],
) -> List[dict]:
    temps = list(temperatures)
    instances = list(instances)
    results: List[dict] = []

    for temp in temps:
        alignments: List[float] = []
        incoherences: List[float] = []
        per_instance: List[dict] = []
        for inst in instances:
            alignment, incoh, J_naive, J_star = _misalignment_for_instance(inst, temp)
            alignments.append(alignment)
            incoherences.append(incoh)
            per_instance.append(
                {
                    "spec_name": inst.spec.name,
                    "seed": inst.seed,
                    "alignment": alignment,
                    "misalignment": 1.0 - alignment,
                    "incoherence": incoh,
                    "J_naive": J_naive,
                    "J_star": J_star,
                }
            )
        align_arr = np.array(alignments)
        incoh_arr = np.array(incoherences)
        if len(align_arr) > 1 and np.std(align_arr) > 0 and np.std(incoh_arr) > 0:
            corr = float(np.corrcoef(1.0 - align_arr, incoh_arr)[0, 1])
        else:
            corr = float("nan")
        printable_corr = corr if not math.isnan(corr) else float("nan")
        print(
            f"Temp {temp} for Correlation between misalignment (1 - alignment) and incoherence "
            f"{printable_corr:.4f}"
        )
        results.append(
            {
                "temperature": temp,
                "correlation": None if math.isnan(corr) else corr,
                "num_instances": len(instances),
                "instances": per_instance,
            }
        )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Correlation between policy misalignment and incoherence")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/misalignment.yaml"),
        help="Path to the misalignment YAML configuration.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the JSON summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_misalignment_config(args.config)
    dataset = load_effective_horizon_dataset(config.dataset_path)
    instances = instances_from_records(dataset, config.env_specs)
    results = run_alignment_vs_incoherence_suite(instances, config.temperatures)
    output_path = args.output or (config.results_dir / "misalignment_vs_incoherence.json")
    output_path.write_text(json.dumps(results, indent=2))
    print(f"Saved misalignment summary to {output_path}")


if __name__ == "__main__":
    main()
