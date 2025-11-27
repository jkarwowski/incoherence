# %%
import numpy as np  # type: ignore
from typing import Dict, Tuple

from incoherence.mdp import (
    MDP,
    Policy,
    State,
    Action,
    create_random_mdp,
    make_uniform_policy,
    occupancy_measure_states_time,
    compute_J,
)
from incoherence.training import retrain_agent
from incoherence.policy import boltzmann_incoherence_causal
from incoherence.experiments import TEMP_INCOH


# ------------------------------------------------------
# 1. Tabular Q^pi and Q^* via dynamic programming
# ------------------------------------------------------

def compute_q_v_pi(mdp: MDP, policy: Policy
                   ) -> Tuple[Dict[State, Dict[Action, float]],
                              Dict[State, float]]:
    """
    Finite-horizon Q^pi and V^pi, with undiscounted sum of Bernoulli rewards.

    Q^pi(s,a) = E[ sum_{t >= time(s)} R_t | S = s, A = a, follow pi thereafter ]
    V^pi(s)   = E[ sum_{t >= time(s)} R_t | S = s, follow pi ].
    """
    time_to_states = mdp.time_to_state()

    # max decision time: last time index with any actions
    decision_times = [t for t, states in time_to_states.items()
                      if any(mdp.actions[s] for s in states)]
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
                r = float(mdp.rewards[s][a].expectation())  # Bernoulli mean
                p_next = mdp.transitions[s][a]

                def v_next(s2: State) -> float:
                    return V[s2]

                exp_next = float(p_next.expectation(v_next))
                q_val = r + exp_next
                q_s[a] = q_val

            Q[s] = q_s

            # V^pi(s) = E_{a~pi}[Q(s,a)]
            def q_of_action(a: Action) -> float:
                return Q[s][a]

            V[s] = float(policy[s].expectation(q_of_action))

    return Q, V


def compute_q_v_star(mdp: MDP
                     ) -> Tuple[Dict[State, Dict[Action, float]],
                                Dict[State, float], Policy]:
    """
    Optimal Q^* and V^* via finite-horizon DP, plus one greedy optimal policy.
    """
    time_to_states = mdp.time_to_state()

    decision_times = [t for t, states in time_to_states.items()
                      if any(mdp.actions[s] for s in states)]
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

    # Greedy optimal policy (ties broken arbitrarily)
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

def greedy_action(Q_s: Dict[Action, float], actions: list[Action]) -> Action | None:
    if not actions:
        return None
    return max(actions, key=lambda a: Q_s[a])


def compute_alignment_with_random_q(mdp: MDP) -> float:
    """
    Alignment = E_{(t,s) ~ occupancy of uniform policy}
                [ 1[ argmax_a Q^{pi_rand}(s,a) == argmax_a Q^*(s,a) ] ].
    """
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

def run_alignment_vs_incoherence(
    num_mdps: int = 40,
    T: int = 4,
    A: int = 3,
    temp: float = .8,
    seed: int | None = 0,
) -> None:
    """
    For each random deterministic MDP:
      - compute alignment between greedy(Q^{pi_rand}) and greedy(Q^*)
      - compute incoherence of the naively goal-conditioned policy G(p)
      - report correlation between misalignment and incoherence
    """
    print("A")
    if seed is not None:
        np.random.seed(seed)

    alignments: list[float] = []
    incoherences: list[float] = []
    j_naive_vals: list[float] = []
    j_star_vals: list[float] = []

    # print("idx | alignment | incoh( G(p) ) | J_naive | J_star")
    # print("-" * 70)

    for idx in range(num_mdps):
        mdp = create_random_mdp(A, T, deterministic_transitions=True)
        uniform = make_uniform_policy(mdp)

        # alignment between greedy(Q_rand) and greedy(Q*)
        alignment = compute_alignment_with_random_q(mdp)

        # naive goal-conditioned policy = one application of G to the prior
        naive_policy = retrain_agent(mdp, uniform)  # π^G_1
        incoh = float(
            boltzmann_incoherence_causal(mdp, naive_policy, temperature=temp)
        )

        J_naive = compute_J(mdp, naive_policy)
        # optimal J via DP (slightly cheaper than policy_iteration)
        _, V_star, _ = compute_q_v_star(mdp)
        J_star = V_star[mdp.initial_state]

        alignments.append(alignment)
        incoherences.append(incoh)
        j_naive_vals.append(J_naive)
        j_star_vals.append(J_star)

        # print(
        #     f"{idx:3d} | {alignment:9.4f} | {incoh:13.6f} | "
        #     f"{J_naive:7.3f} | {J_star:7.3f}"
        # )

    align_arr = np.array(alignments)
    incoh_arr = np.array(incoherences)

    if len(align_arr) > 1 and np.std(align_arr) > 0 and np.std(incoh_arr) > 0:
        corr = float(np.corrcoef(1.0 - align_arr, incoh_arr)[0, 1])
    else:
        corr = float("nan")

    # print()
    print(
        f"Temp {temp} for Correlation between misalignment (1 - alignment) and incoherence "
        f"{corr:.4f}"
    )
# %%
for temp in [.15, .3, .5, .8, .9, .95]:
    run_alignment_vs_incoherence(temp=temp)
# %%
