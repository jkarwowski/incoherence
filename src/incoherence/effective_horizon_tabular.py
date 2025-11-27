# %%
import math
import random
from typing import Dict, Tuple

import numpy as np  # type: ignore

from incoherence.mdp import (
    MDP,
    Policy,
    State,
    Action,
    create_random_mdp,
    make_uniform_policy,
    compute_J,
)
from incoherence.policy import boltzmann_incoherence_causal
from incoherence.scenarios import create_two_cards_game


# -----------------------------------------------------------
# 1. Optimal Q* via finite-horizon DP
# -----------------------------------------------------------

def optimal_Q_V(
    mdp: MDP,
) -> Tuple[Dict[int, Dict[State, Dict[Action, float]]],
           Dict[int, Dict[State, float]]]:
    """
    Compute optimal Q* and V* for a finite-horizon MDP.

    Q[t][s][a] = E[sum_{u=t}^{T-1} R_u | S_t = s, A_t = a, optimal thereafter]
    V[t][s]    = max_a Q[t][s][a]
    """
    T = mdp.time_horizon
    V: Dict[int, Dict[State, float]] = {
        t: {s: 0.0 for s in mdp.states} for t in range(T + 1)
    }
    Q: Dict[int, Dict[State, Dict[Action, float]]] = {
        t: {s: {} for s in mdp.states} for t in range(T)
    }

    for t in reversed(range(T)):  # decision times 0..T-1
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
                exp_next = sum(
                    p * V[t + 1].get(s2, 0.0) for s2, p in trans.dist.items()
                )
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


def sequences_from(mdp: MDP, start_state: State, k: int):
    """
    All valid action sequences of length 1..k starting from start_state,
    respecting the finite horizon.
    """
    T = mdp.time_horizon
    start_t = mdp.state_time[start_state]
    seqs: list[list[Action]] = []

    def rec(s: State, t: int, depth: int, prefix: list[Action]):
        if depth >= 1:
            seqs.append(prefix.copy())
        if depth == k or t >= T or not mdp.actions[s]:
            return
        for a in mdp.actions[s]:
            s2 = deterministic_next_state(mdp, s, a)
            t2 = mdp.state_time[s2]
            prefix.append(a)
            rec(s2, t2, depth + 1, prefix)
            prefix.pop()

    rec(start_state, start_t, 0, [])
    return seqs


def simulate_GORP_once(
    mdp: MDP,
    k: int,
    m: int,
    Q_star: Dict[int, Dict[State, Dict[Action, float]]],
    rng: random.Random,
) -> bool:
    """
    Simulate a single run of a GORP-like procedure with lookahead depth k
    and m rollouts per candidate sequence. Return True iff the resulting
    policy's greedy actions match Q* along its induced trajectory.
    """
    # Number of decision steps: max time index over states with actions, +1
    decision_T = max(
        mdp.state_time[s] for s in mdp.states if mdp.actions[s]
    ) + 1

    def pi_expl(t: int, s: State) -> Action:
        # Simple random exploration policy
        return rng.choice(mdp.actions[s])

    # Learned policy π: mapping (t, s) -> action
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

    # Build policy one timestep at a time
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
                # follow seq
                for a in seq:
                    # sample Bernoulli reward
                    prob_r1 = mdp.rewards[s][a].dist[1]
                    r = 1 if np.random.rand() < prob_r1 else 0
                    total += r
                    s_next = deterministic_next_state(mdp, s, a)
                    s = s_next
                    t = mdp.state_time[s]
                    if t >= decision_T or not mdp.actions[s]:
                        break
                # then exploration until horizon
                while t < decision_T and mdp.actions[s]:
                    a = pi_expl(t, s)
                    prob_r1 = mdp.rewards[s][a].dist[1]
                    r = 1 if np.random.rand() < prob_r1 else 0
                    total += r
                    s = deterministic_next_state(mdp, s, a)
                    t = mdp.state_time[s]
            q_hat = total / m
            if a0 not in best_val_by_a or q_hat > best_val_by_a[a0]:
                best_val_by_a[a0] = q_hat

        # choose first action greedily
        max_val = max(best_val_by_a.values())
        best_actions = [a for a, v in best_val_by_a.items() if v == max_val]
        chosen = rng.choice(best_actions)
        pi[(t_i, s_i)] = chosen

    # Check if π is optimal by comparing to Q*
    s = mdp.initial_state
    t = mdp.state_time[s]
    while t < decision_T and mdp.actions[s]:
        key = (t, s)
        if key not in pi:
            return False
        a = pi[key]
        qs = Q_star[t][s]
        max_q = max(qs.values())
        optimal_actions = [aa for aa, q in qs.items()
                           if abs(q - max_q) < 1e-9]
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
    n_runs: int = 2000,
    seed: int = 0,
) -> float:
    """
    Estimate P(GORP^(k,m) recovers an optimal policy).
    """
    rng = random.Random(seed)
    np.random.seed(seed)
    successes = 0
    for _ in range(n_runs):
        if simulate_GORP_once(mdp, k, m, Q_star, rng):
            successes += 1
    return successes / n_runs


# -----------------------------------------------------------
# 3. Effective horizon estimate H_hat from GORP
# -----------------------------------------------------------

def estimate_effective_horizon(
    mdp: MDP,
    ks=(1, 2, 3, 4),
    m_grid=(1, 2, 3, 5, 10, 20),
    n_runs: int = 2000,
    base_seed: int = 0,
):
    """
    Approximate effective horizon H as in Laidlaw et al.:

        H_k = k + log_A m*_k

    where m*_k is the smallest m for which a GORP run with depth k
    recovers an optimal policy with probability >= 0.5.

    Returns:
        H_hat:  min_k H_k over ks (or inf if none finite)
        results: dict[k] -> {
            "best_m": m*_k or None,
            "H_k": H_k,
            "probs": {m: est success prob}
        }
    """
    Q_star, V_star = optimal_Q_V(mdp)
    A = max(len(acts) for acts in mdp.actions.values() if acts)

    results: dict[int, dict] = {}
    for k in ks:
        # guard against silly k larger than number of decisions
        decision_T = max(
            mdp.state_time[s] for s in mdp.states if mdp.actions[s]
        ) + 1
        if k > decision_T:
            continue

        best_m: int | None = None
        probs: dict[int, float] = {}
        for j, m in enumerate(m_grid):
            seed = base_seed + 97 * k + 131 * j
            p = estimate_success_prob(mdp, k, m, Q_star, n_runs=n_runs, seed=seed)
            probs[m] = p
            if p >= 0.5 and best_m is None:
                best_m = m
        if best_m is None:
            H_k = math.inf
        else:
            H_k = k + math.log(best_m, A)
        results[k] = {"best_m": best_m, "H_k": H_k, "probs": probs}

    finite = [v["H_k"] for v in results.values() if math.isfinite(v["H_k"])]
    H_hat = min(finite) if finite else math.inf
    return H_hat, results


# -----------------------------------------------------------
# 4. Demo: effective horizon vs incoherence
# -----------------------------------------------------------

def demo_effective_horizon_vs_incoherence(
    num_mdps: int = 10,
    T: int = 4,
    A: int = 2,
    temp: float = 0.15,
    ks=(1, 2, 3, 4),
    m_grid=(1, 2, 3, 5, 10, 20),
    n_runs: int = 2000,
    seed: int | None = 0,
) -> None:
    """
    Small demo:
        - Sample `num_mdps` random deterministic MDPs with A actions and horizon T.
        - For each:
            * estimate effective horizon H_hat via GORP
            * find k* achieving min H_k
            * compute incoherence of the uniform policy
            * compute optimal return J*
        - Print a one-line summary per MDP.
    """
    if seed is not None:
        np.random.seed(seed)

    print("idx | k_star | H_hat  | incoherence(uniform) | J*(initial_state)")
    print("-" * 78)

    for idx in range(num_mdps):
        mdp = create_random_mdp(A, T, deterministic_transitions=True)
        uniform = make_uniform_policy(mdp)
        incoh = float(
            boltzmann_incoherence_causal(mdp, uniform, temperature=temp)
        )
        H_hat, results = estimate_effective_horizon(
            mdp, ks=ks, m_grid=m_grid, n_runs=n_runs, base_seed=(seed or 0) + idx
        )

        # find k_star achieving H_hat (smallest such k)
        k_star = None
        if math.isfinite(H_hat):
            for k, v in results.items():
                if math.isfinite(v["H_k"]) and abs(v["H_k"] - H_hat) < 1e-9:
                    k_star = k
                    break

        # compute optimal J* using V_star at initial state
        Q_star, V_star = optimal_Q_V(mdp)
        J_star = V_star[0][mdp.initial_state]

        print(
            f"{idx:3d} | "
            f"{str(k_star):>6} | "
            f"{'inf' if not math.isfinite(H_hat) else f'{H_hat:6.3f}'} | "
            f"{incoh:21.6f} | "
            f"{J_star:7.3f}"
        )


# Example: run on the two-cards game as a sanity check
mdp = create_two_cards_game()
H_hat, res = estimate_effective_horizon(
    mdp, ks=(1, 2, 3), m_grid=(1, 2, 3, 5, 10), n_runs=2000, base_seed=0
)
print("Two-cards game: H_hat =", H_hat)
print("Per-k details:", res)

# And on random tree MDPs
demo_effective_horizon_vs_incoherence()

# %%
