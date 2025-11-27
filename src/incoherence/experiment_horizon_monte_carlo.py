# exp_effective_horizon_micro.py
# %%
import math
import random
from typing import Dict, Tuple, List

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
from incoherence.training import retrain_agent
from incoherence.scenarios import create_two_cards_game


# -----------------------------------------------------------
# 1. Optimal Q* via finite-horizon dynamic programming
# -----------------------------------------------------------

def optimal_Q_V(
    mdp: MDP,
) -> Tuple[Dict[int, Dict[State, Dict[Action, float]]],
           Dict[int, Dict[State, float]]]:
    """
    Compute optimal Q* and V* for a finite-horizon MDP.

    Q[t][s][a] = E[sum_{u=t}^{T_dec-1} R_u | S_t = s, A_t = a, optimal thereafter]
    V[t][s]    = max_a Q[t][s][a]

    We treat "decision times" as the time indices of states that have actions.
    """
    # Decision horizon: last time index at which any action is available
    decision_T = max(
        mdp.state_time[s] for s in mdp.states if mdp.actions[s]
    ) + 1

    V: Dict[int, Dict[State, float]] = {
        t: {s: 0.0 for s in mdp.states} for t in range(decision_T + 1)
    }
    Q: Dict[int, Dict[State, Dict[Action, float]]] = {
        t: {s: {} for s in mdp.states} for t in range(decision_T)
    }

    for t in reversed(range(decision_T)):  # decision times 0..decision_T-1
        for s in mdp.states:
            if mdp.state_time[s] != t:
                continue
            acts = mdp.actions[s]
            if not acts:
                continue
            q_s: Dict[Action, float] = {}
            for a in acts:
                # rewards are Bernoulli => expectation is probability of reward 1
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


def sequences_from(mdp: MDP, start_state: State, k: int) -> List[List[Action]]:
    """
    All valid action sequences of length 1..k starting from start_state,
    respecting the finite horizon.
    """
    # decision horizon
    decision_T = max(
        mdp.state_time[s] for s in mdp.states if mdp.actions[s]
    ) + 1
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
) -> bool:
    """
    Simulate a single GORP-like run with lookahead depth k and m rollouts
    per candidate sequence. Returns True iff the resulting policy is greedy
    w.r.t. Q* along its induced trajectory.
    """
    decision_T = max(
        mdp.state_time[s] for s in mdp.states if mdp.actions[s]
    ) + 1

    def pi_expl(t: int, s: State) -> Action:
        # Exploration policy used for rollouts after the prefix:
        # here simply uniform random over available actions.
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

    # Build policy one decision time at a time
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
                # Follow the candidate sequence
                for a in seq:
                    prob_r1 = mdp.rewards[s][a].dist[1]
                    r = 1 if np.random.rand() < prob_r1 else 0
                    total += r
                    s = deterministic_next_state(mdp, s, a)
                    t = mdp.state_time[s]
                    if t >= decision_T or not mdp.actions[s]:
                        break
                # Then explore randomly until horizon
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

        # Greedy in the first action of the best sequence(s)
        max_val = max(best_val_by_a.values())
        best_actions = [a for a, v in best_val_by_a.items() if v == max_val]
        chosen = rng.choice(best_actions)
        pi[(t_i, s_i)] = chosen

    # Check if π is optimal by comparing its greedy actions to Q* along its own trajectory
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
    n_runs: int = 500,
    seed: int = 0,
) -> float:
    """
    Estimate P(GORP^(k,m) recovers an optimal policy).
    """
    rng = random.Random(seed)
    np.random.seed(seed)
    successes = 0
    for _ in range(n_runs):
        if simulate_gorp_once(mdp, k, m, Q_star, rng):
            successes += 1
    return successes / n_runs


# -----------------------------------------------------------
# 3. Effective horizon H_hat via adaptive search in m
# -----------------------------------------------------------

def estimate_effective_horizon(
    mdp: MDP,
    ks=(1, 2, 3, 4),
    success_threshold: float = 0.5,
    m_max: int = 256,
    n_runs: int = 500,
    base_seed: int = 0,
):
    """
    Approximate Laidlaw et al.'s effective horizon H using a GORP-like
    procedure on a single small tabular MDP.

        For each k in ks:
            - find the smallest m (1,2,4,8,...) with success prob >= threshold
            - set H_k = k + log_A(m)

        H_hat = min_k H_k over ks (or inf if no such k).

    Returns:
        H_hat:   float (inf if all H_k are inf)
        results: dict[k] -> {
            "best_m": m*_k or None,
            "H_k": H_k,
            "probs": {m: est success prob}
        }
    """
    Q_star, V_star = optimal_Q_V(mdp)
    A = max(len(acts) for acts in mdp.actions.values() if acts)

    decision_T = max(
        mdp.state_time[s] for s in mdp.states if mdp.actions[s]
    ) + 1

    results: dict[int, dict] = {}
    for k in ks:
        if k > decision_T:
            continue

        best_m: int | None = None
        probs: dict[int, float] = {}

        m = 1
        j = 0
        while m <= m_max:
            seed = base_seed + 997 * k + 131 * j
            p = estimate_success_prob(mdp, k, m, Q_star, n_runs=n_runs, seed=seed)
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
    """
    One application of your control-as-inference operator G:
        π^G_1 = G(p), with p uniform by default.
    """
    if prior is None:
        prior = make_uniform_policy(mdp)
    return retrain_agent(mdp, prior)


def compute_incoherence_and_returns(
    mdp: MDP,
    temperature: float = 1.0,
):
    """
    Compute κ_δ and J for:
        - uniform policy p
        - naive goal-conditioned policy π^G_1 = G(p)

    Also compute optimal J* for comparison.
    """
    uniform = make_uniform_policy(mdp)
    pi_G = naive_goal_conditioned_policy(mdp, uniform)

    kappa_uniform = float(
        boltzmann_incoherence_causal(mdp, uniform, temperature)
    )
    kappa_piG = float(
        boltzmann_incoherence_causal(mdp, pi_G, temperature)
    )

    J_uniform = float(compute_J(mdp, uniform))
    J_piG = float(compute_J(mdp, pi_G))

    # Optimal return J* from DP
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
# 5. Utility: search random MDPs with finite effective horizon
# -----------------------------------------------------------

def find_random_mdps_with_finite_H(
    num_mdps: int,
    T: int,
    A: int,
    ks=(1, 2, 3, 4),
    success_threshold: float = 0.5,
    m_max: int = 256,
    n_runs: int = 500,
    seed_start: int = 0,
):
    """
    Search over random deterministic MDPs (by seed) until we find
    `num_mdps` of them for which H_hat < inf under our GORP budget.
    """
    found: list[tuple[int, float, MDP, dict]] = []
    seed = seed_start
    while len(found) < num_mdps:
        print(f"Trying seed={seed} T={T} A={A}")
        np.random.seed(seed)
        mdp = create_random_mdp(A, T, deterministic_transitions=True)
        H_hat, results = estimate_effective_horizon(
            mdp,
            ks=ks,
            success_threshold=success_threshold,
            m_max=m_max,
            n_runs=n_runs,
            base_seed=seed * 1000,
        )
        if math.isfinite(H_hat):
            print(f"{seed} success")
            found.append((seed, H_hat, mdp, results))
        else:
            print(f"{seed} failed")
        seed += 1
        # safety to avoid infinite loops; adjust if needed
        if seed - seed_start > 500:
            break
    return found


# -----------------------------------------------------------
# 6. Demo / pretty-printing
# -----------------------------------------------------------

def print_effective_horizon_summary(
    name: str,
    H_hat: float,
    results: dict[int, dict],
    incoh_info: dict,
) -> None:
    print(f"\n=== Environment: {name} ===")
    if math.isfinite(H_hat):
        print(f"Estimated effective horizon H_hat: {H_hat:.3f}")
    else:
        print("Estimated effective horizon H_hat: inf (no (k,m) with success >= threshold within budget)")

    print("Per-k details (k : best_m, H_k, probs(m)): ")
    for k in sorted(results.keys()):
        best_m = results[k]["best_m"]
        H_k = results[k]["H_k"]
        probs = results[k]["probs"]
        if best_m is None:
            H_str = "inf"
        else:
            H_str = f"{H_k:.3f}"
        print(f"  k={k}: best_m={best_m}, H_k={H_str}, probs={probs}")

    print("\nIncoherence & returns (δ=1.0 by default):")
    print(f"  κδ(uniform)     = {incoh_info['kappa_uniform']:.6f}")
    print(f"  κδ(π^G_1 = G(p)) = {incoh_info['kappa_piG']:.6f}")
    print(f"  J(uniform)      = {incoh_info['J_uniform']:.4f}")
    print(f"  J(π^G_1)        = {incoh_info['J_piG']:.4f}")
    print(f"  J* (optimal)    = {incoh_info['J_star']:.4f}")


def run_demo(T, A, num, seed_start) -> None:
    # 1) Two-cards game as a sanity check / simple example
    # two_cards = create_two_cards_game()
    # H_two_cards, res_two_cards = estimate_effective_horizon(
    #     two_cards,
    #     ks=(1, 2, 3),
    #     success_threshold=0.5,
    #     m_max=32,
    #     n_runs=1000,
    #     base_seed=0,
    # )
    # incoh_two_cards = compute_incoherence_and_returns(two_cards, temperature=1.0)
    # print_effective_horizon_summary(
    #     "two_cards",
    #     H_two_cards,
    #     res_two_cards,
    #     incoh_two_cards,
    # )

    # 2) A few random tree MDPs (deterministic) where H_hat is finite
    found = find_random_mdps_with_finite_H(
        num_mdps=num,
        T=T,
        A=A,
        ks=(1, 2, 3, 4),
        success_threshold=0.5,
        m_max=256,
        n_runs=500,
        seed_start=seed_start,
    )

    for idx, (seed, H_hat, mdp, results) in enumerate(found):
        incoh_info = compute_incoherence_and_returns(mdp, temperature=1.0)
        name = f"random_T{T}_A{A}_seed{seed}"
        print_effective_horizon_summary(name, H_hat, results, incoh_info)


run_demo(T=4, A=3, num=5, seed_start=15)
run_demo(T=4, A=3, num=5, seed_start=2*20)
run_demo(T=4, A=3, num=5, seed_start=3*25)
run_demo(T=4, A=3, num=5, seed_start=4*30)

run_demo(T=5, A=2, num=5, seed_start=15)
run_demo(T=5, A=2, num=5, seed_start=2*20)
run_demo(T=5, A=2, num=5, seed_start=3*25)
run_demo(T=5, A=2, num=5, seed_start=4*30)

run_demo(T=6, A=2, num=5, seed_start=15)
run_demo(T=6, A=2, num=5, seed_start=2*20)
run_demo(T=6, A=2, num=5, seed_start=3*25)
run_demo(T=6, A=2, num=5, seed_start=4*30)

run_demo(T=3, A=4, num=5, seed_start=15)
run_demo(T=3, A=4, num=5, seed_start=2*20)
run_demo(T=3, A=4, num=5, seed_start=3*25)
run_demo(T=3, A=4, num=5, seed_start=4*30)


# %%
