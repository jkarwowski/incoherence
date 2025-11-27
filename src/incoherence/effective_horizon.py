# %%
import math
import random
import numpy as np
from typing import Dict, Tuple
from incoherence.mdp import (
    MDP,
    Policy,
    State,
    Action,
    create_random_mdp,
    make_uniform_policy,
    compute_J,
)
from incoherence.scenarios import create_two_cards_game


# --- 1. Optimal Q* via DP ---------------------------------------------

def optimal_Q_V(mdp: MDP) -> Tuple[Dict[int, Dict[State, Dict[Action, float]]],
                                   Dict[int, Dict[State, float]]]:
    T = mdp.time_horizon
    V = {t: {s: 0.0 for s in mdp.states} for t in range(T+1)}
    Q = {t: {s: {} for s in mdp.states} for t in range(T)}

    for t in reversed(range(T)):          # decision times 0..T-1
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
                exp_next = sum(p * V[t+1].get(s2, 0.0) for s2, p in trans.dist.items())
                q_s[a] = r + exp_next
            Q[t][s] = q_s
            V[t][s] = max(q_s.values())
    return Q, V

# --- 2. GORP simulation for small MDPs --------------------------------

def deterministic_next_state(mdp: MDP, s: State, a: Action) -> State:
    trans = mdp.transitions[s][a].dist
    positive = [s2 for s2, p in trans.items() if p > 1e-9]
    if not positive:
        return s
    if len(positive) > 1:
        raise ValueError("Non-deterministic transitions not supported here.")
    return positive[0]

def sequences_from(mdp: MDP, start_state: State, k: int):
    """All valid action sequences of length 1..k starting from start_state."""
    T = mdp.time_horizon
    start_t = mdp.state_time[start_state]
    seqs = []

    def rec(s, t, depth, prefix):
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

def simulate_GORP_once(mdp: MDP, k: int, m: int,
                       Q_star, rng: random.Random) -> bool:
    """Return True if this GORP run finds an optimal policy."""
    T = mdp.time_horizon

    def pi_expl(t, s):
        return rng.choice(mdp.actions[s])

    # learned policy π: mapping (t,s) -> action
    pi: Dict[Tuple[int, State], Action] = {}

    def follow_prefix():
        s = mdp.initial_state
        t = mdp.state_time[s]
        while t < T and mdp.actions[s]:
            key = (t, s)
            if key not in pi:
                break
            a = pi[key]
            s = deterministic_next_state(mdp, s, a)
            t = mdp.state_time[s]
        return s, t

    # Build policy one timestep at a time
    for i in range(T):
        s_i, t_i = follow_prefix()
        if t_i >= T or not mdp.actions[s_i]:
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
                    r = 1 if np.random.rand() < mdp.rewards[s][a].dist[1] else 0
                    total += r
                    s_next = deterministic_next_state(mdp, s, a)
                    s = s_next
                    t = mdp.state_time[s]
                    if t >= T or not mdp.actions[s]:
                        break
                # then exploration until horizon
                while t < T and mdp.actions[s]:
                    a = pi_expl(t, s)
                    r = 1 if np.random.rand() < mdp.rewards[s][a].dist[1] else 0
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
    while t < T and mdp.actions[s]:
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

def estimate_success_prob(mdp: MDP, k: int, m: int,
                          Q_star, n_runs: int = 2000, seed: int = 0) -> float:
    rng = random.Random(seed)
    successes = 0
    for _ in range(n_runs):
        if simulate_GORP_once(mdp, k, m, Q_star, rng):
            successes += 1
    return successes / n_runs

# --- 3. High-level driver: estimate H ----------------------------------

def estimate_effective_horizon(mdp: MDP,
                               ks=(1, 2, 3, 4),
                               m_grid=(1, 2, 3, 5, 10, 20),
                               n_runs=2000,
                               seed=0):
    Q_star, V_star = optimal_Q_V(mdp)
    A = max(len(acts) for acts in mdp.actions.values() if acts)
    results = {}
    for k in ks:
        if k > mdp.time_horizon:
            continue
        best_m = None
        probs = {}
        for m in m_grid:
            p = estimate_success_prob(mdp, k, m, Q_star, n_runs=n_runs, seed=seed)
            probs[m] = p
            if p >= 0.5 and best_m is None:
                best_m = m
        if best_m is None:
            Hk = math.inf
        else:
            Hk = k + math.log(best_m, A)
        results[k] = {"best_m": best_m, "H_k": Hk, "probs": probs}
    # global minimum
    finite = [v["H_k"] for v in results.values() if math.isfinite(v["H_k"])]
    H_hat = min(finite) if finite else math.inf
    return H_hat, results

# %%
mdp = create_two_cards_game()
H_hat, details = estimate_effective_horizon(mdp, ks=(1,2,3), m_grid=(1,2,3,5,10))
print("Estimated H:", H_hat)
print(details)

# %%
