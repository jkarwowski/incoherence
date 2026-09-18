from typing import Dict

import numpy as np  # type: ignore

from .distributions import P
from .mdp import Action, MDP, Policy, State, make_uniform_policy
from .metrics import compute_causal_forward_kl_divergence, compute_causal_forward_kl_divergence_belousov


def soft_log(x, soft=True):
    if soft:
        return np.log(x)
    else:
        return x


def soft_exp(x, soft=True):
    if soft:
        return np.exp(x)
    else:
        return x


def soft_Q(
    mdp: MDP[State, Action],
    policy: Policy,
    state: State,
    action: Action,
    t: int,
    soft=True,
) -> float:
    """Recursive computation of soft Q values at time t for a given state and action."""
    if t < mdp.time_horizon - 1:
        nxt = soft_log(
            mdp.transitions[state][action].expectation(
                lambda next_state: soft_exp(
                    soft_V(mdp, policy, next_state, t + 1), soft=soft
                )
            ),
            soft=soft,
        )
    else:
        nxt = 0
    return np.log(mdp.rewards[state][action].expectation()) + nxt


def soft_V(mdp: MDP, policy: Policy, state: State, t: int, soft=True) -> float:
    """Recursive computation of soft V values at time t for a given state."""
    if not mdp.actions[state] or t >= mdp.time_horizon:
        return 0.0
    return soft_log(
        policy[state].expectation(
            lambda action: soft_exp(soft_Q(mdp, policy, state, action, t), soft=soft)
        ),
        soft=soft,
    )


def compute_Vs_Qs(
    mdp: MDP, policy: Policy, soft=True
) -> tuple[Dict[State, float], Dict[State, Dict[Action, float]]]:
    """Computes the V and Q values for a given policy."""
    V = {
        state: soft_V(mdp, policy, state, mdp.state_time[state], soft=soft)
        for state in mdp.states
    }
    Q = {
        state: {
            action: soft_Q(mdp, policy, state, action, mdp.state_time[state], soft=soft)
            for action in mdp.actions[state]
        }
        for state in mdp.states
    }
    return V, Q


def print_Q(Q: Dict[State, Dict[Action, float]]):
    """Prints the Q values in a human-readable format."""
    for state, actions in Q.items():
        for action, value in actions.items():
            print(f"Q({state}, {action}) = {value}")


def print_V(V: Dict[State, float]):
    """Prints the V values in a human-readable format."""
    for state, value in V.items():
        print(f"V({state}) = {value}")


def boltzmann_rational_policy(Q, temperature=1.0):
    """Computes the Boltzmann rational policy given Q values and a temperature."""
    policy = {}
    for state, actions_q_values in Q.items():
        maximum = max(actions_q_values.values(), default=0.0)
        for action, q_value in actions_q_values.items():
            if state not in policy:
                policy[state] = {}
            policy[state][action] = np.exp((q_value - maximum) / temperature)

    # Normalize the policy to form a proper probability distribution
    for state in policy:
        policy[state] = P(policy[state])
    return policy


def kl_divergence(p: np.ndarray, q: np.ndarray):
    """Computes the KL divergence D(p || q) for two probability distributions."""
    return sum(
        p_val * np.log(p_val / q_val) for p_val, q_val in zip(p, q) if p_val != 0
    )


def compute_kl_policies_flat(p1: Policy, p2: Policy) -> float:
    """Computes the KL divergence between two policies. KL(p1 || p2)"""
    kl_div = 0
    for state in p1:
        actions = p1[state].dist.keys()
        kl_div += kl_divergence(
            np.array([p1[state].dist[a] for a in actions]),
            np.array([p2[state].dist[a] for a in actions]),
        )
    return kl_div


def boltzmann_incoherence_flat(mdp: MDP, policy: Policy, temperature: float) -> float:
    """Computes Boltzmann incoherence using KL divergence."""
    _, Q = compute_Vs_Qs(mdp, policy)
    br_policy = boltzmann_rational_policy(Q, temperature)
    return compute_kl_policies_flat(policy, br_policy)


def boltzmann_incoherence_causal(mdp: MDP, policy: Policy, temperature: float) -> float:
    _, Q = compute_Vs_Qs(mdp, policy)
    br_policy = boltzmann_rational_policy(Q, temperature)
    return compute_causal_forward_kl_divergence(mdp, policy, br_policy)


def converge_boltzmann_coherence(mdp: MDP, policy: Policy, temperature: float) -> Policy:
    iterations = mdp.time_horizon
    for _ in range(iterations):
        _, Q = compute_Vs_Qs(mdp, policy)
        br_policy = boltzmann_rational_policy(Q, temperature)
        policy = br_policy
    return policy


def iterated_boltzmann_coherence_flat(
    mdp: MDP, policy: Policy, temperature: float, num_iterations: int
):
    """Computes the Boltzmann coherence over multiple iterations with flat KL."""
    coherence = []
    for _ in range(num_iterations):
        _, Q = compute_Vs_Qs(mdp, policy)
        br_policy = boltzmann_rational_policy(Q, temperature)
        coherence.append(compute_kl_policies_flat(policy, br_policy))
        policy = br_policy
    return coherence


def iterated_boltzmann_coherence_causal(
    mdp: MDP, policy: Policy, temperature: float, num_iterations: int
):
    """Computes the Boltzmann coherence over multiple iterations with causal KL."""
    coherence = []
    for _ in range(num_iterations):
        _, Q = compute_Vs_Qs(mdp, policy)
        br_policy = boltzmann_rational_policy(Q, temperature)
        coherence.append(compute_causal_forward_kl_divergence(mdp, policy, br_policy))
        policy = br_policy
    return coherence


def iterated_boltzmann_coherence_causal_belousov(
    mdp: MDP, policy: Policy, temperature: float, num_iterations: int
):
    """Computes the Boltzmann coherence over multiple iterations using alternative definition."""
    coherence = []
    for _ in range(num_iterations):
        _, Q = compute_Vs_Qs(mdp, policy)
        br_policy = boltzmann_rational_policy(Q, temperature)
        coherence.append(compute_causal_forward_kl_divergence_belousov(mdp, policy, br_policy))
        policy = br_policy
    return coherence


def policy_improvement(mdp: MDP, Q):
    """Generates a new policy using the current value function."""
    new_policy = {}
    for state in mdp.states:
        best_action = max(list(Q[state].keys()), key=Q[state].get)
        new_policy[state] = P(
            {action: 1 if action == best_action else 0 for action in mdp.actions[state]}
        )
    return new_policy


def policy_iteration(mdp):
    """Repeatedly evaluates and improves a policy until it converges to the optimal policy."""
    policy = make_uniform_policy(mdp)
    while True:
        _, Q = compute_Vs_Qs(mdp, policy, soft=False)
        new_policy = policy_improvement(mdp, Q)
        if new_policy == policy:
            break
        policy = new_policy
    return policy


__all__ = [
    "boltzmann_incoherence_causal",
    "boltzmann_incoherence_flat",
    "boltzmann_rational_policy",
    "compute_Vs_Qs",
    "compute_kl_policies_flat",
    "converge_boltzmann_coherence",
    "iterated_boltzmann_coherence_causal",
    "iterated_boltzmann_coherence_causal_belousov",
    "iterated_boltzmann_coherence_flat",
    "kl_divergence",
    "policy_improvement",
    "policy_iteration",
    "print_Q",
    "print_V",
    "soft_Q",
    "soft_V",
    "soft_exp",
    "soft_log",
]
