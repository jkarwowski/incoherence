import numpy as np  # type: ignore

from .distributions import P
from .mdp import MDP, Policy, compute_prob_over_trajectories, occupancy_measure_states_time, log_reward, compute_J


def H(p: P):
    """Computes the entropy of a probability distribution."""
    return -p.expectation(lambda x: np.log(p.dist[x]))


def forward_kl_divergence_P(p: P, q: P) -> float:
    """Computes the KL divergence D(p || q) for two probability distributions."""
    if any(p.dist[k] > 0 and q.dist[k] == 0 for k in p.dist):
        return float("inf")
    return sum(
        (p.dist[k] * np.log(p.dist[k] / q.dist[k]))
        for k in p.dist.keys()
        if p.dist[k] != 0
    )


def compute_causal_entropy(mdp: MDP, policy: Policy) -> float:
    # causal entropy (from MCE Primer, Gleave et al)
    occ_measures = occupancy_measure_states_time(mdp, policy)
    entropy_per_time = {
        t: prob.expectation(lambda state: H(policy[state]))
        for t, prob in occ_measures.items()
    }
    return sum(entropy_per_time.values())


def compute_causal_forward_kl_divergence_belousov(mdp, policy1, policy2) -> float:
    # causal forward KL divergence of KL(p1 || p2)
    occ_measures = occupancy_measure_states_time(mdp, policy1)
    kl_per_time = {
        t: prob.expectation(lambda state: forward_kl_divergence_P(policy1[state], policy2[state]))
        for t, prob in occ_measures.items()
    }
    return sum(kl_per_time.values())


def compute_causal_forward_kl_divergence(mdp, policy1, policy2) -> float:
    # The occupancy identity avoids underflow in full-trajectory probabilities.
    return compute_causal_forward_kl_divergence_belousov(mdp, policy1, policy2)


def compute_J_entropy(mdp: MDP, policy: Policy) -> float:
    # computes J normalised with entropy
    # that is J = E_pi[sum_t r(s_t, a_t) + H(pi(a_t|s_t))]
    # where H is the entropy of the policy in the given state
    prob = compute_prob_over_trajectories(mdp, policy)
    return prob.expectation(
        lambda traj: sum(log_reward(mdp, state, action) + H(policy[state]) for state, action, _ in traj)
    )


def compute_J_causal_entropy(mdp: MDP, policy: Policy) -> float:
    # computes J normalised with entropy
    # that is J = E_pi[sum_t r(s_t, a_t) + H(pi(a_t|s_t))]
    # where H is the entropy of the policy in the given state
    return compute_J(mdp, policy) + compute_causal_entropy(mdp, policy)


__all__ = [
    "H",
    "compute_J_causal_entropy",
    "compute_J_entropy",
    "compute_causal_entropy",
    "compute_causal_forward_kl_divergence",
    "compute_causal_forward_kl_divergence_belousov",
    "forward_kl_divergence_P",
]
