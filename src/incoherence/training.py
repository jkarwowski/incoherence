from collections import defaultdict
from dataclasses import replace
from math import prod

import numpy as np  # type: ignore

from .distributions import P, bernoulli
from .mdp import MDP, Policy, Rewards, compute_marginals, compute_prob_over_trajectories, posterior_cond_R, sample_trajectory, make_uniform_policy

RETRAIN_STEPS = 100


def retrain_agent_MC(
    mdp: MDP, policy: Policy, *, retrain_mc_steps=RETRAIN_STEPS
):
    """Collects roll-outs, filters them based on rewards, and updates the policy."""
    filtered_actions = {state: [] for state in mdp.states}
    sampled_trajectories = [
        sample_trajectory(mdp, policy) for _ in range(retrain_mc_steps)
    ]
    for trajectory in sampled_trajectories:
        reward = prod(reward for _, _, reward in trajectory)
        if reward == 1:
            for state, action, _ in trajectory:
                filtered_actions[state].append(action)

    # Update policy based on filtered actions
    new_policy = {
        state: {action: 0.0 for action in mdp.actions[state]} for state in mdp.states
    }
    for state in filtered_actions:
        if filtered_actions[state]:
            total = len(filtered_actions[state])
            for action in filtered_actions[state]:
                new_policy[state][action] += 1.0 / total
        else:
            for action in mdp.actions[state]:
                # Default to uniform distribution if no rewards
                new_policy[state][action] = 1.0 / len(mdp.actions[state])

    return {state: P(actions) for state, actions in new_policy.items()}
    # , make_distribution_from_empirical(sampled_trajectories)


def retrain_agent(mdp: MDP, policy: Policy):
    """Collects roll-outs, filters them based on rewards, and updates the policy."""
    trajectories = compute_prob_over_trajectories(mdp, policy)
    filtered_trajectories = posterior_cond_R(trajectories, R=1)
    conditioned_policy = compute_marginals(mdp, filtered_trajectories)

    # For states absent from successful rollouts, condition on rollouts starting there.
    visited_states = {state for traj in filtered_trajectories.dist for state, _, _ in traj}
    for state in mdp.states:
        if mdp.actions[state] and state not in visited_states:
            suffix_mdp = replace(
                mdp, initial_state=state,
                time_horizon=mdp.time_horizon - mdp.state_time[state],
            )
            trajectories = compute_prob_over_trajectories(suffix_mdp, policy)
            filtered_trajectories = posterior_cond_R(trajectories, R=1)
            # If success is impossible, preserve the prior as a convention.
            conditioned_policy[state] = (
                compute_marginals(suffix_mdp, filtered_trajectories)[state]
                if filtered_trajectories.dist else policy[state]
            )
    return conditioned_policy


def iterate_G(mdp: MDP, policy: Policy, iterations: int) -> list[Policy]:
    """Convenience helper that applies the control-as-inference operator repeatedly."""
    history = [policy]
    current = policy
    for _ in range(iterations):
        current = retrain_agent(mdp, current)
        history.append(current)
    return history


def retrain_agent_filter(mdp: MDP, policy: Policy):
    """Collects roll-outs, filters them based on rewards, and updates the policy."""
    trajectories = compute_prob_over_trajectories(mdp, policy)
    filtered_trajectories = posterior_cond_R(trajectories, R=1)
    new_policy = compute_marginals(mdp, filtered_trajectories)
    return new_policy


def increase_temp(mdp: MDP, alpha=1.0) -> MDP:
    rewards = {
        state: {
            action: bernoulli(np.power(prob.dist[1], alpha))
            for action, prob in actions_rewards.items()
        }
        for state, actions_rewards in mdp.rewards.items()
    }
    return MDP(
        mdp.states,
        mdp.actions,
        mdp.transitions,
        rewards,
        mdp.state_time,
        mdp.time_horizon,
        mdp.initial_state,
    )


def fold_policy_into_reward(mdp: MDP, policy: Policy, prior: Policy | None = None) -> Rewards:
    """Fold log(policy/prior), normalizing by a common factor at each time."""
    if prior is None:
        prior = make_uniform_policy(mdp)

    # Adding log(policy/prior) to r means multiplying q = exp(r) by policy/prior.
    probabilities = defaultdict(dict)
    for state in mdp.states:
        for action in mdp.actions[state]:
            if prior[state].dist[action] == 0:
                probabilities[state][action] = 0.0
            else:
                probabilities[state][action] = (
                    mdp.rewards[state][action].dist[1]
                    * policy[state].dist[action] / prior[state].dist[action]
                )

    # One common scale per time step keeps probabilities <= 1 without changing conditioning.
    new_rewards = defaultdict(lambda: dict())
    for states in mdp.time_to_state().values():
        normalizer = max([1.0] + [q for state in states for q in probabilities[state].values()])
        for state in states:
            for action in mdp.actions[state]:
                new_rewards[state][action] = bernoulli(probabilities[state][action] / normalizer)
    return new_rewards


def fold_posterior_into_reward(
    orig_mdp: MDP, mdp: MDP, policy: Policy
):
    marginals = retrain_agent(mdp, policy)

    return MDP(
        orig_mdp.states,
        orig_mdp.actions,
        orig_mdp.transitions,
        fold_policy_into_reward(orig_mdp, marginals, policy),
        orig_mdp.state_time,
        orig_mdp.time_horizon,
        orig_mdp.initial_state,
    ), marginals


__all__ = [
    "RETRAIN_STEPS",
    "fold_policy_into_reward",
    "fold_posterior_into_reward",
    "increase_temp",
    "retrain_agent",
    "retrain_agent_MC",
    "retrain_agent_filter",
]
