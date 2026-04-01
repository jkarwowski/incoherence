from collections import defaultdict
from dataclasses import dataclass
from math import prod
from typing import Dict, Generic, List, Mapping, Tuple, TypeVar

import numpy as np  # type: ignore

from .distributions import Bool, P, bernoulli

State = TypeVar("State")
Action = TypeVar("Action")
Reward = int
Rewards = Dict[State, Dict[Action, P[Bool]]]
Policy = Dict[State, P[Action]]
Trajectory = Tuple[Tuple[State, Action, Reward]]

J_SAMPLES = 100  # 100000


def _invert_dict(d):
    inv_d = defaultdict(list)
    for k, v in d.items():
        inv_d[v].append(k)
    return dict(inv_d)


@dataclass
class MDP(Generic[State, Action]):
    states: List[State]
    actions: Dict[State, List[Action]]  # Actions available for each state
    transitions: Dict[State, Dict[Action, P[State]]]
    rewards: Rewards
    state_time: Dict[State, int]
    time_horizon: int  # Time horizon for the MDP
    initial_state: State

    def time_to_state(self) -> Mapping[int, List[State]]:
        return _invert_dict(self.state_time)


def sample_trajectory(mdp: MDP, policy: Policy) -> Trajectory:
    """Generates a trajectory following a policy"""
    state = mdp.initial_state
    trajectory = []
    for _ in range(mdp.time_horizon):
        action = policy[state].sample()
        reward = mdp.rewards[state][action].sample()
        trajectory.append((state, action, reward))
        next_state = mdp.transitions[state][action].sample()
        state = next_state
    return tuple(trajectory)


def compute_J_MC(mdp: MDP, policy: Policy, num_samples=J_SAMPLES) -> float:
    """Computes J by sampling trajectories"""
    total_reward = 0
    for _ in range(num_samples):
        trajectory = sample_trajectory(mdp, policy)
        total_reward += sum(reward for _, _, reward in trajectory)
    return total_reward / num_samples


def compute_J(mdp: MDP, policy: Policy) -> float:
    """Computes J analytically"""
    trajectory_dist = compute_prob_over_trajectories(mdp, policy)
    return trajectory_dist.expectation(
        lambda traj: sum(reward for _, _, reward in traj)
    )


def _extend_trajectory(mdp: MDP, policy: Policy, state: State, prob: float): # type: ignore
    extensions = []
    for action, action_prob in policy[state].dist.items():
        for reward, reward_prob in mdp.rewards[state][action].dist.items():
            for next_state, transition_prob in mdp.transitions[state][action].dist.items():
                p = prob * action_prob * reward_prob * transition_prob
                if p > 0:
                    extensions.append((p, (state, action, reward), next_state))
    return extensions


def compute_prob_over_trajectories(mdp: MDP, policy: Policy) -> P[Trajectory]:
    """Computes the posterior over trajectories given a policy and an initial state"""

    stack = [(1.0, [], mdp.initial_state)]
    trajectories = []
    while stack:
        prob, trajectory, state = stack.pop()
        if len(trajectory) == mdp.time_horizon:
            trajectories.append((prob, trajectory))
        else:
            stack += [
                (new_prob, trajectory + [ext], new_state)
                for new_prob, ext, new_state in _extend_trajectory(
                    mdp, policy, state, prob
                )
            ]
    return P({tuple(traj): prob for prob, traj in trajectories})


def posterior_cond_R(trajectories: P[Trajectory], R: float) -> P[Trajectory]:
    """Computes the posterior over trajectories given a reward R"""
    return P(
        {
            traj: prob
            for traj, prob in trajectories.dist.items()
            if prod(reward for _, _, reward in traj) == R
        }
    )


def compute_marginals(mdp: MDP, trajectories: P[Trajectory]) -> Dict[State, P[Action]]: # type: ignore
    """Computes the marginal distribution over states and actions"""
    marginals = defaultdict(lambda: defaultdict(lambda: float()))
    for traj, prob in trajectories.dist.items():
        for state, action, _ in traj:
            marginals[state][action] += prob

    for state in mdp.states:
        if state not in marginals:
            marginals[state] = {action: 1.0 for action in mdp.actions[state]}  # type: ignore
    return {state: P(action_prob) for state, action_prob in marginals.items()}


def make_uniform_policy(mdp: MDP[State, Action]) -> Policy[State, Action]:
    """Creates a uniform policy for the given MDP."""
    return {
        state: P({action: 1 for action in actions})
        for state, actions in mdp.actions.items()
    }


def occupancy_measure_states_time(mdp: MDP[State, Action], policy: Policy) -> Mapping[int, P[State]]:
    # occupancy measure disaggregated over time
    # i.e. d_\pi^t(s_t) = prob of ending in state s_t at time t following policy pi
    occ_measures = {t: {s: 0.0 for s in states} for t, states in mdp.time_to_state().items()}
    prob_traj = compute_prob_over_trajectories(mdp, policy)
    for traj, p in prob_traj.dist.items():
        for state, _, _ in traj:
            occ_measures[mdp.state_time[state]][state] += p
    occ_measures = {t: P(v) for t, v in occ_measures.items() if t < mdp.time_horizon}
    return occ_measures


def stochastify(mdp: MDP):
    time_to_states = mdp.time_to_state()
    for time, states in time_to_states.items():
        if time == mdp.time_horizon:
            continue
        for state in states:
            for action in mdp.actions[state]:
                mdp.transitions[state][action] = P({s2: np.random.rand() for s2 in time_to_states[time + 1]})
                # mdp.transitions[state][action] = P({'a': 1})
    return mdp


def DFS(T, num_actions):
    if T == 0:
        return []
    return [f"{a}{s}" for s in (DFS(T - 1, num_actions) + [""]) for a in range(num_actions)]


def num_to_state(num, actions_num):
    # recursively build the state string from the number
    if num == 0:
        return ""
    return num_to_state(num // actions_num, actions_num) + str(num % actions_num)


def state_to_num(state, actions_num):
    # recursively build the number from the state string
    if state == "":
        return 0
    return actions_num * state_to_num(state[:-1], actions_num) + int(state[-1])


def test_state_num(actions_num=4):
    for i in range(10):
        state = num_to_state(i, actions_num)
        num = state_to_num(state, actions_num)
        print(i, state, num, num == i)
        assert num == i


def create_random_mdp(actions_num: int, time_horizon: int, deterministic_transitions: bool, seed: None | int = None):
    # sample the transition dynamics and rewards randomly
    # states are strings in a form f"{a_0}{a_1}, ..., {a_t}" where t < time_horizon
    # for example, states for 2-action, 2-time horizon MDP are "0", "1", "00", "01", "10", "11"
    if seed is not None:
        np.random.seed(seed)
    
    states = DFS(time_horizon, actions_num) + [""]
    state_time = {
        state: len(state) for state in states
    }

    # Very hacky
    def _next_state(state: str, action: str, horizon: int) -> str:
        if len(state + action) > horizon:
            return 'end'
        else:
            return state + action

    actions = {state: list(map(str, range(actions_num))) for state in states}
    if deterministic_transitions:
        # transitions are deterministic
        transitions = {
            state: {
                action: P({_next_state(state, action, time_horizon): 1})  # _next_state(state, action, time_horizon)
                for action in actions[state]
            }
            for state in states
        }
    else:
        # transitions probabitilies are sampled randomly
        transitions = {
            state: {
                action: P({_next_state(state, a, time_horizon): np.random.rand() for a in actions[state]})  # a distribution over the next available states
                for action in actions[state]
            }
            for state in states
        }

    rewards = {
        state: {action: bernoulli(np.random.rand()) for action in actions[state]}
        for state in states
    }

    states.append('end')
    state_time['end'] = time_horizon + 1
    actions['end'] = []
    transitions['end'] = {}
    rewards['end'] = {}

    return MDP(states, actions, transitions, rewards, state_time, time_horizon + 1, "")


__all__ = [
    "Action",
    "MDP",
    "Policy",
    "Rewards",
    "State",
    "Trajectory",
    "compute_J",
    "compute_J_MC",
    "compute_marginals",
    "compute_prob_over_trajectories",
    "create_random_mdp",
    "make_uniform_policy",
    "occupancy_measure_states_time",
    "posterior_cond_R",
    "sample_trajectory",
    "stochastify",
    "test_state_num",
]
