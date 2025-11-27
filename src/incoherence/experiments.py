from dataclasses import dataclass
from functools import cached_property
from typing import Dict
import numpy as np  # type: ignore
from matplotlib import pyplot as plt
from .mdp import MDP, Policy, compute_J, compute_marginals, compute_prob_over_trajectories, create_random_mdp, make_uniform_policy, posterior_cond_R, sample_trajectory, stochastify
from .metrics import compute_J_causal_entropy, compute_J_entropy, compute_causal_forward_kl_divergence, compute_causal_forward_kl_divergence_belousov
from .policy import boltzmann_incoherence_causal, boltzmann_incoherence_flat, boltzmann_rational_policy, compute_Vs_Qs, converge_boltzmann_coherence, iterated_boltzmann_coherence_causal, iterated_boltzmann_coherence_causal_belousov, iterated_boltzmann_coherence_flat, policy_iteration, print_Q, print_V
from .scenarios import create_counterexample, create_two_cards_game
from .training import fold_posterior_into_reward, increase_temp, retrain_agent
from .utils import print_js, print_policy, print_trajectory, print_trajectory_prob

def run_compute_prob_over_trajectories():
    mdp = create_two_cards_game()
    policy = make_uniform_policy(mdp)
    prob = compute_prob_over_trajectories(mdp, policy)
    print('Initial dist')
    print_trajectory_prob(prob)
    print()
    for i in range(3):
        prob = posterior_cond_R(prob, R=1)
        print('Posterior dist ', i)
        print_trajectory_prob(prob)
        policy = compute_marginals(mdp, prob)
        print_policy(mdp, policy)
        print()

def run_qv_diagnostics():
    mdp = create_two_cards_game()
    policy = make_uniform_policy(mdp)
    print('---------- Trajectories -----------')
    for _ in range(8):
        print_trajectory(sample_trajectory(mdp, policy))
    V, Q = compute_Vs_Qs(mdp, policy)
    print('---------- Q values -----------')
    print_Q(Q)
    print('---------- V values -----------')
    print_V(V)
    print('B-incoherence flat', boltzmann_incoherence_flat(mdp, policy, temperature=1.0))
    print('B-incoherence caus', boltzmann_incoherence_causal(mdp, policy, temperature=1.0))
    print('J', compute_J(mdp, policy))
    print('Iterated Boltzmann policy flat', iterated_boltzmann_coherence_flat(mdp, policy, temperature=1.0, num_iterations=3))
    print('Iterated Boltzmann policy causal', iterated_boltzmann_coherence_causal(mdp, policy, temperature=1.0, num_iterations=3))
    print('Iterated Boltzmann policy causal belousov', iterated_boltzmann_coherence_causal_belousov(mdp, policy, temperature=1.0, num_iterations=3))
    new_policy = policy_iteration(mdp)
    print('---- Optimal policy ----')
    print_policy(mdp, new_policy)
    print('J', compute_J(mdp, new_policy))

TEMP_INCOH: float = 0.8

@dataclass
class Result:
    name: str
    data: dict
    policy: Policy
    mdp: MDP

    @staticmethod
    def new_result(name: str, mdp: MDP, policy: Policy, **data):
        return Result(name=name, data=data, policy=policy, mdp=mdp)

    @cached_property
    def j(self) -> float:
        return compute_J(self.mdp, self.policy)

    @cached_property
    def j_entropy(self) -> float:
        return compute_J_entropy(self.mdp, self.policy)

    @cached_property
    def j_causal_entropy(self) -> float:
        return compute_J_causal_entropy(self.mdp, self.policy)

    def incoherence_flat(self, temp: float) -> float:
        return boltzmann_incoherence_flat(self.mdp, self.policy, temp)

    def incoherence_causal(self, temp: float) -> float:
        return boltzmann_incoherence_causal(self.mdp, self.policy, temp)

    def __eq__(self, other):
        return self.policy == other.policy and self.mdp == other.mdp

def run_retrain(n, mdp: MDP) -> list[Result]:
    policy = make_uniform_policy(mdp)
    results = []
    results.append(Result.new_result(name='retrain', step=0, mdp=mdp, policy=policy))
    retrained_policy = policy
    for k in range(n):
        retrained_policy = retrain_agent(mdp, retrained_policy)
        results.append(Result.new_result(name='retrain', step=k + 1, mdp=mdp, policy=retrained_policy))
    return results

def run_boltzmann(n):
    mdp = create_two_cards_game()
    policy = make_uniform_policy(mdp)
    js = []
    js.append(compute_J(mdp, policy))
    temps = [1 / 2 ** k for k in range(n)]
    for temp in temps:
        retrained_policy = converge_boltzmann_coherence(mdp, policy, temperature=temp)
        js.append(compute_J(mdp, retrained_policy))
    print_js(js)

def run_increase_temp(n, mdp: MDP, schedule=None):
    orig_mdp = mdp
    uniform = make_uniform_policy(mdp)
    results = []
    results.append(Result.new_result(name='temp', temp=np.inf, mdp=mdp, policy=uniform))
    schedule = (lambda n: range(1, n + 1)) if schedule is None else schedule
    for temp in schedule(n):
        mdp = increase_temp(orig_mdp, temp)
        retrained_policy = retrain_agent(mdp, uniform)
        results.append(Result.new_result(name='temp', temp=temp, mdp=orig_mdp, policy=retrained_policy))
    return results

def run_fold_posterior_into_reward_and_policy(n, mdp: MDP):
    orig_mdp = mdp
    policy = make_uniform_policy(mdp)
    js = []
    js.append(compute_J(orig_mdp, policy))
    for i in range(n):
        mdp, policy = fold_posterior_into_reward(mdp, mdp, policy)
        js.append(compute_J(orig_mdp, policy))
    print_js(js)

def run_fold_posterior_into_reward(n, mdp: MDP):
    orig_mdp = mdp
    uniform = make_uniform_policy(mdp)
    results = []
    results.append(Result.new_result(name='fold', step=0, mdp=orig_mdp, policy=uniform))
    for i in range(n):
        mdp, policy = fold_posterior_into_reward(mdp, mdp, uniform)
        results.append(Result.new_result(name='fold', step=i, mdp=orig_mdp, policy=policy))
    return results

def run_fold_posterior_into_reward_orig_mdp(n, mdp: MDP):
    orig_mdp = mdp
    uniform = make_uniform_policy(mdp)
    results = []
    results.append(Result.new_result(name='fold_orig', step=0, mdp=orig_mdp, policy=uniform))
    for i in range(n):
        prob = compute_prob_over_trajectories(mdp, uniform)
        posterior = posterior_cond_R(prob, R=1)
        policy = compute_marginals(orig_mdp, posterior)
        results.append(Result.new_result(name='fold_orig', step=i + 1, mdp=orig_mdp, policy=policy))
        mdp, _ = fold_posterior_into_reward(orig_mdp, mdp, uniform)
    return results

def print_results(results: Dict[str, list[Result]], *, axs=None, temp=0.15, alpha=0.3):
    colors = {'retrain': 'blue', 'temp': 'orange', 'fold_orig': 'green'}
    markers = {'retrain': '.', 'temp': '+', 'fold_orig': 'v'}
    results = {k: v for k, v in results.items() if k in colors}
    for name, res in results.items():
        print(f'J   {name:<10} {[round(float(r.j), 8) for r in res]}')
    print()
    if axs is None:
        fig, axs = plt.subplots(2, 3)
    for name, res in results.items():
        data = [round(float(r.j), 8) for r in res]
        axs[0, 0].plot(data, label=name, c=colors[name], linewidth=2, alpha=alpha, marker=markers[name], markersize=10)
    axs[0, 0].set_title('J')
    for name, res in results.items():
        data = [round(float(r.incoherence_flat(temp)), 8) for r in res]
        axs[0, 1].plot(data, label=name, c=colors[name], linewidth=2, alpha=alpha, marker=markers[name], markersize=10)
    axs[0, 1].set_title('Incoherence flat')
    for name, res in results.items():
        data = [round(float(r.incoherence_causal(temp)), 8) for r in res]
        axs[0, 2].plot(data, label=name, c=colors[name], linewidth=2, alpha=alpha, marker=markers[name], markersize=10)
    axs[0, 2].set_title('Incoherence causal')
    for name, res in results.items():
        data = [round(float(r.j_entropy), 8) for r in res]
        axs[1, 0].plot(data, label=name, c=colors[name], linewidth=2, alpha=alpha, marker=markers[name], markersize=10)
    axs[1, 0].set_title('J entropy')
    for name, res in results.items():
        data = [round(float(r.j_causal_entropy), 8) for r in res]
        axs[1, 1].plot(data, label=name, c=colors[name], linewidth=2, alpha=alpha, marker=markers[name], markersize=10)
    axs[1, 1].set_title('J causal entropy')

def run_mdp_suite(mdp: MDP, k: int):
    results = {}
    results['retrain'] = run_retrain(k, mdp=mdp)
    results['temp'] = run_increase_temp(k, mdp=mdp)
    results['fold_orig'] = run_fold_posterior_into_reward_orig_mdp(k, mdp=mdp)
    results['temp_exp'] = run_increase_temp(k, mdp=mdp, schedule=lambda n: [2 ** i for i in range(0, n)])
    results['fold'] = run_fold_posterior_into_reward(k, mdp=mdp)
    print_results(results)

def run_two_cards_suite(k):
    print('Two cards')
    mdp = create_two_cards_game()
    run_mdp_suite(mdp, k)
    print('Test two cards OK')

def run_two_cards_stoch_suite(k):
    print('Two cards stoch 5')
    mdp = create_two_cards_game()
    mdp = stochastify(mdp)
    run_mdp_suite(mdp, k)

def run_random_det_transitions(k, T=3, A=4):
    print('Random det transition')
    mdp = create_random_mdp(A, T, deterministic_transitions=True)
    run_mdp_suite(mdp, k)

def run_random_stoch_transitions(k, T=3, A=4):
    print('Random stoch trans')
    mdp = create_random_mdp(A, T, deterministic_transitions=True)
    mdp = stochastify(mdp)
    run_mdp_suite(mdp, k)

def run_three_state_counterexample():
    mdp = create_counterexample()
    mdp = create_random_mdp(2, 1, deterministic_transitions=False)
    k = 2
    results = {}
    results['retrain'] = run_retrain(k, mdp=mdp)
    results['temp'] = run_increase_temp(k, mdp=mdp)
    print('Retrain')
    print_policy(mdp, results['retrain'][-1].policy)
    print('Temp')
    print_policy(mdp, results['temp'][-1].policy)

def run_incoherence_limit():
    mdp = create_random_mdp(2, 4, True)
    k = 8
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    results = {}
    results['retrain'] = run_retrain(k, mdp=mdp)
    results['temp'] = run_increase_temp(k, mdp=mdp)
    results['fold_orig'] = run_fold_posterior_into_reward_orig_mdp(k, mdp=mdp)
    for temp in [0.6, 0.4, 0.2, 0.1]:
        print_results(results, temp=temp, axs=axs, alpha=temp)

def get_j(results: list[Result]) -> np.ndarray:
    return np.array([results[i].j for i in range(len(results))])

__all__ = ['TEMP_INCOH', 'Result', 'get_j', 'print_results', 'run_qv_diagnostics', 'run_boltzmann', 'run_compute_prob_over_trajectories', 'run_fold_posterior_into_reward', 'run_fold_posterior_into_reward_and_policy', 'run_fold_posterior_into_reward_orig_mdp', 'run_incoherence_limit', 'run_increase_temp', 'run_mdp_suite', 'run_random_det_transitions', 'run_random_stoch_transitions', 'run_retrain', 'run_three_state_counterexample', 'run_two_cards_suite', 'run_two_cards_stoch_suite']
