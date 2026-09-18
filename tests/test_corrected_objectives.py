import math

import numpy as np
import pytest

from incoherence.distributions import P, bernoulli
from incoherence.mdp import (
    MDP, compute_J, compute_J_MC, compute_success_probability,
    create_random_mdp, make_uniform_policy,
    compute_marginals, compute_prob_over_trajectories, posterior_cond_R,
)
from incoherence.metrics import forward_kl_divergence_P, compute_causal_forward_kl_divergence
from incoherence.training import fold_posterior_into_reward, increase_temp, retrain_agent


def one_step(q=(0.4, 0.9)):
    return MDP(
        ["s", "end"], {"s": ["A", "B"], "end": []},
        {"s": {a: P({"end": 1}) for a in ["A", "B"]}, "end": {}},
        {"s": {a: bernoulli(v) for a, v in zip(["A", "B"], q)}, "end": {}},
        {"s": 0, "end": 1}, 1, "s",
    )


def test_return_is_log_reward_not_bernoulli_mean():
    mdp = one_step((0.5, 0.5))
    pi = make_uniform_policy(mdp)
    assert compute_J(mdp, pi) == pytest.approx(math.log(0.5))
    assert compute_J_MC(mdp, pi, num_samples=10) == pytest.approx(math.log(0.5))
    assert compute_success_probability(mdp, pi) == pytest.approx(0.5)


def test_stochastic_success_improvement_does_not_imply_return_improvement():
    mdp = MDP(
        ["s", "a", "good", "bad", "end"],
        {"s": ["A", "B"], "a": ["done"], "good": ["done"], "bad": ["done"], "end": []},
        {"s": {"A": P({"a": 1}), "B": P({"good": 0.6, "bad": 0.4})},
         **{s: {"done": P({"end": 1})} for s in ["a", "good", "bad"]}, "end": {}},
        {"s": {a: bernoulli(1) for a in ["A", "B"]},
         **{s: {"done": bernoulli(q)} for s, q in [("a", 0.5), ("good", 1), ("bad", 0.01)]}, "end": {}},
        {"s": 0, "a": 1, "good": 1, "bad": 1, "end": 2}, 2, "s",
    )
    pi = make_uniform_policy(mdp)
    updated = retrain_agent(mdp, pi)
    assert compute_J(mdp, updated) < compute_J(mdp, pi)
    assert compute_success_probability(mdp, updated) > compute_success_probability(mdp, pi)


def test_fold_preserves_nonuniform_prior_when_reward_is_neutral():
    mdp = one_step((1, 1))
    prior = {"s": P({"A": 0.8, "B": 0.2}), "end": P({})}
    current = mdp
    for _ in range(3):
        current, pi = fold_posterior_into_reward(mdp, current, prior)
        assert pi["s"].dist["A"] == pytest.approx(0.8)


def test_ratio_above_one_is_normalized_without_changing_policy():
    mdp = one_step()
    prior = {"s": P({"A": 0.8, "B": 0.2}), "end": P({})}
    current, first = fold_posterior_into_reward(mdp, mdp, prior)
    assert first["s"].dist["B"] == pytest.approx(0.36)
    # Raw q*pi/p is (0.32,1.62), so a common scale is necessary.
    assert current.rewards["s"]["A"].dist[1] == pytest.approx(0.32 / 1.62)
    assert current.rewards["s"]["B"].dist[1] == pytest.approx(1)
    _, second = fold_posterior_into_reward(mdp, current, prior)
    assert second["s"].dist["A"] == pytest.approx(retrain_agent(mdp, first)["s"].dist["A"])


def test_fold_preserves_zero_prior_support():
    mdp = one_step()
    prior = {"s": P({"A": 1, "B": 0}), "end": P({})}
    current, _ = fold_posterior_into_reward(mdp, mdp, prior)
    _, pi = fold_posterior_into_reward(mdp, current, prior)
    assert pi["s"].dist["A"] == 1
    assert pi["s"].dist["B"] == 0


@pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_full_policy_equivalences_for_nonuniform_prior(deterministic, seed):
    mdp = create_random_mdp(2, 2, deterministic, seed=seed)
    prior = {s: P({a: 0.8 if i == 0 else 0.2 for i, a in enumerate(mdp.actions[s])})
             for s in mdp.states}
    current = mdp
    pi_g = prior
    for _ in range(4):
        current, pi_f = fold_posterior_into_reward(mdp, current, prior)
        pi_g = retrain_agent(mdp, pi_g)
        for s in mdp.states:
            for a in mdp.actions[s]:
                assert pi_f[s].dist[a] == pytest.approx(pi_g[s].dist[a], abs=1e-12)
    if deterministic:
        current = mdp
        for exponent in [1, 2, 4, 8]:
            current, pi_h = fold_posterior_into_reward(current, current, prior)
            pi_temp = retrain_agent(increase_temp(mdp, exponent), prior)
            for s in mdp.states:
                for a in mdp.actions[s]:
                    assert pi_h[s].dist[a] == pytest.approx(pi_temp[s].dist[a], abs=1e-12)


def test_forward_kl_to_deterministic_limit_is_infinite():
    assert math.isinf(forward_kl_divergence_P(P({0: 0.99, 1: 0.01}), P({0: 1, 1: 0})))


def test_small_positive_kl_denominator_is_not_clipped():
    actual = forward_kl_divergence_P(P({0: 0.5, 1: 0.5}), P({0: 1, 1: 1e-30}))
    assert actual == pytest.approx(15 * math.log(10) - math.log(2))


def test_causal_kl_matches_full_trajectory_kl():
    mdp = create_random_mdp(2, 2, False, seed=8)
    prior = make_uniform_policy(mdp)
    updated = retrain_agent(mdp, prior)
    direct = forward_kl_divergence_P(
        compute_prob_over_trajectories(mdp, prior),
        compute_prob_over_trajectories(mdp, updated),
    )
    assert compute_causal_forward_kl_divergence(mdp, prior, updated) == pytest.approx(direct, abs=1e-12)


def test_future_conditioning_at_unvisited_states():
    mdp = create_random_mdp(2, 2, True, seed=0)
    prior = {s: P({a: 1.0 if i == 0 else 0.0 for i, a in enumerate(mdp.actions[s])})
             for s in mdp.states}
    updated = retrain_agent(mdp, prior)
    for s in mdp.states:
        for a in mdp.actions[s]:
            assert updated[s].dist[a] == prior[s].dist[a]
    # State "11" is unvisited, but its own future conditional is well-defined.
    prior["11"] = P({"0": 0.8, "1": 0.2})
    updated = retrain_agent(mdp, prior)
    q0, q1 = (mdp.rewards["11"][a].dist[1] for a in ["0", "1"])
    assert updated["11"].dist["0"] == pytest.approx(0.8 * q0 / (0.8 * q0 + 0.2 * q1))


@pytest.mark.parametrize("deterministic", [False, True])
def test_conditioning_matches_trajectory_enumeration(deterministic):
    mdp = create_random_mdp(2, 2, deterministic, seed=7)
    prior = make_uniform_policy(mdp)
    expected = compute_marginals(mdp, posterior_cond_R(compute_prob_over_trajectories(mdp, prior), 1))
    actual = retrain_agent(mdp, prior)
    for s in mdp.states:
        for a in mdp.actions[s]:
            assert actual[s].dist[a] == pytest.approx(expected[s].dist[a], abs=1e-12)
