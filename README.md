# Incoherence in goal-conditioned autoregressive models

This is an implementation and numerical verification of the main results in the paper "Incoherence in goal-conditioned autoregressive models", Jacek Karwowski and Raymond Douglas, 2025.

## Objective conventions

Stored Bernoulli variables are auxiliary success indicators with mean `q`. The paper's reward is `r = log(q)`, and `compute_J` reports expected additive log return. `compute_success_probability` separately reports the probability that every auxiliary indicator is one. Re-conditioning improves the latter for arbitrary dynamics; additive-return improvement is guaranteed here for deterministic dynamics.

Both folding variants use `log(posterior / reference_prior)`, with a shared normalization at each time step. Exact conditioning is computed backward at every state, including states not visited from the initial state. Conditioning on an impossible event keeps the input policy as a convention; the equivalence statements assume positive conditioning probability.

## Experiments

All experiments are driven by YAML configuration files stored in `configs/`. Run effective horizon first: misalignment and return/incoherence read its saved instances, so all three comparisons use the same 140 MDPs. The horizon search retains 20 instances per family with finite estimates within the configured search budget; the reported correlations are conditional on that selection.

- **Effective horizon vs. incoherence**

  ```bash
  uv run python experiments/effective_horizon.py --config configs/effective_horizon.yaml
  ```

- **Policy misalignment vs. incoherence**

  ```bash
  uv run python experiments/misalignment.py --config configs/misalignment.yaml
  ```

- **Iterated incoherence (deterministic vs. stochastic)**

  ```bash
  uv run python experiments/iterated_incoherence.py --config configs/iterated_incoherence.yaml
  ```

- **Success-probability and deterministic-return improvement**

  ```bash
  uv run python experiments/strong_return_improvement.py --config configs/strong_return_improvement.yaml
  ```

- **Return/incoherence diagnostic (the same 140 configured MDPs, four updates)**

  ```bash
  uv run python experiments/return_incoherence.py --config configs/misalignment.yaml
  ```

Run `uv run pytest -q` for the regression checks, including nonuniform priors, zero prior support, full-policy equivalences and the stochastic additive-return counterexample.

## Figures

- ![Effective horizon vs. incoherence for π₁](results/effective_horizon_vs_incoherence_pi1.png)
  Correlation r≈0.419 with 95% bootstrap CI [0.307, 0.529].
- ![Policy misalignment vs. incoherence](results/misalignment_vs_incoherence.png)
  Correlation curve across temperatures with bootstrap confidence intervals.
- ![Misalignment vs. incoherence scatter at temperature 1](results/misalignment_vs_incoherence_temp1.png)
  Per-environment scatter with r≈0.269 and 95% bootstrap CI [0.106, 0.417].
- ![Temperature sweep — deterministic](results/corollary_5_11.png)
  Incoherence at fixed temperatures for a representative deterministic MDP; finite curves do not establish an unrestricted joint limit.
- ![Strong return improvement trajectories — deterministic](results/strong_return_improvement_deterministic.png) and ![Strong return improvement trajectories — stochastic](results/strong_return_improvement_stochastic.png)
  Additive log return J for deterministic environments and all-success probability S for stochastic environments. Both histories are retained in the JSON results.
