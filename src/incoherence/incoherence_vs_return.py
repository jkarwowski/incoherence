# %%
import numpy as np  # type: ignore

from incoherence.mdp import MDP, create_random_mdp
from incoherence.experiments import run_retrain, TEMP_INCOH
from incoherence.policy import boltzmann_incoherence_causal


def run_incoherence_vs_return_random_mdps(
    num_mdps: int = 50,
    retrain_steps: int = 4,
    T: int = 4,
    A: int = 4,
    temp: float = TEMP_INCOH,
    seed: int | None = 0,
) -> None:
    """
    Simple diagnostic experiment:

    - Sample `num_mdps` random deterministic MDPs with `A` actions and horizon `T`.
    - For each MDP, start from a uniform policy and apply `retrain_steps` iterations
      of the retraining procedure from the paper (condition on R=1, recompute marginals).
    - At each iteration, record the return J (Result.j) and the
      Boltzmann incoherence (causal KL) of the current policy.
    - Print correlations between -incoherence and J, plus a simple monotonicity statistic.
    """
    if seed is not None:
        np.random.seed(seed)

    all_js: list[float] = []
    all_incohs: list[float] = []
    per_mdp_corrs: list[float] = []
    monotone_pairs = 0
    total_pairs = 0

    for mdp_idx in range(num_mdps):
        mdp = create_random_mdp(A, T, deterministic_transitions=False)
        # Run the retraining dynamics defined in the paper
        results = run_retrain(retrain_steps, mdp=mdp)

        js = np.array([float(r.j) for r in results])
        incohs = np.array(
            [float(boltzmann_incoherence_causal(mdp, r.policy, temperature=temp))
             for r in results]
        )

        all_js.extend(js.tolist())
        all_incohs.extend(incohs.tolist())

        # Per-MDP correlation between -incoherence and return
        if len(js) > 1 and np.std(js) > 0 and np.std(incohs) > 0:
            corr = float(np.corrcoef(-incohs, js)[0, 1])
            per_mdp_corrs.append(corr)

        # Track how often J increases while incoherence decreases
        for t in range(1, len(js)):
            if js[t] >= js[t - 1] and incohs[t] <= incohs[t - 1]:
                monotone_pairs += 1
            total_pairs += 1

        # Print one example MDP's trajectory for quick inspection
        if mdp_idx == 0:
            print("Example MDP (index 0)")
            print("  J trajectory:         ", [round(float(x), 4) for x in js])
            print("  Incoherence trajectory:", [round(float(x), 4) for x in incohs])
            print()

    # Pooled correlation across all MDPs / iterations
    all_js_arr = np.array(all_js)
    all_incohs_arr = np.array(all_incohs)
    if len(all_js_arr) > 1 and np.std(all_js_arr) > 0 and np.std(all_incohs_arr) > 0:
        pooled_corr = float(np.corrcoef(-all_incohs_arr, all_js_arr)[0, 1])
    else:
        pooled_corr = float("nan")

    print(f"Pooled corr(-incoherence, J) over all MDPs/steps: {pooled_corr:.4f}")
    if per_mdp_corrs:
        mean_corr = float(np.mean(per_mdp_corrs))
        std_corr = float(np.std(per_mdp_corrs))
        print(
            f"Mean per-MDP corr(-incoherence, J): {mean_corr:.4f} ± {std_corr:.4f} "
            f"(over {len(per_mdp_corrs)} MDPs)"
        )
    else:
        print("Not enough variability to compute per-MDP correlations.")

    if total_pairs > 0:
        frac_monotone = monotone_pairs / total_pairs
        print(
            f"Fraction of time-steps where J increased while incoherence decreased: "
            f"{frac_monotone:.3f} ({monotone_pairs}/{total_pairs})"
        )

# %%
run_incoherence_vs_return_random_mdps()
# %%
