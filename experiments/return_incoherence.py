"""Reproduce the paper's four-update return/incoherence diagnostic on its dataset."""
import argparse
import json
from pathlib import Path

import numpy as np

from incoherence.experiments import instances_from_records, load_effective_horizon_dataset, load_misalignment_config
from incoherence.mdp import compute_J, create_random_mdp, make_uniform_policy
from incoherence.policy import boltzmann_incoherence_causal
from incoherence.training import iterate_G


def run(config_path):
    config = load_misalignment_config(config_path)
    records = []
    instances = instances_from_records(
        load_effective_horizon_dataset(config.dataset_path), config.env_specs
    )
    for inst in instances:
        mdp = create_random_mdp(inst.spec.num_actions, inst.spec.horizon, inst.spec.deterministic, seed=inst.seed)
        policies = iterate_G(mdp, make_uniform_policy(mdp), 4)
        returns = [float(compute_J(mdp, pi)) for pi in policies]
        kappas = [float(boltzmann_incoherence_causal(mdp, pi, 1.0)) for pi in policies]
        corr = float(np.corrcoef(returns, -np.asarray(kappas))[0, 1])
        records.append({"spec_name": inst.spec.name, "seed": inst.seed,
                        "return_history": returns, "kappa_history": kappas,
                        "correlation": corr})
    mean = float(np.mean([r['correlation'] for r in records]))
    pooled = float(np.corrcoef([x for r in records for x in r['return_history']],
                              [-x for r in records for x in r['kappa_history']])[0, 1])
    payload = {"reward_definition": "log_q", "updates": 4, "temperature": 1.0,
               "num_instances": len(records), "mean_correlation": mean,
               "pooled_correlation": pooled, "instances": records}
    path = config.results_dir / 'return_incoherence.json'
    path.write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps({k: v for k, v in payload.items() if k != 'instances'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/misalignment.yaml'))
    run(parser.parse_args().config)
