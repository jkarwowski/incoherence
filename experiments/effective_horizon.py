from __future__ import annotations

import argparse
from pathlib import Path

from incoherence.experiments import EffectiveHorizonConfig, load_effective_horizon_config, load_effective_horizon_dataset
from incoherence.experiment_horizon_monte_carlo import (
    collect_effective_horizon_data,
    save_effective_horizon_data,
)
from incoherence.mdp import create_random_mdp
from incoherence.reporting import (
    plot_effective_horizon,
    plot_effective_horizon_complexity,
    records_to_dicts,
    write_effective_horizon_summary,
)


def run(config: EffectiveHorizonConfig) -> Path:
    output_path = config.results_dir / "effective_horizon.json"
    if not output_path.exists():
        records = collect_effective_horizon_data(
            env_specs=config.env_specs,
            settings=config.settings,
            temperature=config.temperature,
            global_seed=config.global_seed,
            verbose=True,
        )
        save_effective_horizon_data(output_path, records)
        print(f"Saved effective horizon dataset to {output_path}")
        record_dicts = records_to_dicts(records)
        summary_path = config.results_dir / "effective_horizon_summary.csv"
        write_effective_horizon_summary(summary_path, record_dicts)
    else:
        record_dicts = load_effective_horizon_dataset(output_path)

    size_metric = {}
    complexity = {}
    for spec in config.env_specs:
        mdp = create_random_mdp(
            spec.num_actions,
            spec.horizon,
            deterministic_transitions=spec.deterministic,
            seed=config.global_seed,
        )
        total_slots = float(sum(len(actions) for actions in mdp.actions.values()))
        size_metric[spec.name] = total_slots
        complexity[spec.name] = {
            "states": float(len(mdp.states)),
            "actions": float(sum(len(mdp.actions[s]) for s in mdp.states)),
            "slots": total_slots,
        }

    plot_effective_horizon(
        record_dicts,
        size_metric,
        config.results_dir / "effective_horizon_vs_incoherence_uniform.png",
        config.results_dir / "effective_horizon_vs_incoherence_pi1.png",
    )
    plot_effective_horizon_complexity(
        record_dicts,
        complexity,
        config.results_dir / "effective_horizon_vs_complexity.png",
    )
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Effective horizon experiment")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/effective_horizon.yaml"),
        help="Path to the effective horizon YAML configuration.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_effective_horizon_config(args.config)
    run(config)


if __name__ == "__main__":
    main()
