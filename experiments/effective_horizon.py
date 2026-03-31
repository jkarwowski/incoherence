from __future__ import annotations

import argparse
from pathlib import Path

from incoherence.experiments import EffectiveHorizonConfig, load_effective_horizon_config
from incoherence.experiment_horizon_monte_carlo import (
    collect_effective_horizon_data,
    save_effective_horizon_data,
)


def run(config: EffectiveHorizonConfig) -> Path:
    records = collect_effective_horizon_data(
        env_specs=config.env_specs,
        settings=config.settings,
        temperature=config.temperature,
        global_seed=config.global_seed,
        verbose=True,
    )
    output_path = config.results_dir / "effective_horizon.json"
    save_effective_horizon_data(output_path, records)
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
    output_path = run(config)
    print(f"Saved effective horizon dataset to {output_path}")


if __name__ == "__main__":
    main()
