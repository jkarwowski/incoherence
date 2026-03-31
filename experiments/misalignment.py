from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from incoherence.experiments import (
    instances_from_records,
    load_effective_horizon_dataset,
    load_misalignment_config,
)
from incoherence.correlation_misalignment_incoherence import (
    run_alignment_vs_incoherence_suite,
)


def run(config_path: Path) -> Path:
    config = load_misalignment_config(config_path)
    try:
        dataset = load_effective_horizon_dataset(config.dataset_path)
    except FileNotFoundError as exc:
        raise SystemExit(
            f"Effective horizon dataset not found at {config.dataset_path}. "
            "Run experiments/effective_horizon.py first."
        ) from exc
    instances = instances_from_records(dataset, config.env_specs)
    results = run_alignment_vs_incoherence_suite(
        instances,
        temperatures=config.temperatures,
    )
    output_path = config.results_dir / "misalignment_vs_incoherence.json"
    output_path.write_text(json.dumps(results, indent=2))
    csv_path = config.results_dir / "misalignment_summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["temperature", "correlation", "num_instances"])
        for entry in results:
            writer.writerow([
                entry["temperature"],
                "" if entry["correlation"] is None else entry["correlation"],
                entry["num_instances"],
            ])
    print(f"Saved misalignment results to {output_path}")
    print(f"Saved misalignment summary to {csv_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Misalignment vs incoherence experiment")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/misalignment.yaml"),
        help="Path to the misalignment experiment YAML configuration.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
