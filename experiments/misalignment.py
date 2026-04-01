from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import math

from incoherence.mdp import create_random_mdp

from incoherence.experiments import (
    generate_instances,
    load_misalignment_config,
    records_from_instances,
)
from incoherence.reporting import plot_misalignment, plot_misalignment_scatter
from incoherence.correlation_misalignment_incoherence import (
    run_alignment_vs_incoherence_suite,
)


def _size_metric(env_specs, seed: int) -> dict[str, float]:
    metric: dict[str, float] = {}
    for idx, spec in enumerate(env_specs):
        mdp = create_random_mdp(
            spec.num_actions,
            spec.horizon,
            deterministic_transitions=spec.deterministic,
            seed=seed + idx,
        )
        total_slots = float(sum(len(actions) for actions in mdp.actions.values()))
        metric[spec.name] = total_slots
    return metric


def run(config_path: Path) -> Path:
    config = load_misalignment_config(config_path)
    size_metric = _size_metric(config.env_specs, config.global_seed)
    instances = generate_instances(config.env_specs, config.global_seed)
    records = records_from_instances(instances)
    config.dataset_path.parent.mkdir(parents=True, exist_ok=True)
    config.dataset_path.write_text(json.dumps(records, indent=2))
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
    plot_misalignment(results, config.results_dir / "misalignment_vs_incoherence.png")

    target_temp = 1.0
    for entry in results:
        if entry["correlation"] is None:
            continue
        if math.isclose(entry["temperature"], target_temp, rel_tol=1e-9, abs_tol=1e-9):
            scatter_path = config.results_dir / "misalignment_vs_incoherence_temp1.png"
            plot_misalignment_scatter(
                entry.get("instances", []),
                entry["temperature"],
                scatter_path,
                size_metric=size_metric,
            )
            break
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
