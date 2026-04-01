from __future__ import annotations

import argparse
import shutil
from pathlib import Path


FIGURE_SOURCES = {
    "effective_horizon_vs_incoherence_uniform.png": Path("results/effective_horizon_vs_incoherence_uniform.png"),
    "effective_horizon_vs_incoherence_pi1.png": Path("results/effective_horizon_vs_incoherence_pi1.png"),
    "effective_horizon_vs_complexity.png": Path("results/effective_horizon_vs_complexity.png"),
    "misalignment_vs_incoherence.png": Path("results/misalignment_vs_incoherence.png"),
    "iterated_incoherence_deterministic.png": Path("results/iterated_incoherence_deterministic.png"),
    "iterated_incoherence_stochastic.png": Path("results/iterated_incoherence_stochastic.png"),
    "corollary_5_10.png": Path("results/corollary_5_10.png"),
    "corollary_5_11.png": Path("results/corollary_5_11.png"),
    "strong_return_improvement_deterministic.png": Path("results/strong_return_improvement_deterministic.png"),
    "strong_return_improvement_stochastic.png": Path("results/strong_return_improvement_stochastic.png"),
}

DATA_SOURCES = {
    "effective_horizon.json": Path("results/effective_horizon.json"),
    "effective_horizon_summary.csv": Path("results/effective_horizon_summary.csv"),
    "misalignment_vs_incoherence.json": Path("results/misalignment_vs_incoherence.json"),
    "misalignment_summary.csv": Path("results/misalignment_summary.csv"),
    "misalignment_instances.json": Path("results/misalignment_instances.json"),
    "iterated_incoherence.json": Path("results/iterated_incoherence.json"),
    "strong_return_improvement.json": Path("results/strong_return_improvement.json"),
}


def ensure_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required artifact missing: {path}. Run the corresponding experiment first.")


def copy_artifacts(output_dir: Path) -> None:
    figures_dir = output_dir / "figures"
    data_dir = output_dir / "data"
    figures_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    for name, source in FIGURE_SOURCES.items():
        ensure_exists(source)
        shutil.copy2(source, figures_dir / name)

    for name, source in DATA_SOURCES.items():
        ensure_exists(source)
        shutil.copy2(source, data_dir / name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect precomputed artifacts for the paper")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper/artifacts"),
        help="Destination directory for collected artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    copy_artifacts(args.output)
    print(f"Copied artifacts to {args.output}")


if __name__ == "__main__":
    main()
