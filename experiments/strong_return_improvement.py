from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable

import matplotlib.pyplot as plt
import numpy as np


def _load_records(dataset_path: Path) -> list[dict]:
    if not dataset_path.exists():
        raise SystemExit(
            f"Strong return improvement dataset not found at {dataset_path}. "
            "Run experiments/iterated_incoherence.py first."
        )
    return json.loads(dataset_path.read_text())


def _gather_series(records: Iterable[dict]) -> Dict[str, np.ndarray]:
    grouped: Dict[str, list[np.ndarray]] = {"deterministic": [], "stochastic": []}
    for record in records:
        key = "deterministic" if record.get("deterministic") else "stochastic"
        grouped[key].append(np.asarray(record["return_history"], dtype=float))
    return {k: np.vstack(v) for k, v in grouped.items() if v}


def _plot_return_improvement(series: Dict[str, np.ndarray], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    styles = {
        "deterministic": {"color": "C0", "linestyle": "-", "label": "Deterministic"},
        "stochastic": {"color": "C1", "linestyle": "--", "label": "Stochastic"},
    }

    for key, data in series.items():
        iterations = np.arange(data.shape[1])
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        style = styles.get(key, {})
        ax.plot(iterations, mean, marker="o", linewidth=2, **style)
        ax.fill_between(
            iterations,
            mean - std,
            mean + std,
            color=style.get("color", "C0"),
            alpha=0.2,
        )

    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"Return $J(\pi_k)$")
    ax.set_title("Strong return improvement lemma")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize="small")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def run(dataset_path: Path, output_path: Path) -> Path:
    records = _load_records(dataset_path)
    series = _gather_series(records)
    if not series:
        raise SystemExit("Dataset does not contain any return trajectories to plot.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _plot_return_improvement(series, output_path)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strong return improvement visualisation")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("results/iterated_incoherence.json"),
        help="Path to the iterated incoherence dataset (JSON).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/strong_return_improvement.png"),
        help="Destination for the output plot.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = run(args.dataset, args.output)
    print(f"Saved strong return improvement figure to {output_path}")


if __name__ == "__main__":
    main()
