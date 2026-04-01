#!/usr/bin/env bash

set -euo pipefail

# echo "[1/5] Effective horizon experiment"
# uv run python experiments/effective_horizon.py --config configs/effective_horizon.yaml

echo "[2/5] Misalignment experiment"
uv run python experiments/misalignment.py --config configs/misalignment.yaml

echo "[3/5] Iterated incoherence experiment"
uv run python experiments/iterated_incoherence.py --config configs/iterated_incoherence.yaml

echo "[4/5] Strong return improvement figure"
uv run python experiments/strong_return_improvement.py --dataset results/iterated_incoherence.json

echo "[5/5] Collect paper artifacts"
uv run python experiments/generate_paper_artifacts.py --output paper/artifacts

echo "All experiments completed."
