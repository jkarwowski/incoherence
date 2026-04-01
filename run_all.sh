#!/usr/bin/env bash

set -euo pipefail

export MPLBACKEND=${MPLBACKEND:-Agg}

echo "[1/4] Effective horizon experiment"
uv run python experiments/effective_horizon.py --config configs/effective_horizon.yaml

echo "[2/4] Misalignment experiment"
uv run python experiments/misalignment.py --config configs/misalignment.yaml

echo "[3/4] Iterated incoherence experiment"
uv run python experiments/iterated_incoherence.py --config configs/iterated_incoherence.yaml

echo "[4/4] Strong return improvement figure"
uv run python experiments/strong_return_improvement.py --config configs/strong_return_improvement.yaml

echo "All experiments completed."
