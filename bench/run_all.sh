#!/bin/bash
# Runs the smoke test, then both models. Called through heavy.sh (one model job at a time).
set -e
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1
python3 bench/bench.py mlx-community/Qwen2.5-0.5B-Instruct-4bit results/smoke.json --quick
python3 bench/bench.py mlx-community/Qwen2.5-0.5B-Instruct-4bit results/qwen2.5-0.5b-4bit.json
python3 bench/bench.py mlx-community/Qwen2.5-1.5B-Instruct-4bit results/qwen2.5-1.5b-4bit.json
rm -f results/smoke.json
