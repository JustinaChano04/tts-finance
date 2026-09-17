#!/usr/bin/env python
"""CLI entrypoint: python experiments/run_baseline.py --config configs/baseline.yaml"""

import argparse

from tts_finance.runner import run_experiment


def main():
    parser = argparse.ArgumentParser(description="Run a test-time scaling FinQA experiment.")
    parser.add_argument("--config", required=True, help="Path to an experiment YAML config.")
    args = parser.parse_args()
    run_experiment(args.config)


if __name__ == "__main__":
    main()
