#!/bin/bash
set -e

echo "Starting Experiment 1: Early Integration"
python3 train_v26_early_operator.py --config v26_early_operator_config.json

echo "Starting Experiment 2: Separate Integration"
python3 train_v26_separate_operator.py --config v26_separate_operator_config.json

echo "All experiments finished!"
