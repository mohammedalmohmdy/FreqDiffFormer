#!/usr/bin/env bash
# reproduce.sh
# Simple script to reproduce a demo experiment (train on sketchy_mini and evaluate)
set -e
CONFIG=configs/default.yaml
CHECKPOINT=outputs/checkpoint_last.pth

echo "Starting demo training..."
python scripts/train.py --config $CONFIG

echo "Evaluating model..."
python scripts/evaluate_mAP.py --config $CONFIG --checkpoint $CHECKPOINT --dataset sketchy_mini --k 200

echo "Done."
