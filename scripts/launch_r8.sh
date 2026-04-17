#!/bin/bash
# Round 8 overnight launcher:
#   1) Exp1 (Gated) on GPU0 + Exp2 (FastLeak) on GPU1 — parallel
#   2) Wait for both to finish (auto-detect via model_4999.pt)
#   3) Baseline on GPU0

set -e
cd /workspace/isaaclab

TRAIN_CMD="./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py"
COMMON_ARGS="--num_envs 2048 --max_iterations 5000 --headless --logger wandb --log_project_name full_dof_trpo"

echo "[$(date)] Starting Round 8 experiments"

# Phase 1: Exp1 + Exp2 in parallel
echo "[$(date)] Phase 1: Launching Exp1 (Gated) on GPU0 and Exp2 (FastLeak) on GPU1"

CUDA_VISIBLE_DEVICES=0 $TRAIN_CMD --task Isaac-FullDOF-R8-Gated-v0 \
    $COMMON_ARGS --run_name r8_gated &
PID_EXP1=$!

CUDA_VISIBLE_DEVICES=1 $TRAIN_CMD --task Isaac-FullDOF-R8-FastLeak-v0 \
    $COMMON_ARGS --run_name r8_fastleak &
PID_EXP2=$!

echo "[$(date)] Exp1 PID=$PID_EXP1, Exp2 PID=$PID_EXP2"
echo "[$(date)] Waiting for both experiments to complete..."

# Wait for both background jobs
wait $PID_EXP1
EXIT1=$?
echo "[$(date)] Exp1 (Gated) finished with exit code $EXIT1"

wait $PID_EXP2
EXIT2=$?
echo "[$(date)] Exp2 (FastLeak) finished with exit code $EXIT2"

# Phase 2: Baseline on GPU0
echo "[$(date)] Phase 2: Launching Baseline on GPU0"

CUDA_VISIBLE_DEVICES=0 $TRAIN_CMD --task Isaac-FullDOF-R8-Baseline-v0 \
    $COMMON_ARGS --run_name r8_baseline

echo "[$(date)] All Round 8 experiments complete!"
