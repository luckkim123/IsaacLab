#!/usr/bin/env bash
# r14 final training run: r13_B baseline + entropy reduction + aggressive DR + action latency.
# See docs/superpowers/specs/2026-04-21-r14-final-design.md.
set -e

cd /workspace/isaaclab

EXPERIMENT_NAME="r14"
STAMP=$(date +%Y%m%d_%H%M%S)
STDOUT_LOG="/workspace/isaaclab/logs/archive/launch_scripts/${EXPERIMENT_NAME}_${STAMP}.log"
mkdir -p "$(dirname "$STDOUT_LOG")"

echo "[r14 $(date)] START experiment_name=${EXPERIMENT_NAME}" | tee -a "$STDOUT_LOG"
CUDA_VISIBLE_DEVICES=0 ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-FullDOF-TRPO-v0 \
    --num_envs 4096 \
    --max_iterations 20000 \
    --headless \
    --logger wandb \
    --log_project_name full_dof_trpo \
    --experiment_name "${EXPERIMENT_NAME}" \
    2>&1 | tee -a "$STDOUT_LOG"
RC=${PIPESTATUS[0]}
echo "[r14 $(date)] END rc=${RC}" | tee -a "$STDOUT_LOG"
exit "$RC"
