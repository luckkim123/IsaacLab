# Lagrangian Baseline: 3 Constraint + Fixed Entropy

## Goal

Replicate the 3/6 run's success conditions (attitude error <5 deg) on top of
the current Lagrangian codebase. This establishes whether Lagrangian enforcement
can match IPO performance under equivalent constraint/entropy conditions.

## Motivation

- Run 2026-03-06_18-26-36 (IPO, 3 constraints, entropy_coef=0.005): 3.7 deg roll, 3.6 deg pitch, reward 74.6
- Run 2026-03-16_15-09-42 (Lagrangian, 8 constraints, target_entropy=2.0): 17.3 deg roll, 20.6 deg pitch, reward 31.2
- Root causes: target_entropy keeping noise_std at 1.0, 8 constraints producing excessive lambda growth
- This experiment isolates the variable: same 3 constraints + fixed entropy, IPO vs Lagrangian

## Changes

### 1. config.py: HeroAgentConstrainedEncoderEnvCfg

Reduce constraint terms from 8 to 3 (matching 3/6 run):
- Keep: accum_rot (budget=0.02), attitude_abs (budget=0.01), singularity (budget=0.15)
- Remove: effort_limit, joint_vel, oscillation, yaw_vel, cob_cog

### 2. rsl_rl_ppo_cfg.py: RslRlConstraintTRPOAlgorithmCfg

- num_constraints: 8 -> 3
- constraint_budgets: (0.02, 0.01, 0.15)
- alpha_entropy_lr: 0.01 -> 0.0 (freeze alpha = fixed entropy_coef)
- alpha_entropy_init: 0.001 -> 0.005 (equivalent to entropy_coef=0.005)

### 3. rsl_rl_ppo_cfg.py: RslRlPpoActorCriticEncoderConstrainedCfg

- num_constraints: 8 -> 3

### 4. hero_agent.py (asset): velocity_limit_sim

- 4.19 -> 6.28 rad/s (restore 3/6 value)

## What stays unchanged (code improvements retained)

- Lagrangian dual update with lambda warmup, d_k normalization, LS-gating
- std detach from cost gradient
- Reward advantage normalization
- Asymmetric critic (NORBC)
- d_k^2 cost value normalization
- Encoder z detach from cost surrogate
- Actuator DR ranges (datasheet-based: Kp 40-120, Kd 0.5-5.0, effort 0.7-1.0)

## Success criteria

- roll/pitch error < 10 deg within 500 iterations
- noise_std converges to 0.2-0.5 range (not stuck at 1.0)
- grad_cost/grad_reward ratio < 5x for majority of training
- No LS failure cascades

## Verification

Run training, monitor WandB/TensorBoard, compare with 3/6 baseline.
