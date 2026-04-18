# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## Historical changelogs

- [Round 8: Error-Gated Integration](docs/hero/experiments/round8_gated_integral.md) (BEST POLICY)
- [Round 7: Integral Observation + EpsSmooth](docs/hero/experiments/round7_integral_obs.md)
- [Round 6: Axis-Specific Shape Calibration](docs/hero/experiments/round6_axis_calibration.md)
- [Round 5: Constraint Budget Tuning](docs/hero/experiments/round5_constraint_tuning.md)
- [Round 4: Saturating Penalty](docs/hero/experiments/round4_saturating_penalty.md)
- [Round 3: SS Error Structural Fixes](docs/hero/experiments/round3_ss_structural.md)
- [Round 2: PerDimEnt Validation](docs/hero/experiments/round2_perdiment_validation.md)
- [Round 1: Per-Dim Noise Comparison](docs/hero/experiments/round1_noise_comparison.md)
- [Pre-Round Infrastructure](docs/hero/experiments/pre_round_infrastructure.md) (2026-04-04 ~ 2026-04-13)
- [Full ALBC early development](docs/hero/changelog_full_albc_early.md) (2026-03-31 ~ 2026-04-02)
- [Constrained ALBC development](docs/hero/changelog_constrained_albc.md) (2026-03-27 ~ 2026-03-31)
- [Legacy development](docs/hero/changelog_legacy.md) (2026-03-05 ~ 2026-03-26)
- [Encoder ablation study](docs/hero/experiments/encoder_ablation.md) (Steps 0-19)

---

## [2026-04-18] Starting Point: Code Cleanup + Current Baseline

Previous 8 rounds of experiments (R1-R8) completed. R8-Gated confirmed as best policy.
Codebase underwent major simplification (16 files, -2071 / +565 lines).
This entry documents the current code state as the baseline for all future work.

### Current Best: R8-Gated

Error-gated 6D integral integration이 SS error와 overshoot를 동시에 개선한 유일한 configuration.
Model checkpoint는 log cleanup 사고로 소실 -- 재학습 필요.

Key results (eval_dr_fulldof):
- Aggregate: SS=0.131 (best), OS=13.1% (best), n>20%=16.0% (best)
- Attitude: SS=0.370 (-15% vs R7I), OS=9.3% (-48% vs R7I)
- Velocity: SS=0.014 (-53% vs R7I), OS=10.4% (-52% vs R7I)
- Yaw SS=0.001 (6D integral), Yaw OS=34.4% (sole remaining weakness)

### Architecture

```
Task: Isaac-FullDOF-TRPO-v0 (single registered task)
Action: 8D (2D arm revolute + 6D thruster wrench)
Observation: 87D policy (26D proprio + 55D temporal history + 6D integral)
Privileged: 24D (DR parameters, static min-max normalized)

Encoder:  p_t(24D) -> static_minmax -> MLP[256,128,64] -> LayerNorm -> softsign -> z(9D)
Actor:    cat([o_t(87D), z(9D)]) = 96D -> MLP[256,128,64] -> 8D (Gaussian)
Critic:   cat([o_t(87D), z(9D), p_t(24D)]) = 120D -> MLP[512,256,128] -> 1D (asymmetric)
Cost:     same 120D -> MLP[512,256,128] -> K (multi-head, one per constraint)
```

### Algorithm

- **ConstraintTRPO + IPO** (Interior-Point Optimization)
- max_kl=0.005, cg_iters=10, cg_damping=0.1
- GAE: gamma=0.99, lam=0.95
- Value: Adam lr=1e-3, 5 epochs, 4 mini-batches
- Barrier: t=100.0, alpha=0.05

### Entropy Management

- entropy_coef_per_dim: arm=(0.01, 0.01), thr=(0.001 x6) -- PerDimEnt, validated R2
- min_std_per_dim: arm=(0.10, 0.10), thr=(0.05 x6)
- max_std=2.0, min_std=0.05 (scalar fallback)
- init_noise_std=0.7

### Reward

```
r = r_att + r_lin + r_yaw + r_tau + r_thr + r_s

Tracking: r = k * (exp(-e^2/2s^2) - q*e^2)
  att_rp:  k=9.0, sigma=0.10, quad=0.833, roll_weight=1.5
  lin_vel: k=4.0, sigma=0.10, quad=1.0
  yaw_vel: k=3.5, sigma=0.10, quad=1.0

Saturating penalty fields (tanh_coef, arctan_coef) exist but default to 0.0.
Penalty: k_tau=-0.01, k_thr=-0.35, k_s=-0.1
```

### Constraints (10 terms: 5 Prob + 5 Avg)

| Type | Name | Budget |
|------|------|--------|
| Prob | attitude_limit (80 deg) | 0.01 |
| Prob | arm_torque (9.5 Nm) | 0.08 |
| Prob | arm_joint_vel (4.189 rad/s) | 0.02 |
| Prob | joint1_pos (4*pi rad) | 0.01 |
| Prob | cumulative_yaw (8*pi rad) | 0.01 |
| Avg | thruster_util | 0.40 |
| Avg | rp_rate (1.0 rad/s) | 0.10 |
| Avg | yaw_rate (0.7 rad/s) | 0.10 |
| Avg | rp_vel_settling (0.087 rad) | 0.20 |
| Avg | manipulability (w=0.3) | 0.05 |

### DORAEMON DR Curriculum

- kl_ub=0.04, performance_lb=90.0, step_interval=250
- SLSQP optimizer, log-space Beta parameterization, 15D physics-only
- Binary success criterion (episode_return >= performance_lb)

### Observation Detail

Current proprioception (26D):
- Command (6D): vel_cmd_lin(3), ang_cmd(3) [att_rp(2) + yaw_rate(1)]
- Body State (9D): euler(3), ang_vel(3), lin_vel(3)
- Arm State (5D): joint_pos(2), joint_vel(2), manipulability(1)
- Thruster (6D): filtered output (T0-T5)

Temporal history (55D): ring buffer, stride=3
- Joint tracking (12D): (q_des-q_actual, joint_vel) x 3 steps
- Body tracking (27D): (lin_vel_err, ang_err, rpy) x 3 steps
- Action (16D): full_action(8D) x 2 steps

Integral error (6D): leaky integrator (leak=0.99, clamp=+-2.0)
- 6 channels: roll_err, pitch_err, vx_err, vy_err, vz_err, yaw_rate_err
- Error-gated: accumulate only when |err| < sigma (R8-Gated configuration)

### Registered Tasks

| Task | Algorithm | Encoder | Purpose |
|------|-----------|---------|---------|
| Isaac-FullDOF-TRPO-v0 | ConstraintTRPO + IPO | Yes (24D->9D) | Production |
| Isaac-FullDOF-NoEncoder-v0 | ConstraintTRPO + IPO | No | Ablation baseline 1 |
| Isaac-FullDOF-PPO-v0 | Standard PPO | No | Ablation baseline 2 |

### Code Simplification (this session)

16 files modified (-2071 / +565 lines). Key changes:
- Removed experiment-specific task registrations (R5/R6/R7/R8 tasks)
- Consolidated runner configs into production + 2 ablation baselines
- Extracted shared PolicyBase for ActorCriticEncoder and ActorCriticAsymConstrained
- Simplified config.py: removed unused experiment configs
- Cleaned up reward, constraint, and observation modules

### Open Questions

- R8-Gated model needs retraining (checkpoint lost in log cleanup)
- Yaw OS (34.4%): channel-specific gate configuration
- Roll SS high per-env variance (std > mean at all DR levels)
- Entropy collapse in all R8 runs (Gated: 0.03): PerDimEnt tuning needed?

---

## [2026-04-18] R9 Plan: Roll Oscillation / Yaw Overshoot + Refactor Bug Discovery

### Context

R8-Gated eval_dr_fulldof revealed three residual problems that block a production-ready policy: roll oscillation (high SS jitter), vz undershoot, and isolated yaw overshoot. Needed to launch new experiments addressing these while also retraining baseline (R8-Gated checkpoint lost) and exercising the disabled `normalize_value` feature for the first time.

### Experiments

Hard-DR evidence from R8-Gated archive (enhanced_summary.json):
- roll SS=0.855 / jitter=0.275 / OS=14.6% vs pitch SS=0.320 / jitter=0.082 / OS=5.6% -> roll is ~3x worse across every metric despite same reward structure.
- vz SS=0.040 (std=0.099) / undershoot=4.19% / OS=14.1% -> vz is the worst lin_vel axis by 2-3x on every metric, consistent with buoyancy-F_bu 26.24 N + heave-added-mass being 10x smaller than surge/sway.
- yaw SS=0.0017 rad/s and rise=0.014 s (essentially perfect) but OS=23.8% with n_gt20=33% -> control authority is not the issue; yaw_rate constraint threshold of 0.7 rad/s authorizes the overshoot given cmd range +-0.5 rad/s.

R9 queue launched on GPU0/1 (2048 envs, 5000 iter, WandB project `fulldof_albc`):
- **r9_baseline** (GPU0 first): control, zero code change on top of r8_gated config.
- **r9_normval** (GPU1 first): `normalize_value=True` (HORA-style running mean/std for critic targets, previously-disabled path).
- **r9_tightrates** (GPU0 second): rp_rate soft_threshold 1.0 -> 0.5, yaw_rate 0.7 -> 0.55. Hypothesis: both rate constraints had 3x / 1.4x margin over command, leaving rate damping inactive.
- **r9_symatt** (GPU1 second): `att_roll_weight` 1.5 -> 1.0. Hypothesis: the 1.5x multiplier is in a middle zone where it neither compensates the 20x TAM moment-arm gap (0.007 m vs 0.145 m) nor avoids being a competing sharp signal.

### Decisions

- **Minimum-change per run, orthogonal hypotheses** over a single "combined fix" run. Separate isolation lets the decision tree for R10 read off cleanly (e.g., if tightrates helps roll/yaw but symatt does not, rate threshold was the bottleneck, not reward asymmetry).
- **Root-cause fix for refactor bugs** chosen over a lazy null-check guard, on user pushback. The proper contract is "after `_reset_idx`, env is in a valid observation state", restored by populating `_euler_cache` at reset-time just like `_get_dones` does per step. An `if is None` check would have masked the ordering violation.
- **Git worktree + PYTHONPATH prepend** over argparse CLI overrides for run isolation. Argparse route would have required modifying config parse logic, violating minimum-change. PYTHONPATH takes precedence over pip editable-install .pth entries, so each worktree's source is found first without touching the main install.
- **Sequential queue over 4 concurrent**. RTX 4060 (8GB) cannot host two runs simultaneously at ~6-7 GB each. Halving num_envs to fit 2-concurrent would double iterations to reach equivalent samples, yielding zero net wall-time gain.

### Latent bugs surfaced by first fresh training since refactor eafca264

The -2071 / +565 line refactor removed lazy-init guards that previously hid three bugs. R8-Gated only evaluated an archived checkpoint, so none of these had been exercised until this session's fresh train:

1. **`_euler_cache` uninitialized at first `_get_observations`**. Root: init=None in `__init__`, population only in `_get_dones` which does not run during `env.reset()`. Symptom: `TypeError: cannot unpack non-iterable NoneType`. Fix: populate at tail of `_reset_idx` so post-reset observation contract holds.
2. **Encoder static-min-max constants were plain Python attributes**. `_enc_obs_range` and `_enc_obs_midpoint` were assigned with `self.x = ...`, not `register_buffer`, so `module.to(cuda)` left them on CPU while inputs were on cuda. Symptom: `RuntimeError: tensors on different devices`. Fix: wrap both in `register_buffer`.
3. **`normalize_value` pipeline referenced a PPO-only attribute**. `self.alg.normalize_advantage_per_mini_batch` does not exist on `ConstraintTRPO`. Symptom: `AttributeError`. Only triggered when the flag was flipped on for the first time in r9_normval. Fix: `getattr(..., False)` fallback in `_compute_returns_with_value_norm`.

Lesson: checkpoint-reload eval passes do not validate init paths. Before trusting a refactor, run fresh training at least once with each toggleable feature enabled.

### Open Questions

- vz structural undershoot (4.19% hard DR, highest of all axes): candidate fixes deferred to R10 (per-axis lin_vel sigma, vz-only undershoot penalty, or revisiting added-mass DR bounds for heave).
- Rise time improvement: user marked low priority this round; may revisit after R9 results if rate-tightening regresses it.
- `normalize_value` fix is on r9_normval branch only; main branch still has the latent AttributeError when the flag is True. Landing the fallback on feat/encoder-tdc-integration would unblock future opt-in without per-branch fixes.

---
