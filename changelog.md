# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

For entries before 2026-03-27, see [changelog_legacy.md](changelog_legacy.md).

---

## [2026-03-29] Q1+Q3 Encoder Fix: HORA-style Normalization + Strided Proprio History

### Context

Systematic analysis of why encoder destabilizes PPO training identified two structural
issues: (1) `actor_obs_normalizer` applies EmpiricalNorm to `cat([o_t, z])`, normalizing
the already-bounded softsign output z with non-stationary running stats, and (2) z/actor_input
ratio of 48.1% (13D z / 27D total) causes excessive mu shift per encoder update.

HORA reference comparison revealed: HORA normalizes only policy obs, passes z raw (no
double normalization), and has z/input ratio of 1.4-7.7%. HORA also does NOT normalize
privileged info before encoder (raw p_t to MLP).

Two experiments run (Steps 8a, 8b), both with 4096 envs:

| Step | Config | Roll/Pitch | noise_std | z_range | LR | KL iter-0 |
|------|--------|-----------|-----------|---------|-----|-----------|
| 8a | Q1+Q3 (enc norm kept) | 23/20 deg | 0.96 CEIL | [-0.86, 0.85] | 1.5e-5 | ~0.8 |
| 8b | Q1+Q3 (enc norm removed) | 26/32 deg | 0.98 CEIL | [-1.00, 1.00] SAT | 4.0e-5 | ~0.8 |

Both experiments show same failure pattern as Steps 4-7: iter-0 KL spike (~0.8) crashes
adaptive LR, noise_std stays at ceiling, policy effectively random. The Q1 normalize fix
did prevent z saturation in 8a (z_range [-0.86, 0.85] vs previous [-0.99, 0.99]), but
KL spike magnitude unchanged. Removing p_t normalization (8b) caused z saturation,
confirming encoder_obs_normalizer is necessary for 23D privileged input.

Key finding: Q1 and Q3 alone are insufficient. The iter-0 KL spike is caused by the
encoder's first gradient step magnitude, not by normalization or z/input ratio.
Next steps: Q2 (critic gradient to encoder) and Q4 (KL management).

### Added
- `config.py`: `proprio_history_stride` field on `ALBCEnvCfg` (default 1). Controls
  stride for proprioceptive history recording. stride=N records every N-th control step.
- `config.py`: `ALBCDebugEncoderHistStrideEnvCfg` -- Step 8a env config with
  `proprio_history_len=15`, `proprio_history_stride=5` (10Hz sampling, 1.5s window).
- `agents/rsl_rl_ppo_cfg.py`: `_Q1Q3EncoderPolicyCfg` (proprio_hist_dim=120,
  z_bounds_coef=0.0), `_Q1Q3AlgorithmCfg` (PPO, entropy_coef=0.0, desired_kl=0.02),
  `ALBCDebugPPOQ1Q3RunnerCfg` (Step 8a runner).
- `agents/rsl_rl_ppo_cfg.py`: Phase 1b configs: `_Q1Q3NoEncNormPolicyCfg`
  (encoder_obs_normalization=False), `ALBCDebugPPOQ1Q3NoEncNormRunnerCfg` (Step 8b).
- `__init__.py`: Registered `Isaac-Constrained-ALBC-Debug-PPO-Q1Q3-v0` (Step 8a)
  and `Isaac-Constrained-ALBC-Debug-PPO-Q1Q3-NoEncNorm-v0` (Step 8b).

### Changed
- `encoder/actor_critic_encoder.py`: HORA-style normalization -- `actor_obs_normalizer`
  now covers only `o_t + hist` dimensions (excludes z). Normalization moved inside
  `_get_actor_obs()`: normalizes obs part, then concatenates raw z. Previously normalized
  full `cat([o_t, z])` including the bounded softsign output.
  - `__init__`: `EmpiricalNormalization(num_actor_obs)` -> `EmpiricalNormalization(num_actor_obs_norm)`
    where `num_actor_obs_norm = policy_obs_dim + proprio_hist_dim` (z excluded).
  - `_get_actor_obs()`: builds `obs_part = cat([o_t, hist])`, normalizes it, then `cat([obs_normed, z])`.
  - `act()`, `act_inference()`, `evaluate()`: removed external `actor_obs_normalizer()` call.
  - `update_normalization()`: updates normalizer on `o_t + hist` only (not z).
  - `load_state_dict()`: added migration logic for normalizer dimension change (old->new reset).
- `albc_env.py`: Strided proprioceptive history recording. Added `_proprio_step_counter`
  (per-env torch.long). `_update_proprio_hist()` now increments counter and only records
  on stride boundary (`counter % stride == 0`). Counter reset on episode reset.

### Notes
- Step 8a confirmed: removing z from EmpiricalNorm prevents z saturation (z_range
  [-0.86, 0.85] vs previous saturated runs). This fix is sound and should be kept.
- Step 8b confirmed: encoder_obs_normalization is necessary for 23D p_t (removing it
  causes z saturation to [-1.00, 1.00]). Unlike HORA's 9D p_t, 23D benefits from normalization.
- Next experiments: Q2 (critic gradient to encoder via value loss path) and Q4
  (KL management: desired_kl, min_lr, init_lr adjustments).

---

## [2026-03-27] Encoder Ablation Study (Steps 0-7)

### Summary

Systematic ablation to isolate why full constrained ALBC (TRPO+IPO+Encoder+DR) stagnates at
17-27 deg attitude error. Components added incrementally: PPO (0.7 deg) -> +DR (3.7 deg)
-> +TRPO (5.1 deg) -> +Barrier (6.3 deg) -> **+Encoder (45 deg, DIVERGED)**. 14 encoder
experiments across TRPO, PPO, shared/separate backbone, large/small encoder, and with/without
history all failed with the same pattern: encoder update at iter 1 creates ~0.14 KL (7x
desired_kl), crashing adaptive LR. History-only PPO (no encoder, 254D input) converges to
3.3 deg, confirming encoder integration as the sole problem.

### Steps 0-3: Baseline Components

| Step | Config | Roll | Pitch | Iters | Verdict |
|------|--------|------|-------|-------|---------|
| 0 | Pure PPO (no DR/encoder/constraints) | 0.6 | 0.7 | 75 | PASS |
| 1 | PPO + DR | 3.9 | 3.7 | 66 | PASS |
| 2 | TRPO + DR | 5.4 | 5.1 | 83 | PASS (slower) |
| 3 | TRPO + DR + Barrier (4 constraints) | 8.4 | 6.3 | 162 | PASS (tighter=slower) |

Barrier works correctly: 0 spikes, all margins positive. Constraint budgets tightened from
ablation data (torque 0.20->0.08, velocity 0.10->0.02, yaw_vel 0.785->0.40).
Nominal position (0,pi)->(0,pi/2) tested: no difference (asymmetry from encoder, not kinematics).

### Step 4: TRPO+Encoder (FAIL -- Pitch Diverges)

Roll 16.4, pitch 45.2 deg (diverged in 54 iters). Encoder and actor share TRPO KL budget
(max_kl=0.005). Fisher info ~0 for encoder params + CG damping=0.1 amplifies encoder gradient
10x, consuming KL budget and leaving actor unable to improve.

### Step 4b: PPO+Encoder (FAIL -- Different Mechanism)

| Metric | Step 4 (TRPO+Enc) | Step 4b (PPO+Enc) |
|--------|-------------------|--------------------|
| Roll / Pitch | 14.7 / 46.2 deg | 32.5 / 26.3 deg |
| z_std | 0.265 | 0.975 (saturated) |
| LR | N/A (TRPO) | 1e-5 (crashed) |
| Failure mode | Fisher amplification | z saturation -> KL -> LR death |

PPO: 20 steps/iter (5 epochs x 4 minibatches) cause z_std 0.17->0.63 in 10 iters, KL to 0.04
(4x desired), LR crashes to 1e-5.

**z/actor_input ratio -- root cause of sensitivity:**

| | HORA | ALBC |
|--|------|------|
| Base obs / z / Actor input | 96D / 8D / 104D | 14D / 13D / 27D |
| z ratio | 7.7% | 48.1% |

Solution: add proprio history (30x8D=240D) -> z ratio 48.1% -> 4.9%.

### Step 4c: PPO+Encoder+History -- 6 Ablations (All Failed)

**HORA vs ALBC key differences:**

| Parameter | HORA | ALBC |
|-----------|------|------|
| entropy_coef | 0.0 | 0.01 |
| init_lr / min_lr | 5e-3 / 1e-6 | 3e-4 / 1e-5 |
| kl_threshold | 0.02 | 0.01 |
| horizon | 8 | 64 |
| normalize_value | yes | no |
| reward_scale | 0.01x | 1x |

HORA's init_lr=5e-3 allows 21 consecutive LR decreases before min_lr; ALBC's 3e-4 dies after 9.

**Single-variable ablation (all 267D actor input):**

| Exp | Changed | LR death | Roll | Pitch | Observation |
|-----|---------|:--------:|-----:|------:|-------------|
| baseline | (none) | YES | 41.5 | 32.5 | noise_std 0.97 ceiling |
| 4c-1 | entropy_coef=0.0 | YES | 29.8 | 37.2 | noise_std downtrend, LR=5.1e-5 |
| 4c-2 | ent=0+lr=5e-3 | YES | 16.2 | 47.1 | roll improved, pitch worsened |
| 4c-3 | desired_kl=0.02 | YES | 15.9 | 40.3 | Best reward, z SAT returned |
| 4c-4 | steps_per_env=8 | YES | 13.3 | 53.0 | Anti-phase oscillation, NaN |
| 4c-5 | normalize_value | YES | 24.3 | 22.0 | Best balanced (both improved) |
| 4c-6 | fixed schedule | N/A | NaN | NaN | Diverged -- adaptive LR was safety net |

All noise_std > 0.94 (policy effectively random).

### Step 4d: History-Only PPO -- No Encoder (SUCCESS)

Actor: policy(14D) + history(240D) = 254D. Standard ActorCritic, no encoder.

| Metric | ent=0.01 | ent=0.0 |
|--------|----------|---------|
| Roll / Pitch | 3.57 / 3.27 | 3.03 / 3.83 |
| reward | -6.71 | -5.57 |
| noise_std | 0.81 (rising) | 0.20 (falling) |

entropy_coef=0.0 (matching HORA) resolved sigma plateau. 254D input works fine.

### Steps 5-7: Architecture Experiments (All Failed)

**Step 5a-5b: Shared backbone (6 variants)**

Consistent pattern: iter 0 KL ~0.02 -> iter 1 KL 0.3-1.5 -> LR crashes -> pitch diverges.

| Variant | Key change | KL iter 1 | Result |
|---------|-----------|-----------|--------|
| 5a-v1 | 2-group opt, lr=1e-3 | 0.318 | LR death |
| 5a-v2 | single group | 0.517 | NaN (surr 5.9e22) |
| 5a-v3 | +log_ratio clamp | 0.835 | LR death |
| 5a-v4 | +per-minibatch refresh | 0.835 | LR death |
| 5b-v1 | +history(10) | 0.367 | LR death |
| 5b-v2 | +asymmetric LR | 0.367 | LR death |

Root cause: value loss shifts backbone features -> mu shifts -> unbounded KL not bounded by
surrogate advantage. At 2D actions, KL concentrates on 2 dims (HORA's 16D disperses it).

**Step 6: Separate network + per-minibatch refresh + combined hyperparams**

iter 1: KL=0.139 (7x desired), LR crashes to 5.9e-5. Pitch 19->48 deg.
Per-minibatch refresh reduced iter-1 KL from shared backbone's 0.835 to 0.139 (6x), still
insufficient.

**Step 7: Small encoder [256,128]->8D (15% of policy, matching HORA fraction)**

iter 1: KL=0.144, nearly identical to Step 6's 0.139. Encoder SIZE is not the differentiator.

### All 14 Experiments Summary

| Step | Architecture | Encoder | KL iter1 | Outcome |
|------|-------------|---------|----------|---------|
| 0 | PPO | none | - | 0.7 deg (PASS) |
| 1 | PPO+DR | none | - | 3.7 deg (PASS) |
| 2 | TRPO+DR | none | - | 5.1 deg (PASS) |
| 3 | TRPO+DR+Barrier | none | - | 6.3 deg (PASS) |
| 4 | TRPO+DR+Enc | [256,128,64]->13 | N/A | 45 deg (FAIL) |
| 4b | PPO+Enc | [256,128,64]->13 | high | z sat + LR death |
| 4c (x6) | PPO+Enc+Hist | [256,128,64]->13 | high | 6 ablations all failed |
| 4d | PPO+Hist (no enc) | none | - | 3.3 deg (PASS) |
| 5a (x6) | PPO+Enc shared BB | various | 0.3-1.5 | shared BB amplifies KL |
| 6 | PPO+Enc+Hist separate | [256,128,64]->13 | 0.139 | LR death |
| 7 | PPO+Enc+Hist separate | [256,128]->8 | 0.144 | LR death |

**Invariant finding**: Encoder update at iter 1 creates KL ~0.14 (7x desired_kl=0.02)
regardless of encoder size, architecture, or optimizer configuration.

### Unresolved Directions

- (a) Cosine-decaying encoder LR (starts high, decays to near-zero)
- (b) Freeze encoder for N iterations, let actor converge, then unfreeze
- (c) Encoder inside actor MLP as conditional input (not concatenated)
- (d) Abandon online encoder; use offline system identification

### Added

- `encoder/actor_critic_encoder.py`: `shared_backbone` mode (backbone MLP + linear heads),
  `z_bounds_loss()` method (soft quadratic penalty on |z| > 0.85)
- `encoder/actor_critic_constrained.py`: ActorCritic + cost critic wrapper (no encoder)
  for barrier-only ablation
- `algorithms/ppo.py`: `_update_encoder_ppo()` with per-minibatch mu/sigma refresh,
  per-epoch LR adaptation. Single optimizer group. Log-ratio clamp(-20, 20).
- `config.py`: `proprio_history_len` (default 0), `proprio_feature_dim` (8);
  debug env configs: `ALBCDebugEnvCfg`, `ALBCDebugDREnvCfg`, `ALBCDebugBarrierEnvCfg`,
  `ALBCDebugEncoderEnvCfg`, `ALBCDebugEncoderHistEnvCfg` (4c, history_len 30->10),
  `ALBCDebugHistOnlyEnvCfg` (4d, `state_space=0`)
- `albc_env.py`: `_get_proprio_features()` (8D per step), `_update_proprio_hist()` ring buffer,
  `_get_observations()` exposes `proprio_hist` as flat `(N, 240)`
- `encoder/actor_critic_encoder.py`: `proprio_hist_dim`, `_proprio_hist_key` parsing,
  `_get_actor_obs()` concatenates `cat([o_t, hist_flat, z])`.
  Added `nan_to_num` + `clamp(-10, 5)` on `log_std`.
- `agents/rsl_rl_ppo_cfg.py`: Runner/algorithm configs for Steps 4b/4c/4d/5a/5b/6/7.
  `_PPOHistOnlyAlgorithmCfg` (`entropy_coef=0.0`),
  `_PPOEncoderHistAlgorithmCfg` (4c ablation)
- `runners/constraint_encoder_runner.py`: `normalize_value` flag with Welford running mean/std
- `__init__.py`: Registered ablation tasks: `Isaac-Constrained-ALBC-Debug-v0` (0),
  `-DR-v0` (1), `-TRPO-v0` (2), `-Barrier-v0` (3), `-Encoder-v0` (4),
  `-PPO-Encoder-v0` (4b), `-PPO-Enc-Hist-v0` (4c), `-PPO-Hist-Only-v0` (4d),
  `-PPO-SB-v0` (5a), `-PPO-SB-Hist-v0` (5b), `-PPO-Sep-Enc-Hist-v0` (6)

### Changed

- `config.py`: `nominal_joint_pos` (0,pi)->(0,pi/2); constraint budgets tightened
  (torque 0.20->0.08, velocity 0.10->0.02, yaw_vel 0.785->0.40);
  reward weights (k_tau -0.01->-0.005, k_s -0.2->-0.1)
- `albc_env.py`: `_get_observations()` flattens `proprio_hist` to `(N,240)`.
  Guard `compute_all_costs()` with `num_constraints > 0`.
- `encoder/actor_critic_encoder.py`: `_get_actor_obs()` no longer flattens hist (already flat)
- `algorithms/constraint_trpo.py`: `num_constraints > 0` guards in `act()`,
  `process_env_step()`, `compute_returns()`, `_update_values()`

### Key Lessons

1. **RL fundamentally sound**: PPO solves 2-DOF in <75 iters (0.7 deg). All complexity from
   encoder integration.
2. **Encoder destabilizes any optimizer**: TRPO (Fisher amplification), PPO (z expansion ->
   KL -> LR death). Same iter-1 KL ~0.14 regardless of architecture.
3. **z/actor_input ratio**: HORA 7.7% vs ALBC 48.1%. History reduces to 4.9%, insufficient.
4. **Shared backbone incompatible with 2D actions**: value gradient -> unbounded KL via
   backbone feature shift. HORA's 16D disperses KL.
5. **HORA success non-transferable**: 16D actions, 16384 envs, reward_scale=0.01, horizon=8
   provide stability margins ALBC cannot match.
6. **Per-minibatch mu/sigma refresh**: reduces KL 6x, insufficient alone.
7. **entropy_coef=0.0 required**: positive entropy_coef pushes sigma up while LR death
   prevents pushing down. Resolved sigma plateau in Step 4d.
8. **Adaptive LR death = failure mode AND safety net**: prevents learning but also NaN.
9. **normalize_value**: only single variable improving both roll and pitch simultaneously.

---

## [2026-03-27] Action Parameterization & Reward Tuning

### Summary

Three sequential fixes addressing action jitter and constraint feasibility:
(1) Torque constraint measured PD controller's unbounded internal computation instead of actual
motor output, making it 100% violated and unsatisfiable. (2) Gaussian policy noise in absolute
joint targets created 115 deg/step jitter, causing 91% effort saturation. Switched to delta
action where noise is bounded per step. (3) Tuned delta_scale and reward weights from first
delta run analysis.

### Fix: Torque Constraint (computed_torque -> applied_torque)

`torque_limit_cost()` checked `computed_torque` (PD output, 326-554 Nm) against 9.5 Nm limit
-- 100% violated on every step, fundamentally unsatisfiable.

| Metric | computed_torque | applied_torque |
|--------|----------------|----------------|
| Range | 326-554 Nm | 12.0-12.5 Nm |
| Violation rate | ~100% | ~70-80% (improvable) |
| effort_saturation | 78-95% | - |

Impact: constant barrier gradient with no directional info, dominated reward signal (4:1),
collapsed exploration (noise_std 0.61->0.41), encoder grad_norm spikes to 19680.

#### Fixed
- `mdp/constraints.py`: `torque_limit_cost()` uses `applied_torque` instead of `computed_torque`

#### Notes
- Velocity constraint (limit=4.189 rad/s) is correct: checks actual joint_vel against motor max.
- Reward `joint_torque` already correctly used `applied_torque`.

### Switch: Absolute -> Delta Action Parameterization

With `action_scale=pi` and `noise_std=0.64`, per-step target jump = 0.64*pi = 2.0 rad = 115 deg.
PD (Kp=100) needs position error < 0.095 rad (5.4 deg) for torque < 9.5 Nm. Even at min_std=0.2,
noise = 0.2*pi = 36 deg -- 7x constraint-feasible range.

Reference: TDC achieves 0.2-6 deg using small incremental IK deltas. NORBC uses sigma_a=0.4
(8x smaller), but absolute scaling doesn't suit continuous-rotation arm.

Delta action: limits per-step change, allows any absolute position via accumulation. At 50Hz
with delta_scale=0.05, max velocity = 2.5 rad/s (within 4.189 constraint). With min_std=0.2,
noise = 0.65 deg/step (within PD tracking range).

#### Changed
- `config.py`: `action_scale: float = pi` -> `delta_scale: float = 0.05`
- `albc_env.py`: `_apply_joint_pd_action()` from absolute (`q_des = q_nominal + scale * a_t`)
  to delta accumulation (`q_des += delta_scale * a_t`, clamped to joint limits)

#### Notes
- Smoothness reward now penalizes acceleration (change in velocity command) rather than
  change in absolute position -- more physically meaningful with delta actions.
- delta_scale=0.10 rejected: PD torque = 10 Nm exceeds 9.5 limit.

### Tune: delta_scale and Reward Weights

First delta run (`2026-03-27_02-40-36`, 139 iters) -- dynamics success, attitude regression:

| Category | Metric | Absolute | Delta |
|----------|--------|----------|-------|
| Dynamics | effort_saturation | 91% | 2.2% |
| | applied_torque_max | 12.3 Nm | 6.5 Nm |
| | torque cost_return | 92 | 4.5 (within budget!) |
| | velocity cost_return | 91 | 0.02 |
| Attitude | Roll / Pitch | 17 / 13 deg | 21.6 / 18.8 deg |
| Reward | command:smoothness:torque | 97.3%:2.3%:0.5% | - |

Issues: delta_scale=0.05 too slow (0.62s to reach 90 deg offset), 160:1 reward imbalance.

#### Changed
- `config.py`: `delta_scale` 0.05 -> 0.08 (bandwidth +60%, 0.39s to 90 deg, PD torque 8.0 Nm
  within 9.5 limit)
- `config.py`: `k_tau` -0.001 -> -0.01 (10x), `k_s` -0.05 -> -0.2 (4x).
  Target ratio: command ~85%, smoothness ~10%, torque ~5%.

### Key Lessons

1. **Constraint must measure actual output**: computed_torque (PD internal) is unbounded;
   applied_torque (post-clamp) is the physical quantity.
2. **Gaussian noise in absolute action = structural jitter**: noise amplitude > 7x
   constraint-feasible range even at min_std. Delta action bounds per-step change.
3. **Reward weight balance matters**: 97%:2%:0.5% gives no incentive for smoothness/efficiency.

---

## [2026-03-27] TRPO+IPO Algorithm Fixes (NORBC Paper Alignment)

### Summary

Six structural fixes aligning ConstraintTRPO with the NORBC paper (Muller et al., ICML 2025).
Fixes applied in order: (1) logging artifact, (2) cost critic normalization + encoder
starvation, (3) encoder trust region integration, (4) missing 1/(1-gamma) factor,
(5) cost advantage standardization, (6) barrier_alpha tuning.

Combined effect: reward -78.80 -> -37.36 (2x), roll 29.2 -> 18.0 deg (38%), pitch 26.5 ->
11.9 deg (55%), z saturation eliminated ([-0.99,0.99] -> [-0.53,0.40]).

### Fix 1: Line Search Logging Artifact

`surrogate()` closure sets `_last_barrier_penalty` and `_last_mean_entropy` on every call.
During backtracking (up to 10 attempts), monitoring vars retain last rejected candidate's
values -- inflated barrier from near-constraint-boundary proposals.

#### Fixed
- `algorithms/constraint_trpo.py`: Recalculate `surrogate()` with reverted params after
  line search failure

### Fix 2: Cost Critic d_k^2 Normalization + Encoder LS Gating

**d_k^2 normalization**: Intended to prevent large-budget constraints from dominating.
Actually ineffective: yaw_vel (d_k=78.5, d_k^2=6162) contributed 98.6% of loss. Raw MSE
scales O(d_k^2), division merely cancels scaling. Non-standard -- OmniSafe, CPO, FOCOPS,
IPO all use plain MSE.

**Encoder LS gating**: Encoder received zero gradient on line search failure. No precedent
in HORA/RMA/Extreme Parkour/RSL-RL/PPG. Creates starvation loop: bad z -> constraint
violation -> LS fails -> encoder frozen -> worse z. Longest freeze: 8 iters, reward dropped
4.3x faster.

#### Changed
- `algorithms/constraint_trpo.py`: Cost value loss `(per_k_mse / d_k^2).mean()` ->
  `per_k_mse.mean()`
- `algorithms/constraint_trpo.py`: Removed `ls_success` gate on encoder update

### Fix 3: Encoder Integration into TRPO Trust Region

Separate Adam encoder update (5 epochs, lr=3e-4) was destroying trust region:
- Pre-encoder KL: 0.0035 avg (within budget)
- Post-encoder KL: 0.138 avg (**27.6x budget**, max 1153.4x)
- 11.4% of iterations: barrier_penalty = -inf

NORBC trains encoder jointly with actor (same optimizer, same KL constraint). Moved encoder
params into TRPO CG + line search.

#### Changed
- `algorithms/constraint_trpo.py`: Encoder params moved from separate Adam into
  `_policy_params`. Added `_encoder_param_offset`, `_encoder_param_count` for monitoring.
- `utils/logging.py`: `log_encoder_metrics()` reads `_last_encoder_grad_norm` from TRPO
- `runners/constraint_encoder_runner.py`: Removed encoder optimizer save/load, replaced
  `pre_encoder_kl` with `encoder_grad_norm`
- `agents/rsl_rl_ppo_cfg.py`: Removed `num_encoder_epochs`, `encoder_lr` config fields

#### Removed
- `algorithms/constraint_trpo.py`: `_update_encoder()` (22 lines), `encoder_optimizer`,
  `_encoder_params`, `_has_encoder_params`, `_last_pre_encoder_kl`

#### Notes
- CG Fisher matrix automatically captures encoder's KL contribution via natural gradient
  curvature. Encoder weight_decay (1e-5 in Adam) now omitted.

### Fix 4: Missing 1/(1-gamma) in IPO Barrier Cost Surrogate

Paper Eq. 10: `margin_k = d_k^i - J_Ck - [1/(1-gamma)] * E[ratio * A_Ck]`

With cost_gamma=0.99, factor = 100. Barrier estimated margin change 100x too small.
Example (attitude, d_k=1.0, barrier_base=0.5):
- Paper: 0.5 - 100*0.003 = 0.2 (detects shrinking)
- Code: 0.5 - 0.003 = 0.497 (sees no change)

Reward surrogate intentionally omits factor (constant scale, direction-only). Cost term INSIDE
log() changes barrier argument, not just scale.

#### Fixed
- `algorithms/constraint_trpo.py`: Added `inv_one_minus_gamma = 1/(1-cost_gamma)` to
  `cost_surrs` in barrier surrogate

#### Notes
- `margin.clamp(min=1e-8)` kills gradient at margin <= 0 (OK at ratio=1, may need smooth
  barrier if value function accuracy is poor)

### Fix 5: Per-Constraint Cost Advantage Standardization (NORBC Sec IV-B)

Removed during paper-aligned architecture overhaul (`8ba1827c`). Without it, constraints with
different scales (binary 0/1 vs continuous |omega_z|) have vastly different gradient magnitudes.
When deeply infeasible (96% violation), accurate cost value predictions make raw cost advantages
near-zero (A_Ck ~ 0.04) -- barrier gradient dominated by noise.

NORBC Sec IV-B: `A_hat_Ck = (A_Ck - mu) / sigma` per constraint k.

#### Fixed
- `algorithms/constraint_trpo.py`: Restored `(A_Ck - mean) / (std + 1e-8)` per constraint.
  Originally added in `332eff85`, removed in `8ba1827c`.

### Fix 6: Barrier Alpha Adjustment

With 1/(1-gamma)=100 and barrier_t=100, effective barrier weight = 1/margin_k. Four deeply
infeasible constraints at floor margins (0.20-1.57) gave total barrier weight = 9.2 vs
reward = 1.

Increased barrier_alpha 0.02 -> 0.05: enlarges floor margin (alpha*d_k), self-deactivating
when constraints become feasible.
- torque: 0.40->1.0, velocity: 0.20->0.50, yaw_vel: 1.57->3.93
- Total barrier weight: 9.19 -> 2.26 (ratio 2.3:1)

**3-run progression:**

| Run | Changes | reward | noise | entropy | enc_grad max |
|-----|---------|--------|-------|---------|-------------|
| 00-09-23 | baseline | -78.80 | 0.64 | 1.41 | 1.0 |
| 01-15-43 | +Fix 3,4 | -38.77 | 0.60 | 1.29 | 322 |
| 01-38-08 | +Fix 5 | -37.36 | 0.44 | 0.82 | 14097 |

#### Changed
- `agents/rsl_rl_ppo_cfg.py`: `barrier_alpha` 0.02 -> 0.05

### Key Lessons

1. **Separate encoder optimizer nullifies TRPO trust region**: encoder added 27.6x KL budget
   per iteration. Joint CG + line search is mandatory.
2. **1/(1-gamma) is critical in IPO barrier**: without it, barrier 100x too weak to detect
   constraint-violating steps.
3. **Cost advantage standardization required**: raw advantages near-zero when deeply infeasible.
   NORBC Sec IV-B standardization provides balanced gradient across constraints.
4. **d_k^2 normalization non-standard and ineffective**: raw MSE scales O(d_k^2), division
   merely cancels. Use plain MSE (OmniSafe/CPO convention).
5. **Encoder starvation from LS gating**: no precedent in literature, creates positive
   feedback loop.
6. **Monitoring must reflect accepted state**: in IPM, log(margin) diverges as margin->0.
   Rejected candidates' metrics are misleading.
7. **barrier_alpha controls deeply-infeasible behavior**: self-deactivating when feasible.
   Preferable to barrier_t for infeasibility management.
