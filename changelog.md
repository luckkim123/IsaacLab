# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [2026-03-17] Reward restructure + quadratic command reward

### Context
Analysis of 9-constraint run `2026-03-17_08-21-47` (762 iters) showed plateau at roll 8 deg,
pitch 10 deg. Four constraints OVER budget (attitude_err, singularity, yaw_vel, joint_osc).
joint_osc replaced with smoothness reward, PBRS removed, command_sigma tightened 0.35->0.20.

Noise floor 0.15 caused entropy collapse: run `08-48-06` (269 iters) entropy=-0.96, error +80%.
Reverted to 0.20. But even with floor=0.20, run `08-57-07` showed noise_std reaching floor by
iter 70 with entropy=-0.38 (COLLAPSED). Root cause: Laplacian reward `exp(-e/sigma)` with
sigma=0.20 creates gradient=5.0*exp(-e/0.20) that stays strong near zero, driving continuous
noise reduction with no counterforce (alpha_entropy=0).

Switched to quadratic command reward `r_c = -k_c*(roll_err^2 + pitch_err^2)` per reference
paper. Quadratic gradient = -2*k*error weakens near zero, providing natural entropy-friendly
structure. Policy stops compressing noise once error reduction slows. This matches the paper's
3-term reward design (command quadratic + torque penalty + smoothness penalty).

### Added
- `mdp/rewards.py`: `command_type` parameter in `command_reward()` supporting "quadratic"
  and "laplacian" modes. Quadratic: `-(roll_err^2 + pitch_err^2)`, uses per-axis error
  (not L2 norm). Laplacian: existing composite exp + linear ramp (unchanged).
- `mdp/rewards.py`: `command_type` field in `ALBCRewardCfg` (default: "laplacian" for
  backward compatibility with non-constrained envs)

### Changed
- `config.py`: Constrained encoder `command_type` set to "quadratic" (was implicit "laplacian")
- `config.py`: Removed `command_sigma=0.20` override (irrelevant for quadratic mode)
- `config.py`: `smoothness_weight` 0.0 -> -0.5, `progress_weight` 2.0 -> 0.0,
  `settling_weight` -> 0.0, joint_osc constraint removed (9->8 constraints)
- `agents/rsl_rl_ppo_cfg.py`: `num_constraints` 9->8, budgets updated
- `base_env.py`: Pass `reward_type` from config to `command_reward` function

### Fixed
- `runners/base_runner.py`, `algorithms/constraint_trpo.py`: Reverted noise floor 0.15 -> 0.20.
  0.15 caused entropy collapse within 269 iters. Even 0.20 hits floor by iter 70 with Laplacian
  sigma=0.20. Quadratic reward should resolve the underlying pressure.

### Notes
- Noise floor tested: 0.10 (immediate collapse), 0.15 (collapse in 269 iters), 0.20 (floor
  reached by iter 70 with Laplacian). Quadratic should allow noise_std to stay above floor.
- Quadratic gradient at 3deg: 0.52 vs Laplacian(sigma=0.20): 3.85. Weaker gradient is
  actually desirable: constraint system (attitude_err budget=7deg) handles fine control,
  reward provides coarse tracking signal.
- Other envs (Base, Encoder-Base, etc.) unaffected: default command_type="laplacian"
- Quadratic alone did NOT fix entropy collapse: run `09-02-53` still showed entropy=-0.38,
  noise_std=0.20 (floor) by early iterations. Root cause identified as `smoothness_weight=-0.5`:
  E[da^2] contains 2*sigma^2 term, so reducing noise directly reduces smoothness penalty.
  With alpha_entropy=0, this constant downward pressure is uncontested.
  Reduced smoothness_weight -0.5 -> -0.1 (1/5 pressure) to test hypothesis.

## [2026-03-17] Constraint expansion 3→9 + PBRS progress reward

### Context
With alpha_entropy=0, noise_floor=0.2, lambda_warmup=0.3 stabilized (previous session),
expanded constraints for behavioral quality improvement. The prior 8-constraint failure
was caused by target_entropy + continuous constraint interaction (now resolved: alpha=0 +
noise floor). Added PBRS progress reward to accelerate rise time (replacing settling reward).

### Added
- `mdp/constraints.py`: 4 new cost functions:
  - `joint_torque_cost` (alias of effort_limit_cost, clearer name)
  - `joint_velocity_limit_cost` (binary: joint_vel > 4.189 rad/s = 40 RPM)
  - `overshoot_cost` (binary: error sign flip + magnitude > 2 deg threshold)
  - `attitude_error_cost` (continuous: reuses env._potentials L2 norm)
- `mdp/rewards.py`: `progress_reward` PBRS function (prev_potential - gamma * potential),
  `ALBCRewardCfg.progress_weight` and `progress_gamma` fields
- `base_env.py`: `_prev_attitude_error_rp` buffer for overshoot detection (initialized to
  initial error in `_reset_task_and_state` to prevent false positives on first step)

### Changed
- `config.py`: `HeroAgentConstrainedEncoderEnvCfg.constraints` expanded from 3 to 9 terms:
  binary(6): accum_rot(0.02), attitude_abs(0.01), singularity(0.15),
  joint_torque(0.05), joint_vel_limit(0.05), overshoot(0.10);
  continuous(3): attitude_err(0.122=7deg), joint_osc(0.30), yaw_vel(0.15)
- `config.py`: Reward updated: settling_weight=0.0 (replaced by attitude_err constraint),
  progress_weight=2.0 (PBRS, scale_by_dt=False)
- `agents/rsl_rl_ppo_cfg.py`: `num_constraints` 3→9, `constraint_budgets` synced to 9-tuple
  in both `RslRlConstraintTRPOAlgorithmCfg` and `RslRlPpoActorCriticEncoderConstrainedCfg`
- `base_env.py`: `_build_reward_terms()` adds progress term when weight != 0;
  `_get_rewards()` updates overshoot buffer after constraint computation
- `mdp/__init__.py`: Exports updated for all new functions

### Notes
- PBRS is theoretically safe (Ng et al. 1999): does not change optimal policy
- attitude_err budget=0.122 rad (7 deg) is moderately strict; if lambda saturates, relax to 10 deg
- Overshoot false positive prevention: `_prev_attitude_error_rp` set to initial error at reset
- joint_torque_cost is a pure alias of effort_limit_cost (no code duplication)

## [2026-03-17] 3-constraint Lagrangian baseline + disable entropy bonus for TRPO

### Context
Previous run `2026-03-16_15-09-42` (999 iters, Lagrangian, 8 constraints, target_entropy=2.0)
failed with 17-20 deg attitude error, noise_std stuck at 1.0. Compared with successful run
`2026-03-06_18-26-36` (IPO, 3 constraints, entropy_coef=0.005): 3.7 deg error, noise_std 0.2.

**Root cause 1 (8→3 constraints)**: 5 continuous constraints (joint_vel, oscillation, yaw_vel,
cob_cog, effort_limit) produce costs proportional to noise_std. As noise increased, continuous
costs grew, lambda grew, cost gradient dominated reward gradient, creating a vicious cycle.
The 3/6 run used only 3 binary constraints (noise-insensitive).

**Root cause 2 (target_entropy)**: SAC-style alpha kept noise_std at 1.0, preventing the
natural reward-driven noise reduction that the 3/6 run exhibited (converged to 0.2).

**Fix applied**: Reduced to 3 binary constraints (accum_rot, attitude_abs, singularity) +
fixed alpha_entropy_init=0.005. Restored velocity_limit_sim=6.28 (was 4.19).

**Run `2026-03-17_07-15-39` (227 iters)**: Error improved to 5-8 deg (good!), reward peaked
at 69.2 (iter 150). BUT noise_std grew unboundedly: 1.02→4.45, entropy 2.84→5.75. Reward
started declining after iter 150.

**Root cause 3 (entropy bonus in TRPO)**: With all constraints satisfied (lambda=0), the
fixed alpha=0.005 entropy bonus has no counterbalancing force. PPO has clip ratio + adaptive
LR to resist noise growth; TRPO takes max-KL steps every iteration, so any alpha > 0
consistently pushes noise_std up. TRPO's KL constraint alone provides sufficient exploration.

**Fix**: Set alpha_entropy_init=0.0. Also fixed math.log(0) crash in constraint_trpo.py
(added guard: log(max(init, 1e-8))).

### Changed
- `config.py`: Reduced `HeroAgentConstrainedEncoderEnvCfg.constraints.terms` from 8 to 3
  (kept: accum_rot budget=0.02, attitude_abs budget=0.01, singularity budget=0.15;
  removed: effort_limit, joint_vel, oscillation, yaw_vel, cob_cog)
- `agents/rsl_rl_ppo_cfg.py`: `num_constraints` 8→3, `constraint_budgets` updated to
  (0.02, 0.01, 0.15), `alpha_entropy_lr` 0.01→0.0, `alpha_entropy_init` 0.005→0.0
  (TRPO KL constraint provides exploration; entropy bonus causes unbounded noise growth)
- `agents/rsl_rl_ppo_cfg.py`: `RslRlPpoActorCriticEncoderConstrainedCfg.num_constraints` 8→3
- `hero_agent.py`: `velocity_limit_sim` 4.19→6.28 rad/s (restored 3/6 value)

### Fixed
- `algorithms/constraint_trpo.py`: `math.log(alpha_entropy_init)` crashes when init=0.0.
  Added guard: `log(max(init, 1e-8))` so alpha initializes to ~1e-8 (effectively zero).

### Added
- `docs/plans/2026-03-17-lagrangian-baseline-3constraint.md`: Design document for experiment

### Notes
- All Lagrangian code improvements retained: std detach, reward adv normalization,
  lambda warmup, d_k normalization, LS-gated updates, asymmetric critic, z detach from cost
- Key insight: entropy bonus interacts fundamentally differently with TRPO vs PPO. In PPO,
  clip + adaptive LR naturally resist noise growth. In TRPO, max_kl step has no such mechanism.

## [2026-03-17] Raise noise floor 0.1 -> 0.2 + alpha=0 run analysis

### Context
Run `2026-03-17_07-25-13` (alpha=0, 3 constraints, 454 iters) confirmed noise_std fix works:
noise decreased naturally 1.0 -> 0.17, reward 60-66, roll 4-7 deg, pitch 5-9 deg.
All constraints satisfied (lambda=0), line search 100%.

**However, noise_std kept falling without bound** (0.17 at iter 454, still declining).
Entropy went negative (-0.73) -- exploration effectively dead. Reward plateaued at 60-66
(vs 74.6 in 3/6 run) because policy stopped exploring for better strategies.

TRPO entropy bonus dilemma:
- alpha=0.005: noise grows unboundedly (1.0 -> 7.44 in 316 iters)
- alpha=0.0: noise shrinks to floor (1.0 -> 0.17 in 454 iters)

Simplest fix: raise noise floor from 0.1 to 0.2. This matches the 3/6 run's converged
noise_std (0.20-0.24) and guarantees minimum exploration without any entropy bonus tuning.
The floor is a hard clamp -- no interaction with TRPO step dynamics.

### Changed
- `runners/base_runner.py`: `min_std` 0.1 -> 0.2 in `_apply_noise_floor()` -- ensures
  exploration persists at convergence, matching 3/6 run's natural noise level
- `algorithms/constraint_trpo.py`: `min_log_std` from `log(0.1)` to `log(0.2)` -- unified
  with base_runner floor

### Notes
- Next step: verify noise stabilizes at 0.2, then add constraints back
- The 3/6 run (IPO, entropy_coef=0.005) had implicit exploration from barrier pressure;
  Lagrangian with lambda=0 has no equivalent, so the floor is essential

## [2026-03-16 Summary] Lagrangian migration + entropy stabilization (10 sessions)

### Context
Intensive debugging day (10 sessions) focused on migrating constraint enforcement from
IPO log-barrier to Lagrangian primal-dual, and fixing the entropy collapse/explosion cycle.

**Core problem**: IPO log-barrier assumed feasible start, but random policy starts infeasible.
Barrier gradient's easiest path was reducing action variance (kills exploration). Multiple
fixes attempted (barrier_t tuning, noise floors/ceilings, adaptive thresholds, n_active
normalization) were all band-aids. Fundamental fix: Lagrangian primal-dual where lambda
starts at 0 and grows with violations.

**Entropy debugging arc**: (1) Initial 6-bug fix session -> entropy still collapsed.
(2) barrier_t 10->50 + n_active normalization + entropy_coef 0.04 -> explosion (std=1678).
(3) Revert entropy_coef to 0.02 + add noise ceiling -> balanced (std=0.48, best run).
(4) Replace IPO with Lagrangian -> entropy collapsed via std cost gradient path.
(5) Detach std from cost gradient -> entropy locked at ceiling (entropy_coef push only).
(6) Remove entropy_coef entirely -> natural reward-driven std decrease. (7) Lambda warmup
+ d_k^2 cost value normalization + budget recalibration from unconstrained baselines.
(8) Reward advantage normalization + LS-gated lambda/encoder updates. (9) SAC-style target
entropy -> nearly worked but alpha_lr too slow initially. (10) Encoder z detach from cost
surrogate -> LS 100% success, BUT error stuck at 17-20 deg (effort_limit budget too tight).

### Changed
- `algorithms/constraint_trpo.py`: Replaced IPO log-barrier with Lagrangian primal-dual
  (lambda_k dual variables, lr_lambda=0.01, lambda_max=20, lambda_warmup_frac=0.3).
  Detached std from cost gradient (`_log_prob_mean_only()`). Detached encoder z from cost
  surrogate. Added reward advantage normalization. LS-gated lambda/encoder/z_bounds updates.
  d_k-normalized dual update. d_k^2-normalized cost value loss. SAC-style alpha_entropy
  (target_entropy, alpha_entropy_lr, alpha_entropy_init). Removed: barrier params, adaptive
  threshold, n_active normalization, line_search_cost_margin, encoder_value_grad_scale.
- `agents/rsl_rl_ppo_cfg.py`: Replaced barrier params with Lagrangian params (lr_lambda,
  lambda_max, lambda_init, lambda_warmup_frac). Constraint budgets recalibrated from
  unconstrained baseline (oscillation 0.3->0.4, yaw_vel 0.15->0.4). entropy_coef replaced
  by target_entropy/alpha_entropy_lr/alpha_entropy_init (final: all 0.0).
- `runners/constraint_encoder_runner.py`: Lambda checkpoint persistence (lambda_state.pt).
  Logging: barrier metrics -> lambda/violation metrics. log_alpha state in checkpoints.
- `config.py`: energy_weight=0.0, smoothness_weight=0.0 (double-counting with constraints).
  effort_limit budget 0.05->0.25. Removed joint_effort_limit_range override.
- `mdp/constraints.py`: effort_limit_cost uses per-env DR'd limits. Removed dead fields
  (barrier_t, barrier_t_final, barrier_t_schedule_frac, adaptive_threshold_alpha).
- `runners/base_runner.py`: Noise floor 0.15->0.25->0.1, ceiling added then removed.

### Fixed
- `algorithms/constraint_trpo.py`: Cost value loss clamps targets >=0. NaN guard on cost
  advantage normalization. Log_std clamp before KL measurement.
- `mdp/constraints.py`: effort_limit_cost uses per-env DR'd limits (was cached scalar).

### Notes
- Key architectural decisions: (1) std detach from cost = permanent fix for entropy collapse,
  (2) Lagrangian vs IPO = handles infeasible start, (3) LS-gated updates = prevents death
  spiral during line search failures, (4) z detach from cost = prevents encoder instability
- Best run of the day: `04-27-48` (reward 42, roll 11.6 deg, pitch 13.4 deg, std=0.48)
  but that used 8 constraints which were later reduced to 3 for stability

## [2026-03-10] Entropy collapse fix: raise noise floor + remove effort_limit DR conflict

### Context
Analysis of constrained_encoder_base run `klmm0hqj` (372 steps, barrier_t=50/100) showed
policy learning reward (0 -> 1.5, attitude error 30 -> 15 deg) but arm freezing at singularity.

Previous diagnosis was wrong: line_search_success was 99.4% (333/335), NOT 80% failure.
Barrier_t=50/100 is fine. The actual root cause chain:
1. Entropy collapsed by step 50 (2.84 -> -0.69), noise_std hit floor (0.98 -> 0.15)
2. Two inconsistent noise floors existed: constraint_trpo.py (0.1) vs base_runner.py (0.15)
3. std=0.15 -> 95% of actions within +-0.3 of mean -> exploration dies
4. "Don't move arm" is rational under low exploration (avoids 5 arm-related constraint costs)
5. Arm drifts to singularity (joint_pos_mean: 1.6 -> 5.4), costs explode
6. Narrow signal -> encoder z saturates (+-0.98 by step 50)

Additionally, `HeroAgentConstrainedEncoderEnvCfg` overrode `joint_effort_limit_range=(1.3, 1.5)`,
allowing PhysX 130-150% stall torque while `effort_limit_cost` checks against 100% -- a permanent
training conflict where normal actuator behavior always violates the constraint.

Barrier_t reverted from 50/100 back to 10/50 (both values produce >99% ls_success; lower
barrier gives tighter constraint enforcement from the start).

### Changed
- `algorithms/constraint_trpo.py`: `min_log_std` from `log(0.1)` to `log(0.25)` -- unified with base_runner floor
- `runners/base_runner.py`: `min_std` from 0.15 to 0.25 -- at std=0.25, entropy ~0.07 (vs -0.95 at 0.15)
- `config.py`: Removed `joint_effort_limit_range=(1.3, 1.5)` override from `HeroAgentConstrainedEncoderEnvCfg`, restoring unified default (0.7, 1.0)
- `agents/rsl_rl_ppo_cfg.py`: `barrier_t` 50.0 -> 10.0, `barrier_t_final` 100.0 -> 50.0 (revert to original)
- `mdp/constraints.py`: `ALBCConstraintCfg.barrier_t` 50.0 -> 1.0, `barrier_t_final` 100.0 -> 50.0 (dead field, synced with revert)

### Notes
- std=0.25 gives 95% of samples within +-0.5 of mean for [-1,1] actions -- wider exploration without excessive noise
- Two noise floors now unified at 0.25 (constraint_trpo.py post-step + base_runner.py per-iteration)

## [2026-03-09] barrier_t fix: 10->50 initial, 50->100 final

### Context
Constrained-Encoder-Base training (320 iterations) converged to a local optimum where
the arm stopped moving entirely. Full root cause chain: low barrier_t (10-20) caused
cost gradient to dominate policy_loss -> line search failed 80% of the time (success=0
from step ~80) -> actor params reverted every iteration (no learning) -> encoder gated
out (no policy-loss gradient) -> z saturated to [-1, +1] (z_bounds_loss too weak alone)
-> encoder grad_norm -> 0 (dead) -> entropy collapsed (2.0 -> 0 by step 75) ->
noise_std hit floor (0.15) -> arm froze.

### Changed
- `agents/rsl_rl_ppo_cfg.py`: `barrier_t` 10.0 -> 50.0, `barrier_t_final` 50.0 -> 100.0 in `RslRlConstraintTRPOAlgorithmCfg`
- `mdp/constraints.py`: `ALBCConstraintCfg.barrier_t` 1.0 -> 50.0, `barrier_t_final` 50.0 -> 100.0 (dead field, synced for consistency)

---

## [2026-03-09 Summary] NORBC conformance + actuator DR unification

### Context
Systematic NORBC paper comparison and actuator parameter audit. Asymmetric critic (raw
privileged 19D instead of encoder z 13D) was the highest-impact change. Also unified all
environments to use physically-grounded actuator DR ranges from Dynamixel XW540-T260-R
datasheet. Per-constraint cost advantage normalization added (NORBC Sec IV-B).

### Changed
- `encoder/actor_critic_encoder.py`: Asymmetric critic (`_get_critic_obs()` bypassing encoder, 32D input)
- `encoder/actor_critic_encoder_constrained.py`: Cost critic asymmetric, K mismatch detection
- `algorithms/constraint_trpo.py`: Update order threshold->policy->value, instantaneous threshold, pure MSE value, per-constraint cost advantage standardization
- `config.py`: Unified actuator DR (Kp 40-120, Kd 0.5-5.0, effort 0.7-1.0), removed TDC overrides
- `hero_agent.py`: `velocity_limit_sim` 6.28 -> 4.19 rad/s (datasheet 40 rpm)

### Removed
- `algorithms/constraint_trpo.py`: `encoder_value_grad_scale`, `adaptive_ema_alpha`, `_compute_barrier_loss()`

## [2026-03-08 Summary] Constraint system build-out and stabilization

### Context
Built constraint system from 3 to 8 terms, converted 3 binary costs to continuous (NORBC
average type), added cost critic softplus fix, and tuned barrier schedule. Multiple rounds
of budget tuning driven by WandB analysis. Key fix: cost critic negative V_cost (softplus
output activation) caused catastrophic instability at step ~650.

### Changed
- `mdp/constraints.py`: ConstraintTermCfg registry, 8 cost functions (final: accum_rot, attitude_abs, singularity, effort_limit binary; joint_vel, oscillation, yaw_vel, cob_cog continuous)
- `algorithms/constraint_trpo.py`: barrier_t 1.0->10.0, value_lr 3e-4->1e-3, barrier_t_schedule_frac, mean_cost_returns clamp
- `encoder/actor_critic_encoder_constrained.py`: `F.softplus()` on cost critic output, K mismatch detection
- `config.py`: 8-term constraint list, budgets tuned (joint_vel=1.0, singularity=0.15, attitude_err=0.087, attitude_abs limit 60 deg)
- `runners/constraint_encoder_runner.py`: Named constraint logging, auto-sync K, lambda checkpoint

## [2026-03-05/06 Summary] ConstraintTRPO initial implementation

### Context
Full NORBC-style constrained RL (IPO + TRPO) implementation. 6 critical bugs fixed across
5 debugging rounds. Encoder value gradient experiments (shelved: too aggressive at any scale).
Entropy_coef finalized at 0.005. Equilibrium joint init implemented then reverted.

### Added
- `algorithms/constraint_trpo.py`: Full TRPO + IPO (~600 lines)
- `encoder/actor_critic_encoder_constrained.py`: Multi-head cost critic
- `runners/constraint_encoder_runner.py`: Constraint metrics logging
- `mdp/constraints.py`: Initial 3 binary cost functions, `compute_all_costs()`
- `algorithms/ppo_patch.py`, `docs/THEORETICAL_ANALYSIS.md`, `play.py`, `eval_dr_comparison.py`

### Changed
- `base_env.py`: Accumulated rotation tracking, cost computation, constraint buffers
- `config.py`: Registered `Isaac-HeroAgent-Constrained-Encoder-Base-v0`

### Notes
- Baseline run (15-30-39) best overall: mean_reward 80, attitude error 8-10 deg
