# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [2026-03-09] NORBC conformance: asymmetric critic + algorithm fixes

### Context
Systematic equation-by-equation comparison of NORBC paper (Figure 2 / Algorithm 1)
against our ConstraintTRPO implementation revealed 1 structural + 3 algorithmic
differences.

HIGH impact: Critic input path was symmetric (encoder z 13D) instead of asymmetric
(raw privileged 19D). This caused: (1) critics estimated values through a 13D
bottleneck, losing information; (2) value loss gradient flowed through encoder,
potentially accelerating tanh saturation.

MEDIUM: Barrier loss was added to value update (NORBC has barrier only in policy
objective). encoder_value_grad_scale code was now incompatible with asymmetric
critic (value loss no longer touches encoder).

LOW: Adaptive threshold used EMA smoothing instead of NORBC Eq 11's instantaneous
assignment. Update order was value->policy->threshold instead of NORBC's
threshold->policy->value.

Theoretical review confirmed: discounted budget conversion D_k/(1-gamma), IPO
gradient 1/((1-gamma)*t*margin), cost GAE, Fisher/CG are all mathematically correct.

### Changed
- `encoder/actor_critic_encoder.py`: Added `asymmetric_critic` flag (default False for PPO compat), `_get_critic_obs()` method bypassing encoder, critic MLP input 32D when asymmetric, `_handle_critic_dim_mismatch()` for checkpoint compat
- `encoder/actor_critic_encoder_constrained.py`: Cost critic uses `num_critic_obs` (32D asymmetric), `evaluate_costs()` routes through `_get_critic_obs()`, `load_state_dict()` handles cost_critic dim mismatch
- `agents/rsl_rl_ppo_cfg.py`: `RslRlPpoActorCriticEncoderConstrainedCfg` now has `asymmetric_critic=True` (NORBC default)
- `algorithms/constraint_trpo.py`: Update order changed to threshold->policy->value (NORBC Algorithm 1), adaptive threshold now instantaneous (no EMA, Eq 11 exact), value update is pure MSE (no barrier)

### Removed
- `algorithms/constraint_trpo.py`: `encoder_value_grad_scale` parameter and all related code (1b full-batch block, grad merge block, return dict entry) -- incompatible with asymmetric critic
- `algorithms/constraint_trpo.py`: `adaptive_ema_alpha` parameter -- replaced by instantaneous threshold
- `algorithms/constraint_trpo.py`: `_compute_barrier_loss()` method -- was only called from value update barrier (now removed), dead code
- `algorithms/constraint_trpo.py`: `barrier` key from loss_dict return -- barrier no longer in value update
- `agents/rsl_rl_ppo_cfg.py`: `encoder_value_grad_scale` and `adaptive_ema_alpha` fields from `RslRlConstraintTRPOAlgorithmCfg`

### Notes
- PPO encoder (non-constrained) retains symmetric critic (asymmetric_critic=False default) -- HORA/RMA standard
- Existing symmetric checkpoints will trigger graceful reinit of critic layers on load (dimension mismatch detection)
- Cost critic softplus activation retained (V_cost >= 0, independent of this change)
- z_bounds_loss retained (HORA regularization, unrelated to NORBC changes)
- `_compute_barrier_loss()` computed barrier loss (log), while inline code computes barrier-weighted cost surrogate (gradient) -- not a DRY candidate, just dead code

## [2026-03-08] Cost critic softplus fix + budget tuning (post-750-iter crash analysis)

### Context
WandB analysis of ~750 iteration ConstraintTRPO run revealed catastrophic instability
at step ~650: cost_return_joint_vel dropped to -600, cost_return_oscillation to -75,
while cost_return_singularity exploded to 400 and cost_return_attitude_abs to 300.
Attitude error regressed from 12-15 deg back to 20-25 deg.

Root cause: cost value function (MLP with linear output) predicted negative V_cost
values. Since cost GAE return = V_cost + advantage, negative V_cost made the GAE
return negative despite all per-step costs being non-negative. Negative mean_cost_returns
inflated the barrier margin (d_k - (-X) = d_k + X), effectively disabling constraint
pressure. Without constraints, the policy violated singularity and attitude limits.

NORBC theory confirmation: J_{C_k} = E[sum gamma^t * C_k] >= 0 by definition (C_k >= 0,
gamma > 0). The negative values were purely estimation error from the cost value function.

Also tightened joint_vel budget (50% headroom was excessive) and loosened singularity
budget (cost_return was crossing d_k at step 375).

### Fixed
- `encoder/actor_critic_encoder_constrained.py`: Added `F.softplus()` on cost critic output -- ensures V_cost >= 0 (root cause fix)
- `algorithms/constraint_trpo.py`: Clamped `mean_cost_returns` to non-negative (safety net for residual GAE errors)

### Changed
- `config.py`: joint_vel budget 1.5 -> 1.0 rad/s (cost_return was only 50% of d_k=150)
- `config.py`: singularity budget 0.10 -> 0.15 (cost_return 13 was crossing d_k=10)
- `agents/rsl_rl_ppo_cfg.py`: constraint_budgets tuple synced (1.5->1.0, 0.10->0.15)

### Notes
- softplus chosen over relu: smooth gradient everywhere, no dead neurons, softplus(x) ~ x for large x
- Checkpoint compatible: MLP architecture unchanged, only output activation added
- Two-layer defense: softplus prevents V_cost < 0, clamp catches edge cases in GAE computation

## [2026-03-08] Constraint redesign: 3 binary costs -> continuous (NORBC average type)

### Context
NORBC paper analysis identified a gradient vanishing problem with tight binary constraints:
when a binary cost fires 100% of the time (all trajectories equally violated), cost_advantage
approaches zero everywhere, providing no gradient signal. This affected 3 of 6 constraints
(joint_vel, oscillation, attitude_err) which are quality metrics, not hard safety limits.

Solution: convert these 3 from binary indicator (0/1) to continuous cost (raw physical value),
with cost_type="average" and budgets in physical units instead of violation probabilities.
Continuous costs naturally produce trajectory-level variance (different magnitudes), ensuring
meaningful cost_advantage gradients regardless of how tight the budget is set.

Budget rationale: joint_vel=1.5 rad/s (~36% of 4.19 no-load speed), oscillation=0.3 rad/s
(half of old binary threshold), attitude_err=0.087 rad (5 deg convergence target).

### Changed
- `mdp/constraints.py`: `joint_velocity_cost` -- removed `limit` param, returns raw max |vel| (rad/s) instead of binary
- `mdp/constraints.py`: `joint_oscillation_cost` -- removed `limit` param, returns raw HF RMS (rad/s) instead of binary
- `mdp/constraints.py`: `attitude_error_cost` -- removed `limit` param, returns raw max |err| (rad) instead of binary
- `config.py`: `joint_vel` constraint -- budget 0.15 (prob) -> 1.5 (rad/s), cost_type "binary" -> "average", params cleared
- `config.py`: `oscillation` constraint -- budget 0.15 (prob) -> 0.3 (rad/s), cost_type "binary" -> "average", params cleared
- `config.py`: `attitude_err` constraint -- budget 0.02 (prob) -> 0.087 (rad, ~5 deg), cost_type "binary" -> "average", params cleared
- `agents/rsl_rl_ppo_cfg.py`: constraint_budgets default (0.15, 0.02, 0.15, 0.01, 0.02, 0.10) -> (1.5, 0.02, 0.3, 0.01, 0.087, 0.10)

### Notes
- No algorithm changes needed: ConstraintTRPO treats binary and average costs identically (d_k = D_k / (1-gamma) works for both)
- 3 remaining binary constraints unchanged: accum_rot (hard safety), attitude_abs (capsizing), singularity (kinematic limit)
- Amplification sanity check (barrier_t=10): attitude_err ~11.5x, oscillation ~3.3x, joint_vel ~0.67x -- manageable

## [2026-03-08] Constraint tuning: attitude_abs warning band + attitude_err budget

### Context
WandB analysis of ConstraintTRPO run (~600 iter) after barrier_t fix. Key findings:

attitude_err (limit=7 deg, budget=0.02): massively violated (cost_return 60-100 vs d_k=2).
100% violation rate meant cost_advantage = 0 everywhere -- zero gradient signal. Error
reduction (30->15 deg) was driven entirely by reward, not the constraint. Reverted limit
to 15 deg (0.262 rad). Budget kept at 0.02 for stronger amplification when active.

attitude_abs (limit=80 deg, budget=0.01): never active -- 10 deg warning band (80->90 deg)
gave only 1-3 cost steps before termination. Lowered limit to 60 deg (1.047 rad) for a
30 deg warning band.

### Changed
- `config.py`: attitude_error_cost limit 0.122 rad (7 deg) -> 0.262 rad (15 deg), budget 0.05 -> 0.02
- `config.py`: attitude_absolute_cost limit 1.396 rad (80 deg) -> 1.047 rad (60 deg)
- `agents/rsl_rl_ppo_cfg.py`: constraint_budgets attitude_err slot 0.05 -> 0.02

### Notes
- Encoder z_std=0.85, grad_norm 0.008->0.001: saturation vs convergence unclear. Per-dimension z-DR correlation needed.
- ConstraintTRPO encoder optimizer has weight_decay=0 (PPO fix was 1e-4). Not yet addressed.

## [2026-03-08] ConstraintTRPO gradient balance fix + barrier schedule refactor

### Context
IPO cost surrogate amplification `1/((1-gamma) * barrier_t * margin_k)` with `barrier_t=1.0`
produced ~193x total gradient amplification across 6 constraints, overwhelming the 1x reward
gradient. Policy collapsed to "minimize all activity".

Fix: `barrier_t` 1.0 -> 10.0 (reduces to ~19x). Changed `barrier_t_schedule_iters` to
fraction-based `barrier_t_schedule_frac=0.4` for max_iterations-independent scheduling.
Also `value_lr` 3e-4 -> 1e-3 (TRPO standard for separate value optimizer).

### Changed
- `algorithms/constraint_trpo.py`: `barrier_t` 1.0 -> 10.0, `value_lr` 3e-4 -> 1e-3, `barrier_t_schedule_iters` -> `barrier_t_schedule_frac: float = 0.4`, store `barrier_t_init`
- `agents/rsl_rl_ppo_cfg.py`: Matching config defaults updated
- `mdp/constraints.py`: `barrier_t_schedule_frac` in ALBCConstraintCfg
- `runners/base_runner.py`: Call `alg.set_max_iterations()` to resolve fraction to absolute iters

### Added
- `algorithms/constraint_trpo.py`: `set_max_iterations()` method

## [2026-03-08] Constraint system expansion + registry pattern + logging cleanup

### Context
Expanded constraints from 3 (joint_vel, accum_rot, oscillation) to 6 by adding attitude_absolute,
attitude_error, and singularity. Refactored from hardcoded fields to registry pattern
(ConstraintTermCfg list). Reduced WandB metrics from 33 to 14.

### Added
- `mdp/constraints.py`: `ConstraintTermCfg` registry, `attitude_absolute_cost`, `attitude_error_cost`, `singularity_cost`, `action_smoothness_cost`, `angular_velocity_cost` (last two reserve for Phase 2)
- `encoder/actor_critic_encoder_constrained.py`: K mismatch detection in load_state_dict
- `runners/constraint_encoder_runner.py`: Auto-sync num_constraints from env config

### Changed
- `mdp/constraints.py`: `ALBCConstraintCfg` to `terms: list[ConstraintTermCfg]` with derived properties
- `config.py`: 6-term constraint list, attitude_abs budget 0.10 -> 0.01
- `agents/rsl_rl_ppo_cfg.py`: num_constraints 3 -> 6, budgets updated
- `runners/constraint_encoder_runner.py`: Named constraint logging (not numeric indices)

### Removed
- `mdp/constraints.py`: `joint_position_cost` (replaced by `singularity_cost`)
- `runners/constraint_encoder_runner.py`: 19 redundant WandB metrics

### Notes
- Phase 2 reserve costs (action_smoothness, angular_velocity) implemented, just need config registration
- K=3 checkpoint backward compatibility: cost_critic auto-reinitializes on K mismatch

---

## [2026-03-05/06 Summary] ConstraintTRPO implementation, stabilization, and encoder experiments

### Context
Full implementation and stabilization of NORBC-style constrained RL (IPO + TRPO) for Hero Agent.
Initial implementation had 6 critical bugs requiring 5 rounds of debugging to reach 100% line
search success. Also explored encoder value gradient injection and equilibrium joint initialization
-- both ultimately reverted/disabled after experimental evidence showed they were counterproductive.

Key bugs fixed in ConstraintTRPO (discovery order):
1. Cost advantage normalization amplified noise 1000x when constraints satisfied
2. z_bounds_loss updated encoder 20x/iter during value loop (violated TRPO old-policy assumption)
3. Barrier margin floor 1e-6 caused gradient explosion (fixed to 0.1*d_k)
4. Gradient/line-search objective mismatch
5. Missing 1/(1-gamma) factor (cost gradient 100x too weak)
6. TRPO step direction was +F^{-1}g (ascent) instead of -F^{-1}g (descent)

Encoder value gradient experiments: PPO encoder gets 60 updates/iter (surrogate+value+z_bounds),
TRPO gets only 2 (policy+z_bounds). Attempted to restore value gradient via mini-batch
accumulation (20:1 ratio imbalance, z collapse) then full-batch autograd.grad (1:1 ratio,
still too aggressive at scale>=0.01). Final state: `encoder_value_grad_scale=0.0` (disabled
by default), full-batch infrastructure preserved.

Entropy experiments: entropy_coef tested at 0.005, 0.01, 0.02. Higher entropy (0.02) stabilized
noise_std at 0.40 but HURT control quality (2x torque, 2x oscillation). Conclusion: entropy
collapse is desirable for 2-DOF attitude control. Final: entropy_coef=0.005.

Equilibrium joint init: analytical IK from roll/pitch, verified correct. Reverted to random
default due to sim-to-real coverage gap (no large-mismatch training at episode start).

### Added
- `algorithms/constraint_trpo.py`: Full TRPO + IPO (~600 lines) -- CG solver, Fisher-vector product, line search, log-barrier, adaptive thresholds, `encoder_value_grad_scale` param, `kl_trpo` metric, min log_std floor
- `encoder/actor_critic_encoder_constrained.py`: Multi-head cost critic (K outputs)
- `runners/constraint_encoder_runner.py`: Barrier schedule + constraint metrics logging
- `mdp/constraints.py`: 3 binary cost functions (joint_velocity, accumulated_rotation, joint_oscillation), `compute_all_costs()`
- `algorithms/ppo_patch.py`: Monkey-patch for RSL-RL PPO encoder optimizer (WD=1e-5)
- `docs/THEORETICAL_ANALYSIS.md`: TDC, rewards, NORBC pipeline analysis
- `mdp/events.py`: `compute_equilibrium_joint_positions()` (opt-in via `joint_init_mode`, default="random")
- `play.py` + `eval_dr_comparison.py`: ConstraintEncoderRunner + ActorCriticEncoderConstrained support

### Changed
- `base_env.py`: Accumulated rotation tracking, cost computation, constraint buffers, TDE prev-step pattern, reordered reset (pose before joints), 3-way joint init dispatch
- `config.py`: `HeroAgentConstrainedEncoderEnvCfg` registered, `joint_init_mode` (default="random"), `linear_error_weight=0.0` in constrained env, `target_attitude_range` capped to 20 deg
- `__init__.py`: Registered `Isaac-HeroAgent-Constrained-Encoder-Base-v0`
- `encoder/adaptation.py`: Phase 2 critic evaluate() uses z_hat
- `controllers/tdc.py`: F_bu accepts per-env tensor; extracted `_set_param()` helper
- `constraint_trpo.py`: Sign fix, 1/(1-gamma) scaling, barrier margin floor, deferred encoder update, linearized surrogate, full-batch value gradient (gated behind scale=0.0)
- `rsl_rl_ppo_cfg.py`: ConstraintTRPO algorithm config (entropy_coef=0.005 final, encoder_value_grad_scale=0.0)
- `eval_dr_comparison.py`: Added buoy perturbation DR fields, uses training timestamp for output dir

### Removed
- `base_env.py`: `_cumulative_effort` buffer, `HeroAgentEnvWindow` class
- `mdp/rewards.py`: `action_rate_penalty()`, `angular_velocity_penalty()` (weight=0 everywhere)
- MPC docstring references from controllers/encoder/runners `__init__.py`
- `eval_dr_comparison.py`: SAC-MPC dead code (~40 lines), dead DR enable line
- `constraint_trpo.py`: Superseded objective functions, cost advantage normalization, diagnostic print()
- `base_runner.py`: Dead `_save_best_model()` and `_best_mean_reward`

### Fixed
- `constraint_trpo.py`: Barrier loss gradient path, encoder grad isolation, NaN guard (shs <= 0), line search margin floor, squeeze(-1) for B=1 safety
- `mdp/events.py`: IndexError in equilibrium joint positions (1D tensor indexed as 2D)
- `mdp/rewards.py`: Restored `termination_penalty` accidentally removed during cleanup
- `eval_dr_comparison.py`: Steady-state comment ("last 50% of segment")

### Notes
- Encoder sensitivity ~4x weaker than PPO (z_range 0.29 vs 1.24) due to 2 vs 60 updates/iter. Value gradient approach shelved (too aggressive at any tested scale).
- Baseline run (15-30-39) best overall: mean_reward 80, attitude error 8-10 deg, smooth control.
