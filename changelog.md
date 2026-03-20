# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [2026-03-20] Remove constrained_encoder_base code from hero_agent

### Context
All constrained RL code has been fully migrated to `constrained_albc/` package.
This session removes the remaining constrained_encoder_base references from
hero_agent to eliminate dead imports and prevent runtime confusion between the
two packages. 4 files deleted, 10 files edited, ruff check clean.

### Changed
- `base_env.py`: Removed `compute_all_costs` import, constraint cost computation block,
  `_prev_attitude_error_rp` buffer (init + 3 reset/update sites), `_check_dr_infeasibility()`
  method and its call site, `log_dr_infeasibility` import.
- `config.py`: Removed constraint imports (ALBCConstraintCfg, ConstraintTermCfg, 6 cost functions).
  Deleted `HeroAgentConstrainedEncoderEnvCfg` class (~70 lines).
- `agents/rsl_rl_ppo_cfg.py`: Removed ConstraintTRPO/ConstraintEncoderRunner/
  ActorCriticEncoderConstrained imports and module registrations. Deleted 3 config classes
  (RslRlConstraintTRPOAlgorithmCfg, RslRlPpoActorCriticEncoderConstrainedCfg,
  HeroAgentConstrainedEncoderRunnerCfg, ~95 lines).
- `__init__.py`: Removed gym.register for Isaac-HeroAgent-Constrained-Encoder-Base-v0,
  config import, and __all__ entry.
- `encoder/__init__.py`, `agents/__init__.py`, `runners/__init__.py`,
  `algorithms/__init__.py`, `mdp/__init__.py`: Removed all constrained-related imports/exports.
- `utils/logging.py`: Deleted `log_dr_infeasibility()` and `_get_dr_infeasibility_logger()`.
- `utils/__init__.py`: Removed `log_dr_infeasibility` from imports and __all__.
- `runners/base_runner.py`: Removed ConstraintTRPO reference in docstring comment.

### Removed
- `encoder/actor_critic_encoder_constrained.py`: Constrained encoder network (4.5KB)
- `algorithms/constraint_trpo.py`: C-TRPO algorithm (42KB)
- `runners/constraint_encoder_runner.py`: Constrained encoder runner (6.4KB)
- `mdp/constraints.py`: Constraint cost functions (12.5KB)

## [2026-03-20] C-TRPO mode oscillation fix: EMA smoothing + cost critic LR gating

### Context
C-TRPO training exhibited rapid mode oscillation (loss/mode flipping 0<->1 every
5-10 iterations in later training). Root cause analysis identified 5 layers:
- RC1 (HIGH): Cost critic decoupling -- `_update_values()` runs 20 gradient steps
  regardless of actor update success. When actor is frozen (ls_success=False), cost
  critic drifts, changing margin without policy change -> "phantom" mode switches.
- RC2 (HIGH): Weak barrier beta=0.01 -- barrier penalty negligible until margin < 1.
- RC3 (MEDIUM): Hard binary mode switch with no continuous interpolation.
- RC4 (MEDIUM): Narrow hysteresis band (0.8) allowing rapid safe<->recovery cycling.
- RC5 (LOW): Binary constraint volatility (minor with 4096-env averaging).

Implemented Approach B (B1 + B2, ~30 lines) targeting RC1, plus Approach A tuning
targeting RC2 + RC4. Approach C (soft mode transition, ~120 lines) reserved as
follow-up if oscillation persists.

### Changed
- `algorithms/constraint_trpo.py`: B1 -- EMA smoothing (alpha=0.3) on mean_cost_returns
  before margin computation. `_compute_margins()` now receives smoothed values instead
  of raw per-iteration cost returns. ~3-iteration lag, sufficient for real violation
  detection while filtering single-iteration cost value jumps.
- `algorithms/constraint_trpo.py`: B2 -- Cost critic LR gated on actor update success.
  When `ls_success=False`, value optimizer LR reduced to 10% of base LR. Prevents
  cost critic from drifting while actor is frozen, eliminating the primary source of
  phantom mode switches. LR restored after `_update_values()` completes.
- `agents/rsl_rl_ppo_cfg.py`: Added `ema_cost_alpha=0.3` parameter. Updated `beta`
  default 0.01->0.05 (5x barrier strengthening). Updated `recovery_threshold_frac`
  default 0.8->0.6 (wider hysteresis band).
- `runners/constraint_encoder_runner.py`: EMA state (ema_cost_returns, ema_initialized)
  persisted in barrier_state.pt checkpoint. Backward-compatible with old checkpoints
  (missing EMA keys handled gracefully). Added per-constraint `ema_cost_return` metric
  to WandB/TensorBoard logging for monitoring smoothed vs raw cost returns.

### Notes
- B3 (adaptive beta based on min margin) is designed but not implemented -- add if
  beta=0.05 proves insufficient after B1+B2 stabilize mode switching.
- Approach C (soft mode transition with sigmoid alpha_k blending) is the structural
  solution if oscillation fundamentally persists. ~120 lines, replaces binary
  safe/recovery with continuous interpolation. Reserved as follow-up.
- Verification: run with same config, check Loss/mode switch period > 30 iters in
  step 100-300 range, Constraint/cost_return stable near d_k, Policy/line_search_success > 70%.

## [2026-03-20] Remove PBRS progress reward (redundant with quadratic command)

### Context
Analysis of reward structure revealed that `progress_reward` (PBRS: prev_potential -
gamma * potential) is redundant with `command_reward` (quadratic: -(roll_err^2 + pitch_err^2)).
Both derive from the same attitude_error variable, producing identical gradient directions.

Key findings from mathematical analysis:
- PBRS telescopes to initial minus discounted final error (path-independent),
  while command integrates error over entire trajectory (path-dependent).
  However, for stabilization tasks with monotonic error decrease, both produce
  mirror-image WandB curves with no independent learning signal.
- PBRS theorem (Ng et al. 1999) guarantees progress does not change optimal policy.
- Progress was NOT dt-scaled, making it ~2x stronger per step than command --
  effectively dominating the reward signal while adding no new information.
- C-TRPO uses full-batch natural gradient, so the "early training value function
  noise" argument for PBRS is weaker than in PPO.

Dry run verified: 2 iterations, 4 envs, headless -- clean execution with only
command + smoothness reward terms logged.

### Removed
- `mdp/rewards.py`: Deleted `progress_reward()` function and `progress_weight`/`progress_gamma`
  fields from `ALBCRewardCfg`. Reward architecture simplified from 3-term to 2-term.
- `albc_env.py`: Removed `_potentials` and `_prev_potentials` buffers (only used by progress).
  Renamed `_update_potentials()` to `_update_attitude_error()` (simpler, reflects actual purpose).
  Removed progress term construction in `_build_reward_terms()`.
  Removed potential initialization/reset in `_reset_task_and_state()`.
- `config.py`: Removed `progress_weight=2.0` from `ALBCRewardCfg` instantiation.
- `mdp/__init__.py`: Removed `progress_reward` from imports and `__all__`.

## [2026-03-20] config.py code review: dead fields, barrier singularity, documentation

### Context
Post-simplification code review of `config.py` plus cross-referenced issues in
`constraint_trpo.py`. 8 verified issues found (3 exploration agents, false positives
filtered). Issues 1-4 in config.py (dead field, undocumented budget scaling, magic numbers,
missing validation); Issues 5-8 in constraint_trpo.py (barrier near-singularity, any-recovery
design, kwargs absorption, dead clamp guard). Issue 4 (DR range validation) skipped as low ROI.

Key finding: barrier penalty has a singularity gap -- when margin is small positive (0, ~0.01)
but recovery mode hasn't triggered (threshold is margin <= 0), phi_pp = 1/m^2 can reach 1e6,
causing explosive gradients. Clamping margin to min=0.01 caps phi_pp at 1e4 (barrier penalty
~100 with beta=0.01 and cost_surrogate=0.1).

### Changed
- `config.py`: Added budget D_k vs d_k documentation. Per-step budget D_k is scaled to
  discounted d_k = D_k / (1 - cost_gamma) = D_k * 100 by the algorithm. This relationship
  was undocumented, making budget tuning non-obvious.
- `config.py`: Added inline unit comments for constraint magic numbers: `1.396` -> `# ~80 deg`,
  `4.189` -> `# 40 RPM (Dynamixel XW540 no-load)`.
- `algorithms/constraint_trpo.py`: Added design note comment on "any-recovery" policy trade-off
  at lines 569-578. Documents that per-constraint blend is a known alternative but adds
  complexity with shared trust region interaction.

### Fixed
- `algorithms/constraint_trpo.py`: Barrier margin clamped to min=0.01 in
  `_compute_barrier_penalty()`. Prevents phi_pp explosion when margin is small positive but
  recovery hasn't triggered. Old: `1/(m^2 + 1e-8)` -> New: `1/max(m, 0.01)^2`.
- `algorithms/constraint_trpo.py`: `**_kwargs` now logs ignored kwargs at debug level instead
  of silently absorbing. Aids diagnosis when RSL-RL passes unexpected parameters.
- `algorithms/constraint_trpo.py`: Added comment explaining d_k^2 clamp (min=0.01) is a
  defensive guard that never activates with default cost_gamma=0.99 (min d_k=1.0).

### Removed
- `config.py`: Deleted dead `enable_payload: bool = True` field. Payload is always initialized
  and computed unconditionally since the simplification removed its conditional logic.

## [2026-03-20] MDP code review: 3 critical bugs + 5 theoretical fixes

### Context
Systematic code review of Constrained ALBC MDP modules (rewards.py, constraints.py,
observations.py, events.py, albc_env.py) using 3 parallel code-explorer agents.
Identified 3 critical bugs, 7 theoretical issues, 7 design items, and 6 minor items.
Fixed all critical bugs and 5 of 7 theoretical issues in this session.

BUG-1 root cause: `_update_action_buffers()` stored `self._actions` (current step a_t)
into `_prev_actions_obs` instead of `self._prev_actions` (a_{t-1}). This caused a causal
violation -- policy obs[11:13] contained the current action, while encoder proprio history
correctly used the previous action. Temporal inconsistency between policy and encoder.

BUG-2: `effort_limit_cost` compared `max(torques)` against `max(limits)` instead of
per-joint comparison. When joints have different DR'd limits, a violation on the weaker
joint could be masked by the stronger joint's higher limit.

BUG-3: Joint gain/friction randomization ran unconditionally even with `rand_cfg.enable=False`
(debug/eval mode). Comment claimed "ranges collapse to defaults" but actual DR ranges were
wide (Kp 40-120, Kd 0.5-5.0), so debug envs had randomized actuator properties.

### Fixed
- `albc_env.py`: BUG-1 -- `_prev_actions_obs` now stores `_prev_actions` (a_{t-1}) instead
  of `_actions` (a_t). Both sliced and full-clone paths corrected.
- `mdp/constraints.py`: BUG-2 -- `effort_limit_cost` uses per-joint comparison
  `(computed.abs() > limits).any(dim=-1)` instead of `max(dim=-1)` reduction on both sides.
- `albc_env.py`: BUG-3 -- Joint actuator DR (gains, effort limits, friction) wrapped in
  `if rand_cfg.enable:` guard. Debug/eval envs now keep default actuator properties.
- `mdp/constraints.py`: THEO-1 -- `overshoot_cost` checks `prev.abs() > threshold` (departure
  magnitude) instead of `curr.abs() > threshold` (landing magnitude). Catches small overshoots
  where zero crossing lands below threshold (e.g., prev=+0.04rad -> curr=-0.01rad).

### Changed
- `albc_env.py`: THEO-3 -- `_get_attitude_error()` returns cached `self._attitude_error` instead
  of recomputing. Safe because `_get_rewards()` -> `_update_potentials()` always runs first in
  Isaac Lab's step order (line 393 before 410 in direct_rl_env.py). Eliminates duplicate
  `compute_attitude_error()` call per step.
- `mdp/events.py`: THEO-6 -- Payload restoring moment clamp uses horizontal (xy) norm instead
  of 3D norm. Roll/pitch restoring moment depends only on horizontal offset; Z-component payload
  was being over-constrained.
- `mdp/events.py`: THEO-7 -- `_HydroBaseCache.inertia` fallback (0.5 * added_mass[3:6]) now
  emits `logger.warning()` when `rigid_body_inertia` is None. Added `import logging`.

### Removed
- `mdp/constraints.py`: THEO-2 -- Removed dead `cost_type` field from `ConstraintTermCfg`.
  Never read by `compute_all_costs()` or `constraint_trpo.py`. Removed `cost_type="average"`
  from `yaw_velocity_cost` term in `config.py`. Updated module docstring.

### Notes
- THEO-4 (PBRS L2 norm vs quadratic gradient mismatch): Not fixed -- reward landscape change
  would require full retraining. Documented only.
- THEO-5 (yaw in obs but not in reward): Not fixed -- obs dimension change affects encoder/actor
  architecture. Separate task.
- DES-1~7 and M-1~6: Out of scope (no functional impact). Documented in review plan.

## [2026-03-20] Full package code review: encoder optimizer resume + NaN guard

### Context
Systematic code review of entire constrained_albc package (6 review areas: agents,
encoder, algorithms, mdp, config+env, runners+utils). Found 1 confirmed bug
(encoder optimizer state not saved on checkpoint), 2 theoretical concerns (recovery
drops reward globally, standardization-barrier inverse variance), and 2 design issues
(unused cost_type field, missing isfinite guard on encoder loss).

BUG-1 root cause: `self.optimizer = self.value_optimizer` alias (constraint_trpo.py:210)
means OnPolicyRunner.save() only persists value_optimizer state_dict. The separate
`encoder_optimizer` (Adam with lr=3e-4, wd=1e-5) loses momentum (exp_avg, exp_avg_sq)
on resume, causing a transient gradient magnitude spike as Adam re-estimates statistics.

### Fixed
- `runners/constraint_encoder_runner.py`: Save/load `encoder_optimizer.pt` alongside
  `barrier_state.pt` in checkpoint. Uses existing `_save_aux_state`/`_load_aux_state`
  helpers. Load respects `load_optimizer` flag (skip during eval/play).
- `algorithms/constraint_trpo.py`: Added `torch.isfinite(total_loss)` guard before
  `.backward()` in `_update_encoder()`. NaN/Inf loss skips the epoch with warning
  instead of corrupting encoder parameters irreversibly.

### Notes
- THEORY-1 (recovery drops reward surrogate globally): conservative valid choice per
  C-TRPO paper (Muller et al. Sec 4.1). Monitor reward stalls during recovery.
- THEORY-2 (standardization-barrier inverse variance): tight constraints get stronger
  barrier (arguably correct). Undocumented interaction, monitor per-constraint magnitude.
- DESIGN-1 (cost_type field in ConstraintTermCfg): dead field, never consumed by
  algorithm. Deferred cleanup.

## [2026-03-20] Constrained ALBC encoder code review: DRY, perf, backward compat

### Context
Code review of constrained_albc encoder directory (`actor_critic_encoder.py`,
`actor_critic_encoder_constrained.py`) after the simplification session. Found 7 issues
(5 required code changes, 2 already fixed). Key findings: (1) `_encode()` defined but
never called -- `_get_combined_obs()` duplicated its logic inline (DRY violation from
simplification that inlined `_build_encoder_input()` but forgot to delegate). (2)
`update_normalization()` ran encoder forward pass with grad tracking on every env step
(262K unnecessary grad-enabled passes per iteration). (3) `load_state_dict()` had all
backward compatibility stripped, causing silent partial loads on architecture mismatch.
(4) softplus on cost critic biased gradient for near-zero costs. (5) z_bounds_loss
returned CPU tensor on fallback path.

### Changed
- `encoder/actor_critic_encoder.py`: `_get_combined_obs()` now delegates to `_encode()`
  instead of duplicating encoder forward-pass logic inline (DRY fix).
- `encoder/actor_critic_encoder.py`: `update_normalization()` wraps encoder call in
  `torch.no_grad()` and only runs when `actor_obs_normalization=True`. Saves 262K
  unnecessary grad-tracked encoder passes per iteration (4096 envs x 64 steps).
- `encoder/actor_critic_encoder_constrained.py`: Cost critic activation `F.softplus()`
  -> `F.relu()`. softplus required x -> -inf for zero output (gradient vanishing for
  healthy constraints); ReLU allows exact zero with finite MLP values.

### Fixed
- `encoder/actor_critic_encoder.py`: `z_bounds_loss()` device fallback uses
  `next(self.parameters()).device` instead of hardcoded "cpu" when `_last_z is None`.
- `encoder/actor_critic_encoder.py`: Restored `_handle_critic_dim_mismatch()` and full
  `load_state_dict()` with backward compatibility: encoder_obs_normalizer injection for
  old checkpoints, critic input dim mismatch detection + reinitialization, unknown key
  filtering with logging, missing essential key warnings.
- `encoder/actor_critic_encoder_constrained.py`: Restored `load_state_dict()` override
  with cost_critic handling: K mismatch detection (different num_constraints), input dim
  mismatch via parent `_handle_critic_dim_mismatch()`, missing cost_critic key injection.

### Notes
- MEDIUM-3 (num_encoder_epochs default=5) and LOW-2 (dead encoder_output_activation
  config field) already fixed in prior sessions
- Agent-reported "missing encoder gradient from cost surrogate" verified as by-design
  (encoder role is information compression, cost avoidance is actor's responsibility)

## [2026-03-20] albc_env.py code review: _prev_joint_pos timing + control_dt fix

### Context
Code review of `albc_env.py` (1016 lines) identified 2 bugs and 1 question requiring
user confirmation. BUG-1: `_prev_joint_pos` was set in `_reset_action_buffers()` (during
`_reset_framework` phase) but joint positions are subsequently changed by
`_reset_task_and_state()` (equilibrium/random init). This injected a false delta (~0.5 rad)
into `_accumulated_rotation` on the first step after reset, skewing IPO constraint budget.
BUG-2: `control_dt` used `physics_dt` instead of `step_dt` -- latent bug currently masked
by `decimation=1` but would cause position_delta underestimation if decimation changed.
BUG-3: encoder weight_decay 1e-5 vs hero_agent's 1e-4 -- deferred to user confirmation.

### Fixed
- `albc_env.py`: Added `_prev_joint_pos` re-sync at end of `_reset_task_and_state()` after
  joint positions are set to equilibrium/random. Prevents false delta in
  `_accumulated_rotation` (IPO constraint) on first post-reset step.
- `albc_env.py`: Changed `control_dt = self.physics_dt * ...` to `self.step_dt * ...`.
  Latent bug -- no behavioral change at current `decimation=1`, but correct for any value.

### Notes
- BUG-3 (encoder weight_decay 1e-5 in constraint_trpo.py vs 1e-4 in hero_agent) pending
  user confirmation on whether the difference is intentional.
- 10 additional items verified correct (VERIFY-1~6, DESIGN-1~3, MINOR-1~2).

## [2026-03-20] Constrained ALBC algorithms code review + runtime integration fixes

### Context
Executed constrained ALBC algorithms code review plan targeting mathematical correctness
and latent bugs. Plan identified 3 fixes (overshoot_cost cross-axis false positive,
num_encoder_epochs default mismatch, barrier beta docstring). During dry run verification,
5 additional runtime integration bugs were discovered that prevented the constrained_albc
task from running at all -- the previous simplification session removed compatibility shims
that were actually load-bearing, and the auto-sync mechanism was silently broken.

Root cause of runtime failures: (1) hero_agent and constrained_albc registered identical
class names into `_runner_module` namespace; alphabetical import order caused hero_agent to
overwrite constrained_albc's classes. (2) `train.py` `_RUNNER_MAP` only had hero_agent
paths. (3) Runner auto-sync used `hasattr()` on plain dicts (from `to_dict()`), silently
skipping the `num_constraints` sync. (4) RSL-RL `OnPolicyRunner` expected `rnd` and
`multi_gpu_cfg` attributes. (5) `storage.dones` shape was `(T,N,1)` not `(T,N)`.

Dry run verified: 5 iterations of `Isaac-Constrained-ALBC-Encoder-v0` with 64 envs
completed successfully after all fixes.

### Fixed
- `mdp/constraints.py`: `overshoot_cost` per-axis conjunction -- sign flip and magnitude
  now checked on the SAME axis. Previously `any(dim=-1)` for sign flip and `max(dim=-1)`
  for magnitude could match different axes, triggering false positive overshoot cost.
- `algorithms/constraint_trpo.py`: `num_encoder_epochs` default 5 -> 1 to match config.
  Default=5 would cause stale importance sampling ratio when instantiated without config.
- `algorithms/constraint_trpo.py`: Barrier beta docstring now documents re-parametrization
  `beta_code = beta_paper / (2*t)` absorbing the 1/2 and 1/t factors.
- `algorithms/constraint_trpo.py`: Added `**_kwargs` to `__init__()` for RSL-RL
  `multi_gpu_cfg` compatibility, and `self.rnd = None` for `OnPolicyRunner.learn()` line 84.
- `algorithms/constraint_trpo.py`: Fixed `_compute_cost_returns` dones shape -- added
  `.squeeze(-1)` before `unsqueeze(-1)` to handle `(T,N,1)` storage dones.
- `agents/rsl_rl_ppo_cfg.py`: ALBC-prefixed `_runner_module` registration names
  (`ALBCConstraintEncoderRunner`, `ALBCConstraintTRPO`, `ALBCActorCriticEncoderConstrained`)
  to avoid namespace collision with hero_agent's identically-named registrations.
- `runners/constraint_encoder_runner.py`: Changed `num_constraints` auto-sync from
  `hasattr()`/attribute access to dict key access (`in`/`[]`). `train_cfg` is a plain dict
  from `agent_cfg.to_dict()`, so `hasattr()` always returned False, silently skipping sync.
- `scripts/.../train.py`: Added `ALBCConstraintEncoderRunner` to `_RUNNER_MAP` pointing
  to `constrained_albc.runners.ConstraintEncoderRunner`.

### Notes
- Previous simplification session removed `**kwargs`, `self.rnd = None`, and `hasattr`
  guards from `ConstraintTRPO` (commit cbd2dd24) -- these were actually required for
  RSL-RL `OnPolicyRunner` compatibility when not running through hero_agent's BaseRunner.
- hero_agent's `ConstraintEncoderRunner` has the same `hasattr()` auto-sync bug but was
  masked because hero_agent's `BaseRunner` chain handles the initialization differently.
- The `num_encoder_epochs` default mismatch existed since the C-TRPO migration (2026-03-17)
  but was never triggered because config always provided the value explicitly.

## [2026-03-20] Constrained ALBC code review fixes

### Context
Post-simplification code review of constrained ALBC found 8 issues. Three required
code changes (Fix 1, 5, 8); five are design issues documented for future work.
Fix 8 (num_encoder_epochs default) was already applied in the simplification session.

### Fixed
- `algorithms/constraint_trpo.py`: Initialize 8 `_last_*` / `_cached_*` monitoring
  attributes in `__init__()`. Without initialization, `ConstraintEncoderRunner` calling
  `_log_constraint_metrics()` before first `update()` would raise `AttributeError`.
  Also fixes `_cached_barrier_penalty` missing when first iteration enters recovery mode.
- `utils/logging.py`: Fixed `log_encoder_metrics` docstring claiming "Metrics kept (3)"
  when the function actually logs 5 metrics (z_mean, z_std, z_min, z_max, grad_norm).

### Notes
- Issues 2 (any-recovery mode), 4 (KL distribution overwrite), 6 (encode/get_combined_obs
  duplication), 7 (log_encoder_metrics env.get_observations() cost) documented as design
  issues for future work. No code changes needed -- current behavior is correct.

## [2026-03-20] Constrained ALBC algorithm simplification (10-step plan)

### Context
Comprehensive simplification of the `constrained_albc/` package across 10 sessions on 2026-03-20.
The codebase (23 files, ~7,000 lines) had accumulated complexity from being forked from hero_agent
and supporting multiple unused features. Only 1 task is registered (`Isaac-Constrained-ALBC-Encoder-v0`),
but the code supported 4 config levels, optional DORAEMON DR, TDE observation, legacy encoder modes,
and backward checkpoint compatibility. Goal: remove all unused code paths so root cause analysis is
straightforward when problems occur. Final package: ~4,900 lines (~2,100 lines removed, ~30% reduction).

### Added
- `constrained_albc/` package (23 files): Extracted from hero_agent as standalone C-TRPO + encoder
  constrained RL. Registered as `Isaac-Constrained-ALBC-Encoder-v0`. Zero runtime dependency on hero_agent.

### Changed
- `config.py`: Merged 4-class hierarchy (`ALBCEnvCfg` + `ALBCTrainEnvCfg` + `ALBCEncoderTrainEnvCfg`
  + `ConstrainedALBCEncoderEnvCfg`) into single `ALBCEnvCfg`. All fields at final production values.
  `DomainRandomizationCfg` removed `fixed_pose()`/`half_strength()` classmethods and buoy perturbation fields.
  Backward-compat alias `ConstrainedALBCEncoderEnvCfg = ALBCEnvCfg` for gym registration.
- `albc_env.py`: Removed `enable_payload` conditional (always True), `_payload_enabled` property,
  `state_space` vs `enable_payload` validation. Payload wrench always computed (no None checks).
  Extracted `_collect_termination_metrics()`, `_collect_dynamics_metrics()`, `_reset_action_buffers()`,
  `_reset_perturbation_buffers()` from monolithic methods. Removed DR infeasibility logging.
- `mdp/observations.py`: Removed `state_space >= 18/19` guards (always 19). Payload and added mass
  always included in privileged obs.
- `mdp/rewards.py`: Reduced `ALBCRewardCfg` from 10 to 5 fields. Removed penalty curriculum, settling/
  energy rewards, laplacian command branch. `RewardManager` simplified.
- `mdp/events.py`: `DRSampler.get()` simplified: removed `_key` string parameter and `**_kwargs`.
  All 14 call sites updated. Removed DORAEMON integration from `_DRSampler`.
- `mdp/constraints.py`: Removed `joint_torque_cost` alias and 5 unused cost functions.
- `encoder/actor_critic_encoder.py`: Removed no-history mode, symmetric critic, sigmoid activation,
  backward-compat `load_state_dict()` (key filtering, dim mismatch handling). 465 -> 295 lines.
- `encoder/actor_critic_encoder_constrained.py`: Removed `load_state_dict()` override with K-mismatch
  and input-dim-mismatch handling. 131 -> 78 lines.
- `algorithms/constraint_trpo.py`: Removed `**kwargs` catch-all, `self.rnd = None`, 3 `hasattr`
  guards around `evaluate_costs()`, 3 standalone surrogate methods (inlined as closures), entropy_coef
  (dead code). Vectorized cost GAE. 1043 -> 780 lines.
- `runners/`: Flattened 3-level hierarchy (`BaseRunner` -> `EncoderRunner` -> `ConstraintEncoderRunner`)
  to single `ConstraintEncoderRunner(OnPolicyRunner)`. Removed DORAEMON scheduling, noise floor/LR
  methods. Deleted `base_runner.py` and `encoder_runner.py`.
- `agents/rsl_rl_ppo_cfg.py`: Removed `asymmetric_critic`, `encoder_output_activation` from policy
  config (unused by encoder). Flattened 3-level config hierarchy to 2-level.
- `utils/logging.py`: Removed `log_dr_infeasibility()`, `_get_dr_infeasibility_logger()`,
  `connect_encoder_to_env()`.

### Removed
- `doraemon.py`: Deleted entirely (728 lines, unused adaptive DR scheduler).
- `runners/base_runner.py`, `runners/encoder_runner.py`: Deleted (merged into ConstraintEncoderRunner).
- `encoder/history_tcn.py`: Deleted (HistoryTCN replaced by raw flatten concat for TRPO OOM fix).
- Dead code: penalty curriculum, settling/energy rewards, laplacian command, TDE obs, buoy perturbation,
  symmetric critic, sigmoid activation, backward-compat checkpoint loading, DORAEMON integration.

### Notes
- ruff check + ruff format clean across entire package (13 files)
- All public API preserved: ALBCEnv, ALBCEnvCfg, gym registration, encoder architecture
- No checkpoint breaking: state_dict keys unchanged (removed only backward-compat loading logic)
- Pre-existing issue: runner namespace collision between hero_agent and constrained_albc (not addressed)

## [2026-03-18] C-TRPO encoder fix experiments: multi-step caused KL instability

### Context
The encoder gradient fix (2026-03-17) restored enc_grad from 8.3e-4 to 0.04, but introduced
new problems. Three experimental runs were conducted:

1. **Post-mod baseline** (5-epoch, lr=1e-3, torque=15, min_std=0.25, 514it): Best error 5.3-5.5 deg
   at iter 500 but plateaued. kl=0.13~2.87 (vs pre-mod 0.013). Entropy crashed to floor by iter 100.
2. **entropy_coef=0.005** (178it): Performance degraded (r_cmd=-0.97 vs -0.37). Entropy bonus too
   strong for TRPO single-step; slowed convergence without preventing collapse. Reduced to 0.001.
3. **entropy_coef=0.001, torque=20, min_std=0.18** (2500it): Roll 6.2 deg, pitch 6.1 deg. Slower
   convergence than baseline (~5x). kl still unstable (0.04~5.6). min_std had no effect (noise
   never reached 0.18 floor). mode oscillation: 146 transitions, yaw_vel in recovery 23% of time.

Root cause identified: **5-epoch encoder update causes uncontrolled distribution shift**. TRPO
trust region constrains actor KL to 0.01, but encoder changes z (actor input), causing indirect
distribution shift of kl=0.08~5.6 (up to 560x larger). Pre-mod code had kl=0.013 because encoder
barely changed (recovery bug suppressed updates + single cached gradient).

### Changed
- `agents/rsl_rl_ppo_cfg.py`: Reverted `num_encoder_epochs: 5 -> 1`. Multi-step encoder
  update incompatible with TRPO trust region (indirect KL not bounded).
- `agents/rsl_rl_ppo_cfg.py`: Reverted `encoder_lr: 1e-3 -> 3e-4`. Even with 1 epoch,
  3.3x higher LR caused excessive distribution shift vs pre-mod baseline (kl=0.013).
- `algorithms/constraint_trpo.py`: Matched default `encoder_lr` to 3e-4.
- `agents/rsl_rl_ppo_cfg.py`: `joint_torque` budget 0.15 -> 0.20 (reduces constraint floor).
- `config.py`: Synced `joint_torque` budget to 0.20.

## [2026-03-17 Summary] C-TRPO migration, encoder architecture, eval fairness

### Context
Major architecture day: migrated from Lagrangian to C-TRPO (Muller et al., ICML 2025), fixed
encoder gradient death (50x drop), designed and implemented history-augmented encoder (TCN -> raw
flatten), unified eval_dr conditions for fair TDC vs Encoder comparison. Seven Lagrangian tuning
sessions concluded the approach was fundamentally flawed (lambda hysteresis, entropy dilemma).

### Changed
- `algorithms/constraint_trpo.py`: Full rewrite from Lagrangian to C-TRPO barrier-based.
  Removed lambda_k, log_alpha, alpha_optimizer. Added barrier penalty, margin tracking,
  safe/recovery modes. Fixed encoder gradient death: multi-step fresh forward passes
  (num_encoder_epochs=5, later reverted to 1). Removed recovery mode encoder blocking.
- `encoder/actor_critic_encoder.py`: Replaced HistoryTCN with raw flatten concat
  (TRPO OOM fix: 960MB -> 260MB). hist(N,30,8) -> flatten(N,240) -> concat to MLP.
- `agents/rsl_rl_ppo_cfg.py`: Lagrangian params replaced with barrier params (beta=0.01,
  recovery_threshold_frac=0.8). num_encoder_epochs and encoder_lr added.
- `runners/constraint_encoder_runner.py`: lambda_state.pt -> barrier_state.pt checkpoint.
- `eval_dr.py`: Unified DR conditions (removed is_tdc override). Pre-allocate latency buffers.
- `tdc_env.py`: Added TDC output latency buffer for fair evaluation.
- `config.py`: Constraints finalized at 6 terms. Budgets tuned (joint_torque 0.05->0.20,
  yaw_vel 0.15->0.35). PBRS progress_weight=2.0, quadratic command.

### Removed
- `encoder/history_tcn.py`: Deleted (OOM on TRPO full-batch).
- Lagrangian mechanism: lambda_k, alpha_entropy, dual update, warmup logic.
- `_cache_encoder_grads()` method (replaced by multi-step fresh passes).

### Notes
- Encoder z sweep (2500 iter run): 10/13 z dims near-constant, cosine sim 0.9482.
  Encoder not learning useful DR representations from privileged info alone.
- History-augmented architecture provides command-response relationship needed for informative z.
- TRPO entropy dilemma unsolvable in Lagrangian framework: alpha>0 -> unbounded growth, alpha=0 -> collapse.

## [2026-03-16 Summary] Lagrangian migration + entropy stabilization (10 sessions)

### Context
Intensive debugging day (10 sessions) focused on migrating constraint enforcement from
IPO log-barrier to Lagrangian primal-dual, and fixing the entropy collapse/explosion cycle.

**Core problem**: IPO log-barrier assumed feasible start, but random policy starts infeasible.
Barrier gradient's easiest path was reducing action variance (kills exploration). Multiple
fixes attempted (barrier_t tuning, noise floors/ceilings, adaptive thresholds, n_active
normalization) were all band-aids. Fundamental fix: Lagrangian primal-dual where lambda
starts at 0 and grows with violations.

### Changed
- `algorithms/constraint_trpo.py`: Replaced IPO log-barrier with Lagrangian primal-dual.
  Detached std from cost gradient, detached encoder z from cost surrogate. Added reward
  advantage normalization, LS-gated updates, d_k-normalized dual update.
- `agents/rsl_rl_ppo_cfg.py`: Barrier params -> Lagrangian params. Budgets recalibrated.
- `runners/constraint_encoder_runner.py`: Lambda checkpoint persistence.
- `config.py`: energy_weight=0.0, smoothness_weight=0.0. effort_limit budget 0.05->0.25.
- `mdp/constraints.py`: effort_limit_cost uses per-env DR'd limits. Removed dead barrier fields.

### Fixed
- `algorithms/constraint_trpo.py`: Cost value loss clamps targets >=0. NaN guard on cost
  advantage normalization.

## [2026-03-05/10 Summary] ConstraintTRPO build-out, stabilization, NORBC conformance

### Context
Built NORBC-style constrained RL (IPO -> Lagrangian TRPO) from scratch. Key milestones:
initial 3-constraint IPO implementation (3/5-6), constraint expansion to 8 terms with
cost critic softplus fix (3/8), NORBC conformance audit with asymmetric critic and actuator
DR unification (3/9), barrier_t fix 10->50/100 for arm freeze (3/9), noise floor unification
0.1/0.15 -> 0.25 + effort_limit DR conflict removal (3/10).

### Changed
- `algorithms/constraint_trpo.py`: Full TRPO + IPO (~600 lines). barrier_t tuning (1->10->50->10). min_log_std unified to log(0.25).
- `encoder/actor_critic_encoder.py`: Asymmetric critic (raw privileged input)
- `encoder/actor_critic_encoder_constrained.py`: Multi-head cost critic, F.softplus() output
- `runners/constraint_encoder_runner.py`: Named constraint logging, auto-sync K, lambda checkpoint
- `mdp/constraints.py`: 3->8 cost functions, effort_limit uses per-env DR'd limits
- `config.py`: 8-term constraints, unified actuator DR (Kp 40-120, Kd 0.5-5.0, effort 0.7-1.0)
- `hero_agent.py`: velocity_limit_sim 6.28->4.19 rad/s (datasheet 40 rpm)
- `base_env.py`: Accumulated rotation tracking, cost computation, constraint buffers
