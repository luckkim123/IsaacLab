# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [2026-03-20] Simplify constraint_trpo.py internal code duplication

### Context
`constrained_albc/algorithms/constraint_trpo.py` (1043 lines) contained four
categories of code duplication: importance sampling ratio calculation (3 lines x 4
sites), Gaussian KL formula (inlined in both `_kl_divergence` and
`_fisher_vector_product`), line search logic (~40 lines each in `_line_search_safe`
and `_line_search_recovery`), and TRPO step orchestration (~50 lines each in
`_trpo_step_safe` and `_trpo_step_recovery`). The safe/recovery pairs differed only
in which surrogate objective was evaluated, so the common structure was extracted
into shared methods with callable injection.

### Changed
- `constraint_trpo.py`: Extracted `_compute_ratio()` helper (4 call sites reduced from 3 lines to 1)
- `constraint_trpo.py`: Extracted `_gaussian_kl()` static method, used by `_kl_divergence` and `_fisher_vector_product`
- `constraint_trpo.py`: Unified `_line_search_safe` + `_line_search_recovery` into `_line_search(surrogate_fn)`
- `constraint_trpo.py`: Unified `_trpo_step_safe` + `_trpo_step_recovery` into `_trpo_step(compute_loss_fn, compute_surrogate_fn, mode_name)`
- `constraint_trpo.py`: `update()` now passes mode-specific logic as closures to `_trpo_step`

### Removed
- `constraint_trpo.py`: `_line_search_safe`, `_line_search_recovery` (replaced by `_line_search`)
- `constraint_trpo.py`: `_trpo_step_safe`, `_trpo_step_recovery` (replaced by `_trpo_step`)

### Notes
- 1043 lines -> 921 lines (-122), matching the ~920-940 estimate
- No functional change: ruff check + format pass, AST verification confirms old methods removed and new methods present
- Checkpoint compatibility unaffected (ConstraintTRPO has no own state_dict; RSL-RL OnPolicyRunner manages policy/optimizer)
- Full Isaac Sim import not tested (requires runtime); syntax + structure verified via AST parse

## [2026-03-20] Extract constrained_albc + simplify agents/ config hierarchy + mdp cleanup

### Context
The constrained encoder environment (`Isaac-HeroAgent-Constrained-Encoder-Base-v0`) was embedded
in `hero_agent/`, sharing `base_env.py`, `config.py`, `mdp/`, `encoder/`, `runners/`, `utils/`
with 9 other tasks. This made hero_agent a monolithic dependency: any change to shared modules
could break constrained training, and the constrained pipeline couldn't run without the full
hero_agent package installed.

Extracted into `constrained_albc/` as a fully independent package with zero runtime dependency
on hero_agent. Renamed `HeroAgent*` -> `ALBC*` (Active Link Buoyancy Control) throughout.
Registered as `Isaac-Constrained-ALBC-Encoder-v0`. TDC-specific code (controllers, TDC rewards,
TDC logging, adaptation module) was excluded since the constrained pipeline uses only base RL
with C-TRPO + encoder.

Verified: grep for `hero_agent|HeroAgent` shows only external asset references
(`HeroAgentHydrodynamicsCfg` from `isaaclab_assets`). All ruff lint + format checks pass.

### Added
- `constrained_albc/` package (23 files): standalone C-TRPO + encoder constrained RL
- `constrained_albc/__init__.py`: gym registration for `Isaac-Constrained-ALBC-Encoder-v0`
- `constrained_albc/albc_env.py`: `ALBCEnv` (renamed from `HeroAgentEnv`)
- `constrained_albc/config.py`: 5 config classes (`DomainRandomizationCfg`, `ALBCEnvCfg`, `ALBCTrainEnvCfg`, `ALBCEncoderTrainEnvCfg`, `ConstrainedALBCEncoderEnvCfg`)
- `constrained_albc/doraemon.py`: DORAEMON adaptive DR scheduler (verbatim)
- `constrained_albc/mdp/`: observations, events, rewards, constraints (TYPE_CHECKING renames)
- `constrained_albc/encoder/`: `ActorCriticEncoder`, `ActorCriticEncoderConstrained` (verbatim)
- `constrained_albc/algorithms/`: `ConstraintTRPO` only (no ppo_patch)
- `constrained_albc/runners/`: `BaseRunner`, `EncoderRunner`, `ConstraintEncoderRunner` (no AdaptRunner)
- `constrained_albc/agents/rsl_rl_ppo_cfg.py`: `ConstrainedALBCEncoderRunnerCfg` + constrained policy/algorithm configs
- `constrained_albc/utils/`: logging (no TDC functions), debug_vis (verbatim)

### Changed
- `direct/__init__.py`: Added `from . import constrained_albc` for gym auto-registration
- `constrained_albc/agents/rsl_rl_ppo_cfg.py`: Flattened 3-level config hierarchy to 2-level.
  Merged `_ALBCPolicyCfg` + `_RslRlPpoEncoderBaseCfg` into single `_EncoderPolicyCfg`.
  Inlined `_ALBCBaseRunnerCfg` (PPO algorithm settings never used, always overridden by C-TRPO)
  into `ConstrainedALBCEncoderRunnerCfg`. Inlined `_HISTORY_PRIVILEGED_OBS_GROUPS` constant.
  Removed unused `RslRlPpoActorCriticEncoderCfg` (dead code, no runner references it).
  Removed unused `RslRlPpoAlgorithmCfg` import. 223 -> 180 lines, 8 classes -> 4 classes.
- `constrained_albc/agents/__init__.py`: Removed re-exports of network classes
  (`ActorCriticEncoder`, `ActorCriticEncoderConstrained`) already available via
  `constrained_albc.encoder`. Removed dead `RslRlPpoActorCriticEncoderCfg` export.
  23 -> 18 lines, 6 exports -> 3 exports.

### Removed
- (From copied code) All TDC controller imports, TDC reward functions (`tdc_torque_penalty`, `_compute_M_true`, `mhat_accuracy_reward`, `compute_stability_gate`), TDC logging functions (`log_tdc_*`), adaptation module, ppo_patch
- `utils/logging.py`: Deleted `pearson_r()` (only hero_agent uses it) and `_WandbTBWriter`
  class (never instantiated in constrained_albc)
- `utils/__init__.py`: Removed `pearson_r` from imports and `__all__` (8 -> 7 exports)
- `mdp/constraints.py`: Removed 5 unused cost functions (no config reference):
  `joint_velocity_cost`, `joint_oscillation_cost`, `singularity_cost`,
  `attitude_error_cost`, `cob_cog_alignment_cost`. Also removed `quat_apply`,
  `quat_apply_inverse` imports (only used by `cob_cog_alignment_cost`). 391 -> 251 lines.
- `mdp/__init__.py`: Removed 5 deleted functions from exports

### Changed (config cleanup, separate session)
- `agents/rsl_rl_ppo_cfg.py`: `num_constraints` default 6 -> 0 in both
  `RslRlPpoActorCriticEncoderConstrainedCfg` and `RslRlConstraintTRPOAlgorithmCfg`.
  `constraint_budgets` default `(0.02, 0.01, 0.20, 0.05, 0.10, 0.35)` -> `()`.
  These were duplicated from `config.py`; `ConstraintEncoderRunner` already auto-syncs
  both fields from env config at init time (lines 41-56), making hardcoded defaults
  a maintenance hazard.
- `config.py`: Removed 2 disabled-constraint comments (`singularity: disabled`,
  `attitude_err: disabled`) from `ConstrainedALBCEncoderEnvCfg.constraints`. Design
  rationale already documented in `constraints.py` function docstrings.

### Fixed
- `mdp/__init__.py`: Added missing `randomize_joint_effort_limit` to exports
  (was defined in events.py but not re-exported; albc_env.py imported directly)
- `utils/logging.py`: Section header said "4 essential metrics" but actually logs 5; fixed
  to match. TDC references replaced with ALBC terminology ("TDC lambda" -> "ALBC torque
  capacity", "TDC stability" -> "rotational dynamics") since constrained_albc has no TDC
- `utils/debug_vis.py`: Module docstring referenced "Hero Agent" instead of "constrained ALBC"
- `utils/debug_vis.py`: Extracted 4 magic numbers into class constants (`_PAYLOAD_COG_RADIUS`,
  `_PAYLOAD_STEM_RADIUS`, `_MIN_STEM_DIST`, `_HIDDEN_POS`) for clarity

### Refactored (runners/ simplification, 7 steps)
- `runners/base_runner.py`: Docstring "Hero Agent" -> "constrained ALBC". Added `_doraemon`
  property (replaces 3x `hasattr(raw_env, "_doraemon") and raw_env._doraemon is not None`),
  `_should_log` property (replaces 4x `self.log_dir is not None and not self.disable_logs`),
  `_save_aux_state`/`_load_aux_state` static methods (replaces manual `os.path.join` + `torch.save/load`
  in save/load). DORAEMON logging switched from raw `writer.add_scalar` loop to `flush_metrics()`.
- `runners/encoder_runner.py`: 2x logging guard replaced with `self._should_log`.
- `runners/constraint_encoder_runner.py`: Removed 8 `hasattr(alg, "_last_...")` guards
  (ConstraintTRPO.update() always sets these before log()). Removed lambda_state.pt backward
  compat code (4 lines, legacy from hero_agent copy, never relevant in constrained_albc).
  save/load uses `_save_aux_state`/`_load_aux_state`. Removed unused `os`/`torch` imports.

### Changed (encoder/ structural refactoring, separate session)
- `encoder/actor_critic_encoder.py`: Unified activation path -- removed conditional
  `last_activation="tanh"` from MLP construction. Both tanh and sigmoid now flow through
  `_activate_z()` unconditionally. Checkpoint-safe (nn.Tanh has 0 params, state_dict identical).
  501 -> 464 lines.
- `encoder/actor_critic_encoder.py`: Extracted `_build_encoder_input()` helper that returns
  `(encoder_input, hist_flat)` tuple. Eliminates double `_get_hist_flat()` call in
  `_get_combined_obs()` and duplicate encoder-input concatenation in `update_normalization()`.
  `_encode()`, `_get_combined_obs()`, `_get_critic_obs()`, `update_normalization()` all refactored.
- `encoder/actor_critic_encoder.py`: `_handle_critic_dim_mismatch()` simplified -- removed
  `prefix` parameter and `cost_critic` knowledge. Base class now handles only `critic.` prefix.
- `encoder/actor_critic_encoder_constrained.py`: Cost critic input dim mismatch check inlined
  in `load_state_dict()` (was delegated to base class via `_handle_critic_dim_mismatch()`).
  117 -> 130 lines.

### Removed (encoder/ dead code, separate session)
- `encoder/actor_critic_encoder.py`: Removed `act_with_z_hat()` (13 LOC, AdaptRunner-only method
  not applicable to constrained_albc which has no Phase 2 adaptation pipeline)
- `encoder/actor_critic_encoder.py`: Removed `self._proprio_history_len` and
  `self._proprio_feature_dim` instance variable storage (ActorCriticEncoderAdapt-specific,
  never referenced within this class). Signature params kept to absorb config kwargs.

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

barrier_penalty was ~0.0000 throughout all runs -- effectively not constraining anything.
Mode oscillation (safe/recovery switching) destabilized training without clear benefit.

### Changed
- `agents/rsl_rl_ppo_cfg.py`: Reverted `num_encoder_epochs: 5 -> 1`. Multi-step encoder
  update incompatible with TRPO trust region (indirect KL not bounded).
- `agents/rsl_rl_ppo_cfg.py`: Reverted `encoder_lr: 1e-3 -> 3e-4`. Even with 1 epoch,
  3.3x higher LR caused excessive distribution shift vs pre-mod baseline (kl=0.013).
- `algorithms/constraint_trpo.py`: Matched default `encoder_lr` to 3e-4.
- `agents/rsl_rl_ppo_cfg.py`: `joint_torque` budget 0.15 -> 0.20 (reduces constraint floor).
- `config.py`: Synced `joint_torque` budget to 0.20.
- `runners/base_runner.py`: `min_std` 0.25 -> 0.18 (no observed effect yet).
- `algorithms/constraint_trpo.py`: Added `entropy_coef` parameter (default 0.0, disabled).
  Entropy bonus in safe-mode surrogate. Tested at 0.005 and 0.001; both ineffective for TRPO.

### Removed from cfg
- `entropy_coef` removed from `RslRlConstraintTRPOAlgorithmCfg` (kept in algorithm code,
  disabled by default).

### Open questions
- C-TRPO barrier_penalty ~0 throughout: barrier may need higher beta or different formulation.
- Mode oscillation (146 transitions/2500it) suggests recovery threshold tuning needed.
- Whether to continue with C-TRPO or revert to Lagrangian approach needs evaluation after
  conservative restoration (1-epoch + recovery fix only).

## [2026-03-17] Fix encoder gradient death in C-TRPO (50x drop)

### Context
First C-TRPO run (2026-03-17_16-32-32, 359 iters) showed severe performance regression vs
Lagrangian baseline: pitch error 13.43 deg (was 9.31), enc_grad 8.3e-4 (was 0.04, a 50x drop),
z_std 0.42 and declining (was 0.61). 4 constraint costs diverging in last 50 iters. The
encoder was effectively frozen, unable to learn useful DR representations.

Root cause analysis identified three structural issues:

1. **Update frequency mismatch (~20-25x)**: PPO gives the encoder 20 gradient steps per
   iteration (5 epochs x 4 mini-batches). C-TRPO's single full-batch TRPO step gave the
   encoder exactly 1 gradient computation via `_cache_encoder_grads()`.

2. **Recovery mode killed encoder gradient entirely**: `_trpo_step_recovery()` set
   `self._encoder_grads_cache = []`, giving the encoder zero policy gradient when ANY
   constraint entered recovery mode. Only z_bounds_loss remained, which is a regularizer
   that pushes z toward zero -- explaining the z_std decline.

3. **z_bounds_loss dominated**: Without opposing policy gradient, z_bounds_loss (penalizes
   |z| > 0.9) collapsed z toward zero, making the encoder output uninformative.

Combined effect: 20x (update freq) x 2x (ratio drift absence) x 1.5x (full-batch variance
reduction) = ~50-60x, matching the observed 0.04 / 0.00083 = 48x drop.

### Changed
- `algorithms/constraint_trpo.py`: Rewrote `_update_encoder()` with multi-step fresh
  forward/backward passes (`num_encoder_epochs`, default 5). Each epoch does a full
  forward pass through encoder+actor, computes reward_surrogate gradient to encoder,
  adds z_bounds gradient, clips (max_norm=0.2), and steps Adam. Replaces single cached
  gradient approach.
- `algorithms/constraint_trpo.py`: Added `num_encoder_epochs` and `encoder_lr` as proper
  `__init__` parameters (were hardcoded). encoder_lr default 1e-3 (was hardcoded 3e-4).
- `algorithms/constraint_trpo.py`: Removed recovery mode encoder gradient blocking in
  `_trpo_step_recovery()`. Encoder uses separate Adam optimizer (not TRPO trust region),
  so no theoretical reason to block it during cost-minimization steps.
- `algorithms/constraint_trpo.py`: Changed `retain_graph=True` to `False` in
  `_trpo_step_recovery._flat_grad()` (CG solver builds own fresh graphs).
- `agents/rsl_rl_ppo_cfg.py`: Added `num_encoder_epochs: int = 5` and
  `encoder_lr: float = 1e-3` to `RslRlConstraintTRPOAlgorithmCfg`.

### Removed
- `algorithms/constraint_trpo.py`: Deleted `_cache_encoder_grads()` method and
  `_encoder_grads_cache` attribute. No longer needed with multi-step fresh forward passes.
- `algorithms/constraint_trpo.py`: Removed `_cache_encoder_grads(reward_surrogate)` call
  from `_trpo_step_safe()`.

### Notes
- Effective encoder update per iteration: 5 epochs x 1e-3 LR = 5e-3 (vs PPO: 20 x 3e-4 = 6e-3).
  Slightly conservative but close to PPO's total update magnitude.
- Encoder grad clip (max_norm=0.2) preserved per epoch to prevent z instability from ratio drift.
- Code verified working: 64-env test (3 iters, 2.69s) passed. 4096-env training launched.

## [2026-03-17] C-TRPO migration: Lagrangian → barrier-based trust region

### Context
After extensive debugging of the Lagrangian primal-dual approach (10+ sessions on 2026-03-16,
7+ sessions on 2026-03-17), the fundamental structural problems remained unsolved:

1. **Lambda oscillation/hysteresis**: Transient constraint violations caused lambda to spike,
   collapsing noise_std. Even when lambda decreased (constraints satisfied), noise never
   recovered -- a one-way ratchet on exploration.
2. **Entropy collapse/explosion dilemma**: alpha_entropy=0 → noise collapses to floor;
   alpha_entropy>0 → noise rises unboundedly in TRPO (no clip mechanism like PPO).
3. **All band-aids failed**: noise floor, alpha tuning, detached-std cost ratio, warmup
   schedules -- each addressed symptoms but not root cause.

Migrated to C-TRPO (Muller et al., ICML 2025, arXiv:2411.02957) which eliminates lambda
entirely. Barrier function naturally shapes the trust region geometry: steps become more
conservative near constraint boundaries. Two modes: safe (reward + barrier penalty) and
recovery (cost minimization for infeasible constraints). Option C variant: barrier curvature
only in objective gradient, FVP stays pure KL for CG stability.

### Changed
- `agents/rsl_rl_ppo_cfg.py`: Removed 7 Lagrangian params (`lr_lambda`, `lambda_max`,
  `lambda_init`, `lambda_warmup_frac`, `target_entropy`, `alpha_entropy_lr`,
  `alpha_entropy_init`). Added 2 barrier params: `beta=0.01` (barrier coefficient),
  `recovery_threshold_frac=0.8` (hysteresis for mode switching). Updated class docstring.
- `algorithms/constraint_trpo.py`: Full rewrite (~500 lines). Removed: `lambda_k`,
  `log_alpha`, `alpha_optimizer`, `_log_prob_mean_only()`, lambda warmup logic, dual
  update block, alpha entropy update block. Added: `_compute_margins()` (per-constraint
  margin tracking + safe/recovery mode switching with hysteresis), `_compute_barrier_penalty()`
  (linearized log-barrier: `beta * phi''(margin) * cost_surr^2`), `_linearized_surrogate_safe()`
  and `_linearized_surrogate_recovery()` (mode-specific objectives), `_line_search_safe()`
  (KL only) and `_line_search_recovery()` (KL + cost decrease), `_trpo_step_safe()` and
  `_trpo_step_recovery()` (mode-specific TRPO steps). Encoder update, value update, noise
  floor, z_bounds, LS-gated encoder updates all preserved.
- `runners/constraint_encoder_runner.py`: Checkpoint: `lambda_state.pt` → `barrier_state.pt`
  (saves `_in_recovery` flags + `_margins`). Backward compat: old `lambda_state.pt` silently
  ignored. Logging: removed `lambda_k`, `alpha_entropy`; added `margin_k`, `in_recovery_k`,
  `barrier_penalty`, `mode` (0=all safe, 1=mixed, 2=all recovery).

### Notes
- Barrier 1/2 factor from paper's Bregman divergence absorbed into beta (effective barrier
  strength = 2x paper semantics at same beta value). Documented in code comment.
- `_flat_grad` changed to `retain_graph=False` after encoder grad caching to reduce
  memory during CG loop (CG builds own fresh graphs via FVP).
- `_encoder_grads_cache` initialized in `__init__` (code review fix: prevented potential
  AttributeError on edge case).
- Recovery mode priority: if ANY constraint infeasible, entire step is recovery (cost
  minimization). Conservative but simple; mixed mode deferred to future if needed.
- All existing mechanisms preserved: noise floor (0.25), z_bounds loss, per-constraint
  cost advantage standardization (NORBC Sec IV-B), d_k^2-normalized cost value loss,
  encoder grad clip (0.2), reward advantage normalization.

## [2026-03-17] Fair eval_dr: unified DR conditions for TDC vs Encoder comparison

### Context
Evaluated latest constrained encoder run (2026-03-17_10-01-14, 2499 iter) and discovered
three fairness issues when comparing TDC vs Encoder policies via eval_dr.py:

1. **Joint gains**: TDC received fixed high gains (Kp=160-240, Kd=8-12) while Encoder
   used DR-interpolated gains (Kp=40-120, Kd=0.5-5.0). Higher gains = easier control.
2. **Action latency**: TDC had `action_latency_range=(0,0)` override while Encoder got 0-4
   steps. BUT investigation revealed latency was never actually applied to EITHER task
   because eval_dr creates the env with "none" DR (latency=0), and latency ring buffers
   are allocated during `__init__` -- switching DR level later doesn't re-allocate them.
3. **Structural latency**: TDC bypasses RL actions entirely (`_pre_physics_step` ignores
   actions and runs TDC controller directly), so even with correct buffer allocation,
   base_env's `_get_delayed_actions()` has no effect on TDC.

### Changed
- `eval_dr.py`: Removed `is_tdc` parameter from `build_dr_config()` and `apply_dr_config()`.
  All tasks now receive identical DR interpolation (joint gains, latency, perturbation, etc.)
- `eval_dr.py`: Pre-allocate action latency buffers at env creation by setting
  `action_latency_range` to full DR max before `gym.make()`, then letting per-level DR
  override the config on each reset. Without this, buffers stay None at all DR levels.
- `eval_dr.py`: Load agent params from run's `params/agent.yaml` via `yaml.full_load()`
  for correct model architecture (ConstraintEncoderRunner vs EncoderRunner). Falls back
  to task registry config on YAML parse failure.
- `eval_dr.py`: Save eval_dr output to `<run_dir>/eval_dr/` instead of separate
  `logs/eval_dr/` tree, keeping results alongside training logs.
- `tdc_env.py`: Added TDC output latency buffer (`_tdc_target_history`) that delays
  `_joint_pos_targets` by N control steps using `base_env._action_latency` values.
  Anti-windup uses immediate targets (controller's view); delay models actuator comm latency.
- `tdc_env.py`: Reset latency buffer in `_reset_idx` with current joint positions to
  prevent stale values on episode reset.

### Notes
- TDC's TDE partially compensates for lower joint gains (that's TDE's purpose), so gain
  equalization may not dramatically change results. The key fairness improvement is latency.
- eval_dr now uses `DomainRandomizationCfg` defaults as SSOT for all tasks. Same class
  used in training, so hard DR level = training conditions for all policies.
- hero-agent-analysis skill updated: "evaluate" now defaults to eval_dr + PNG plots.
  Text metric summaries are not evaluation.

## [2026-03-17] Replace HistoryTCN with raw flatten concat (TRPO OOM fix)

### Context
Previous session designed a history-augmented encoder using a shared HistoryTCN
(temporal convolution) to produce h_embed (32D) from proprioception history (30, 8).
Three consecutive CUDA OOM errors occurred during constrained encoder training:

1. TCN called 3x per step -> added h_embed caching -> OOM (960MB backward)
2. Added gradient checkpointing -> OOM (recomputation needs same memory)
3. Fundamental: Conv1d intermediate activations on TRPO full-batch (4096x64=262K samples
   x 30 timesteps) require ~960MB, exceeding RTX 4070 available VRAM (~0.6GB free after
   model + rollout buffer)

User questioned whether TCN and embedding are necessary at all. Decision: eliminate TCN
entirely, flatten proprio_hist (N, 30, 8) -> (N, 240) and concatenate directly to MLP
inputs. Adds only ~260MB input tensor (vs ~960MB TCN intermediates). The encoder/actor/
critic MLPs learn from raw history without any preprocessing.

Phase 2 adaptation (ProprioAdaptTConv) retains its TCN because it runs in PPO minibatch
mode via AdaptRunner (not TRPO full-batch), so Conv1d memory is manageable.

Post-fix: training crashed with `IndexError: too many indices for tensor of dimension 2`
in `log_encoder_metrics()` because `_encode()` now expects TensorDict, not raw tensor.
Fixed by passing full obs TensorDict.

### Changed
- `encoder/actor_critic_encoder.py`: Complete rewrite. Removed HistoryTCN, h_embed caching,
  gradient checkpointing, shared_tcn, hist_normalizer, h_embed_dim. Added `_get_hist_flat()`
  (simple flatten). New dimensions: Encoder 272D, Actor 266D, Critic 272D (with history)
- `encoder/adaptation.py`: Removed dual-mode (adapt_head vs adapt_tconv). Always uses
  ProprioAdaptTConv for Phase 2. `_get_combined_obs`/`evaluate` use hist_flat from parent
- `encoder/__init__.py`: Removed HistoryTCN export and docstring reference
- `algorithms/constraint_trpo.py`: Removed `shared_tcn`/`hist_normalizer` from
  `encoder_prefixes`. Removed 2x `clear_h_embed_cache()` calls
- `agents/rsl_rl_ppo_cfg.py`: Removed `h_embed_dim: int = 32` from `_RslRlPpoEncoderBaseCfg`
- `adapt_base_env.py`: Updated docstring (shared_tcn/h_embed -> adapt_tconv/hist_flat)

### Fixed
- `utils/logging.py`: `log_encoder_metrics()` called `policy._encode(privileged)` with raw
  tensor, but `_encode()` now expects TensorDict. Changed to `policy._encode(obs)`

### Removed
- `encoder/history_tcn.py`: Deleted (no longer imported by any module)

### Notes
- Memory budget: TCN backward ~960MB -> raw flatten input ~260MB (fits in RTX 4070)
- ProprioAdaptTConv (Phase 2) kept: runs in PPO minibatch mode, not TRPO full-batch
- Architecture now: hist(N,30,8) -> flatten(N,240) -> concat to MLP inputs directly
- No embedding, no normalization module for history -- MLPs learn from raw data

## [2026-03-17] Encoder z sweep analysis + history-augmented encoder design

### Context
Final constrained encoder run `2026-03-17_10-01-14` (2500 iters) achieved roll=5.46 deg,
pitch=6.74 deg -- a plateau comparable to TDC baseline (6 deg avg under hard DR). Gradient
ratio healthy (0.08-0.15), constraints mostly satisfied, training stable. But encoder z
analysis revealed the encoder is NOT learning useful DR representations.

Encoder z sweep analysis across 8 DR conditions:
- 10/13 z dimensions near-constant (std < 0.12)
- Cosine similarity = 0.9482 (z vectors nearly identical regardless of DR)
- Max Pearson correlation with physics params: |r| = 0.239 (very weak)
- added_mass_surge: no z dimension had |r| > 0.1

Root cause: privileged info alone (static hydrodynamic parameters) does not reveal dynamic
response characteristics. The encoder cannot distinguish DR conditions without observing
how the system responds to commands. The same physical parameters produce different
dynamics depending on operating point, coupling, and transient behavior.

Designed a new history-augmented encoder architecture that adds shared proprioception
history (TCN) to all modules. Key principle: proprioception history provides command-response
relationship (same action + different physics -> different angular velocity) that the encoder
needs to produce informative z.

### Added
- `docs/plans/2026-03-17-history-encoder-architecture.md`: Implementation plan for
  history-augmented encoder architecture. 13 tasks across 4 chunks.

### Notes
- New architecture:
  - Proprio History (30, 8D) -> shared HistoryTCN -> h_embed (32D)
  - Encoder: [policy(13D), h_embed(32D), privileged(19D)] -> z(13D)
  - Actor: [policy(13D), h_embed(32D), z(13D)] -> actions(2D) (no privileged)
  - Critics: [policy(13D), h_embed(32D), privileged(19D)] -> value/cost
- Actor alone cannot see privileged; accesses DR info only via z
- Phase 2 simplified: shared_tcn(frozen) + adapt_head(h_embed->z_hat) replaces full TCN
- No code changes in this session -- design and planning only
- TDC baseline comparison: classic controller achieves ~6 deg under hard DR, matching
  current RL+encoder (5.5-6.7 deg). Encoder contributes negligible improvement.

## [2026-03-17] Lagrangian tuning arc (7 sessions, superseded by C-TRPO)

### Context
Seven debugging sessions tuning the Lagrangian approach before concluding it was
fundamentally flawed for this problem. Key progression: 3-constraint baseline (alpha=0.005
-> noise explosion to std=4.45) -> alpha=0 (noise collapse to 0.17) -> noise floor 0.2
-> constraint expansion 3->9->6 -> budget relaxation -> lambda warmup extension -> PBRS
progress reward -> entropy/noise plateau diagnosis. All approaches encountered the same
lambda hysteresis: transient violation -> lambda spike -> noise collapse -> lambda drop
but noise never recovered. Entire arc superseded by C-TRPO migration (above).

### Changed
- `config.py`: Constraints evolved 3->9->8->6 (final: accum_rot, attitude_abs, joint_torque,
  joint_vel_limit, overshoot, yaw_vel). Budgets tuned: joint_torque 0.05->0.10->0.15,
  yaw_vel 0.15->0.35. Removed singularity + attitude_err (redundant with DLS IK / quadratic
  reward). Added PBRS progress_weight=2.0, command_type="quadratic", smoothness_weight=-0.1.
- `agents/rsl_rl_ppo_cfg.py`: num_constraints 3->9->8->6, budgets synced. alpha_entropy
  0.005->0.0. lr_lambda 0.01->0.005. lambda_warmup_frac 0.3->0.5.
- `runners/base_runner.py`: noise floor 0.1->0.2->0.25 (final: 0.25)
- `algorithms/constraint_trpo.py`: noise floor synced with base_runner (final: log(0.25)).
  Encoder LR 3e-3->3e-4 (272D input desync fix). Encoder grad clip added (max_norm=0.2).
- `mdp/constraints.py`: Added joint_torque_cost, joint_velocity_limit_cost, overshoot_cost,
  attitude_error_cost. Removed singularity + attitude_err constraints.
- `mdp/rewards.py`: Added progress_reward (PBRS), command_type "quadratic"/"laplacian"
- `base_env.py`: Overshoot buffer, progress reward term, command_type passthrough

### Notes
- TRPO entropy dilemma (no solution in Lagrangian framework):
  alpha>0 -> unbounded noise growth (no clip like PPO); alpha=0 -> collapse to floor
- Lambda hysteresis: one-way ratchet on exploration (lambda up -> noise down -> lambda down
  but noise stays down). Root cause of C-TRPO migration decision.
- noise_floor=0.25 kept as safety net in C-TRPO (still relevant)

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

## [2026-03-05/10 Summary] ConstraintTRPO build-out, stabilization, NORBC conformance

### Context
Built NORBC-style constrained RL (IPO -> Lagrangian TRPO) from scratch. Key milestones:
initial 3-constraint IPO implementation (3/5-6), constraint expansion to 8 terms with
cost critic softplus fix (3/8), NORBC conformance audit with asymmetric critic and actuator
DR unification (3/9), barrier_t fix 10->50/100 for arm freeze (3/9), noise floor unification
0.1/0.15 -> 0.25 + effort_limit DR conflict removal (3/10).

### Changed
- `algorithms/constraint_trpo.py`: Full TRPO + IPO (~600 lines). barrier_t tuning (1->10->50->10, tested range). min_log_std unified to log(0.25). Update order, per-constraint cost advantage standardization.
- `encoder/actor_critic_encoder.py`: Asymmetric critic (raw privileged 32D input)
- `encoder/actor_critic_encoder_constrained.py`: Multi-head cost critic, F.softplus() output, K mismatch detection
- `runners/constraint_encoder_runner.py`: Named constraint logging, auto-sync K, lambda checkpoint
- `runners/base_runner.py`: min_std 0.15->0.25 (unified with constraint_trpo floor)
- `mdp/constraints.py`: 3->8 cost functions, effort_limit uses per-env DR'd limits
- `config.py`: 8-term constraints, unified actuator DR (Kp 40-120, Kd 0.5-5.0, effort 0.7-1.0). Removed joint_effort_limit_range override.
- `hero_agent.py`: velocity_limit_sim 6.28->4.19 rad/s (datasheet 40 rpm)
- `base_env.py`: Accumulated rotation tracking, cost computation, constraint buffers
