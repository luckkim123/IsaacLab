# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [2026-03-05] NORBC Constrained RL (IPO + TRPO) Code Review Fixes

### Context
Systematic code review of constraint_trpo.py identified 9 issues (2 critical, 1 moderate-critical,
3 moderate, 3 minor). The most significant finding: cost constraints were computed via GAE but
never incorporated into the policy gradient -- the policy optimized reward only, making the
"constrained" RL functionally equivalent to unconstrained TRPO with a barrier that only nudged
value estimates. Additionally, encoder params were updated via TRPO natural gradient (FIM step
size), but encoder affects the distribution only indirectly through z, making FIM calibration
inappropriate.

Runtime fixes during testing:
- z_bounds_loss crashed (`RuntimeError: does not require grad`) because `_last_z` was stored
  from inference-mode rollout collection. Fixed by adding fresh forward pass before z_bounds_loss.
- Line search crashed (`RuntimeError: normal expects all elements of std >= 0.0`) because
  encoder Adam step was applied between gradient computation and line search, corrupting params.
  Fixed by deferring encoder Adam step until after line search completes.

### Changed
- `constraint_trpo.py`: **[Critical]** Added barrier-weighted cost surrogate to TRPO policy loss.
  Each constraint k contributes `(1/t) * E[ratio * cost_adv_k] / margin_k`, making policy aware
  of constraints via IPO gradient
- `constraint_trpo.py`: **[Critical]** Added cost feasibility check to line search. New third
  condition: rejects steps where any cost surrogate exceeds 50% of constraint margin
- `constraint_trpo.py`: Added cost value bootstrapping on episode time-outs (matching reward
  bootstrapping pattern: `costs + cost_gamma * V_C * timeout_mask`)
- `constraint_trpo.py`: Barrier loss amortized by `num_epochs * num_mini_batches` to prevent
  20x amplification from being applied as constant in every mini-batch update
- `constraint_trpo.py`: Separated encoder params from TRPO natural gradient into dedicated
  Adam optimizer (lr=3e-3). Actor uses TRPO, encoder uses Adam, value/cost_critic use Adam.
  Encoder gradients computed eagerly but Adam step deferred until after TRPO line search
- `constraint_trpo.py`: Fixed KL epsilon from `log(sigma/old_sigma + 1e-5)` to
  `log((sigma/old_sigma).clamp(min=1e-5))` removing positive bias
- `constraint_trpo.py`: Adaptive threshold now uses EMA toward target (allows tightening)
  instead of `max()` (loosening only)
- `constraint_trpo.py`: `z_bounds_coef` parameter now stored and applied as coefficient
  (was previously accepted but unused)
- `constraint_trpo.py`: z_bounds_loss moved from value optimizer (gradient waste on encoder
  params) to encoder_optimizer with fresh forward pass to populate `_last_z` with grad_fn

### Notes
- All Pyright warnings are expected (RSL-RL/Isaac Sim imports, dynamic storage attributes)
- First training run pending to verify all fixes work together end-to-end

## [2026-03-05] NORBC Constraint Logging Refactor

### Context
Constraint monitoring metrics (barrier_t, d_k_adaptive, cost_return, line_search_success) were
logged twice per iteration: once via loss_dict (prefixed `Loss/` by OnPolicyRunner.log() at
on_policy_runner.py:212) and again via `_log_constraint_metrics()` (prefixed `Constraint/`).
This caused 8 redundant scalar writes and polluted the `Loss/` WandB panel with non-loss metrics.

### Changed
- `constraint_encoder_runner.py`: Expanded `_log_constraint_metrics()` to read all constraint
  metrics from algorithm instance attributes and log with proper prefixes (`Constraint/` for
  barrier/cost metrics, `Policy/` for line_search_success). Now uses `flush_metrics()` utility
  instead of raw `writer.add_scalar()` calls
- `constraint_encoder_runner.py`: Added `flush_metrics` import from utils.logging

### Added
- `constraint_encoder_runner.py`: New metrics -- `Constraint/barrier_schedule_progress` (0~1),
  `Constraint/cost_return_std_{k}`, `Constraint/feasibility_rate_{k}` (fraction of envs
  satisfying constraint budget)

### Removed
- `constraint_trpo.py`: Removed `cost_return_{k}`, `d_k_adaptive_{k}`, `barrier_t`,
  `line_search_success` from loss_dict (these were non-loss metrics causing `Loss/` prefix
  pollution). Now stored as `_last_*` instance attributes read by runner

---

## [2026-03-05] Theoretical/Logical Error Fixes

### Context
Conducted systematic theoretical audit of the Hero Agent codebase against TDE/HORA
theory after NORBC paper analysis. Three exploration agents reported 32 potential issues;
direct code reading and formula verification classified them into 3 critical, 2 high,
4 medium issues and 4 false positives (pitch T_b derivation, action latency indexing,
z activation mismatch, Lambda_inv math -- all verified correct).

Key findings:
- C1: TDE observation computed instantaneous residual (current-step values) instead of
  time-delayed estimate (previous-step values). TDC controller correctly uses _prev
  buffers but the observation function did not, violating the TDE identity H_t ~ H_{t-L}.
- C2: Phase 2 (Adapt-Base) critic used encoder z while actor used z_hat from adapt_tconv.
  PPO assumes V(s) evaluates the same state the actor sees -- using different z representations
  biases the advantage estimate.
- C3: TDCController initialized with F_bu.mean() (scalar), losing per-env DR variation.
  First episode steps used averaged Lambda matrix.
- H1/M4: z_bounds_loss and encoder weight_decay patches only existed in site-packages
  rsl_rl/algorithms/ppo.py (not git-tracked, lost on container rebuild).
- M2 (EMA reset to 0): Re-verified as correct -- joint velocities are reset to 0 in
  events.py, so EMA=0 matches the post-reset state. No fix needed.

### Added
- `algorithms/ppo_patch.py`: Monkey-patch module for RSL-RL PPO. Adds encoder-aware
  optimizer (separate param groups with WD=1e-5 for encoder, WD=0 for actor/critic)
  and z_bounds_loss integration. Auto-applied at import time via `algorithms/__init__.py`.
  Idempotent: detects if site-packages is already patched and skips.

### Changed
- `base_env.py`: TDE observation now uses previous-step Lambda*p_EE and T_b buffers
  (2 new history tensors: `_tde_Lambda_p_EE_prev`, `_tde_T_b_prev`), matching TDC
  controller's TDE pattern. Added control frequency validation (decimation >= 1,
  frequency in [10Hz, 1000Hz]).
- `encoder/adaptation.py`: Phase 2 critic `evaluate()` changed from encoder z to
  z_hat (detached), ensuring actor and critic see the same state representation.
- `controllers/tdc.py`: `__init__` F_bu parameter type changed from `float` to
  `torch.Tensor | float`, supporting per-env tensor initialization.
- `tdc_env.py`: Passes full per-env F_bu tensor to TDCController instead of
  `F_bu.mean().item()`.
- `algorithms/__init__.py`: Added `apply_ppo_patch()` auto-invocation on import.

### Notes
- TDE-Base-v0 training should be re-evaluated after C1 fix (observation semantics changed)
- Phase 2 Adapt-Base value loss convergence may improve with C2 fix (critic sees actor's state)
- M2 (EMA reset) verified as non-issue: joint velocities reset to 0 in events.py
- False positives documented: pitch T_b formula, action latency indexing, z activation
  consistency, Lambda_inv DLS math -- all verified correct against derivation docs

---

## [2026-03-05] IPO + TRPO Constrained RL Implementation

### Context
Implemented NORBC-style Interior-point Policy Optimization (IPO) with TRPO natural
gradient for the Hero Agent Encoder-Base pipeline. The underwater environment has
physical constraints (joint velocity limits, rotation limits, oscillation) previously
handled as soft reward penalties. IPO separates constraints from rewards using explicit
cost budgets and log-barrier penalties, producing more robust controllers with simpler
reward design. Three binary indicator constraints (K=3): joint velocity (|vel|>3 rad/s),
accumulated rotation (>2 full rotations), and joint oscillation (HF RMS >1.5 rad/s).

During integration, fixed multiple runtime compatibility issues:
- `agents/__init__.py` missing exports for new runner/algorithm/policy configs
- `train.py` `_RUNNER_MAP` missing `ConstraintEncoderRunner` entry
- `ConstraintTRPO` missing `rnd` and `optimizer` attributes expected by RSL-RL OnPolicyRunner
- Inference tensors from rollout storage (collected under `torch.inference_mode()`) cannot
  be used in autograd -- fixed by `.clone()` on all storage tensors before backward passes
- `EncoderRunner._update_encoder_lr()` assumes Adam param groups for encoder -- overridden
  to no-op since ConstraintTRPO uses natural gradient (no optimizer) for policy/encoder

### Added
- `mdp/constraints.py`: ALBCConstraintCfg dataclass + 3 binary cost functions (joint_velocity_cost, accumulated_rotation_cost, joint_oscillation_cost, compute_all_costs)
- `algorithms/__init__.py`: New module for algorithm exports
- `algorithms/constraint_trpo.py`: Full TRPO + IPO implementation (~600 lines). Conjugate gradient solver, Fisher-vector product via Hessian-free double backprop, backtracking line search with KL + feasibility checks, log-barrier on constraint margins, adaptive thresholds
- `encoder/actor_critic_encoder_constrained.py`: ActorCriticEncoderConstrained with multi-head cost critic (K outputs), backward-compatible load_state_dict
- `runners/constraint_encoder_runner.py`: ConstraintEncoderRunner with barrier schedule update, constraint metrics logging, encoder LR override (no-op for TRPO)

### Changed
- `base_env.py`: Added `_accumulated_rotation` / `_prev_joint_pos` buffers, delta rotation tracking in `_pre_physics_step()`, cost computation via `compute_all_costs()` in `_get_rewards()`, reset in `_reset_framework()`
- `config.py`: Added `HeroAgentConstrainedEncoderEnvCfg` (inherits EncoderTrain, adds constraints, zeros joint_velocity/oscillation reward weights)
- `agents/rsl_rl_ppo_cfg.py`: Added `RslRlConstraintTRPOAlgorithmCfg`, `RslRlPpoActorCriticEncoderConstrainedCfg`, `HeroAgentConstrainedEncoderRunnerCfg` + module namespace injections for RSL-RL eval resolution
- `agents/__init__.py`: Added exports for constrained configs (HeroAgentConstrainedEncoderRunnerCfg, etc.)
- `__init__.py`: Registered `Isaac-HeroAgent-Constrained-Encoder-Base-v0` gym environment
- `encoder/__init__.py`: Added `ActorCriticEncoderConstrained` export
- `runners/__init__.py`: Added `ConstraintEncoderRunner` export
- `mdp/__init__.py`: Added constraint function exports
- `train.py`: Added `ConstraintEncoderRunner` to `_RUNNER_MAP`

### Notes
- Training partially verified (env loads, rollout completes, update runs to TRPO policy step)
- Still debugging: may have additional runtime issues in TRPO line search or logging
- Architecture: value params (critic + cost_critic) use Adam; policy params (actor + encoder) use TRPO natural gradient

---

## [2026-03-05] Code Simplification

### Context
Code simplification session for hero_agent codebase (~7,700 lines, 27 Python files).
Focused on dead code removal, duplicate code consolidation, and unused reward function cleanup.
During post-simplification diff analysis against run 2026-03-03_09-24-38, discovered that
`termination_penalty = -10.0` was mistakenly removed as "unused" -- it was actually active
(default -10.0, applied on early termination). Restored immediately.

### Changed
- `base_env.py`: Consolidated `_update_perturbation()` main/buoy logic into `_apply_perturbation_cycle()` helper
- `base_env.py`: Added `_iter_noise_params()` static method; simplified `_pad_noise_cfg_for_tde()` and `_convert_noise_cfg_tuples()` from nested loops to single-line iterations
- `base_env.py`: Replaced verbose termination logging with `_term_rate()` helper
- `config.py`: Removed stale MPC docstring reference, removed redundant `ocean_current` and `enable_payload` overrides that matched parent class
- `config.py`: Observation noise tuples use `[val] * N` pattern for readability
- `controllers/tdc.py`: Extracted `_set_param()` static helper for `update_controller_params()`/`update_gains()` deduplication
- `controllers/tdc.py`: Consolidated 11-buffer `reset()` into `_zero_buffers` list + loop
- `mdp/events.py`: Added `_apply_xyz_offset_with_doraemon()` helper to merge CoB/CoG DORAEMON branches (~16 lines x2 -> 2 calls)
- `mdp/events.py`: Removed unused `_apply_xyz_offset()` function (26 lines)

### Removed
- `base_env.py`: Removed `_cumulative_effort` buffer (logging-only, never used in reward)
- `base_env.py`: Removed `HeroAgentEnvWindow` class and `BaseEnvWindow` import
- `mdp/rewards.py`: Removed `action_rate_penalty()` and `angular_velocity_penalty()` functions (both had weight=0.0 in all configs)
- `mdp/__init__.py`: Removed corresponding imports and `__all__` exports
- `controllers/__init__.py`, `encoder/__init__.py`, `runners/__init__.py`: Removed MPC docstring references
- `direct/__init__.py`: Removed `from . import hero_agent_mpc` (directory was deleted previously)

### Fixed
- `mdp/rewards.py`: Restored `termination_penalty: float = -10.0` field that was incorrectly removed during cleanup (was active, not unused)
- `base_env.py`: Restored termination penalty application code in `_get_rewards()`

### Notes
- `encoder_tdc_env.py` kept as reference code (not registered, not simplified)
- All changes verified with `ruff check` and `ruff format`
- Full step-by-step log: `hero_agent/docs/code-simplification-log.md`
