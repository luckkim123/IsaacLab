# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

For constrained_albc development history (Steps 1-8, 2026-03-27 -- 2026-03-31), see
[changelog_constrained_albc.md](docs/hero/changelog_constrained_albc.md).

For earlier development history (Phase 1-8, 2026-03-05 -- 2026-03-26), see
[changelog_legacy.md](docs/hero/changelog_legacy.md).

For the encoder ablation study (Steps 0-19), see
[encoder_ablation.md](docs/hero/experiments/encoder_ablation.md).

---

## [2026-04-01] Logging System Overhaul + Spin-Out Death Spiral Fix

### Context
First training run of `constrained_full_albc` (Isaac-FullDOF-TRPO-v0) showed 100%
early termination from angular velocity (`too_fast_ang=1.0`, `time_out=0.0`). Min
episode length converged DOWN to 4 steps, indicating the policy was actively learning
to spin out.

Root cause analysis revealed two issues:
1. **Reward death spiral**: All-negative rewards with `termination_penalty=-10` made
   early death optimal when per-step penalty > -0.10 (common with DR). Discounted
   return from dying at step 4 (-10.4) beat surviving (-15 to -20 with bad tracking).
2. **Missing angular velocity soft constraint**: Hard termination at `max_angular_velocity=pi`
   had no corresponding soft constraint. The policy received zero gradient signal before
   hitting the wall. Only `yaw_rate` (1 axis) was constrained; roll/pitch rate were
   completely unprotected.

Contributing factor: Hero Agent allocation matrix has all-same-sign yaw torque row
`(+0.144, +0.144, +0.144, +0.144)` (unlike BlueROV's alternating `+0.19, -0.19`),
combined with extremely low yaw inertia (Izz=0.037 kg*m^2). Max yaw angular
acceleration = 319 rad/s^2, reaching pi in ~4-5 env steps even through first-order
thruster filter.

Separately, the WandB logging system was reviewed and overhauled. The system had
~141 metrics inherited from constrained_albc, with mixed 8D action norms, no
per-axis velocity tracking, no thruster diagnostics, and one duplicate metric.

### Added
- `mdp/constraints.py`: New `angular_velocity_cost(soft_threshold=1.5)` function.
  Uses `max(|p|,|q|,|r|)` (not mean) to match the hard termination condition.
  Provides IPO barrier gradient on ALL 3 axes before the hard wall at pi.
- `config.py`: Added `[5] ang_vel` constraint term (budget=0.10, threshold=1.5).
  Constraint count: 9 -> 10 (5 prob + 5 avg). Runner auto-syncs via env config.
- `albc_env.py`: Added per-axis velocity tracking: `Vel_Tracking/lin_err_x/y/z`,
  `ang_err_roll/pitch/yaw` for surge/sway/heave and roll/pitch/yaw rate diagnostics.
- `albc_env.py`: Added thruster diagnostics: `Thruster/utilization_mean`, `_max`,
  `_std` (std captures whether thrust is distributed vs concentrated).
- `albc_env.py`: Added `Control/cumulative_yaw_deg` (tether wrapping indicator).
- `runners/constraint_encoder_runner.py`: Added `Policy/surrogate_loss` (was computed
  in constraint_trpo.py but never logged).

### Changed
- `mdp/rewards.py`: `termination_penalty` -10.0 -> -50.0. Moves death spiral breakeven
  from per-step penalty -0.10 to -0.50, covering all realistic DR scenarios.
- `albc_env.py`: Replaced mixed 8D `Action/size_mean` + `Action/rate_mean` with
  per-subsystem split: `Action/arm_norm`, `arm_rate` (2D) and `Action/thruster_norm`,
  `thruster_rate` (6D). The 8D combined norm was meaningless (arm delta scale=0.08
  mixed with thruster commands in [-1,1]).
- `utils/logging.py`: Removed `alg` parameter from `log_encoder_metrics()` (no longer
  needed after grad_norm dedup). Updated docstring: 5 metrics -> 4 metrics.

### Removed
- `utils/logging.py`: Removed duplicate `Encoder/grad_norm` (identical to
  `Policy/encoder_grad_norm` in runner, both read `alg._last_encoder_grad_norm`).
- `albc_env.py`: Removed `Vel_Tracking/lin_vel_cmd_norm` and `ang_vel_cmd_norm`
  (command magnitude context metrics, low diagnostic value vs per-axis errors).

### Notes
- New constraint defense structure:
  `ang_vel 0 --[1.5 soft]-- pi [hard kill, -50 penalty]`
  vs previous: `ang_vel 0 -------------- pi [hard kill, -10 penalty]` (no soft)
- `yaw_rate` (threshold 1.0) kept alongside `ang_vel` (threshold 1.5): yaw gets
  tighter protection due to Hero Agent's structurally weak yaw axis (Izz=0.072 total).
- Mean vs max for angular velocity cost: max chosen to match termination condition.
  Mean would allow single-axis spin (2.5 rad/s) to go unpenalized if other axes are 0.
- Thruster `apply_dynamics` uses `physics_dt` (0.005s) instead of `step_dt` (0.02s),
  making thruster 4x slower than intended. Not fixed this session -- accidentally
  helpful (slower ramp = less spin-out), but is a real bug for future reference.
- Total WandB metrics: ~141 -> ~151 (net +10). More metrics but better organized
  with per-axis, per-subsystem splits replacing mixed aggregates.

## [2026-03-31] Gap Analysis: constrained_full_albc vs Historical Lessons

### Context
Systematic gap analysis comparing `constrained_full_albc` implementation against
25 key lessons from 85+ commits across 8 phases (changelog_legacy) and 8 steps
(changelog_constrained_albc). Cross-referenced encoder ablation study (20+ experiments),
arm freeze root cause analysis, and all architecture/experiment documentation in
`docs/hero/`.

Result: 17/25 lessons correctly implemented (PASS), 5 potential issues analyzed
in depth and cleared (OK after numerical verification), 3 minor issues found and fixed.

### Changed
- `algorithms/constraint_trpo.py`: Removed `clamp(min=0.0)` on cost critic targets
  in `_update_values()`. The clamp created systematic positive bias in cost value
  predictions: critic learned 0+ targets while cost advantages could be negative,
  making the IPO barrier slightly more permissive than intended during early training.
  Without clamp, critic can predict negative cost values (valid when cost GAE delta
  is negative due to over-prediction), improving barrier accuracy.

### Fixed
- `__init__.py`: Fixed stale module docstring "33D obs" to "81D obs"
- `albc_env.py`: Fixed stale docstring "23D privileged" to "24D" in `_get_observations()`
- `albc_env.py`: Fixed stale comment "decimation=40, 40 times (2000Hz PD)" to
  "decimation=4, 4 times (200Hz PD)" in `_pre_physics_step()`

### Notes
- `barrier_alpha=0.05` (vs paper's 0.02): Verified correct for 8D action space.
  Barrier gradient is 2.4x weaker but reward gradient is ~4x stronger due to
  action dimensionality. Thruster constraint (d_k=40) dominates gradient budget.
- `torque_limit_cost` uses `applied_torque` (post-clamp): Verified correct.
  `computed_torque` (pre-clamp) would fire on normal PD operation (Kp=100, err=0.1
  -> 10 Nm > 9.5 threshold). ~10% of DR envs (effort_limit < 9.5 Nm) have
  structurally inactive torque constraint -- acceptable.
- Encoder weight_decay=0: Safe under TRPO (LayerNorm + softsign + trust region
  provide 3-layer defense vs PPO where WD was needed).
- Encoder static-only input (24D privileged): Intentional HORA Phase 1 design.
  z/actor_input ratio = 10% (vs 48% that caused KL dominance in 2D ALBC).
  4096 unique z per iteration is sufficient with `ca_std.clamp(min=1.0)` guard.

## [2026-03-31] Code Simplification: constrained_full_albc

### Context
Systematic code cleanup of `constrained_full_albc` (4,968 lines, 19 files). Package was
forked from `constrained_albc` through 8 rapid development steps. Three parallel code
explorer agents analyzed: dead/legacy code, code duplication, and structural complexity.

Result: Package was already quite clean (no true dead code found). Changes focused on
removing backward compat shims, splitting long methods, and improving documentation.
Behavioral changes: none -- readability/maintainability only.

### Changed
- `albc_env.py`: Split `_collect_episode_metrics` (97 lines) into dispatcher + 4 helpers:
  `_log_tracking_metrics`, `_log_action_metrics`, `_log_dynamics_metrics`,
  `_log_midep_metrics`. Main method reduced to ~30 lines.
- `albc_env.py`: Split `_init_state_buffers` (59 lines) into dispatcher + 5 helpers:
  `_init_action_buffers`, `_init_history_buffers`, `_init_velocity_buffers`,
  `_init_tracking_buffers`, `_init_force_buffers`. Main method reduced to 5 lines.
- `doraemon.py`: Simplified `load_state_dict` from ~35 lines (dimension mismatch recovery,
  field rename compat, buffer partial restore) to ~15 lines (strict loading with ValueError
  on dimension mismatch). New package has no legacy checkpoints to support.
- `doraemon.py`: Improved PARAM_SPECS formatting (aligned columns) and added SYNC comment
  noting manual synchronization requirement with `DomainRandomizationCfg`.
- `config.py`: Added observation dimension breakdown comment
  (cmd=6 + body=9 + arm=5 + thruster=6 = 26D current + 55D history).
- `config.py`: Fixed stale comment "ReLU threshold" to "soft threshold".

### Removed
- `config.py`: Removed unused `max_joint_velocity` field (not referenced anywhere).
- `encoder/actor_critic_encoder.py`: Removed `proprio_hist_dim` backward compat shim
  (`kwargs.pop("proprio_hist_dim", None)`). No config in this package uses that field.
- `runners/constraint_encoder_runner.py`: Removed duplicate
  `_runner_module.FullDOFActorCriticEncoder = ActorCriticEncoder` class registration
  (already done in `agents/rsl_rl_ppo_cfg.py`). Removed now-unused `_runner_module` and
  `ActorCriticEncoder` imports.
- `doraemon.py`: Removed checkpoint backward compat handling: `current_threshold_deg`
  field rename fallback, buffer dimension mismatch partial restore, distribution dimension
  mismatch partial restore.

### Notes
- DORAEMON `PARAM_SPECS` bounds are duplicated with `DomainRandomizationCfg`. Dynamic
  derivation was considered but rejected as over-engineering (parameter name mapping is
  non-trivial, nominal values differ from config midpoints). Added SYNC comment instead.
- `ruff check` passed, `ruff format` applied to 5 files.
