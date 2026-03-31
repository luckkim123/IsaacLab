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
