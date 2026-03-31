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
