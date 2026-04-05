# Changelog: constrained_full_albc Early Development

Development history for `constrained_full_albc` (Full 6-DOF ALBC) from initial
creation through DORAEMON stabilization. Covers 2026-03-31 to 2026-04-02.

For the active development changelog (2026-04-04+), see the main
[changelog.md](/workspace/isaaclab/changelog.md).

---

## [2026-04-02] DORAEMON Reference Implementation Alignment (4 commits)

### Context
Cloned reference DORAEMON implementation (Tiboni et al., ICLR 2024) and performed
detailed code comparison. Found critical structural divergences.

**Key fixes**:
1. Binary success criterion (`episode_return >= J_LB`) matching paper Algorithm 1
2. IS denominator: switched to `prev_dist.log_prob(xi)` (ring buffer compatible)
3. `keep_feasible=False` for inverted problem (trust-constr stalls with True)
4. `step_interval=250` restored (implements Algorithm 1's "train for N steps")
5. `kl_ub=0.5` with step_interval=250 matches reference total KL budget

### Changed
- `doraemon.py`: Complete restructure -- single constrained optimization + inverted problem
- `albc_env.py`: Binary success, removed settling error infrastructure
- `config.py`: `DoraemonCfg(kl_ub=0.5, performance_lb=80.0, step_interval=250)`

---

## [2026-04-02] DORAEMON Success Criterion: Fake Episode Fix

### Context
`env.reset()` at training start produced 4096 fake episodes with settling_err=0
(success~1.0) filling the 2000-capacity ring buffer. As real episodes replaced fake
ones over ~11 iterations, success appeared to crash from 0.97 to real levels.

After fix, added 3-component weighted settling error (att=0.5, ang=0.3, lin=0.2)
with threshold=0.40.

### Fixed
- `albc_env.py`: Filter episodes with `settling_idx < settling_window` from buffer

---

## [2026-04-02] DORAEMON Optimizer Fix (trust-constr -> SLSQP) + Step Interval

### Context
DORAEMON frozen since iter 674: `kl_step=0` for 290+ iterations despite success_rate=0.72.
Root cause: `keep_feasible=True` on KL constraint in scipy trust-constr caused optimizer
to stall in 36-parameter Beta space. SLSQP converged in ~30 iterations vs trust-constr failing.

Also: DORAEMON updated every RL iteration caused success crashes. Added `step_interval=50`.

### Fixed
- `doraemon.py`: trust-constr -> SLSQP, maxiter 50 -> 200

### Changed
- `config.py`: `kl_ub` 0.18 -> 0.01
- `runners/`: DORAEMON step gated by `iteration % step_interval == 0`

---

## [2026-04-02] DORAEMON Buffer Fix, KL Budget, Settling Constraint

### Context
Buffer.clear() after every step() discarded all data (44% skip rate). kl_ub=0.01
shared across 18 dims gave per-dim budget 0.00056 (too small).

### Changed
- `doraemon.py`: Removed buffer.clear() (ring buffer naturally overwrites)
- `config.py`: kl_ub 0.01 -> 0.18

### Added
- `mdp/constraints.py`: `rp_vel_settling_cost` (budget initially 0.05, later raised to 0.20)
- `albc_env.py`: lin vel command logging

---

## [2026-04-02] Reward Revert, URDF Continuous Joints, Eval Tooling, Docs Reorganization

### Changed
- `agent.urdf`: joint1/joint2 revolute -> continuous (cable-aware constraints)
- Reverted lin_vel/yaw from exp kernel back to quadratic penalty
- `eval_dr.py`: Refactored from hero_agent to constrained_albc support
- `compare_dr.py`: Per-segment SS computation
- `train.py`: Added FullDOF runner mapping

### Added
- `scripts/demos/test_full_dof_env.py`: Smoke test for full-DOF env
- `docs/hero/`: Centralized documentation structure

### Removed
- 13 standalone hero_agent docs (consolidated into docs/hero/)
- `constrained_albc/encoder/actor_critic_constrained.py` (unused)

---

## [2026-04-02] Exploration Recovery + Command Difficulty Curriculum

### Context
noise_std collapsed to 0.01 (floor) by step 5000. DORAEMON contracted DR slower than
policy degraded. Three fixes: (1) entropy_coef for sigma, (2) DORAEMON command scales,
(3) faster DR adaptation.

### Changed
- `constraint_trpo.py`: Added `entropy_coef=0.005` to sigma optimizer
- `agents/rsl_rl_ppo_cfg.py`: min_std 0.01 -> 0.05
- `doraemon.py`: PARAM_SPECS 15 -> 18 dims (added cmd_lin/att/yaw_scale)
- `albc_env.py`: Per-env command scale buffers from DORAEMON
- `config.py`: kl_ub 0.0015 -> 0.01

---

## [2026-04-01] Attitude Command Review: Reward Unification, Constraint Redesign

### Changed
- All 3 tracking rewards unified to exp kernel (att_rp k=6.0, lin_vel k=4.0, yaw k=4.0)
- `angular_velocity_cost` -> `rp_rate_cost(max(|p|,|q|) > 1.0)`

### Fixed
- DORAEMON settling error normalization (dimensionless)
- `_OBS_BIAS_MIN` symmetry fix

---

## [2026-04-01] Revert Wrench-Space, Remove Velocity Termination

### Context
Wrench-space adds complexity without benefit over lowering init_noise_std. Velocity
termination caused death spiral (all-negative rewards make early death optimal).

### Changed
- Reverted wrench-to-thruster transformation
- Termination: only bad_state + excessive_tilt (>90 deg)
- init_noise_std 0.3 -> 0.5, termination_penalty -50 -> 0

---

## [2026-04-01] Wrench-Space Experiment + init_noise_std Root Cause

### Context
100% too_fast_ang termination. Root cause: Hero Agent TAM yaw row all-same-sign
(+0.144 x4) + low yaw inertia -> 84 rad/s^2 max yaw acceleration. init_noise_std=0.3
resolved immediately.

Wrench-space implemented and tested but reverted (see above).

---

## [2026-04-01] Logging Overhaul + Spin-Out Death Spiral Fix

### Context
First training run showed 100% early termination. Two root causes:
1. Reward death spiral (all-negative + weak penalty)
2. No angular velocity soft constraint

### Added
- `angular_velocity_cost(threshold=1.5)` constraint
- Per-axis velocity tracking, thruster diagnostics, cumulative yaw logging

### Changed
- termination_penalty -10 -> -50
- Per-subsystem action split (arm 2D + thruster 6D)

---

## [2026-03-31] Gap Analysis + Code Simplification

### Gap Analysis
17/25 historical lessons correctly implemented, 5 cleared after verification, 3 minor
fixes applied. Removed cost critic `clamp(min=0.0)` (positive bias). Fixed stale docstrings.

### Code Simplification
Package cleanup (4,968 lines, 19 files). Split long methods, removed backward compat shims.
No behavioral changes.
