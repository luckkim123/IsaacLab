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

## [2026-04-02] DORAEMON Reference Implementation Alignment (4 commits)

### Context
Cloned reference DORAEMON implementation (github.com/gabrieletiboni/doraemon) and performed
detailed code-level comparison. Found critical structural divergences and incorrectly
removed "heuristic" settings that were actually part of the reference algorithm.

**Commit 1: Reference alignment** -- Binary success, single optimization, trust-constr.
Paper-analysis AI confirmed DORAEMON uses binary `sigma = 1{J >= J_LB}` with cumulative
reward threshold. J_LB=80 chosen (~53% of mature reward ~150).

**Commit 2: IS denominator fix** -- Run 22-13-00 (pre-fix) showed success_rate >1.0 in
33/224 iterations (max 1.20), ESS ratio crashed to 4%, 81/224 ESS reverts. Root cause:
stored per-episode log_probs became stale as distribution expanded (entropy -45 -> -29,
17 nats divergence). Switched IS denominator to prev_dist.log_prob(xi), matching reference
which uses current_distr as denominator with fresh data. Ring buffer data is mostly from
recent distributions close to prev_dist, so this is equivalent.

**Commit 3: Inverted problem keep_feasible** -- trust-constr with keep_feasible=True on
KL constraint with kl_ub=0.01 caused persistent "Inverted problem not successful" warnings.
Same root cause as the original main optimization stall. Changed to keep_feasible=False.

**Commit 4: step_interval + kl_ub restoration** -- Run 22-13-00 analysis (405 iter) showed
entropy reached maximum (-26.95, uniform) by iter 277 -- only 175 iterations after policy
crossed J_LB. Policy reward declined from 117 to 93 as DR became too hard too fast.
Root cause: step_interval was incorrectly removed as "heuristic" but is actually Algorithm 1
step 1 ("train policy for N steps"). Reference trains for 1M SAC steps (n_iters=5, budget=5M)
between each DORAEMON update with kl_ub=1.0 (default). Our kl_ub=0.01 was set for
per-iteration updates; with step_interval=250 giving ~10 total DORAEMON updates,
kl_ub=0.5 matches reference total KL budget (10*0.5=5.0 vs 5*1.0=5.0).

### Changed
- `doraemon.py`: Complete DoraemonScheduler restructure. Single constrained optimization
  (max entropy s.t. Ghat >= alpha, KL <= eps) + inverted problem when infeasible.
  trust-constr with analytical Jacobians and keep_feasible=False on all constraints.
  IS denominator: prev_dist.log_prob(xi) instead of stored per-episode log_probs.
  DoraemonCfg: `performance_lb=80.0`, `hard_performance_constraint=True`,
  `step_interval=250`, `kl_ub=0.5`.
- `albc_env.py`: Binary success `episode_return >= performance_lb`. Removed settling
  error infrastructure. Episode filter: `episode_length > 0`.
- `config.py`: `DoraemonCfg(enable=True, kl_ub=0.5, performance_lb=80.0, step_interval=250)`.

### Removed
- `doraemon.py`: threshold annealing (`success_threshold*`, `_anneal_threshold()`),
  backup-then-expand pattern, SLSQP helper (`_run_scipy_step`), self-normalized IS,
  `traversability_tau`, settling error parameters.
- `albc_env.py`: `_settling_window`, `_settling_errors`, `_settling_idx`, weighted
  settling error computation.

### Notes
- step_interval=250 was initially removed as "heuristic" then restored after reference
  analysis confirmed it implements Algorithm 1's policy training sub-routine. TRPO needs
  ~100-250 iterations to adapt (vs SAC's 1M env steps), so step_interval=250 is the
  correct TRPO-adapted value, not the raw env-step conversion (~4 iterations).
- kl_ub=0.01 was previously set because kl_ub=0.18 crashed success rate. The crash was
  caused by absence of step_interval (distribution changed every iteration), not kl_ub
  being too large. With step_interval=250, kl_ub=0.5 is safe.
- IS denominator choice: stored log_probs (theoretically correct for heterogeneous buffer)
  caused ESS collapse as distribution drifted. prev_dist (reference approach) keeps IS
  weights near 1.0 since ring buffer data is mostly from recent similar distributions.
- Reference repo at `/workspace/references/doraemon/` for ongoing comparison.

## [2026-04-02] DORAEMON Success Criterion: Fake Episode Fix + 3-Component Weighted Settling Error

### Context
Continued DORAEMON investigation after SLSQP fix (previous session). New training runs
showed success_rate crashing from 0.97 to 0.02 within 15 iterations despite DR distribution
being frozen (step_interval=50 prevented optimization). Multiple failed hypotheses tested:

1. **threshold/tau too strict** (0.25/0.035): Raised to 0.6/0.1 -- still crashed.
2. **lin_vel dominates settling error** (66% of range): Removed lin_vel, att-only -- still crashed.
3. **Exploration increases error**: User corrected -- TB showed att_err=25.7 deg from iter 0.

Root cause found via discrepancy analysis: at iter 0, DORAEMON success=0.976 but TB
att_err=25.7 deg. For success=0.976, settling_err must be ~0.23 (implying ~10 deg), not
25.7 deg. The real cause: `learn()` calls `env.reset()` which triggers `_log_and_reset_rewards()`
for all 4096 envs with `settling_idx=0` (no steps taken). `mean_settling_err = 0/1 = 0` gives
`success = sigmoid(-(0 - threshold)/tau) ~= 1.0`. These 4096 fake episodes (success ~1.0)
fill the 2000-capacity ring buffer. As real episodes (with actual errors) replace fake ones over
~11 iterations, buffer-average success drops from 0.97 to real levels (~0.09), creating the
appearance of a crash.

IS diagnostic on previous run's checkpoint (17-38-38, frozen at iter 674) confirmed SLSQP
works correctly: success_constraint feasible at x0 (0.78 vs alpha=0.5), optimizer found
+0.20 entropy improvement using full KL budget (0.01), ESS=1959/2000, IS weights nearly
uniform (Gini=0.076). IS estimation is NOT a problem.

After the settling filter fix, run 21-18-55 showed correct DORAEMON behavior: BACKUP at
iter 2 (success=0.38), policy learns, success crosses alpha=0.5 at iter 35, EXPAND at
iter 52 (entropy -45.76 -> -45.35). However, with att-only threshold=0.6, success saturated
at 0.975 while distribution barely changed -- criterion too loose.

Final design: 3-component weighted settling error (att=0.5, ang=0.3, lin=0.2) with
threshold=0.40, tau=0.08. At iter 170 policy level (att=7.7 deg, lin=0.31 m/s,
ang=17.8 deg/s), settling_err=0.396 gives success=0.514 -- right at EXPAND boundary.

### Fixed
- `albc_env.py`: Filter out episodes with `settling_idx < settling_window` from DORAEMON
  buffer recording. Initial `env.reset()` produced 4096 fake episodes with settling_err=0
  (success ~1.0) that polluted the buffer and caused artificial success rate crash.

### Changed
- `albc_env.py`: Settling error from `0.5*(att+lin)` to weighted `0.5*att + 0.3*ang + 0.2*lin`.
  Added yaw_rate_err component. Priority: attitude > angular velocity > linear velocity.
- `doraemon.py`: `success_threshold` 0.25 -> 0.40 (weighted formula calibrated to current
  policy level). `traversability_tau` 0.035 -> 0.08 (softer sigmoid for multi-component error).
- `doraemon.py`: `step()` restructured -- metrics always returned every call, optimization
  gated by `step_count % step_interval`. Fixes WandB logging gap from previous session's
  runner-side interval check.
- `doraemon.py`: Per-parameter distribution stats (mean/std) now logged every iteration
  (not just on optimization steps).

### Notes
- Settling filter verified: buffer starts at 32 episodes (not 2000 fake), fills to 2000 by iter 12
- Success trajectory with filter: 0.00 (iter 0-6) -> 0.16 (iter 49) -> policy learning dependent
- BACKUP during early training is expected (policy can't track yet, not DR's fault)
- BACKUP on already-tight initial distribution (concentration=30) is effectively a no-op
- User's final performance targets: att < 5 deg, lin_vel < 0.05 m/s, ang_vel < 5 deg/s
- Attempted and reverted: warmup_iters (unnecessary after settling filter fix),
  entropy floor clamp (no literature basis)

## [2026-04-02] DORAEMON Optimizer Fix (trust-constr -> SLSQP) + Step Interval

### Context
Deep analysis of run `full_dof_trpo/2026-04-02_17-38-38` (960+ iter) revealed DORAEMON
was completely frozen since iter 674: `kl_step=0` for 290+ consecutive iterations despite
`success_rate=0.72` (well above `alpha=0.5`) and `mode=1` (EXPAND). All 18 dimensions
froze simultaneously -- not a per-dimension issue.

Root cause investigation disproved initial IS weight collapse hypothesis (ESS ratio >97%
even at 10% perturbation). Gradient was non-zero (norm=0.37). Constraints had ample slack
(success 45% above alpha). The actual root cause: `keep_feasible=True` on the KL
`NonlinearConstraint` in scipy `trust-constr`. This forced every intermediate iteration to
satisfy `KL <= kl_ub`, causing the optimizer to rely on inaccurate numerical Jacobians
(36 params, 72+ finite-difference evals per iteration) for feasibility projection. Result:
optimizer consistently moved to WORSE objective (+0.136 entropy loss) and the update was
discarded (`result.success=False` AND `result.fun > x0_fun`).

Verified fix empirically: `keep_feasible=False` -> entropy improved 0.775 nats in 113 iter.
SLSQP optimizer -> same improvement in just 30 iterations with `result.success=True`.
SLSQP has no `keep_feasible` concept and handles constraints via SQP final-solution check.

Additionally, even with kl_ub reduced to 0.01, success still crashed from 0.93 to 0.0002
in 13 iterations because DORAEMON was updating every single RL iteration. Policy needs
~50+ iterations to adapt to any difficulty change. Added `step_interval` parameter.

Paper reference: Appendix A.2 (gradient vanishing near Beta(1,1)) confirmed -- 13/18 dims
at concentration 2.6-2.7 with per-param gradient ~0.06. But this was secondary to the
optimizer bug. Appendix A.2 epsilon sensitivity advice (smaller epsilon) was applied but
insufficient alone without fixing the optimizer.

### Fixed
- `doraemon.py`: Replaced `trust-constr` optimizer with `SLSQP` in `_run_scipy_step()`.
  Converts `NonlinearConstraint` objects to SLSQP dict format. Eliminates `keep_feasible`
  overhead that caused complete optimizer stall in 36-parameter Beta space. SLSQP converges
  in ~30 iterations vs trust-constr failing after 50-300. `maxiter` 50 -> 200.

### Changed
- `config.py`: `kl_ub` 0.18 -> 0.01. Previous value caused success to crash from 0.93 to
  ~0 in a few EXPAND steps. Paper Figure 8 recommends 0.005-0.01 for stability.
- `runners/constraint_encoder_runner.py`: DORAEMON `step()` now gated by
  `iteration % step_interval == 0`. Previously called every RL iteration.

### Added
- `doraemon.py`: `step_interval: int = 50` in `DoraemonCfg`. Controls how often DORAEMON
  optimization runs. Policy needs time to adapt between DR distribution changes. Default 50
  gives ~50 RL iterations of stable training between each DORAEMON update.

### Notes
- Diagnostic script at `/tmp/doraemon_diagnostic.py` was used for IS weight, gradient, and
  constraint feasibility analysis. Not committed.
- The paper (Tiboni et al., ICLR 2024) does not specify its optimizer implementation.
  Likely does not use `keep_feasible=True` or uses a different solver than scipy trust-constr.
- Entropy utilization was 96.6% at freeze point (-29.44 vs max -26.93). Gap breakdown:
  71.8% in 3 concentrated dims (cog_offset_z, volume_scale, cmd_att_scale), 14.4% moderate,
  13.8% near-uniform. All dims frozen simultaneously despite different concentrations.

## [2026-04-02] DORAEMON Buffer Fix, KL Budget Scaling, Settling Constraint, Reward Tuning

### Context
Deep analysis of run `full_dof_trpo/2026-04-02_16-36-08` (712 iter) cross-referenced with
DORAEMON paper (Tiboni et al., ICLR 2024) revealed critical implementation issues. Paper
Algorithm 1 uses importance sampling to reuse ALL collected data for distribution optimization,
but code was calling `buffer.clear()` after every `step()`, discarding all data. This caused
44% of iterations to skip DORAEMON optimization entirely (buffer_size < min_episodes=200).
Even when optimization ran, `kl_ub=0.01` shared across 18 dimensions gave per-dim KL budget
of 0.00056 -- too small for meaningful Beta distribution change. Result: cmd_scale only moved
0.30 -> 0.42 in 712 iterations, DORAEMON effectively stalled. `kl_step=0` in 63% of all
iterations. ESS revert never triggered (ess_ratio ~1.0), confirming the bottleneck was
skip + tiny KL budget, not IS weight variance.

Also added rp_vel_settling constraint (agreed in prior session): roll/pitch are position
targets, so angular velocity should average near zero. Reduced thruster penalty weight to
improve lin_vel_z tracking. Fixed missing lin vel command logging.

### Changed
- `doraemon.py`: Removed `buffer.clear()` from `step()` method. Ring buffer (capacity=2000)
  naturally overwrites old data. Paper's IS mechanism corrects for distribution shift. ESS
  check guards against extreme staleness. This is the single most impactful fix -- eliminates
  44% skip rate.
- `config.py`: `kl_ub` 0.01 -> 0.18 (per-dim 0.01 * 18 dims). Allows meaningful per-dim
  Beta distribution change while maintaining trust region stability. Previous value gave
  per-dim budget of 0.00056, practically freezing the distribution.
- `mdp/rewards.py`: `k_thr` -0.5 -> -0.35 (0.7x reduction). Thruster energy penalty was
  suppressing lin_vel_z tracking -- policy avoided vertical thrust to minimize penalty.

### Added
- `mdp/constraints.py`: `rp_vel_settling_cost()` -- average constraint `(|p|+|q|)/2` for
  roll/pitch angular velocity settling. Budget=0.05 (~3 deg/s average). Complements existing
  `rp_rate_cost` which only penalizes >1.0 rad/s. Total constraints: 10 -> 11.
- `config.py`: Registered `rp_vel_settling` constraint term (5 prob + 6 avg = 11 total).
- `albc_env.py`: Added `Command/lin_vel_x,y,z` logging in `_log_tracking_metrics()`. Previously
  only att_roll_deg, att_pitch_deg, yaw_rate were logged -- lin vel commands were missing.

### Notes
- `num_constraints` auto-synced by runner (constraint_encoder_runner.py:42-61), no manual
  update needed in rsl_rl_ppo_cfg.py.
- DORAEMON paper vs code: structure matches (entropy max + backup + KL trust region). Key
  differences: continuous sigmoid success (vs binary), self-normalized IS (vs unnormalized).
  Both are acceptable variants. The buffer.clear() was the only critical divergence.
- Checkpoint incompatibility: existing checkpoints have 10 constraints, new config has 11.
  Fresh training run required.
- **rp_vel_settling budget 0.05 -> 0.20**: Run `2026-04-02_17-29-18` (120 iter) showed budget
  0.05 was catastrophically tight. cost_return=11.34 >> d_k=5.0, barrier gradient overwhelmed
  reward signal. Roll/pitch errors diverged to 20-25 deg (vs 7 deg in previous run). DORAEMON
  success collapsed to 2.8e-06. Root cause: budget 0.05 allows only 2.9 deg/s average angular
  velocity, but attitude tracking with +-45 deg commands requires ~6-11 deg/s.

## [2026-04-02] Reward Revert, URDF Continuous Joints, Eval Tooling, Docs Reorganization

### Context
Post-session review found multiple uncommitted changes across sessions that were not
recorded in the changelog. Key decisions: (1) reverting exp kernel unification for
lin_vel/yaw tracking back to quadratic penalties based on training observation that
quadratic provides stronger gradient at large errors, (2) changing URDF joint types
from revolute to continuous for cable-aware constraint modeling, (3) refactoring
eval/compare scripts from hero_agent to constrained_albc, and (4) consolidating
scattered hero_agent documentation into centralized `docs/hero/` structure.

### Changed
- `agent.urdf`: joint1/joint2 changed from `revolute` to `continuous` type, joint
  limits (`lower`/`upper`) removed. Enables cable-aware constraint modeling without
  artificial +-2*pi limits. Software constraints must enforce any needed range.
- `constrained_full_albc/mdp/rewards.py`: Reverted lin_vel and yaw tracking from exp
  kernel back to quadratic penalty. `k_lin` 4.0 -> -4.0 (quadratic), `k_yaw` 4.0 ->
  -2.0 (quadratic). Removed `lin_sigma` and `yaw_sigma` fields. Only `att_rp` retains
  exp kernel (positive [0,1]). Contradicts 2026-04-01 exp unification -- later decision
  based on training results showing quadratic is better for velocity/yaw.
- `scripts/analysis/eval_dr.py`: Refactored from hero_agent to constrained_albc task
  support. Updated imports and class registrations to `ALBC*` prefix. Added
  `matplotlib.use("Agg")` before pyplot import for headless stability. Removed
  hero_agent-specific classes (BaseRunner, EncoderRunner, AdaptRunner).
- `scripts/analysis/compare_dr.py`: Per-segment steady-state computation (last 50% of
  each segment) replacing whole-trajectory last-50% averaging. Backward compat fallback
  for old .npz files without `steps_per_segment`. ruff format applied.
- `scripts/reinforcement_learning/rsl_rl/train.py`: Added `FullDOFConstraintEncoderRunner`
  mapping for constrained_full_albc runner resolution in the global runner dispatch table.
- `hero_agent/algorithms/ppo_patch.py`: ruff format applied (no behavioral change).

### Added
- `scripts/demos/test_full_dof_env.py`: Verification script for full-DOF velocity
  tracking ALBC environment (smoke test, obs dims, thruster motion, reward response,
  manipulability index, command resampling, constraint thresholds).
- `docs/hero/`: Centralized documentation structure with subdirectories (architecture/,
  archive/, environment/, experiments/, history/, plans/). Consolidates previously
  scattered hero_agent and constrained_albc docs.
- `constrained_albc/docs/README.md`, `hero_agent/docs/README.md`: Documentation index.

### Removed
- `hero_agent/docs/`: 13 standalone design documents (ARCHITECTURE.md, TDC_CONTROL_LAW.md,
  DOMAIN_RANDOMIZATION.md, DYNAMICS_ANALYSIS.md, PHYSICS_ENVIRONMENT.md, REWARD_FUNCTIONS.md,
  RL_TDC_COMPARISON.md, SAC_MPC_MONITORING.md, SIM_TO_REAL.md, TDC_LITERATURE_SURVEY.md,
  THEORETICAL_ANALYSIS.md, TRAINING_PIPELINE.md, code-simplification-log.md,
  tdc-tuning-history.md) -- consolidated into `docs/hero/`.
- `constrained_albc/docs/arm-freeze-analysis.md`, `constrained_albc/docs/changelog.md`:
  Archived into `docs/hero/`.
- `constrained_albc/encoder/actor_critic_constrained.py` (42 lines): Unused encoder variant.
- `docs/plans/`: 4 old plan files (2026-02 through 2026-03).
  `docs/superpowers/plans/`: 2 old plan files.
- `changelog_legacy.md` (root): Moved to `docs/hero/changelog_legacy.md`.

### Notes
- URDF `continuous` joint type removes PhysX joint limit enforcement. Constraint cost
  functions or action clamping must handle range limits in software.
- Reward revert timeline: 04-01 unified all to exp kernel -> later reverted lin_vel/yaw
  to quadratic. The exp kernel entry in 04-01 changelog reflects the state at that commit,
  not the final decision.
- Thruster `apply_dynamics` dt bug (physics_dt vs step_dt) still present.

## [2026-04-02] Exploration Recovery + Command Difficulty Curriculum

### Context
Analysis of run `full_dof_trpo/2026-04-01_06-03-44` (10k iter) revealed a coupled failure
mode: noise_std collapsed to 0.01 (floor) by step 5000 because the sigma optimizer had
no entropy incentive. With deterministic policy, DORAEMON success_rate dropped 0.60 -> 0.13
despite DR contraction (entropy -29 -> -62). Root cause: score-function gradient equilibrium
drove sigma to floor when reward plateaued (step ~1300). DORAEMON contracted DR slower
(kl_ub=0.0015) than policy degraded. Additionally, command difficulty was fixed at full
range regardless of policy capability. Three changes address the coupled problem: (1) entropy
bonus for sigma adaptive exploration, (2) DORAEMON-managed command difficulty, (3) faster
DR adaptation.

### Changed
- `algorithms/constraint_trpo.py`: Added `entropy_coef` parameter (default 0.005) to sigma
  optimizer. sigma_surrogate now includes `- entropy_coef * log_std.sum()`, providing soft
  upward pressure on sigma proportional to how low it is. Decoupled from TRPO trust region --
  only affects the separate Adam sigma optimizer. Score-function equilibrium shifts higher:
  when learning is poor (weak reward gradient), entropy bonus dominates -> sigma increases.
  When learning is good, reward gradient dominates -> sigma decreases for precision.
- `agents/rsl_rl_ppo_cfg.py`: `min_std` 0.01 -> 0.05 (safety floor, not primary mechanism).
  Added `entropy_coef: float = 0.005` to algorithm config.
- `doraemon.py`: PARAM_SPECS expanded from 15 to 18 dimensions. Added 3 command range
  scale parameters: `cmd_lin_scale`, `cmd_att_scale`, `cmd_yaw_scale` (range [0.1, 1.2],
  nominal=0.3). Commands start at ~30% of full range. DORAEMON entropy maximization pushes
  toward harder commands as success allows; backup contracts toward easier commands when
  policy struggles. Settling error normalization uses global (config) range, so easy commands
  naturally yield higher DORAEMON success -- no changes needed to success metric.
- `albc_env.py`: Added per-env command scale buffers (`_cmd_lin_scale`, `_cmd_att_scale`,
  `_cmd_yaw_scale`). Stored from DORAEMON sample at reset. `_sample_velocity_command()`
  now samples `uniform(-1, 1) * (max_range * per_env_scale)` instead of fixed
  `uniform(range_min, range_max)`. Mid-episode resampling uses same per-env scale.
- `config.py`: DORAEMON `kl_ub` 0.0015 -> 0.01 (~7x faster DR adaptation). Per 18 dims:
  0.00056 KL per dimension per step, enough for meaningful distribution shifts in 50-100 iter.

### Notes
- Existing checkpoints have 15-dim DORAEMON state; new 18-dim config will fail on load
  (intentional -- new runs require fresh start).
- No parameters scale with max_iterations. 2500 and 20000+ iter runs both work without
  config changes (use `--save_interval 200` for long runs).
- entropy_coef=0.005 matches RSL-RL PPO default for AnymalC/Allegro. In TRPO context this
  is safe because sigma is fully decoupled from trust region.
- DORAEMON cmd scales auto-logged as `DORAEMON/mean/cmd_lin_scale` etc. via get_stats().

---

## [2026-04-01] Attitude Command Review: Reward Unification, Constraint Redesign, Bug Fixes

### Context
Code review of the velocity-to-attitude command conversion in `constrained_full_albc`.
Three categories of issues addressed:

1. **Reward asymmetry**: Only roll/pitch used exp kernel (positive [0,1]), while lin_vel
   and yaw used quadratic penalties (negative, unbounded). This caused inconsistent reward
   scale across command types and made zero-command episodes structurally more rewarding.
   Decision: unify all tracking rewards to exp kernel for consistent [0,1] positive rewards.

2. **Constraint mismatch**: `angular_velocity_cost` used max(|p|,|q|,|r|) > 1.5, treating
   all axes identically. With attitude commands for roll/pitch, angular velocity is a tracking
   byproduct (not the command), so roll/pitch should have a tighter constraint. Yaw already
   had its own `yaw_rate_cost(r > 1.0)`. Decision: replace with `rp_rate_cost(max(|p|,|q|) > 1.0)`.

3. **Bug fixes**: DORAEMON settling error mixed m/s + radians; `_OBS_BIAS_MIN` had asymmetry
   from partial update; stale `ang_vel_err` comments.

### Changed
- `mdp/rewards.py`: All 3 tracking rewards now use exp kernel (positive [0,1]):
  `lin_vel` k=4.0 sigma=0.3 m/s, `att_rp` k=6.0 sigma=0.4 rad, `yaw_vel` k=4.0 sigma=0.3 rad/s.
  Previously lin_vel (k=-4.0 quadratic) and yaw (k=-1.0 quadratic) were negative penalties.
  Weights tuned after unification: att_rp 4.0->6.0 (x1.5), yaw 1.0->4.0 (x4) to emphasize
  attitude stability and yaw tracking relative to linear velocity.
- `mdp/constraints.py`: `angular_velocity_cost(max(p,q,r) > 1.5)` replaced by
  `rp_rate_cost(max(p,q) > 1.0)`. Roll/pitch-only, tighter threshold for attitude tracking.
  Yaw covered by existing `yaw_rate_cost(r > 1.0)`.
- `config.py`: Import and constraint list updated for `rp_rate_cost`. Section header updated.
  Noise comments `ang_vel_err` -> `ang_err [att_rp+yaw_rate]`.

### Fixed
- `albc_env.py`: DORAEMON settling error now normalizes each channel by its command range
  before averaging: `0.5 * (lin_err/0.5 + att_err/(pi/4))`. Dimensionless metric makes
  threshold=0.25 and tau=0.035 scale-independent.
- `config.py`: `_OBS_BIAS_MIN` body tracking history `[-0.03]*3` -> `[-0.04]*3` to match
  `_OBS_BIAS_MAX`, restoring symmetry.
- `doraemon.py`: Updated threshold/tau comments from "m/s" to "dimensionless".

### Notes
- Exp kernel sigma rationale: ~60% of command range gives good gradient at typical errors.
  sigma=0.3 for lin_vel/yaw (0.5 m/s or rad/s range), sigma=0.4 for att_rp (0.785 rad range).
- `attitude_limit_cost(limit=80 deg)` unchanged -- well above +-45 deg command range.
- DORAEMON threshold=0.25 retains equivalent behavior for pure velocity tracking.

---

## [2026-04-01] Revert Wrench-Space, Remove Velocity Termination, Direct Thruster + std=0.5

### Context
Following the wrench-space experiment (see below), analysis concluded that:
1. Wrench-space adds complexity (TAM inverse, saturation handling, roll singularity) without
   clear benefit over simply adjusting init_noise_std.
2. Velocity-based hard termination (too_fast_ang > pi, too_fast_lin > 2 m/s) is the root cause
   of the death spiral: all-negative rewards make early death optimal. Soft constraints already
   provide per-step gradient for velocity control.
3. PhysX rigid body `max_angular_velocity=720 deg/s (4*pi)` provides the hard physical clamp.
   The Python termination at pi was 4x more conservative and redundant.
4. Arm uses delta parameterization (rate-limited by delta_scale), but thrusters use absolute
   commands with no scaling. Applying action_scale to thrusters would permanently limit thrust
   authority, which is theoretically incorrect. The proper control is init_noise_std.

Decision: revert wrench-space, remove velocity termination, keep direct thruster control with
lower init_noise_std. This eliminates the death spiral while preserving full thrust authority.

### Changed
- `albc_env.py`: Removed wrench-to-thruster transformation (`_init_wrench_transform`,
  `_wrench_to_thruster`), reverted to direct `apply_dynamics(actions[:, 2:])`.
- `albc_env.py`: `_get_dones()` now terminates only on `bad_state` (NaN/Inf) and
  `excessive_tilt` (>90 deg). Velocity flags (`too_fast_ang`, `too_fast_lin`) computed
  for diagnostics only, no longer trigger termination.
- `agents/rsl_rl_ppo_cfg.py`: `init_noise_std` 0.3 -> 0.5 (balance exploration vs stability).
- `mdp/rewards.py`: `termination_penalty` -50.0 -> 0.0 (no velocity termination = no penalty needed).
- `config.py`: Reverted action_space docstring to "6D thruster".
- `albc_env.py`: Reverted logging labels `Action/wrench_*` -> `Action/thruster_*`.

### Notes
- `too_fast_ang` and `too_fast_lin` diagnostic flags are still computed and logged via
  `_collect_termination_metrics`. They just don't trigger episode reset.
- Remaining termination conditions: `bad_state` (PhysX failure), `excessive_tilt` (>90 deg,
  buoyancy/gravity reversal). Both are non-recoverable simulation states.
- Soft constraints providing velocity gradient: `ang_vel_cost` (threshold 1.5),
  `yaw_rate_cost` (threshold 1.0), `body_lin_vel_cost` (threshold 1.0).
- Thruster `apply_dynamics` dt bug (physics_dt vs step_dt) still present.

---

## [2026-04-01] Wrench-Space Experiment + init_noise_std Root Cause Analysis

### Context
Previous session's fixes (ang_vel constraint + termination_penalty -50) did not resolve
100% `too_fast_ang` termination. Systematic debugging revealed the root cause:

1. **NOT reward/constraint**: penalty=-50 mathematically prevents death spiral (breakeven
   at per-step -0.50, typical is -0.10).
2. **Physical TAM structure**: Hero Agent yaw row is all-same-sign (+0.144 x4). Any
   horizontal thrust produces yaw torque. Random policy with std=1.0 generates
   |T0+T1+T2+T3| ~ 1.05, causing yaw acceleration of 84 rad/s^2.
3. **TRPO KL constraint**: std 1.0->0.3 requires ~40 iterations at KL=0.005. Intermediate
   std values (0.5-0.7) produce WORSE returns than dying early (reward valley).
4. **Root cause confirmed**: init_noise_std=0.3 resolved too_fast_ang=1.0 immediately.

Wrench-space action transformation was implemented (TAM pseudo-inverse mapping
policy output to [surge,sway,heave,roll,pitch,yaw]) to structurally decouple yaw.
However, analysis showed:
- TAM roll row (Mx) is linearly dependent on sway (Fy) for co-planar thrusters
  (Mx = -0.0099 * Fy), making 4x4 sub-TAM singular.
- Per-axis scaling with std=0.7 still caused 41% thruster saturation, corrupting
  wrench allocation. std=0.3 was needed regardless.
- Wrench-space adds complexity (TAM inverse, saturation handling) without clear
  benefit over simply lowering init_noise_std.

Decision: wrench-space will be reverted in favor of direct thruster control + lower std.

### Changed
- `albc_env.py`: Added wrench-to-thruster transformation (`_init_wrench_transform`,
  `_wrench_to_thruster`) using subsystem decomposition: horizontal 4x4 pinv (rank 3)
  + vertical 2x2 inv. Per-axis scaling normalizes policy output to max achievable wrench.
- `albc_env.py`: Logging labels `Action/thruster_*` -> `Action/wrench_*`.
- `config.py`: Updated action_space docstring for wrench-space layout.
- `agents/rsl_rl_ppo_cfg.py`: `init_noise_std` 1.0 -> 0.3.

### Notes
- PhysX rigid body `max_angular_velocity=720 deg/s (4*pi)`, while Python termination
  fires at pi. The Python check is 4x more conservative than PhysX.
- Arm uses delta parameterization (rate-limited by delta_scale=0.10), but thrusters
  use absolute commands with no scaling -- this asymmetry is the fundamental cause of
  the spin-out at high init_noise_std.
- Thruster `apply_dynamics` dt bug (physics_dt vs step_dt) still present.

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
