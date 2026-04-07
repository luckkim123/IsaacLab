# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## Historical changelogs

- [Full ALBC early development](docs/hero/changelog_full_albc_early.md) (2026-03-31 -- 2026-04-02): DORAEMON stabilization, wrench-space experiment, logging overhaul, code simplification
- [Constrained ALBC development](docs/hero/changelog_constrained_albc.md) (2026-03-27 -- 2026-03-31): Steps 1-8
- [Legacy development](docs/hero/changelog_legacy.md) (2026-03-05 -- 2026-03-26): Phase 1-8, hero_agent TDC/encoder
- [Encoder ablation study](docs/hero/experiments/encoder_ablation.md) (Steps 0-19)

---

## [2026-04-08] Full-DOF Comparison Baselines (Phases 1-3)

### Context
Set up three ablation baselines for `Isaac-FullDOF-TRPO-v0` to isolate the
contribution of each component (encoder, constraints, learning itself). All
three reuse `ALBCEnv` so DR, reward, action space, command sampling, and
DORAEMON match the production task exactly.

| Phase | Task                      | What it removes                          |
|-------|---------------------------|------------------------------------------|
| 1     | `Isaac-FullDOF-NoEncoder-v0` | Encoder only (TRPO + IPO kept)        |
| 2     | `Isaac-FullDOF-PPO-v0`       | Encoder + IPO constraint (plain PPO)  |
| 3     | `Isaac-FullDOF-TDC-v0`       | All RL (classical TDC + 6-DOF PD)     |

### Added
- `constrained_full_albc/encoder/actor_critic_asym_constrained.py`: NoEncoder
  policy class for Phase 1. Same asymmetric critic + cost critic as the
  encoder variant, just without the privileged compression head.
- `constrained_full_albc_tdc/`: new module for Phase 3 (sibling of
  `constrained_full_albc/`). Subclasses `ALBCEnv` and overrides only
  `_pre_physics_step` to inject classical controller output as an 8D
  pseudo-action; the parent pipeline handles observation history, reward,
  thruster lag, etc. unchanged.
- `constrained_full_albc_tdc/controllers/thruster_pd.py`: stateless 6-DOF
  thruster PD (`ThrusterPDController`). Drives Fx/Fy/Fz (lin vel),
  **Tx/Ty (roll/pitch attitude PD)**, and Tz (yaw rate) so the baseline has
  full thruster authority on all six wrench components, matching what the RL
  policy has. Initial Tx=Ty=0 design was rejected mid-session as a baseline
  weakening.
- `constrained_full_albc_tdc/{tdc_env.py,config.py,__init__.py,...}`: TDC env
  glue. Reuses `hero_agent.controllers.tdc.TDCController` and
  `hero_agent.controllers.kinematics.ALBCKinematics` as-is. Overrides
  `TDCControllerCfg` with `ik_num_iterations=1, ik_learning_rate=1.0`
  (single-step DLS) -- the rate limiter only allows 0.05 rad/step which the
  single-step solver tracks accurately, and the 100-iter accurate mode adds
  ~30 ms/step of CUDA launch overhead on GPU.

### Changed
- `constrained_full_albc/__init__.py`: registers
  `Isaac-FullDOF-NoEncoder-v0` and `Isaac-FullDOF-PPO-v0`. (Phase 3 task
  is registered in its own module's `__init__.py`.)
- `constrained_full_albc/agents/rsl_rl_ppo_cfg.py`: adds
  `FullDOFNoEncoderRunnerCfg` (Phase 1) and `FullDOFPPORunnerCfg` (Phase 2).
  PPO baseline uses asymmetric obs routing via `obs_groups` so the standard
  rsl-rl `ActorCritic` can have a 81D actor and a 105D critic without a
  custom policy class.
- `constrained_full_albc/encoder/__init__.py`: exports the new
  `ActorCriticAsymConstrained` class.
- `constrained_full_albc/runners/constraint_encoder_runner.py`: encoder
  logging is now optional so the runner works for the NoEncoder ablation
  without spurious zeros in the encoder-specific metrics.

### Phase 3 eval (`eval_dr_fulldof.py`, 64 envs, 4 DR levels)

Stored at `logs/rsl_rl/tdc_pd_baseline/`. Survival 100% at every DR level
(none/soft/medium/hard). Attitude SS error 2.8-7.1 deg (best at DR 100%
because random damping increases happen to help). Linear velocity has the
expected P-only steady-state floor: ~0.11 m/s on vx/vy and 0.25-0.40 m/s
on vz (heave), which scales as `F_drag / kp_lin`. Yaw rate degrades from
0.013 to 0.13 rad/s on hard DR as the P controller is overwhelmed by the
widened yaw damping range.

Post-eval gain bump committed but **not re-validated**: `kp_lin 30 -> 100`,
`kp_yaw 8 -> 25`, `kp_att 8 -> 20`, `kd_att 2 -> 5`. Predicted ~3x SS error
reduction; saturation budget is safe.

### Notes
- Eval artifacts (npz + 9 PNGs) are kept on disk, not committed.
- Phase 1 and 2 task registrations and runner cfgs were created in earlier
  sessions but never committed; they are bundled into this commit because
  the changelog covers all three baselines together.
- Reviewer-caught fixes applied to Phase 3 env: `_coerce_env_ids` instead
  of an asymmetric assert in `_reset_idx`, `_tdc_dt = step_dt` (drop the
  redundant `* control_decimation`), `no_grad` scope extended to cover the
  parent `_pre_physics_step` call, and a no-op `ThrusterPDController.reset`
  removed.
- Wallclock optimization for `hero_agent`'s 100-iter IK loop is the obvious
  follow-up: ~600 sequential GPU kernel launches per step add ~30 ms of
  pure CUDA launch overhead. `torch.compile` fusion or an analytic 2-link
  IK would remove it entirely.

---

## [2026-04-07] Revert rp_vel_settling Budget (Constraint vs Reward Conflict)

### Context
Mid-training analysis of run `2026-04-07_22-24-20` at iter ~833 (post linear-penalty
revert) showed att_rp Episode_Reward stuck near zero while lin_vel learned normally.
Comparison against OLD baseline (`2026-04-06_21-24-43`) at the same iter:

| Metric                       | OLD@800 | CURR@800 | Delta            |
|------------------------------|---------|----------|------------------|
| Train/mean_reward            |  +97.1  |  +11.7   | -85.4 (-88%)     |
| Episode_Reward/att_rp        |  +1.602 |  -0.855  | **sign flipped** |
| Episode_Reward/lin_vel       |  +1.482 |  +1.724  | +0.24 (better)   |
| Episode_Reward/yaw_vel       |  +0.695 |  +0.233  | -0.46            |
| Episode_Reward/smoothness    |  -0.327 |  -0.544  | -0.22 (66% worse)|
| DORAEMON/success_rate        |   0.490 |   0.041  | -0.45 (mode -2)  |

The selective regression (att_rp dead, lin_vel fine) ruled out "wider DR is harder
for everything" -- only the channel that requires roll/pitch rotation suffered. Att_rp
slope in the last 200 iter dropped to +0.025/100, essentially frozen. At this rate
recovery to OLD's +1.60 would need ~9800 more iters (12x current), confirming the
plateau is structural rather than transient.

**Mechanism (code-level)**: `rp_vel_settling_cost` is `(|p|+|q|)/2` averaged over the
episode (`mdp/constraints.py:236-250`). It is an Average-type IPO constraint, and at
budget=0.12 the cost was 9.86/12 = 82% of budget -- in the active barrier-binding
region. The IPO barrier (`barrier_t=100.0`) injects strong negative gradient on the
actor whenever cost approaches budget, directly opposing the att_rp_tracking reward
gradient (which requires p,q angular velocity to follow attitude commands).

OLD ran with budget=0.20 and the same cost was 16.85/20 = 84% (similar binding
ratio), but the absolute limit of 0.20 rad/s was sufficient for a 60-deg traverse
in ~5.2s. With budget=0.12 a 60-deg traverse needs ~8.7s, deep into the IPO
binding region for most of the trajectory. lin_vel does not interact with this
constraint and learns normally; yaw_vel interacts only weakly (yaw_rate constraint
threshold is 0.7 rad/s, well above the 0.5 cmd range). This is the diagnostic
fingerprint: only the channel that fights `rp_vel_settling` is dead.

The constraint's name implies a settling-phase-only behavior, but the implementation
has no time/error gating -- it penalizes |p|+|q| at every timestep regardless of
whether the policy is in a transition or a settling phase. A proper settling-aware
redesign is deferred; the immediate fix is to restore the OLD budget so the next run
isolates the linear-penalty revert as intended.

**Decision**: Revert ONLY `rp_vel_settling.budget` 0.12 -> 0.20 (back to OLD value).
Keep all other 04-07 changes (`performance_lb=80`, HardDR expansion, `yaw_rate=0.7`)
since they did not show selective harm in the data. This is the minimum-change revert
that targets the proximate cause; if att_rp recovers in the next run, the structural
hypothesis is confirmed and the other 3 changes can be evaluated cleanly.

### Changed
- `config.py`: `rp_vel_settling.budget` 0.12 -> 0.20 (single line, matches OLD).
  All other fields unchanged. The constraint definition itself
  (`mdp/constraints.py:rp_vel_settling_cost`) is untouched.

### Notes
- HardDR expansion is NOT the proximate cause despite suspicions. lin_vel learning
  faster than OLD under the wider DR rules out "DR is too hard". The wider DR may
  be a contributing factor but is not load-bearing for the att_rp regression.
- `performance_lb=80` reduction is also exonerated by this evidence. The reason
  DORAEMON success is stuck at 0.04 is that mean_reward (11.7) is far below lb=80,
  not that lb is wrong. Once att_rp recovers, mean_reward should rise into the
  lb-feasible region naturally.
- Smoothness regression (-0.327 -> -0.544) is also expected to recover after the
  revert. It was caused by gradient conflict between IPO barrier and att_rp reward
  pulling the policy in opposite directions every step.
- Next training run: validate att_rp Episode_Reward returns to OLD trajectory
  (+1.6 at iter 800). Watch DORAEMON/success_rate growth and total reward.
- Open question for follow-up: redesign `rp_vel_settling_cost` to be settling-aware
  (gate by command-stationarity AND |attitude_err| < threshold). This would let it
  reduce SS oscillation without fighting the transition-phase reward gradient.
  Deferred until baseline is recovered.

---

## [2026-04-07] Revert Reward Linear Penalty (Dead Zone at Moderate Errors)

### Context
Analyzed the training run `2026-04-07_16-37-45` which was the first run after the
linear penalty addition (previous entry below). Results at iter ~4700:

- `mean_reward` 34 (OLD run `2026-04-06_21-24-43` at same iter: ~142)
- `att_rp` Episode_Reward ~ 0 (OLD: +2.52)
- roll/pitch err 10.6 deg (OLD: 6.55 deg)
- DORAEMON success 0.17, mode=-2 stuck

**Root cause**: The added `att_rp_lin_ratio=0.5` linear penalty interacts badly
with the already-tightened `att_rp_sigma=0.10`. At err=10 deg (=0.175 rad) with
roll-weighted err_sq_w=0.0762:
- `exp_term = exp(-0.0762/0.02) = 0.022` (kernel effectively dead 2 sigma out)
- `quad_pen = 0.833 * 0.0762 = 0.063`
- `linear_pen = 0.5 * (1.5*0.175 + 0.175) = 0.219`
- `raw = 0.022 - 0.063 - 0.219 = -0.260` (negative!)

At err > sigma the exp kernel vanishes while the linear penalty grows, so the
policy is *punished* for any error larger than ~5.7 deg. The optimal strategy
becomes "don't try to track attitude" -- the policy converges with att_rp
Episode_Reward at exactly 0 (confirmed in TB). lin_vel and yaw_vel still get
positive contribution so the episode doesn't collapse, but attitude tracking
is abandoned entirely.

The linear penalty idea was sound in principle (constant gradient at small
errors where exp+quad vanish) but the magnitude was miscalibrated for the
tightened sigma -- it overwhelms the exp kernel in the moderate-error regime
where most of early training happens. Once the policy settles into the
no-attitude-tracking local optimum it cannot escape.

**Decision**: Revert the linear penalty entirely. Keep all other changes from
the prior entry (DORAEMON `performance_lb=80`, HardDR expansion,
`rp_vel_settling` budget 0.12, `yaw_rate` threshold 0.7) so the next run
isolates the linear-penalty effect from the other tunings. The EMA-based SS
bias reward that was previously deferred will be redesigned later -- the user
wants to first run a clean baseline matching the OLD reward shape, then plan
a more optimized reward redesign separately.

### Changed
- `mdp/rewards.py`: Set `att_rp_lin_ratio`, `lin_vel_lin_ratio`, `yaw_vel_lin_ratio`
  all to 0.0 (were 0.5, 0.8, 0.8). Reward formula now matches OLD run baseline:
  `r = k * (exp(-e^2/2s^2) - q_quad*e^2)` with no linear term. Fields left in
  `ALBCRewardCfg` (not deleted) so the linear path stays trivially re-enableable
  for future experiments. Sigma (0.10) and k_lin (4.0) values unchanged --
  they already matched the OLD run per commit 89314422.

### Notes
- Verified against saved `logs/.../2026-04-06_21-24-43/params/env.yaml`: all
  reward fields now match OLD run exactly (k_att_rp=6.0, att_rp_sigma=0.10,
  att_rp_quad_ratio=0.833, att_roll_weight=1.5, k_lin=4.0, lin_vel_sigma=0.10,
  lin_vel_quad_ratio=1.0, k_yaw=3.5, yaw_vel_sigma=0.10, yaw_vel_quad_ratio=1.0).
- config.py intentionally NOT reverted -- DORAEMON `performance_lb=80`, HardDR
  expansion, `rp_vel_settling=0.12`, `yaw_rate=0.7` all retained. Next run
  measures linear-penalty removal in isolation.
- Analytical dead-zone verification (err=10 deg, sigma=0.10, lin_ratio=0.5):
  raw reward = -0.260. At err=4.6 deg (sqrt(2)-scaled from OLD's err_rp_norm=6.5):
  raw = 0.449 - 0.013 - 0.10 = +0.336 (still positive, so OLD's smaller errors
  stayed in the reward-positive regime; NEW never got there because it started
  with larger errors during the learning transient).
- Next training: user will run ~2000 iter with the reverted reward config to
  validate recovery of att_rp Episode_Reward and roll/pitch error tracking.
- Plan: after validating the reward-only revert, redesign the SS-error pressure
  term using an EMA bias signal instead of constant linear penalty (separates
  persistent bias from transient tracking error, avoiding the dead-zone
  pathology). Design session deferred until next conversation.

---

## [2026-04-07] DORAEMON Stuck Fix + HardDR Expansion + Reward Linear Penalty + Trajectory Update

### Context
Following the eval_dr_fulldof bug fixes (earlier today), analyzed `model_9999.pt`
results in detail and identified several remaining issues that motivated targeted
encoder-side improvements before proceeding to no-encoder/TDC baseline comparison:

**Issue 1 (DORAEMON mode -2 stuck)**: Step-aligned trajectory analysis of run
`2026-04-06_21-24-43` revealed phase 7 (iter 8000-9750) showed DORAEMON correctly
shrinking entropy (-18.35 -> -19.69) -- it IS auto-retreating, but slower than the
policy degradation rate. Root cause: `performance_lb=100` is too tight relative to
actual training reward distribution; success_rate plateaus around 0.4 instead of
the alpha=0.5 equilibrium target. Fix: lower lb to 80 so success_rate auto-rises
to ~0.5 and DORAEMON exits mode -2 stuck.

**Issue 2 (HardDR boundary push)**: The new dr_distributions.png plot showed
several DORAEMON-managed parameters where the learned mean +/- 2*std error bars
extended past the HardDR boundary (linear/quadratic damping, cob/cog offsets x).
Fix: conservatively widen 5 fields where DORAEMON pushed against the prior
boundary. Other fields (body_mass, volume, offsets, water_density) kept due to
PhysX added-mass/inertia ratio and buoyancy stability constraints.

**Issue 3 (SS error tolerance)**: User observation that the policy "tolerates" a
certain level of steady-state error and stops trying to reduce it further. Root
cause analysis: both exp kernel `exp(-e^2/2s^2)` and quadratic `q*e^2` have
gradients that vanish as `err -> 0`. At err=0.005 the existing reward gradient is
~0.5/rad while at err=0.05 it is ~4.5/rad -- the policy gets ~9x weaker signal at
small errors and effectively gives up below ~5% of sigma. Fix: add a linear
penalty term `-q_lin * |e|` whose gradient is constant at all error magnitudes.
Verified analytically: linear penalty contributes 50% of the gradient at err=0.005
and 33% at err=0.01, providing the missing constant SS-error pressure.

**Issue 4 (overshoot patterns)**: User observed that overshoot is *lowest* at
hard DR for all channels (att/lin_vel/yaw), and that yaw shows none-worst /
hard-best for *all* metrics. Diagnosis: encoder algorithm is correctly adapting
to DR distribution (positive sign that encoder z is functional), but policy
becomes over-conservative on nominal physics (which is OOD relative to the
DORAEMON-learned mean). Fixes: tighten `rp_vel_settling` budget (0.20 -> 0.12)
to force faster settling, and lower `yaw_rate` soft_threshold (1.0 -> 0.7) so
the policy is no longer free to swing yaw rate up to 1.0 rad/s when commands
are bounded by 0.5 rad/s. `rp_rate` threshold kept at 1.0 (user request).

**Issue 5 (eval trajectory)**: User wanted every block's first logged step to
be at zero command (to clearly visualize the policy at rest before each step
test) and the attitude block's final return to be doubled like lin_vel/yaw.
Fix: insert one zero-command segment after each warmup (3 total) and add one
more `att return (0, 0)` segment, bringing trajectory from 27 to 31 segments.

**Considered but deferred**: An EMA-based SS bias reward term (`-k * |EMA(e)|`)
to provide an explicit signal for "this error is a constant bias, not natural
oscillation". User decided to first run training with the linear penalty alone
and add the EMA term in a later iteration if SS bias persists.

### Changed
- `config.py`: `doraemon.performance_lb` 100.0 -> 80.0. Lower success threshold
  unsticks DORAEMON from mode -2 by raising the IS-estimated success_rate from
  ~0.37 to ~0.5 without changing the underlying physics distribution.
- `config.py` (HardDomainRandomizationCfg): `added_mass_scale` (0.6, 1.4) -> (0.5, 1.5),
  `linear_damping_scale` (0.5, 1.5) -> (0.4, 1.7), `quadratic_damping_scale`
  (0.5, 1.5) -> (0.4, 1.7), `inertia_scale` (0.5, 1.8) -> (0.4, 2.0),
  `payload_mass_range` (0.0, 2.0) -> (0.0, 3.0). All five fields had DORAEMON
  pushing against the prior boundary; conservative expansion gives DORAEMON
  more room without violating PhysX stability constraints.
- `config.py` (constraints): `yaw_rate` soft_threshold 1.0 -> 0.7 (cmd range
  is 0.5 rad/s, prior threshold of 1.0 left too much room for yaw overshoot);
  `rp_vel_settling` budget 0.20 -> 0.12 (40% reduction forces faster settling).
- `mdp/rewards.py`: Added `att_rp_lin_ratio=0.5`, `lin_vel_lin_ratio=0.8`,
  `yaw_vel_lin_ratio=0.8` to `ALBCRewardCfg`. Each tracking reward now subtracts
  `q_lin * |e|`: `att_rp_tracking` uses weighted L1 with `att_roll_weight`
  (consistent with the existing weighted L2 quadratic), `lin_vel_tracking` uses
  L2 norm of the 3D error, `yaw_vel_tracking` uses scalar abs.
- `mdp/rewards.py`: Module docstring rewritten to document the new exp + quad
  + linear formulation and explicitly state the SS-error-tolerance failure
  mode that motivates the linear term.
- `scripts/analysis/eval_dr_fulldof.py`: `build_step_trajectory` now inserts
  three new zero-command segments (one after each of the three warmups) and
  doubles the attitude block's final return. `TRAJECTORY_N_SEGMENTS` 27 -> 31.
  All new segment names start with the matching block prefix (`att zero ...`,
  `vxyz zero ...`, `yaw zero ...`) so `_classify_segment` automatically routes
  them to the correct block. New episode_length_s = 31*5 + 10 = 165s. Per-block
  logged time: attitude 60s (was 50s), lin_vel 55s (was 50s), yaw 25s (was 20s).

### Notes
- Reward gradient verification (analytic, q_lin=0.5 for att, 0.8 for lin/yaw):
  at err=0.005 the linear term is 50% (att) / 61% (lin/yaw) of total gradient;
  at err=0.05 it drops to 10% / 15%. Policy gets a constant ~0.5-0.8/rad signal
  at small errors where exp+quad alone gave near-zero. Total gradient at small
  errors is now 1.5-2.6x the prior value.
- All 5 HardDR-expanded fields stay within physics-stable ranges. PhysX added-mass
  ratio safety check (M_a/I < 1.0) and post-DR per-axis clamp (0.95*I) handle any
  edge-case sample at the new boundary. body_mass/volume/offsets are NOT widened
  because their ratio with each other (buoyancy balance) is more sensitive than
  individual range size.
- Next training run will accumulate 5 simultaneous changes (lb, HardDR, linear
  penalty, yaw_rate threshold, rp_vel_settling, plus the eval-only trajectory
  update). Ablation across these is not planned -- user prioritizes encoder
  improvement over isolating individual contributions before baseline comparison.
- The EMA bias reward term (`-k * |EMA(e)|`, alpha~0.02) is staged for the next
  iteration if SS bias persists. Linear penalty alone is expected to be most of
  the fix; EMA would add SS-phase-specific pressure that linear penalty cannot
  distinguish from transient pressure.
- Eval re-run not required: only training-side configs changed (lb, HardDR,
  rewards, constraints) plus an eval-only trajectory tweak. The new trajectory
  will be exercised at the next eval after the new training run completes.

---

## [2026-04-07] eval_dr_fulldof Two-Bug Fix + DORAEMON DR Visualization

### Context
Re-evaluation of `model_9999.pt` from run `2026-04-06_21-24-43` (the DORAEMON
mode=-2 stuck run) showed all 4 DR levels (none/soft/medium/hard) producing
near-identical results: att SS error 1.8-2.1 deg, lin_vel SS 0.04-0.05 m/s,
100% survival across all levels. Initial interpretation was that the policy
was extremely robust, but deeper inspection of `eval_dr_fulldof.py`
revealed two compounding bugs that made all 4 levels evaluate near-nominal
physics regardless of the requested DR scale.

**Bug 1 (`build_dr_config` fallback)**: The `full` anchor for the 100%-DR
level was `DomainRandomizationCfg()` (the narrow base class), but the actual
training environment uses `HardDomainRandomizationCfg`, which has ranges
1.5-2.67x wider on most fields. The "hard" eval level therefore reached only
~40% of the actual training-time DR width.

**Bug 2 (`load_doraemon_dr` PARAM_SPECS clamp)**: When `--doraemon-dr` was
active, `load_doraemon_dr` clamped the DORAEMON-learned `mean +/- 2*std` into
the bounds of the imported `PARAM_SPECS` constant. But that constant uses
hardcoded base-DR bounds, while the runtime DORAEMON scheduler builds its
specs from `HardDomainRandomizationCfg` via `build_param_specs(dr_cfg)`. So
DORAEMON-learned ranges were being truncated into the narrow base bounds:
e.g. `added_mass_scale` learned (0.544, 1.456) was clamped to (0.85, 1.15),
losing 80% of the learned distribution. After the fix the hard-DR widths
expanded 1.94-3.13x: `payload_mass_range` 0.47 -> 1.47 (3.13x),
`added_mass_scale` 0.30 -> 0.80 (2.67x), `inertia_scale` 0.55 -> 1.06,
`body_mass_scale` 0.20 -> 0.39.

A new `dr_distributions.png` plot visualizes the 4 DR levels per parameter
(normalized to HardDR range), with DORAEMON-learned mean +/- 2*std overlaid
as black star + error bars. This plot makes the relationship between
DORAEMON's learned distribution and the actually-applied hard DR explicit
(any clamp mismatch becomes visually obvious).

### DORAEMON Trajectory Reanalysis (run 2026-04-06_21-24-43)
Step-aligned `mode/success/kl/entropy` trajectory across 40 DORAEMON updates
revealed 7 distinct phases (not the "stuck" interpretation from earlier):

1. iter 0-250: mode -3 (SLSQP failed, gradient=0 at identity)
2. iter 500-750: mode -2 (find feasible, success 0.04 -> 0.38)
3. iter 1000-2500: mode 0 (DR too easy, entropy -34 -> -25, success ~0.97)
4. iter 3000-4750: mode 0 (DR catching up, success 0.93 -> 0.71)
5. iter 5000-6500: mode 0 (entropy frozen at -18.18, KL_step=0, optimizer
   reports zero-step -- success constraint binding)
6. iter 6750-7750: mode +1 (inverted+optimize, success 0.49 -> 0.46)
7. iter 8000-9750: mode -2 (entropy actively shrinking -18.35 -> -19.69,
   policy decay outpacing DORAEMON retreat speed)

Phase 7 entropy *decrease* of 1.34 unit shows DORAEMON IS auto-retreating
when policy can't keep up; the issue is retreat speed (~0.2 entropy units
per 250 iter) vs policy degradation speed (faster). Not a stuck bug.

### Re-evaluation Results (after both bug fixes)
| Level   | DR%  | AttSS | Settling | LinVel | YawSS  | Surv |
|---------|------|-------|----------|--------|--------|------|
| none    |   0% | 1.9d  | 0.30s    | 0.336  | 0.0746 | 100% |
| soft    |  30% | 1.8d  | 0.30s    | 0.335  | 0.0405 | 100% |
| medium  |  60% | 2.2d  | 0.38s    | 0.334  | 0.0384 | 100% |
| hard    | 100% | 2.3d  | 0.41s    | 0.336  | 0.0354 | 100% |

Even with the corrected (much wider) hard-DR anchor that now matches the
true DORAEMON-learned distribution, the policy survives 100% with att SS
error rising only 1.9 -> 2.3 deg and settling time 0.30 -> 0.41s. Yaw SS is
actually *lower* at hard (noise robustness benefit). This is strong evidence
that `model_9999.pt` is genuinely robust across the full HardDR-equivalent
physics range that DORAEMON learned.

### Added
- `eval_dr_fulldof.py`: `_TRUE_NOMINAL_PHYSICS` constant -- explicit physics-true
  nominal for the scale=0 anchor (mass/damping/volume scales = 1.0,
  offsets = 0.0, water_density = 1000.0, payload = 0.0).
- `eval_dr_fulldof.py`: `_DORAEMON_RAW` module-level dict -- stores per-field
  DORAEMON learned (mean, std) for the new visualization.
- `eval_dr_fulldof.py`: `_plot_dr_distributions()` -- horizontal bar plot,
  4 DR levels per parameter normalized to HardDR range, DORAEMON mean +/- 2*std
  overlaid as black star with error bars. Output: `dr_distributions.png`.

### Changed
- `eval_dr_fulldof.py`: `--doraemon-dr` flag now uses
  `argparse.BooleanOptionalAction` with `default=True`, so DORAEMON state is
  auto-loaded from the run dir on every eval. Use `--no-doraemon-dr` to fall
  back to `HardDomainRandomizationCfg` (the static training-time anchor).
- `eval_dr_fulldof.py`: `_make_nominal_dr()` rewritten to use
  `_TRUE_NOMINAL_PHYSICS`. Asset-specific fields (joint_stiffness/damping,
  buoy_moment_arm) still fall back to base-cfg midpoint since they have no
  obvious physics-true value.
- `eval_dr_fulldof.py`: `build_dr_config()` rewritten -- the `full` anchor is
  now `_DORAEMON_FULL_DR or HardDomainRandomizationCfg()` (was base
  `DomainRandomizationCfg()`).
- `eval_dr_fulldof.py`: `load_doraemon_dr()` returns `(cfg, raw)` tuple,
  starts from `HardDomainRandomizationCfg` so non-DORAEMON fields (joint,
  thruster) match training, uses `build_param_specs(HardDR)` to build the
  clamp bounds (was hardcoded `PARAM_SPECS`), and gracefully returns
  `(None, {})` if no DORAEMON tags found in the TB log.

### Fixed
- `eval_dr_fulldof.py`: **Bug 1** -- `build_dr_config` was using base
  `DomainRandomizationCfg` as the 100%-DR anchor instead of
  `HardDomainRandomizationCfg`, causing all 4 DR levels to evaluate near
  nominal (40% of true training DR width).
- `eval_dr_fulldof.py`: **Bug 2** -- `load_doraemon_dr` was clamping
  DORAEMON-learned `mean +/- 2*std` to the imported `PARAM_SPECS` constant
  (which has hardcoded base-DR bounds), truncating DORAEMON's learned
  distribution into the much narrower base DR range. Fix uses
  `build_param_specs(HardDomainRandomizationCfg())` so the clamp matches the
  bounds DORAEMON actually learned over.

### Notes
- The DR distribution plot visually validates the fix: black star error bars
  (unclamped DORAEMON `mean +/- 2*std`) and red hard bars (applied cfg)
  now overlap correctly. Some fields (`linear_damping_scale`,
  `cob/cog_offset_x`, `quadratic_damping_scale`) have stars whose error bars
  extend slightly past [0, 1], indicating DORAEMON tried to push past
  HardDR boundary but was clamped -- evidence that HardDR width is the
  current bottleneck for DORAEMON learning, not the algorithm itself.
- `model_9999.pt` (run 2026-04-06_21-24-43) reaches 100% survival on the
  HardDR-equivalent eval. For the planned encoder vs no-encoder vs TDC
  comparison this is the encoder baseline; the next step is to train and
  evaluate the no-encoder/TDC baselines on the same eval to establish the
  performance gap.
- Open question for next session: should `performance_lb` be lowered (100
  -> 80) to unstick DORAEMON's mode-2 retreat in future runs, and/or should
  HardDR ranges be expanded for fields where DORAEMON pushed against the
  boundary? Decisions deferred until baseline comparison is complete.

---

## [2026-04-06] DORAEMON performance_lb Reduction (200 -> 110)

### Context
Mid-training check on run `2026-04-06_13-43-49` (2142 iter) revealed DORAEMON
stuck at `mode=-2` ("kept max-success dist") for last 6 updates. Success rate
plateaued at 0.035 (vs alpha=0.5), reward plateau at 134.75 since ~45% of
training. Root cause: without command curriculum (cmd_scale fixed at 1.0 since
DORAEMON-managed scales were removed earlier today), task is too hard from
iter 0 to reach `performance_lb=200`. Reward breakdown: att_rp 2.73/6.0 (45%),
lin_vel 1.61/2.7 (60%), yaw_vel 0.93/3.5 (27% -- weakest). Tracking plateau at
roll 11.5 / pitch 12.5 deg, coupled with `rp_vel_settling` constraint at 85%
of budget (17.05/20.0).

DORAEMON mode=-2 behavior: when inverted problem finds max-success direction
but result is still below alpha, DORAEMON keeps that point and skips main
entropy optimization. Physics DR mean contracted (inertia_scale 1.15->1.10,
added_mass_std 0.072->0.054) but success never recovered because the
bottleneck is command difficulty, not physics DR.

Decision: lower `performance_lb` from 200 to 110 (current reward ~135, so lb
below current means most episodes pass -> success_rate will jump to ~60-70%
-> DORAEMON transitions to mode=0 normal -> physics DR re-expands). This
restores DORAEMON functionality at the cost of accepting current tracking
accuracy as the baseline. Tracking accuracy improvement is a separate problem
not addressed here. Command range kept unchanged (att +-30 deg, full lin_vel,
full yaw).

### Changed
- `config.py`: `doraemon.performance_lb` 200.0 -> 110.0 to unstick DORAEMON
  from mode=-2 (max-success dist) fallback. Current reward plateau ~135, so
  new lb brings success_rate from 0.035 to expected ~60-70%, enabling normal
  entropy optimization and DR re-expansion.

### Notes
- Command range intentionally kept at current (att +-30 deg, full lin_vel,
  full yaw) -- per user decision, tracking accuracy improvement is deferred
- Expected new equilibrium: success_rate ~= alpha (0.5), reward 110-130,
  physics DR wider than current (adversarial pressure restored)
- Watch for: DORAEMON/mode transitioning -2 -> 0, entropy_after actually
  moving (currently frozen at -34.55), std/* values growing back
- performance_lb history: 80 -> 200 (2026-04-04) -> 110 (today)
- If tracking accuracy improvement needed later, options: relax
  rp_vel_settling budget, add command curriculum, or Gaussian two-stage
  command sampling (discussed but deferred)

---

## [2026-04-06] Training Analysis + SS Error Tuning + eval_dr_fulldof Overhaul + DORAEMON kl_ub Fix

### Context
Analyzed 20k-iter run (`2026-04-05_01-55-41`). noise_std exploded 0.7->13.95 due to
unbounded entropy gradient in decoupled sigma optimizer. Despite noise, policy mean
was healthy: eval_dr showed SS error 2.4-5.6 deg, 100% survival. Encoder z sweep
confirmed 8/9 latent dimensions active.

SS error analysis revealed reward gradient equilibrium across all 3 channels at
similar magnitudes (~0.15-0.27 per step), preventing further improvement. Roll
SS error 2x worse than pitch (5.4° vs 0.8° at roll+15 target) due to TAM roll
actuation weakness (0.007m arm vs pitch 0.145m).

New run (`2026-04-06_03-20-52`, 2700+ iters) with max_std=2.0: noise_std stable at
0.47 (fix confirmed). However DORAEMON success_rate dropped to 0.31 and stuck --
DR expanded too aggressively (entropy -34->-19 in 1000 iters, 4 updates). Root cause:
kl_ub=1.5 (3x reference default=0.5). Our implementation updates every 250 RL iters
(~16k env steps) vs reference which trains to convergence (~100k steps) between
DORAEMON updates, making same kl_ub effectively much more aggressive. Mode=1
(inverted+optimize) contracts DR then immediately re-expands within same kl budget,
producing near-zero net contraction.

### Added
- `constraint_trpo.py`: `max_std=2.0` parameter -- upper clamp on log_std, serving
  as trust region for sigma (prevents entropy-driven noise explosion)
- `eval_dr.py`: Full-DOF task support + `--doraemon-run` CLI for DORAEMON-learned DR
- `eval_dr_fulldof.py`: Warmup segment exclusion, block-aware trajectory cropping,
  DR-separated row layout for lin_vel/yaw, `error.png`, per-channel summary plots
  (summary_att/lin_vel/yaw), per-axis lin_vel and yaw step-response metrics

### Changed
- `rewards.py`: Tightened sigmas for SS error pressure:
  att_rp_sigma 0.15->0.10, lin_vel_sigma 0.15->0.10, yaw_vel_sigma 0.17->0.10
- `rewards.py`: k_lin 2.7->4.0 (lin_vel gradient was weakest, error gap was largest)
- `rewards.py`: att_roll_weight=1.5 in err_sq (roll gets 1.5x gradient, compensating
  weak TAM actuation)
- `rsl_rl_ppo_cfg.py`: entropy_coef 0.003->0.005, kl_ub 2.0->1.5
- `config.py`: DORAEMON kl_ub 1.5->0.3 (reference-equivalent given our step_interval=250,
  ~16k env steps between updates vs reference ~100k; prevents DR outpacing policy)
- `config.py`: att_cmd_rp_range pi/4->pi/6 (+-45 deg -> +-30 deg)

### Removed
- `doraemon.py`: Removed command scale parameters (cmd_att/lin/yaw_scale) from
  DORAEMON optimization (18D->15D). DORAEMON preferentially shrank commands to boost
  success_rate (cheapest path: less movement = less error = higher return), producing
  degenerate solutions where robot barely moves. Commands are task difficulty knobs,
  not physics parameters -- fixed at scale=1.0
- `albc_env.py`: Removed per-env command scale application from DORAEMON sampling

### Notes
- Eval DR results (DORAEMON DR, none/hard): att SS 2.4/2.7 deg, lin_vel 0.164/0.163,
  yaw 0.058/0.059, rise_time 0.39/0.43s, 100% survival all levels
- noise_std history: 0.005(explosion) -> 0.003(collapse) -> 0.005+max_std=2.0(current)
- kl_ub history: 0.5->1.0->1.5(too fast)->0.3(current, reference-equivalent)
- DORAEMON mode=1 structural note: inverted problem finds feasible point then main
  optimization re-expands, matching reference behavior -- not a bug, but requires
  appropriately sized kl_ub to allow net DR contraction when needed
- kl_ub=0.3 run (8k iters): eval_dr SS error 5.7-6.5 deg, 100% survival all DR levels,
  but DORAEMON collapsed cmd_att_scale to 0.16 (mean), cmd_att_std to 0.05 before fix

---

## [2026-04-04] DORAEMON SLSQP Fix + Constraint/Reward Iterations (sessions 4-6)

### Context
DORAEMON optimizer was completely non-functional. scipy trust-constr stuck because
KL divergence has zero gradient at identity (KL(p||p)=0 -> grad=0). SLSQP handles
this via SQP linearization. Also: log-space parameterization eliminates 72 box
constraints; IS clamp tightened from exp(20) to exp(5).

Multiple constraint/reward iterations in same day:
- thruster_rate constraint added then removed (structurally incompatible with entropy)
- thruster_sat reverted to thruster_util (Average, budget=0.40)
- All tracking rewards unified to exp+quadratic kernels
- Reward weights tuned from run data (k_lin 4.0->2.7, k_yaw 2.0->3.5)

### Changed
- `doraemon.py`: trust-constr -> SLSQP, log-space parameterization, IS clamp 20->5
- `doraemon.py`: ESS min_ess_ratio 0.05->0.01 (prevents excessive reverts)
- `rewards.py`: All 3 tracking terms use exp+quadratic: `k*(exp(-e²/2σ²) - q*e²)`
  att_rp(k=6.0, σ=0.15, q=0.833), lin_vel(k=2.7, σ=0.15, q=1.0), yaw(k=3.5, σ=0.17, q=1.0)
- `config.py`: performance_lb 80->200, constraint list finalized at 10 terms (5 prob + 5 avg)

### Removed
- `thruster_rate_cost`: noise-induced da > threshold every step, barrier suppressed all output
- `body_linear_velocity_cost`: always inactive (cr=0.00)

---

## [2026-04-04] DORAEMON Tuning + PARAM_SPEC Auto-sync (sessions 1-3)

### Context
First successful DORAEMON run after bug fixes. 9/9 scheduled updates succeeded,
entropy -45.66 -> -27.33 (near uniform). Two issues: noise_std collapsed (0.70->0.15,
entropy_coef=0.001 too conservative), DORAEMON used full kl_ub every step (bottleneck).

PARAM_SPEC bounds were hardcoded copies of DomainRandomizationCfg -- DORAEMON couldn't
expand beyond default DR. Fixed with auto-sync from DR config at init time.

eval_dr_fulldof.py created for 6-DOF evaluation (14 segments: att + lin_vel + yaw).

### Changed
- `rsl_rl_ppo_cfg.py`: entropy_coef 0.001->0.003, kl_ub 0.5->1.0
- `doraemon.py`: `build_param_specs(dr_cfg)` for auto-deriving bounds from DR config

### Added
- `eval_dr_fulldof.py`: 6-DOF step trajectory, `--doraemon-dr` flag, per-channel plots
