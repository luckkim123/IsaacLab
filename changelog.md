# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## Historical changelogs

- [Full ALBC early development](docs/hero/changelog_full_albc_early.md) (2026-03-31 -- 2026-04-02): DORAEMON stabilization, wrench-space experiment, logging overhaul, code simplification
- [Constrained ALBC development](docs/hero/changelog_constrained_albc.md) (2026-03-27 -- 2026-03-31): Steps 1-8
- [Legacy development](docs/hero/changelog_legacy.md) (2026-03-05 -- 2026-03-26): Phase 1-8, hero_agent TDC/encoder
- [Encoder ablation study](docs/hero/experiments/encoder_ablation.md) (Steps 0-19)

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
