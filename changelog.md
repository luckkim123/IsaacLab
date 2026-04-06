# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## Historical changelogs

- [Full ALBC early development](docs/hero/changelog_full_albc_early.md) (2026-03-31 -- 2026-04-02): DORAEMON stabilization, wrench-space experiment, logging overhaul, code simplification
- [Constrained ALBC development](docs/hero/changelog_constrained_albc.md) (2026-03-27 -- 2026-03-31): Steps 1-8
- [Legacy development](docs/hero/changelog_legacy.md) (2026-03-05 -- 2026-03-26): Phase 1-8, hero_agent TDC/encoder
- [Encoder ablation study](docs/hero/experiments/encoder_ablation.md) (Steps 0-19)

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
