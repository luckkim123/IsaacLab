# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## Historical changelogs

- [Full ALBC early development](docs/hero/changelog_full_albc_early.md) (2026-03-31 -- 2026-04-02): DORAEMON stabilization, wrench-space experiment, logging overhaul, code simplification
- [Constrained ALBC development](docs/hero/changelog_constrained_albc.md) (2026-03-27 -- 2026-03-31): Steps 1-8
- [Legacy development](docs/hero/changelog_legacy.md) (2026-03-05 -- 2026-03-26): Phase 1-8, hero_agent TDC/encoder
- [Encoder ablation study](docs/hero/experiments/encoder_ablation.md) (Steps 0-19)

---

## [2026-04-13] ERC-TRPO Tested & Reverted + Per-Dim Min_Std

### Context
Entropy collapse investigation continued. ERC-TRPO (Neurocomputing 2024) was implemented,
tested in 3 runs, and reverted due to a fundamental incompatibility with this task.

**ERC-TRPO attempt (3 runs, all failed):**
1. Run `2026-04-13_13-53-43` (absolute H): `KL - beta*H <= delta`. Noise exploded to
   max_std=2.0 because 8D Gaussian has dimension constant ~11.35, giving 24x delta bonus.
2. Run `2026-04-13_15-57-26` (H-H_ref): `KL - beta*(H-H_ref) <= delta`. Fixed explosion
   but created hard entropy floor at `H_ref - kl_limit/beta = 8.498 - 0.75 = 7.748`.
   Line search success dropped to 0% at iter 53 and never recovered. Policy frozen.
   Root cause: effective_kl = KL + beta*(H_ref-H). When entropy drops 0.75 nats, penalty
   alone equals kl_limit (0.0075), leaving zero room for any policy step.
3. Fundamental issue: this task requires entropy drop 8.5 -> 3.1 for precise control.
   ERC-TRPO prevents entropy from dropping more than 0.75 nats. Incompatible by design.

**Baseline reward decline analysis:** Deep investigation of run `2026-04-10_17-20-03`
revealed that reward decline (234 -> 119) tracks DORAEMON DR difficulty increase
(success 0.998 -> 0.589), not arm noise floor. During high DORAEMON success (iter 532-2913),
reward slope was flat even as arm dim0 hit floor at iter 1404. First-differenced correlation
between noise change and reward change was non-significant (r=-0.018, p=0.07).

**Per-dim min_std experiment:** Despite inconclusive correlation, arm noise floor
could affect long-term DR adaptability (not detectable by iteration-level correlation).
Implemented per-dim min_std as experiment: arm(0,1)=0.10, thruster(2-7)=0.05.
TRPO gradient still pushes arm noise down at floor (76% negative steps at min_std=0.05).

### Added
- `constraint_trpo.py`: `min_std_per_dim` parameter (tuple). When provided, per-dim
  log tensor used for clamp instead of scalar. Empty tuple falls back to scalar `min_std`.
- `rsl_rl_ppo_cfg.py`: `min_std_per_dim=(0.10, 0.10, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05)`.

### Removed
- `constraint_trpo.py`: All ERC-TRPO code removed -- `entropy_beta` parameter (absorbed
  by `**_kwargs` for backward compat), `_entropy_ref`, `_entropy_beta`, combined entropy
  gradient (`g + beta*h`), modified line search acceptance (`KL - beta*(H-H_ref)`).
- `rsl_rl_ppo_cfg.py`: `entropy_beta` config field removed.

### Tested and Reverted
- **ERC-TRPO (absolute H, beta=0.01):** Noise explosion, unconstrained trust region.
- **ERC-TRPO (H-H_ref, beta=0.01):** Hard entropy floor at H=7.748, policy frozen after
  iter 53. Mathematically: any beta > 0 creates floor at `H_ref - max_kl*kl_margin/beta`.
  For precise control tasks requiring large entropy reduction, ERC-TRPO is structurally
  incompatible regardless of beta value (smaller beta just lowers the floor).

### Notes
- Entropy collapse experiment history: adaptive entropy (failed), log_std TRPO
  reintegration (failed), ERC-TRPO absolute H (failed), ERC-TRPO H-H_ref (failed),
  per-dim min_std (pending).
- Per-dim min_std arm=0.10 is 1.7x the noise at peak performance (0.058). Chosen as
  moderate value between floor (0.05) and excessive (0.15).
- The question "does entropy collapse cause reward decline?" remains open. DORAEMON
  difficulty increase is the proximate cause, but whether noise floor limits DR adaptability
  requires experimental verification.

---

## [2026-04-10] Log_std TRPO Reintegration + Entropy Collapse Investigation

### Context
Seven sessions investigating and addressing entropy collapse. Key progression:

1. **Adaptive entropy tested and failed:** SAC-style alpha decayed from 0.003 to 0.0014
   during early training (entropy above target). By the time entropy dropped below
   target, alpha was too small to push back. Structural issue: SAC assumes entropy starts
   low; our case has high initial entropy that naturally declines.

2. **HardDR expansion tested and reverted:** Wider bounds degraded tracking (roll 4.59 vs
   2.80 deg at 1500 iter) without compensating benefits.

3. **Log_std TRPO reintegration (key fix):** Our implementation had log_std in a separate
   Adam optimizer instead of the TRPO natural gradient. Every reference TRPO (Spinning Up,
   ikostrikov, SB2/SB3, SafePO, rllab) includes log_std in the natural gradient. With
   log_std outside trust region, KL constraint cannot protect against variance collapse.

4. **Sigma gradient analysis:** Confirmed sigma_step_mean negative in 70/70 iters. Arm
   dims (0-1) drive collapse: 0-1% positive steps, 4-5x larger magnitude than thruster.
   Thruster dims (2-7) oscillate (17-36% positive) but arm dominates aggregate.

5. **kl_ub=0.04 analysis:** Halving DORAEMON expansion rate delayed saturation but did
   NOT prevent the fundamental reward decline pattern (-1.23 from peak vs -1.53).

Run `2026-04-10_17-20-03` (10k iter with log_std in TRPO): entropy still collapsed
(-6.28), arm hit min_std by iter 2000, reward 234->119. Log_std reintegration alone
insufficient -- motivated ERC-TRPO (see 2026-04-13).

Eval of model_9999: att SS 2.5-3.0 deg (none-hard), 100% survival, encoder 9/9 dims active.

### Changed (net, surviving changes only)
- `constraint_trpo.py`: Log_std included in `_policy_params` (TRPO natural gradient).
  Removed separate `std_optimizer` (Adam), sigma update block, and adaptive entropy
  machinery. Extended gradient decomposition to 3-way (sigma/encoder/actor).
- `agents/rsl_rl_ppo_cfg.py`: Removed `std_lr`, `entropy_coef`, `entropy_adaptive`,
  `entropy_target`, `entropy_alpha_lr`. "Three groups" -> "Two groups".
  DORAEMON `kl_ub` 0.08 -> 0.04.
- `config.py`: HardDR bounds restored to OLD values (20 fields reverted).
  DORAEMON `performance_lb` 130.0 -> 90.0 (back to OLD).
- `runners/constraint_encoder_runner.py`: Replaced `Policy/entropy_alpha` with
  `GradDecomp/sigma_{vanilla,natgrad,step}_norm`. Removed adaptive entropy save/load.
  Added `NoiseStd/dim_0` through `dim_7`, `GradDecomp/sigma_step_mean`,
  `SigmaStep/dim_0` through `dim_7`.

### Added
- `eval_dr_fulldof.py`: SS jitter metric, zero-crossing count, sample trajectory overlay,
  summary plots expanded to 3x2 grids.

### Fixed
- `eval_dr_fulldof.py`: Settling time uses correct control-theory definition (permanent
  band crossing, was first crossing). Yaw error uses `|rate-target|` (was `|rate|`).

### Removed
- Separate sigma Adam optimizer and score-function gradient update
- Adaptive entropy (SAC-style): `_log_alpha`, `_alpha_optimizer`, checkpoint save/load
- `Policy/entropy_alpha` metric

### Tested and Reverted
- **Adaptive entropy (SAC-style):** Alpha decayed below fixed entropy_coef (0.0014 < 0.003)
  because entropy started above target. Structural mismatch with declining-entropy regime.
- **HardDR expansion (+30-50%):** Tracking errors genuinely worse (roll 4.59 vs 2.80 deg).
  Wider bounds require longer training, not just more DR.
- **kl_ub=0.04 alone:** Delayed saturation timeline but same decline trajectory. The
  fundamental problem is entropy collapse, not DORAEMON expansion speed.

### Notes
- Entropy literature found: EnTRPO (2021), ERC-TRPO (2024), CSAC-LB (2024).
- sigma_step_mean is always negative: TRPO natural gradient structurally reduces noise.
  Arm dims are primary driver (reward structure couples attitude to arm noise).

---

## [2026-04-09] SS Error + Settling Tuning

### Context
Deep analysis of run `2026-04-07_23-21-27` (10k iter). Cross-eval experiment (OLD model
on NEW DR) proved 70% of hard-DR attitude degradation is policy quality, not DR difficulty.
Even at none-DR: NEW 2.4 deg vs OLD 1.9 deg.

Three fixes based on code-level root cause analysis:
1. `k_att_rp` 6.0->9.0: shifts reward gradient equilibrium toward attitude
2. `rp_vel_settling_cost` redesigned: gated by `|att_err| <= 5 deg` (settling phase only).
   Old: penalized `|p|+|q|` every step (opposed attitude commands during transit).
3. DORAEMON `kl_ub` 0.15->0.08, `performance_lb` 80->90: slows DR expansion.

Cross-eval results:
| Config            | None AttSS | Hard AttSS | Hard Settling | Yaw SS  |
|-------------------|-----------|-----------|---------------|---------|
| OLD model+OLD DR  | 1.9 deg   | 2.2 deg   | 0.38s         | 0.081   |
| OLD model+NEW DR  | 1.9 deg   | 2.9 deg   | 0.39s         | 0.081   |
| NEW model+NEW DR  | 2.4 deg   | 4.5 deg   | 1.74s         | 0.010   |

### Changed
- `mdp/rewards.py`: `k_att_rp` 6.0 -> 9.0
- `mdp/constraints.py`: `rp_vel_settling_cost` gated by `|att_err| <= settling_threshold`.
  Zero during transit, active during settling. `settling_threshold=0.087 rad` (5 deg).
- `config.py`: DORAEMON `kl_ub` 0.15 -> 0.08, `performance_lb` 80.0 -> 90.0

### Notes
- 4 changes not independently ablated. Priority revert order: DORAEMON speed first,
  settling-aware second, k_att_rp last.
- yaw_rate threshold (0.7) retained from previous run (8x improvement confirmed).

---

## [2026-04-08] Full-DOF Comparison Baselines (Phases 1-3)

### Context
Three ablation baselines for component contribution analysis. All reuse `ALBCEnv`
(DR, reward, action space, DORAEMON identical to production task).

| Phase | Task                        | Removes                            |
|-------|-----------------------------|------------------------------------|
| 1     | `Isaac-FullDOF-NoEncoder-v0`| Encoder only (TRPO+IPO kept)       |
| 2     | `Isaac-FullDOF-PPO-v0`      | Encoder + IPO (plain PPO)          |
| 3     | `Isaac-FullDOF-TDC-v0`      | All RL (classical TDC + 6-DOF PD)  |

Phase 3 eval: 100% survival all DR levels, att SS 2.8-7.1 deg, lin_vel ~0.11-0.40 m/s
(P-only floor), yaw degrades 0.013->0.13 at hard DR.

### Added
- `encoder/actor_critic_asym_constrained.py`: NoEncoder policy (Phase 1)
- `constrained_full_albc_tdc/`: Phase 3 module (TDC env, thruster PD controller,
  single-step DLS IK)
- `constrained_full_albc/__init__.py`: Phase 1+2 task registration
- `agents/rsl_rl_ppo_cfg.py`: `FullDOFNoEncoderRunnerCfg`, `FullDOFPPORunnerCfg`

### Notes
- TDC IK: single-step DLS (ik_num_iterations=1) -- rate limiter caps at 0.05 rad/step,
  100-iter mode adds ~30ms CUDA overhead for negligible accuracy gain.
- Post-eval gain bump committed unvalidated: kp_lin 30->100, kp_yaw 8->25,
  kp_att 8->20, kd_att 2->5.

---

## [2026-04-07] eval_dr_fulldof Bug Fixes + Reward/Constraint Tuning Cycle

### Context
Four iterations in one day driven by two critical eval bugs and reward tuning experiments.

**eval_dr_fulldof bugs (fixed first):**
1. `build_dr_config` used base `DomainRandomizationCfg` as 100%-DR anchor instead of
   `HardDomainRandomizationCfg` -- all 4 DR levels evaluated near-nominal (~40% of true width).
2. `load_doraemon_dr` clamped DORAEMON-learned distributions to hardcoded base-DR bounds,
   truncating 60-80% of learned range.

After fix: hard-DR widths expanded 1.94-3.13x. Re-eval of `model_9999.pt` (run
`2026-04-06_21-24-43`): att SS 1.9-2.3 deg, 100% survival all DR levels. Policy is
genuinely robust across full HardDR-equivalent range.

**DORAEMON trajectory reanalysis** (7 phases): mode -3 -> -2 -> 0 (expansion) -> 0
(catching up) -> 0 (frozen, success binding) -> +1 (inverted) -> -2 (retreating).
Phase 7 entropy DECREASE (-18.35 -> -19.69) proves DORAEMON IS auto-retreating when
policy degrades; issue is retreat speed vs degradation speed.

**Reward tuning cycle (linear penalty):** Added `-q_lin * |e|` to provide constant
gradient at small SS errors. Run `2026-04-07_16-37-45` showed dead zone: at err > 5.7 deg
the linear penalty overwhelms the exp kernel (reward goes negative), and the policy
abandons attitude tracking (att_rp Episode_Reward = 0). Reverted same day.

**rp_vel_settling budget cycle:** Tightened 0.20->0.12 to force faster settling. Run
`2026-04-07_22-24-20` showed att_rp sign-flipped to negative (reward -0.855 vs OLD +1.602).
At budget=0.12, a 60-deg traverse needs ~8.7s deep in IPO binding region. Reverted to 0.20.

### Net Changes (surviving after all reverts)
- `eval_dr_fulldof.py`: Two-bug fix (DR anchor + DORAEMON clamp bounds). New
  `dr_distributions.png` visualization. `--doraemon-dr` default=True.
  `_TRUE_NOMINAL_PHYSICS` constant. `_DORAEMON_RAW` for visualization. Trajectory
  updated: 27->31 segments (zero-command segments + doubled att return).
- `config.py`: `performance_lb` 100.0 -> 80.0 (DORAEMON unstick).
  HardDR expanded: added_mass (0.6,1.4)->(0.5,1.5), linear_damping (0.5,1.5)->(0.4,1.7),
  quadratic_damping (0.5,1.5)->(0.4,1.7), inertia (0.5,1.8)->(0.4,2.0),
  payload_mass (0,2.0)->(0,3.0). yaw_rate threshold 1.0->0.7.
- `mdp/rewards.py`: `att_rp_lin_ratio`, `lin_vel_lin_ratio`, `yaw_vel_lin_ratio` fields
  added (set to 0.0 -- linear path retained for future experiments).

### Fixed
- `eval_dr_fulldof.py`: DR anchor bug (base -> HardDR) -- was evaluating at ~40% of
  true training DR width.
- `eval_dr_fulldof.py`: DORAEMON clamp bug -- was truncating learned distribution into
  narrow base-DR bounds.

### Tested and Reverted
- **Linear penalty (`lin_ratio=0.5`):** Dead zone at moderate errors. With sigma=0.10,
  at err=10 deg: exp=0.022, quad=-0.063, linear=-0.219, total=-0.260 (negative reward).
  Policy converges to "don't track attitude" local optimum.
- **rp_vel_settling budget 0.12:** Too tight for transit phase. 60-deg traverse requires
  ~8.7s in IPO binding region. Only the att_rp channel was affected (lin_vel fine),
  confirming the selective constraint-reward conflict.

### Notes
- Re-eval results after bug fix: att SS 1.9-2.3 deg, 100% survival (confirms genuine
  robustness). This is the encoder baseline for Phase 1-3 comparison.
- rp_vel_settling needs settling-aware redesign (gate by att_err proximity, not global).
  Implemented later (2026-04-09).

---

## [2026-04-06] DORAEMON performance_lb + SS Error Tuning + eval_dr_fulldof Overhaul

### Context
Two sessions. Run `2026-04-05_01-55-41` (20k iter): noise_std exploded 0.7->13.95 due to
unbounded entropy in decoupled sigma optimizer. Despite noise, eval showed SS error
2.4-5.6 deg, 100% survival, encoder 8/9 dims active. SS error analysis: reward gradient
equilibrium at ~0.15-0.27/step across all channels. Roll 2x worse than pitch (5.4 vs
0.8 deg) due to TAM roll actuation weakness (0.007m arm vs pitch 0.145m).

New run `2026-04-06_03-20-52` with max_std=2.0: noise stable at 0.47. But DORAEMON
success dropped to 0.31 -- kl_ub=1.5 was 3x reference default. Our step_interval=250
(~16k env steps) vs reference ~100k between updates, making same kl_ub 6x more aggressive.

Mid-training check (`2026-04-06_13-43-49`, 2142 iter): DORAEMON stuck at mode=-2,
success=0.035, reward plateau at 134.75. Without command curriculum, task too hard
from iter 0 to reach performance_lb=200. Lowered to 110.

### Changed
- `constraint_trpo.py`: Added `max_std=2.0` upper clamp on log_std
- `rewards.py`: Tightened sigmas: att_rp 0.15->0.10, lin_vel 0.15->0.10, yaw 0.17->0.10.
  k_lin 2.7->4.0. att_roll_weight=1.5 in err_sq (roll gets 1.5x gradient).
- `rsl_rl_ppo_cfg.py`: entropy_coef 0.003->0.005
- `config.py`: DORAEMON kl_ub 1.5->0.3. performance_lb 200->110. att_cmd_rp_range
  pi/4->pi/6 (+-45->+-30 deg).

### Added
- `eval_dr_fulldof.py`: Full-DOF eval overhaul -- warmup exclusion, block-aware
  trajectory cropping, DR-separated layout, error.png, per-channel summary plots,
  per-axis lin_vel and yaw step-response metrics, `--doraemon-run` CLI.

### Removed
- `doraemon.py`: Command scale parameters removed from DORAEMON optimization (18D->15D).
  DORAEMON shrank commands to boost success (degenerate solution). Commands fixed at
  scale=1.0.
- `albc_env.py`: Per-env command scale application from DORAEMON sampling.

### Notes
- noise_std history: explosion(0.005) -> collapse(0.003) -> max_std=2.0 cap(current)
- kl_ub history: 0.5 -> 1.0 -> 1.5(too fast) -> 0.3(reference-equivalent)
- performance_lb history: 80 -> 200 -> 110
- DORAEMON mode=1: inverted problem finds feasible then re-expands within same kl budget.
  Not a bug, but needs appropriately sized kl_ub for net contraction.

---

## [2026-04-04] DORAEMON Fixes + Constraint/Reward Finalization

### Context
Two sessions. DORAEMON optimizer was non-functional: scipy trust-constr stuck because
KL has zero gradient at identity. SLSQP handles this via SQP linearization. Log-space
parameterization eliminates 72 box constraints. First successful run: 9/9 updates
succeeded, entropy -45.66 -> -27.33.

Issues found: noise_std collapsed (0.70->0.15, entropy_coef=0.001 too conservative),
DORAEMON used full kl_ub every step. PARAM_SPEC bounds were hardcoded copies of
DomainRandomizationCfg -- DORAEMON couldn't expand beyond default DR.

Multiple constraint/reward iterations: thruster_rate added then removed (incompatible
with entropy), thruster_sat reverted to thruster_util (Average, budget=0.40), all
tracking rewards unified to exp+quadratic.

### Changed
- `doraemon.py`: trust-constr -> SLSQP, log-space parameterization, IS clamp 20->5,
  ESS min_ess_ratio 0.05->0.01. `build_param_specs(dr_cfg)` for auto-deriving bounds.
- `rewards.py`: All 3 tracking terms use exp+quadratic: `k*(exp(-e^2/2s^2) - q*e^2)`.
  att_rp(k=6.0, s=0.15, q=0.833), lin_vel(k=2.7, s=0.15, q=1.0), yaw(k=3.5, s=0.17, q=1.0)
- `rsl_rl_ppo_cfg.py`: entropy_coef 0.001->0.003, kl_ub 0.5->1.0
- `config.py`: performance_lb 80->200. Constraint list finalized: 10 terms (5 prob + 5 avg).

### Added
- `eval_dr_fulldof.py`: 6-DOF step trajectory (14 segments), `--doraemon-dr` flag,
  per-channel plots.

### Removed
- `thruster_rate_cost`: noise-induced da > threshold every step, barrier suppressed output.
- `body_linear_velocity_cost`: always inactive (cr=0.00).
