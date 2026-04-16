# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## Historical changelogs

- [Full ALBC early development](docs/hero/changelog_full_albc_early.md) (2026-03-31 -- 2026-04-02): DORAEMON stabilization, wrench-space experiment, logging overhaul, code simplification
- [Constrained ALBC development](docs/hero/changelog_constrained_albc.md) (2026-03-27 -- 2026-03-31): Steps 1-8
- [Legacy development](docs/hero/changelog_legacy.md) (2026-03-05 -- 2026-03-26): Phase 1-8, hero_agent TDC/encoder
- [Encoder ablation study](docs/hero/experiments/encoder_ablation.md) (Steps 0-19)

---

## [2026-04-17] Round 5 Evaluated + Round 6 Launched: Shape Calibration

### Context
Round 4 concluded that reward-shape tuning (Tanh/Arctan) cannot solve SS error
on TAM-coupled axes. User constrained Round 5 to: (a) no new reward terms (keep
6 items), (b) two parallel experiments with single-variable control, (c) must
analyze constraints in addition to rewards, (d) 4-hour budget on GPU0+GPU1.
Targets: hard DR roll/pitch SS < 1.25, lin_vel SS < 0.04 (0.03 for none), yaw
SS < 0.02. Both R5 runs completed in this session; both failed on primary
hypothesis but revealed axis-specific opportunities that Round 6 pursues via
calibrated saturating penalties.

### Experiments
- **Constraint activity audit of Round 4 runs** (`Constraint/cost_return_*` vs
  `Constraint/d_k_*` TB tags): rp_vel_settling cost_return 6.0-6.6 vs budget
  episode-sum 20.0 = **33% utilization** on all 4 runs. Budget is slack, not
  binding. lin_vel_settling and yaw_settling coded in `constraints.py` but NOT
  registered in `_FULL_DOF_CONSTRAINT_TERMS` (default), so Round 4 ran with
  only 10 constraints. `thruster_util` is the only constraint near saturation
  (94% of budget).
- **TAM coupling quantified via Round 4 4-way comparison** (hard DR):
  L1 (far-field yaw penalty grad 0.15): yaw SS 0.019 BEST, roll SS 1.91 WORST.
  Arctan (far-field yaw penalty grad -> 0): yaw SS 0.028 WORST, roll SS 1.42
  BEST. Physical mechanism: horizontal thrusters share yaw moment and lateral
  force; strong yaw penalty -> aggressive differential horizontal thrust ->
  thrust-vector offset from CoM -> roll/pitch disturbance via TAM Mx row
  (0.007, 0.007, -0.007, -0.007, 0, 0). Data supports the coupling.
- **Tanh/Arctan coef mismatch documented**: tanh_coef=1.0 gives gradient 1.0 at
  e=0 (vs L1's 0.15) = 6.7x stronger small-error pressure. Arctan_coef=1.0
  gives 0.637 = 4.2x. Round 4 was NOT a controlled shape comparison; it mixed
  shape effect with magnitude effect. Any future tanh/arctan re-run requires
  coef calibration (~0.2) to L1's gradient at 0.
- **L1 baseline rejected for Round 5**: L1 hard DR SS is roll 1.91/pitch 1.47
  = WORST attitude SS across 4 runs (vs Control 1.68/1.38). L1's velocity SS
  wins (vy 0.043, yaw 0.019) but attitude target gap would be 35% from L1 vs
  26% from Control. Switched to Control baseline for cleaner experiment.

### Decisions
- **Chose Control as Round 5 baseline, not L1.** Rationale: (1) smaller gap to
  attitude target (Control roll 1.68 vs L1 1.91), (2) pure exp+quad reward =
  no reward engineering in baseline = constraint effect can be isolated, (3)
  consistent baseline for both GPU1 and GPU2 enables direct cross-comparison.
- **Two orthogonal constraint interventions, Control reward unchanged on both:**
  - GPU1 (`Isaac-FullDOF-R5-RpVel-v0`, run `r5_rpvel_b008`): tighten
    `rp_vel_settling.budget` 0.20 -> 0.08 to move from 33% to ~80% utilization.
    Targets attitude SS.
  - GPU2 (`Isaac-FullDOF-R5-VelSettling-v0`, run `r5_velsettling_th010`):
    activate `lin_vel_settling` + `yaw_settling` constraints (12 total).
    Targets velocity SS.
  Both use per-dim entropy (arm=0.01, thr=0.001) matching Control run
  `2026-04-14_18-55-20_perdiment_kl06`.
- **Chose settling_threshold = reward_sigma = 0.10 (not 0.04 original)** for
  lin_vel/yaw settling. Rationale: original 0.04 m/s is below Control's hard
  DR SS (0.06-0.07), creating chicken-egg problem (constraint can't activate
  until policy already meets target, but constraint is what drives policy to
  target). Matching threshold to reward sigma follows rp_vel_settling's existing
  design pattern (threshold 0.087 ~= att_sigma 0.1 rad). User initially
  challenged a different threshold rationale (I had circularly used policy's
  current SS state as basis). Corrected to principled sigma-match basis.
- **Budget 0.015 (3x original 0.005) for lin/yaw settling.** Active region is
  2.5x larger with threshold 0.10, so loosen budget proportionally. Can
  tighten in Round 6 if too slack.
- **Rejected: (a)** adding EMA bias / integral error reward (user: no new
  reward terms); **(b)** re-running Tanh/Arctan with calibrated coefs
  (diminishing returns given TAM coupling limit demonstrated in Round 4);
  **(c)** CAPS action-rate penalty (thruster_rate constraint exists but was
  disabled due to "structurally incompatible with entropy_coef>0, noise alone
  violates 5x"; user deferred this path).

### R5 Results (both failed primary, revealed mechanisms)

- **R5 GPU1 (`r5_rpvel_b008`, iter 5000)**: rp_vel_settling utilization moved
  30.8% -> 63.3% (cost_return 6.16 -> 5.07 vs d_k 20.0 -> 8.0). Mechanism bound
  as designed. But eval_dr hard DR showed roll SS 1.68 -> **1.90 (+13%)**,
  pitch 1.38 -> 1.47 (+7%), roll rise_time +29%. Training reward actually
  **improved +3.7%** (150.5 vs 145.1), with `Episode_Reward/lin_vel +27%`.
  Root cause: rp_vel_settling mechanism = "suppress |p|+|q| when near target"
  -> **over-damped control**. Policy reaches ~5 deg then stops, blocking the
  last residual correction. This is a structural mismatch with SS reduction
  (confirmed by rise_time +29% and roll σ ratio 0.94 showing mean shift not
  distribution widening).
  Unexpected win: **vy SS 0.059 -> 0.044 (-24%, σ-ratio 0.70)** — real robust
  improvement. Mechanism: angular stabilization frees horizontal thrust to
  allocate more precisely to linear motion (TAM indirect effect).

- **R5 GPU2 (`r5_velsettling_th010`, iter 5000)**: **Catastrophic failure**.
  Eval yaw SS 0.025 -> **0.308 rad/s (+1117%)**, identical pattern to Round 3
  Settling (0.272-0.337). `Episode_Reward/yaw_vel = -0.002` (policy abandoned
  yaw tracking). yaw US_env_mean = 40% at all DR levels (other 5 runs all 0%).
  yaw σ-ratio 5.78x (env spread massive). Round 3 perverse incentive exactly
  reproduced despite threshold 0.04 -> 0.10 (2.5x) and budget 0.005 -> 0.015
  (3x) relaxation. Policy reward calculus: reward at yaw_err=0.3 is ~0.01
  (exp-tail), yaw_settling cost is 0 when err > threshold (gate=0). Abandoning
  target gives 0 cost vs tracking gives reward ~+1 minus significant cost ->
  policy chooses abandon. **Threshold/budget relaxation does not fix the binary
  gate perverse incentive.** lin_vel_settling utilization stayed at 5.5%
  (binding failed) so vy still improved -17% via the same TAM indirect path
  as GPU1.

- **5-way hard-DR SS comparison (Control/L1/Tanh/Arctan/R5-GPU1/R5-GPU2 six
  runs)**:
  | axis | Control | L1 | Tanh | Arctan | R5-GPU1 | R5-GPU2 |
  |------|---------|------|------|--------|---------|---------|
  | roll | 1.68 | 1.91 | 1.53 | **1.42** | 1.90 | 2.00 |
  | vy   | 0.059 | **0.044** | 0.045 | 0.051 | 0.044 | 0.049 |
  | yaw  | 0.025 | **0.019** | **0.021** | 0.028 | 0.036 | **0.308** |
  | pitch| 1.38 | 1.47 | 1.46 | 1.44 | 1.47 | 1.57 |
  Pattern: Arctan wins roll SS (only winner), L1/Tanh win vy/yaw SS, pitch
  +5-14% across all six (structural limit at pitch SS ~= 0.24 sigma). No
  single intervention wins all axes.

- **Observation structure audit** (`mdp/observations.py:40`): `compute_policy_obs`
  returns only command (6D) + body state (9D) + arm (5D) + thruster (6D) = 26D
  proprioception. **No error, no integral error, no accumulated bias**. Policy
  cannot observe SS bias directly; must infer from command and state. Hwangbo
  2017 (quadrotor) precedent: adding integral error to obs eliminates SS
  offset. Currently unimplemented — deferred as Round 7+ candidate.

- **Reward dead-zone calibration analysis**: `r(e) = exp(-e^2/2σ^2) - q*e^2`,
  `dr/de = -e/σ^2 * exp(-e^2/2σ^2) - 2q*e`, both zero at e=0 (att_rp_lin_ratio
  currently 0). Hard-DR pitch SS 1.38° = 0.24σ; exp gradient there ~63% of
  peak. Pitch cannot be reduced via shape alone; sigma reduction or integral
  error required. Roll is similar but Arctan's smooth gradient (e=0 grad =
  2*coef/pi = 0.19 at coef=0.3) proved sufficient in Round 4 data.

### R5 Decisions
- **Settling-constraint approach declared a structural dead end**. Round 3 +
  R5 GPU1 + R5 GPU2 = three attempts, all failed. rp_vel_settling over-damps;
  lin/yaw_settling trigger perverse incentives. Binary gate `(err < thr)*|dv|`
  is not fixable via parameter tuning. Future work must abandon the settling-
  cost pattern, not iterate on it.
- **Per-env std added to `enhanced_summary.json`** (session code change):
  `os_env_std`, `us_env_std`, `rise_time_std`, `ss_error_std`, `ss_jitter_std`,
  and mirrored fields on `att_norm`. Aggregation = std-across-envs per segment,
  averaged across segments (parallel to existing mean). Backward-compatible
  (mean fields unchanged). Reading σ-ratio clarifies judgments: R5-GPU1 vy
  σ-ratio 0.70 confirms robust improvement; R5-GPU2 yaw σ-ratio 5.78 confirms
  catastrophic regression has concentrated worst-case behavior.
- **Kept rp_vel_settling in Round 6** (user decision). Its over-damping
  contribution suspected (Control's 33% utilization is non-trivial) but
  keeping it preserves single-variable control vs Control baseline. Future
  Round 7 candidate: remove rp_vel_settling once Round 6 shape is validated.

### Round 6 Launched: Axis-Specific Shape Calibration
- **Design rationale**: Round 4 Tanh/Arctan used coef=1.0, giving e=0 gradient
  1.0/0.637 — **6.7x / 4.2x stronger than L1's 0.15**, causing vy reward -40%
  and OS +40%. Round 6 recalibrates to coef=0.3 (Arctan e=0 grad=0.191, Tanh
  0.3), near L1 region. Based on Round 4 5-way evidence, applied axis-
  specifically: Arctan on attitude (because Arctan was the only roll SS
  winner) and Tanh on velocity (because Tanh/L1 were vy/yaw SS winners).
- **GPU0 (`Isaac-FullDOF-R6-AttArctan-v0`, run `r6_attarctan_c03`, wandb
  xtjmnwbk)**: `att_rp_arctan_coef=0.3`, eps=0.10 (=att_rp_sigma). New fields
  `att_rp_arctan_coef/eps` and `att_rp_tanh_coef/eps` added to `ALBCRewardCfg`
  (user-approved: saturating shape is a parameter of the existing att_rp
  penalty, not a new reward term). `att_rp_tracking` function updated to apply
  them (mirrors existing lin_vel_tracking/yaw_vel_tracking structure).
  Target: roll SS < 1.40, pitch SS < 1.30, vy/yaw SS ±5% of Control.
- **GPU1 (`Isaac-FullDOF-R6-VelTanh-v0`, run `r6_veltanh_c03`, wandb
  txroyh8u)**: `lin_vel_tanh_coef=0.3`, `yaw_vel_tanh_coef=0.3`, eps=0.10.
  Uses existing fields only. Target: vy SS < 0.045, yaw SS < 0.022, OS within
  +15-20% (vs Round 4 Tanh coef=1.0's +40%). Attitude untouched.
- **Both runs**: 10 constraints identical to Control (rp_vel_settling retained
  per user decision), per-dim entropy (arm=0.01, thr=0.001), kl_ub=0.06,
  num_envs=2048, max_iter=5000, seed=30. Clean single-variable diff each.

### Round 6 Decisions
- **Recalibrated coef=0.3 instead of re-running Round 4 coefs**. Round 4
  coef=1.0 failure was magnitude, not shape. L1's grad=0.15 worked without
  reward destruction; coef=0.3 (arctan grad 0.19, tanh grad 0.3) sits in that
  region. Rejected coef=0.2 (more conservative, too small gap from L1 to be
  informative) and coef=0.5 (closer to Round 4's failure mode).
- **Axis-specific shape (Arctan att, Tanh vel), not uniform**. Round 4 applied
  single shape to lin+yaw and skipped attitude entirely. 5-way data shows each
  axis has its own winning shape — pitch in particular has never been
  attacked via shape. Applying the 5-way-discovered optimal shape per axis
  is the natural next step.
- **Rejected alternatives**: (a) integral error in obs (Hwangbo 2017 pattern)
  — would change obs dim 26D->32D, full retrain required, 4h budget
  insufficient to characterize; deferred to Round 7 if Round 6 attitude shape
  fails to resolve pitch SS. (b) sigma reduction (0.10->0.05) — narrows reward
  valley, risks slower transit learning; also deferred. (c) CAPS action-rate
  penalty — addresses overshoot not SS, not the priority metric.

### Open Questions (updated after R5, pre-R6 results)
- Does calibrated coef=0.3 shape on attitude actually reduce roll/pitch SS,
  or does the pitch structural ceiling (pitch SS ~0.24 sigma = 63% of exp
  gradient peak) persist regardless of shape? Round 6 GPU0 result will
  answer.
- Does Tanh at coef=0.3 preserve Round 4's vy SS -22% win while avoiding the
  OS +40% penalty? Round 6 GPU1 will answer. The 1/3-magnitude hypothesis is
  testable directly against Round 4 Tanh-coef-1.0 baseline.
- rp_vel_settling's over-damping contribution remains uncharacterized. If
  Round 6 succeeds, Round 7 candidate: rp_vel_settling removed (budget 0.20
  -> null) to measure its standalone impact on SS vs OS.
- Integral error in observation: unimplemented, deferred, remains the highest-
  leverage untested intervention for pitch ceiling.
- Yaw OS ~33% in Control and most runs (R5-GPU2 is 16% only because policy
  abandons yaw entirely) remains unaddressed. Root cause is aggressive yaw
  command ±0.25 rad/s from DORAEMON; whether to tame at the DR level is still
  open.

---

## [2026-04-16] Round 4 Completed + Enhanced Per-Env OS Metric

### Context
Round 4 (Tanh/Arctan saturating penalties) completed after overnight training.
Deep analysis of eval_dr_fulldof results revealed that the stock `OS %` metric
is computed on the ensemble-averaged trajectory peak, which under-reports real
policy behavior when 64 envs' peak timings differ (envelope smoothing) and
silently drops undershoot (target-miss) cases. Recomputed per-env OS
distribution from existing NPZs reverses several prior conclusions.

### Experiments
- **Round 4 completed**: Tanh (`2026-04-16_16-32-12_exp_tanh_ss`) and Arctan
  (`2026-04-16_16-32-44_exp_arctan_ss`) both trained to iter 5000 with kl_ub=0.06,
  num_envs=2048, seed=30 (matching Round 2 PerDimEnt control). Both completed
  cleanly; training reward Tanh=134, Arctan=145 vs Control=151.
- **Deep analysis A (trajectory overlay, 4-way)**: Tanh vy +0.25 step shows
  persistent SS drift (median 0.280 vs target 0.250, +12% above target during
  SS). Tanh vy reversal (-0.25 target) undershoots (peak -0.218 vs target
  -0.250, -12.9% miss). Arctan vy trajectory matches Control.
- **Deep analysis C (reward breakdown, TB Episode_Reward/*)**: Tanh total reward
  4.256 (Control 4.854 = 88%), decomposed as lin_vel -41%, yaw_vel -27%,
  att_rp neutral. Arctan total 4.690 (97%) with only lin_vel -40% hit.
  Saturating penalty erodes exp-kernel reward magnitude without improving SS.
- **Deep analysis D (per-env distribution, hard DR vy+0.25)**: Tanh 29/64 envs
  (45%) above +20% OS vs Control 16/64 (25%). Tanh's overshoot is
  distribution-wide, not outlier-driven.
- **Deep analysis E (thruster usage, TB Action/*)**: Tanh thruster_norm 1.113
  (+2.7% vs Control), thruster_rate 1.287 (+4.0%). Arctan 1.017 (-6.2%) and
  1.146 (-7.4%) with util_margin +31% vs Control. Quantitative confirmation
  that Tanh learns an aggressive controller and Arctan learns a smooth one.
- **Deep analysis F (DORAEMON state)**: All 4 runs converged to
  DORAEMON/entropy_after = -17.808 with identical step trajectory. Tanh's low
  reward is NOT due to harder DR curriculum — DR learning is identical
  across runs, difference is entirely from penalty shape.

- **Methodology audit of eval_dr_fulldof stock metric**: Stock `OS %` uses
  `mean(actual, axis=envs)` then takes peak. Overshoot clamped to >=0, so
  undershoot (reversal lag) silently recorded as 0%. Discrepancy example:
  Tanh vy OS_hard stock=20.6% vs per-env mean=22.7% vs per-env median=22.9%.
  Per-env metric also exposes undershoot via `-(peak - target)/step_mag` when
  negative.
- **Recomputed per-env OS (hard DR) on all 4 runs** from existing NPZ files:

  | Axis  | Control | Exp-L1 | Tanh   | Arctan |
  |-------|---------|--------|--------|--------|
  | roll  | 13.7    | 13.9   | 13.6   | 17.1   |
  | pitch | 9.7     | 12.8   | 9.9    | 9.3    |
  | vx    | 20.0    | 22.5   | 19.3   | 19.9   |
  | vy    | 16.2    | 22.8   | 22.7   | 16.7   |
  | vz    | 20.0    | 18.6   | 16.6   | 17.3   |
  | yaw   | 33.4    | 41.5   | 37.8   | 42.0   |

### Decisions
- **Added `scripts/analysis/recompute_eval_summary.py`**. Reads eval_*.npz,
  produces `enhanced_summary.json` with per-axis stats: OS_env_mean, _median,
  _q90, US_env_mean (target-miss magnitude), n_gt20, n_gt40, n_us_lt_minus20.
  No changes to `eval_dr_fulldof.py` itself (backward compat preserved). Future
  eval runs can call this script as a post-processing step.
- **Retracted "Tanh failed / Arctan succeeded" framing from earlier in this
  session.** Enhanced metric shows neither reached Primary Success:
  - Tanh: vy OS +40% vs Control (real degradation) but vx/vz neutral-or-better,
    yaw neutral. Net: "vy-specific degradation", NOT across-the-board failure.
  - Arctan: roll OS 17.1% (+25% vs Control 13.7%, the worst across runs on
    roll) and yaw OS +26% vs Control. vy/vx match Control, vz better. Net:
    "roll-and-yaw degradation with vy/vz benefit". NOT "smooth winner" as
    stock `AttSS` summary suggested.
  - Exp-L1: across-the-board degradation (vy +41%, yaw +24%, pitch +32%)
    confirmed, consistent with earlier conclusion.
- **Retracted "Arctan thruster-smoother = better controller"**: thruster_norm
  -6.2% is real but does not translate to lower OS on all axes (roll 17.1% is
  the largest across runs). "Smooth" is axis-specific.
- **Confirmed: saturating penalty erodes lin_vel reward without SS benefit**.
  Tanh/Arctan both lose ~40% lin_vel episode reward (penalty cuts into exp
  kernel) yet SS error is unchanged (Tanh vy_SS=0.043 vs Control 0.055;
  within measurement noise).
- **Confirmed: TAM coupling dominates penalty shape**. vz (independent
  vertical thruster) improves on both Tanh and Arctan (-17% and -13% OS vs
  Control). vx/vy/yaw (shared horizontal thrusters) show mixed or degraded
  results. Conclusion: reward-penalty tuning is ineffective for TAM-coupled
  axes; next Round must target action-rate (CAPS) or integral-error
  observation (Hwangbo 2017) instead.

### Open Questions
- Does eval_dr_fulldof.py `compute_metrics` need to be patched in-place to
  emit enhanced metrics directly? (Current: separate script only; future
  runs require manual post-processing call.) Deferred pending decision on
  whether to also move away from mean-trajectory convention entirely or
  keep both.
- Round 5 design: Arctan coef sweep (0.3/0.5) vs CAPS action-rate penalty
  vs integral-error-in-obs. Trajectory overlay + per-env evidence suggests
  **CAPS is highest-leverage** because vy reversal lag (Tanh -12.9%) is an
  action-dynamics issue (thruster commands ramp too slowly vs target sign
  flip), not a reward-shape issue. Deferred to next session.
- Whether to treat "yaw overshoot ~40% across all runs" as a DORAEMON
  scenario artifact (yaw ±0.25 step is aggressive) or a real problem
  needing intervention. Baseline Control is 33% — not small either.

---

## [2026-04-16] Round 3: Structural Fixes for SS Error and Overshoot

### Context
After Round 2 confirmed PerDimEnt as the best entropy config, evaluation revealed
two remaining problems shared across all Round 2 runs: (1) lin_vel SS error and
overshoot even at no-DR, and (2) yaw overshoot. User questioned whether simple
reward weight tuning would suffice or if reward/constraint structure itself
needed redesign.

### Experiments
Deep analysis of reward/constraint mechanics before launching Round 3:

- **Reward gradient dead zone (mathematical proof)**: For `r(e) = exp(-e²/2σ²) - q·e²`,
  `dr/de = -e × (1/σ² × exp(-e²/2σ²) + 2q)`. At e=0 the gradient is exactly 0
  regardless of weights. At e=0.01 the gradient is 16% of peak (at e=σ=0.10).
  Weight tuning (k_lin 4→7) multiplies ALL gradients by 1.75x but preserves the
  zero-at-zero shape. Conclusion: weight tuning cannot fix SS error; structural
  change (L1 term or sigma reduction) required.
- **Constraint asymmetry discovered**: `rp_vel_settling_cost` (constraints.py:236)
  penalizes |ω_rp| when |att_err| < 5° — an explicit anti-overshoot mechanism for
  attitude. No equivalent exists for lin_vel or yaw. This directly explains why
  attitude tracking is satisfactory while lin/yaw show overshoot.
- **Smoothness penalty is structurally weak**: Round 2 data showed smoothness
  contribution = -0.090 per episode with k_s=-0.1 vs mean_reward=151.3 (0.06%).
  10x increase would still only reach 0.6%. The `da.pow(2).mean()` formula over
  normalized 8D actions produces intrinsically small values. Not a viable
  primary lever for overshoot.

Round 3 launched (both at 2026-04-16):
- **Exp-L1** (`Isaac-FullDOF-Exp-L1-v0`, GPU 0): Enable lin_vel_lin_ratio=0.15 and
  yaw_vel_lin_ratio=0.15. Run: `exp_l1_ss`. ETA 3.5h.
- **Exp-Settling** (`Isaac-FullDOF-Exp-Settling-v0`, GPU 1): Add `lin_vel_settling_cost`
  and `yaw_settling_cost` constraints (budget=0.005, settling_threshold=0.04).
  Run: `exp_settling_overshoot`. ETA 4.5h.
- Control: Round 2 PerDimEnt `2026-04-14_18-55-20_perdiment_kl06`. Both new runs
  keep kl_ub=0.06 and num_envs=2048 to enable direct single-variable comparison.

Round 3 completed; eval_dr_fulldof 3-way comparison (64 envs × 4 DR levels):
- **Exp-L1 results**: Mechanism confirmed but tradeoff adverse. Late-training SS
  error reduced on most axes (vxSS -15 to -21%, vySS -18 to -24%, yawSS -10 to
  -21%). But OVERSHOOT WORSE on all tracked axes: attOS +25-60%, vxOS +49-86%,
  vyOS +51-70%, yawOS +16-35%. Rise time 20-38% FASTER. Classic L1 pattern:
  constant far-field gradient makes controller aggressive. Training stable
  (reward 154 vs Control 148, DORAEMON success 0.845 vs 0.811).
- **Exp-Settling results**: Catastrophic yaw failure. yawSS 0.012-0.019 (Control)
  → 0.272-0.337 rad/s (**20x worse**). Yaw rise time = 0s (policy abandoned
  tracking; trajectory plot shows actual yaw_rate = -0.2 rad/s when target =
  +0.3 rad/s). Reward -31% (103 vs Control 148). Entropy collapsed 10x deeper
  (-3.08 vs Control -0.30). Apparent vy overshoot reduction (14.8→10.7%) is NOT
  a real improvement: Jitter 4x higher (0.004 vs 0.000), rise time 8-13% slower,
  ZX count increases at low DR. Pattern is "low-amplitude jittery settling" from
  reduced policy aggressiveness, not cleaner control. TAM-sharing (yaw+sway use
  same 4 horizontal thrusters) explains indirect vy effect; vz (independent
  vertical thrusters) unaffected by yaw dynamics but SS +80% worse from policy
  narrowing.
- **Settling root cause (3-fold structural failure)**:
  (1) `yaw_settling` cost exceeded budget from iter 50 (cost=1.077 vs d_k=0.5),
  barrier gradient activated before policy could learn.
  (2) threshold=0.04 rad/s too tight vs typical yaw_rate_err range 0.1-0.3 →
  near_target gate rarely 1 during normal tracking, but when it flips the
  binary discontinuity creates abrupt penalty.
  (3) Perverse incentive: Policy found local optimum at yaw_rate_err ≈ 0.32
  rad/s where exp kernel reward ~0.006 (abandoned) but yaw_settling gate = 0
  (no constraint cost). Avoiding target became easier than achieving it.
- **lin_vel_settling never bound**: cost=0.010 vs budget=0.50 (2% utilization
  throughout training) — threshold=0.04 m/s too tight for agent to reach. No
  information content from this constraint.

Round 4 launched (both at 2026-04-16, based on Round 3 diagnosis + literature):
- **Exp-Tanh** (`Isaac-FullDOF-Exp-Tanh-v0`, GPU 0): `ρ(e) = coef·eps·tanh(|e|/eps)`
  with coef=1.0, eps=σ=0.10 on lin_vel and yaw. Run: `exp_tanh_ss`. ETA ~3.3h.
- **Exp-Arctan** (`Isaac-FullDOF-Exp-Arctan-v0`, GPU 1):
  `ρ(e) = coef·eps·(2/π)·arctan(|e|/eps)` with same coef=1.0, eps=0.10.
  Safer variant (heavier tail, weaker at e=0). Run: `exp_arctan_ss`. ETA ~4.4h.
- Both num_envs=2048, max_iterations=5000, seed=30, kl_ub=0.06 (Control match).

Literature and gradient analysis supporting Round 4:
- Classical control (Slotine & Li 1991, "Applied Nonlinear Control"): tanh/arctan
  are standard SMC chatter-reducing substitutes for sign(). Finite gradient at
  e=0 with saturation for |e|≫ε.
- Hwangbo et al. 2019 (Science Robotics, ANYmal): logistic kernel with
  near-linear behavior near zero and saturation — only empirically validated
  saturating tracking kernel in RL locomotion.
- Hwangbo 2017 (quadrotor, RA-L): integral-error in observation eliminates SS
  offset without reward shape changes. Alternative approach, deferred.
- CAPS (Mysore et al. 2021, ICRA): derivative/action-rate penalties are the
  standard anti-overshoot tool in RL locomotion — NOT error-shape manipulation.
- Numerical gradient comparison (σ=0.10, exp peak = 6.065 at e=σ):
  At e=0: L1 0.150, Tanh 1.000 (with coef=1.0), Arctan 0.637.
  At e=5σ: L1 persists at 0.150 (causes cruise-phase overshoot), Tanh 0.018,
  Arctan 0.024 (both vanish correctly).
- Predicted Round 4 outcome: Tanh delivers ~50% SS reduction relative to Exp-L1
  while keeping overshoot near Control baseline (far-field force ~1% of L1's).

### Decisions
- **Adopted PerDimEnt as default** (entropy_coef_per_dim = arm=0.01, thr=0.001
  now in `RslRlConstraintTRPOAlgorithmCfg`). Round 2 evidence: reward 151.3 vs
  137.9 baseline, DORAEMON success 0.811 vs 0.775, better noise stability.
- **Rejected simple weight tuning (k_lin↑, k_yaw↑, k_s↑) alone**. Reason:
  mathematically, any smooth symmetric f(|e|) has f'(0)=0. Scaling weights does
  not create gradient at e=0 where SS error lives. The policy has nothing
  pulling it from e=0.01 to e=0. Weight tuning only matters if combined with
  structural change.
- **Chose L1 penalty (ratio=0.15) over sigma reduction for SS error**. L1 gives
  constant gradient at all error magnitudes, directly attacking the dead zone.
  Sigma reduction (σ=0.10→0.05) narrows the reward "valley", risks harder initial
  learning because transit phase gets less signal. L1 is surgical: dominates near
  zero (at e=0.01: L1 gradient 0.15 > exp gradient 1.0×k_lin) but negligible at
  moderate errors where exp+quad dominate. The 0.15 value is low enough to avoid
  the "moderate error dead zone" that caused L1 to be disabled originally.
- **Chose settling-cost constraints over error-derivative reward for overshoot**.
  The settling-cost pattern is already validated by `rp_vel_settling_cost` for
  attitude. Adding the same mechanism for lin_vel and yaw is a proven
  architectural parallel, not a novel invention. Budget=0.005 is tighter than
  att's 0.20, reflecting the smaller scale of velocity changes (m/s vs rad/s).
- **Kept kl_ub=0.06 for Round 3** instead of reverting to 0.04. Reason: direct
  comparison with Round 2 PerDimEnt control requires kl_ub match. kl_ub effect
  study is deferred; tracking precision is the current priority.
- **Deprecated Settling-constraint approach entirely**. Three compounding
  failures (barrier trigger at iter 50, threshold mismatch, perverse incentive
  local optimum) are not fixable with hyperparameter tuning alone. The
  `rp_vel_settling_cost` (attitude) works because the attitude exp kernel is
  strong enough to dominate; the same construction fails for yaw because the
  policy can "escape" constraint activation by staying far from target. Future
  anti-overshoot attempts should use smooth reward penalty (not binary-gated
  hard constraint) or operate via action-rate penalty (CAPS-style).
- **Corrected prior misdiagnosis of Exp-Settling "vy success"**. Initial
  3-way table showed vy overshoot 14.8→10.7% at no-DR as a positive. Deeper
  inspection (Jitter 0.000→0.004, ZX 0.5→2.1, Rise +9.8%) reveals this is
  jittery low-amplitude settling — peaks are smaller but tracking is less
  precise overall. The "improvement" reflects a narrower action distribution
  (symptom of policy abandoning yaw tracking) not an intended settling
  mechanism. Settling approach fails on every axis.
- **Reaffirmed: entropy collapse is symptom, not cause**. The 2026-04-10
  analysis (run `2026-04-10_17-20-03`, r=-0.018 first-differenced correlation
  between entropy and reward) already established this. When recording
  Exp-Settling results, initial note attributed vy effect to entropy collapse;
  corrected to "policy narrowing → TAM-shared thrusters → indirect vy damping"
  with entropy being the narrowing's measurable symptom.
- **Chose saturating penalty (Tanh/Arctan) over 6 alternatives** from gradient
  analysis. Rejected: pseudo-Huber (grad=0 at zero), log-cosh (grad=0 at zero),
  focal |e|^α (grad→∞ at zero; TRPO instability per Engstrom et al. 2020),
  bounded-L1 min(|e|,δ) (discontinuity at e=δ=σ coincides with exp peak),
  Soft-L0 1-exp(-e²/ε²) (huge spike in stopping zone → guaranteed overshoot),
  low-L1 ratio=0.05 (1/3 effect of ratio=0.15, still has far-field force).
  Tanh+Arctan are the only candidates with BOTH (a) non-zero gradient at e=0
  and (b) decay to 0 at far-field — required by gradient-shape analysis.
- **Parameter choice coef=1.0, eps=0.10** for saturating penalties.
  Rationale: at e=0, Tanh gives 16% of exp-kernel peak gradient; this is the
  "enough to kill dead zone but not dominate reward shape" sweet spot derived
  from the gradient table. eps=σ=0.10 aligns saturation scale with where exp
  kernel is strongest — penalty becomes negligible exactly where exp takes over.
- **Launched Tanh AND Arctan together (not Tanh alone)**. Rationale: Arctan
  provides a safety-margin variant with 10.5% peak-at-zero (vs Tanh 16%). If
  Tanh creates any training instability at coef=1.0, Arctan is the fallback
  without re-launching. Time cost: zero (parallel GPUs available).

### Open Questions
- Does saturating penalty (Tanh coef=1.0, eps=0.10) achieve the gradient-analysis
  prediction of ~50% SS reduction relative to Exp-L1 while keeping overshoot
  near Control baseline? Check at iter 5000 via eval_dr_fulldof.
- Does Arctan's weaker peak-at-zero (0.637 vs Tanh 1.000) translate to less SS
  improvement but better overshoot preservation, or does the heavier 1/(1+x²)
  tail cause unexpected interference near cruise? Direct comparison in Round 4.
- If Tanh succeeds, does increasing coef from 1.0 to 2.0 continue improving SS
  linearly, or is there a transition to L1-like overshoot pattern? Deferred to
  Round 5 if Round 4 validates principle.
- Is integral-error-in-observation (Hwangbo 2017 quadrotor precedent) a viable
  complementary approach? Would require encoder input reshape and is higher
  risk than Round 4; deferred pending Round 4 results.
- kl_ub=0.04 vs 0.06 for PerDimEnt still untested. Deferred beyond Round 4.

---

## [2026-04-15] Round 2 Results: Thruster Entropy Reduction is Critical

### Context
Round 2 experiments (PerDimEnt, ArmOnly, Baseline) completed overnight with kl_ub=0.06
and num_envs=2048 at 5000 iters. Goal: isolate whether PerDimEnt's advantage comes from
arm entropy boost (arm=0.01) or thruster entropy reduction (thr=0.001).

### Experiments
- **PerDimEnt** (`2026-04-14_18-55-20_perdiment_kl06`): arm=0.01, thr=0.001.
  Reward 151.3@5000. Arm noise stable (dim0=0.157, dim1=0.244, both above floor).
  All thr dims declining (dim7=0.332). Entropy collapsed to -0.26.
  DORAEMON success 0.990->0.811 (best of 3). Smoothness -0.090, thruster -0.074.
- **ArmOnly** (`2026-04-14_18-55-29_armonly_kl06`): arm=0.01, thr=0.003.
  Reward 130.6@5000 (WORST). Arm noise stable (dim0=0.158, dim1=0.255).
  Thr dims DIVERGED: dim7=1.360, dim6=0.959, dim3=0.909. Entropy grew to 7.63.
  DORAEMON success 0.787. Smoothness -0.326, thruster -0.222 (both worst).
- **Baseline** (`2026-04-14_22-33-43_baseline_kl06`): uniform entropy=0.003.
  Auto-launched by monitor script after PerDimEnt finished (GPU0).
  Reward 137.9@5000. Arm dims at floor (dim0=0.100, dim1=0.114).
  Thr dim7=0.893 (diverging, less than ArmOnly). DORAEMON success 0.775.
  Best roll_deg (10.9) but otherwise middle-of-pack.
- **eval_dr_fulldof** (PerDimEnt vs ArmOnly, Baseline pending):
  Yaw SS error: PerDimEnt 0.019 vs ArmOnly 0.070 rad/s at hard DR (3.7x worse).
  ArmOnly zero-crossings 4.8 at medium DR (yaw oscillation from thr noise divergence).
  Attitude SS: similar (~1.5° none, ~2.4° hard). Lin_vel: similar. Survival: 100% both.

### Decisions
- **ArmOnly (arm boost without thr reduction) is counterproductive.** ArmOnly
  underperformed even Baseline: reward 130.6 vs 138.0, thruster noise diverged 2.19x
  (dim7: 0.621->1.360 vs Baseline 0.606->0.893). Arm entropy boost amplifies thruster
  divergence when thr entropy_coef is at baseline level (0.003). Mechanism: arm exploration
  pressure propagates to thruster dims through shared TRPO update.
- **PerDimEnt's entropy collapse (-0.26) is NOT a liability under DR.** Despite near-zero
  entropy, PerDimEnt maintained highest DORAEMON success (0.811 vs 0.787/0.775) and
  highest reward (151.3 vs 130.6/138.0). Low entropy = precise policy = robust to DR.
- **Preliminary decision: adopt PerDimEnt as default** (arm=0.01, thr=0.001).
  Pending Baseline eval_dr confirmation. Changes needed: add entropy_coef_per_dim to
  FullDOFTRPORunnerCfg, revert kl_ub 0.06->0.04.
- **Monitor script validated.** `monitor_and_launch_baseline.sh` (nohup + pgrep polling)
  successfully detected PerDimEnt completion at 22:33 and auto-launched Baseline on GPU0.

### Open Questions
- Baseline eval_dr results still pending (running). Will complete the 3-way comparison.
- PerDimEnt's roll_deg (14.6°) worse than Baseline (10.9°) in training metrics, but
  eval_dr attitude SS was similar (1.5° vs TBD). Needs eval_dr confirmation.
- Optimal thr entropy_coef: 0.001 confirmed better than 0.003. Is there a sweet spot
  at 0.002? Lower priority -- 0.001 works well.
- kl_ub=0.06 should revert to 0.04 after experiments. Need to confirm before any
  production runs.

---

## [2026-04-14] Per-Dim Noise Analysis + Comparison Experiment Design

### Context
Deep analysis of per-dim noise dynamics in run `2026-04-13_17-15-54` (10k iters).
Arm dims collapse to floor within 285 iters while thr6/7 diverge to 1.22/1.60.
Uniform entropy_coef=0.003 is the root cause: too weak for arm (net gradient -0.004)
but too strong for thrusters (net gradient +0.003 with no natural resistance).

### Experiments
- **Run `2026-04-13_17-15-54` per-dim noise extraction**: arm0 hits floor(0.10)
  at iter 285 (from 0.625). SigmaStep 89% negative in first 300 iters,
  cumulative=-1.95. arm1 near-floor at iter 500.
- **thr6/7 noise divergence**: thr6=1.22, thr7=1.60 at iter 10k. entropy_coef
  +0.003 gradient accumulates unopposed (natural gradient ~0 for these dims).
- **Aggregate entropy masks collapse**: entropy 3.26->5.08 appears healthy,
  but excluding thr6/7 the mean noise is 0.395 (lower than OLD09's 0.553).
- **Eval DR results**: AttSS=3.3deg, LinVel=0.337m/s, Survival=100% at hard DR.
  Performance acceptable but DORAEMON success drops 0.999->0.507 by iter 9750.

### Decisions
- **Rejected Exp-A (min_std 0.10->0.25 clamp raise)** because clamp is applied
  post-TRPO step with no FIM feedback. TRPO wastes KL budget trying to reduce
  arm noise, gets clamped back every iter. Confound: cannot distinguish "forced
  noise hurt precision" from "exploration helped."
- **Rejected accelerated DORAEMON (step_interval=50)** for comparison experiments.
  It creates a fundamentally different training regime (30 DR updates in 1500 iters
  vs 6), not just time-compression. Two variables changing simultaneously.
- **Adopted per-dim entropy_coef** as the primary experiment. arm=0.01 (3.3x
  baseline) reverses arm net gradient to +0.003; thr=0.001 (1/3 baseline) slows
  thr divergence. Works inside the surrogate loss, so FIM sees it and natural
  gradient finds a self-consistent equilibrium. Code: ~15 lines in constraint_trpo.py.
- **max_std=1.0 as secondary experiment** (config-only change, orthogonal to
  per-dim coef). Caps thr6/7 divergence.
- **3-run comparison plan**: Baseline + Exp1(per-dim coef) + Exp2(max_std=1.0),
  all at num_envs=1024, 5000 iters. Baseline+Exp1 parallel on GPU0+GPU1,
  then Exp2 sequential. Total ~4h.

### 3-Run Comparison Results (2500 iters, num_envs=1024)

Baseline (`2026-04-14_15-23-49_baseline`), PerDimEnt (`2026-04-14_15-22-17_perdiment`),
MaxStd1 (`2026-04-14_15-22-40_maxstd1`).

- **PerDimEnt is best at 2500 iters**: reward 204.7 vs 193.8 vs 187.4.
  Best att_rp (5.03), smoothness (-0.11 vs -0.24), thruster (-0.09 vs -0.17).
- **Arm noise equilibrium confirmed**: PerDimEnt arm0=0.144 (not floor), arm1=0.225.
  Slight recovery after iter 750 (0.142->0.147). Baseline/MaxStd1 both hit floor.
- **Thruster noise dropped aggressively in PerDimEnt**: all thr dims 0.25-0.37
  (vs baseline 0.54-0.82). Entropy=0.51 (very low). Primary cause: thr entropy_coef=0.001.
- **PerDimEnt weaknesses**: lin_vel worst (1.11 vs 1.30 vs 1.40). roll_deg=12.76
  (baseline=10.53). Very low entropy may limit DR adaptation.
- **MaxStd1 showed no major benefit**: dim7 still diverged to 0.856 (cap not reached).
  Reward middle of the pack.
- **DORAEMON success stayed >95% in all runs**: kl_ub=0.04 insufficient for DR stress
  in 2500 iters. Reference success drop (0.507) happened after iter 5000.
- **Smoothness reward trajectory** is the clearest differentiator: PerDimEnt improved
  continuously (-0.25->-0.11), baseline flat (-0.25->-0.24), MaxStd1 flat (-0.25->-0.23).

### Round 2 Experiment Design

- **Key question**: Is thr entropy_coef=0.001 too low? Does PerDimEnt survive harder DR?
- **DORAEMON kl_ub raised 0.04->0.06** for both runs. Projected DR entropy at 5000 iters:
  -15.5 (equivalent to reference iter 7250, success ~0.62). kl_ub=0.08 rejected as
  too aggressive (DR entropy -10.1, uncharted territory, collapse risk).
- **Run A** (PerDimEnt: arm=0.01, thr=0.001): current best config + harder DR.
- **Run B** (ArmOnly: arm=0.01, thr=0.003): arm boost only, thr = baseline uniform.
  Tests whether thr noise reduction helps or hurts under harder DR.
- Both num_envs=2048, 5000 iters, kl_ub=0.06. Single variable: thr entropy_coef.
- If A>B: thr=0.001 is optimal. If B>A: thr=0.001 too low, need more exploration.

### Open Questions
- Does PerDimEnt's low entropy (0.51) become a liability under harder DR (kl_ub=0.06)?
  Round 2 will answer.
- Is arm entropy_coef=0.01 sufficient, or would 0.02-0.03 give better equilibrium?
  arm0=0.144 is above floor but still declining.
- Optimal thr entropy_coef: 0.001 (current) vs 0.003 (baseline). Round 2 A vs B comparison.
- For future short-run experiments (<=2500 iter), DORAEMON kl_ub or performance_lb
  must be adjusted to create sufficient DR pressure.

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
  per-dim min_std (pending), entropy_coef restoration (pending).
- Per-dim min_std arm=0.10 is 1.7x the noise at peak performance (0.058). Chosen as
  moderate value between floor (0.05) and excessive (0.15).

## [2026-04-13] Entropy_coef Root Cause Analysis + Restoration

### Context
Continued analysis of per-dim min_std run (2026-04-13_16-34-46) raised critical question:
mean noise_std at iter 227 was nearly identical between new and baseline runs. Investigation
of why TRPO "always reduces noise" led to discovery of run 2026-04-09_16-41-45 where noise
clearly recovered (0.36->0.55 after iter 3758). This contradicted the claim.

### Experiments
- **Per-dim min_std run** (2026-04-13_16-34-46, 645 iter): arm_0 hit 0.10 floor by iter 250,
  arm_1 by iter 550. All thruster dims still DECREASING (slope -0.0002/iter). No dim showed
  increase. Per-dim min_std prevents floor crash but provides no upward pressure.
- **04-09 run** (2026-04-09_16-41-45, 10k iter): noise recovered 0.36->0.55 after iter 3758.
  Entropy recovered 1.08->3.21. Config: `std_lr=0.001, entropy_coef=0.003`.
- **04-10 run** (2026-04-10_17-20-03, 10k iter): noise collapsed to 0.12, entropy to -6.28.
  Config: no std_lr, no entropy_coef. Two changes at once (confounded).
- **sigma_step_mean** over full 04-10 run: 85.6% negative, 14.4% positive (not 100% negative
  as previously claimed from 70-iter subsample). TRPO CAN increase noise, but net direction
  is negative without entropy bonus.

### Decisions
- **Restored entropy_coef=0.003 to TRPO surrogate.** Evidence: 04-09 (with coef) recovered
  noise, 04-10 (without coef) collapsed. The entropy bonus adds +coef gradient to log_std,
  counteracting TRPO's natural noise reduction. This is the only mechanism that pushes noise UP.
- **Kept per-dim min_std alongside entropy_coef.** Roles are complementary: entropy_coef
  provides upward pressure, per-dim min_std provides asymmetric floor (arm sensitivity > thruster).
- **Kept log_std in TRPO natural gradient.** All standard implementations (SpinUp, SB3, rllab)
  do this. No literature supports separating. Commit c0461d8c was right to reintegrate but
  wrong to remove entropy_coef simultaneously.
- **Corrected commit c0461d8c's attribution.** That commit blamed "separate Adam" for entropy
  collapse, but the real cause was entropy_coef removal. Two variables changed simultaneously;
  the wrong one was blamed.

### Literature Findings
- **Per-dim min_std**: No academic precedent. Engineering solution specific to this project.
- **entropy_coef in TRPO**: Non-standard but backed by EnTRPO (Xu et al., 2021) and
  ERC-TRPO (Guo et al., 2024). Standard in PPO but not in TRPO.
- **Constrained TRPO + entropy**: Completely uncharted. NORBC, CPO, IPO, FOCOPS all ignore
  entropy management. This project's low-dim mixed action space (8D arm+thruster) may
  uniquely trigger entropy collapse that higher-dim locomotion tasks avoid.
- **TRPO Fisher(log_sigma) = 2I** (constant). Natural gradient = vanilla gradient / 2.
  No structural protection or attack on entropy. KL limits per-step change (~2.5%/step
  at delta=0.01) but not cumulative decline.

### Open Questions
- Will entropy_coef=0.003 + per-dim min_std + TRPO-integrated sigma reproduce the 04-09
  noise recovery? Next run will test this combination.
- Is per-dim min_std still needed if entropy_coef restores noise recovery? May be redundant
  or may serve as useful safety net for arm dims.
- Optimal entropy_coef value: 0.003 worked in 04-09 but that had separate Adam for sigma.
  Interaction with TRPO-integrated sigma is untested.

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
