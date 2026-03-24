# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [2026-03-24] Encoder learning rate + DORAEMON threshold tuning

### Context
Deep analysis of run `2026-03-24_18-23-55` (853 iter, std_lr=3e-3) showed three issues:

1. **Encoder gradient death (enc_grad=0.003)**: Despite noise_std properly decreasing
   (1.0 -> 0.33), the encoder gradient remained near zero. Initial hypothesis blamed
   softsign activation saturation (z_range [-0.98, 0.97]), but this was debunked:
   batch min/max of 4096 envs will ALWAYS approach +/-1 for any bounded activation
   (statistical inevitability, not saturation). The old successful run (2026-03-17,
   tanh) also had z_max=1.0, z_min=-0.999 yet enc_grad grew to 0.056.

2. **Real cause identified**: Comparison with old successful run revealed encoder_lr
   and num_encoder_epochs were significantly more conservative:
   - encoder_lr: 1e-3 (old) vs 3e-4 (current) -- 3.3x smaller
   - num_encoder_epochs: 5 (old) vs 3 (current) -- 1.7x fewer
   These were the likely bottleneck, not the activation function.

3. **DORAEMON threshold mismatch**: With pitch_err=12.7 deg, the success_threshold_deg=10
   caused most episodes to fail. success_rate=0.36 kept DR distribution overly conservative,
   starving the encoder of diverse training signal.

Retracted suggestions from this session:
- softsign -> tanh revert: NOT justified. Same "saturation" false positive pattern that
  led to sigmoid->tanh->softsign cycle. Batch min/max != saturation.
- max_kl 0.002 -> 0.005: NOT justified. backtracks=0 does NOT mean KL budget has room;
  TRPO step formula scales to exactly max_kl. Both old and new runs use 92-98% of budget.

### Changed
- `agents/rsl_rl_ppo_cfg.py`: `encoder_lr` 3e-4 -> 1e-3, `num_encoder_epochs` 3 -> 5
  (matched to prior successful run 2026-03-17 parameters)
- `algorithms/constraint_trpo.py`: `encoder_lr` default 3e-4 -> 1e-3,
  `num_encoder_epochs` default 1 -> 5 (sync __init__ defaults)
- `doraemon.py`: `success_threshold_deg` 10 -> 15, `success_threshold_deg_final` 10 -> 15

### Notes
- max_encoder_kl=0.003 may be too restrictive with the stronger encoder (lr=1e-3, 5 epochs).
  The old successful run had no KL gating at all. If enc_grad stays dead despite lr increase,
  check whether KL gating is reverting most encoder updates.
- noise_std behavior (monotonic decrease to min_std=0.2) confirmed as intended per
  RMA (Kumar 2021) and NORBC (Kim 2023). No fix needed.
- NORBC uses TRPO natural gradient for std; current ALBC uses separate Adam. Both are
  valid approaches per paper survey (state-independent trainable std is standard).

## [2026-03-24] Increase std_lr 1e-4 -> 3e-3 for faster sigma equilibrium

### Context
Training run `2026-03-24_18-11-41` (310 iter) after sigma decoupling showed the opposite
of the previous collapse problem: noise_std barely moved (1.0 -> 0.98 over 310 iter, 2%
reduction). The score-function gradient `((a-mu)^2 - sigma^2)/sigma^3` was present but
std_lr=1e-4 was too slow for it to take effect.

Consequences of stuck-high sigma (0.98):
1. Actions essentially random noise -> joint_torque OVER (25.48 vs dk=20), joint_vel_limit
   OVER (12.23 vs dk=10)
2. Encoder gradient death (enc_grad=0.00): with random actions, encoder z has no impact on
   advantages, so encoder receives no useful gradient
3. Reward plateau since ~iter 100 (roll=7.28, pitch=12.57 deg, no progress for 200 iter)

Score-function equilibrium is self-correcting: overshooting sigma triggers corrective
gradient in the opposite direction. This makes higher LR safe -- unlike actor/encoder where
overshooting can be catastrophic, sigma naturally oscillates toward equilibrium. Chose 3e-3
(30x increase, same magnitude as encoder_lr but 10x higher to compensate for 1 step/iter
vs encoder's 3 epochs/iter).

### Changed
- `agents/rsl_rl_ppo_cfg.py`: `std_lr` 1e-4 -> 3e-3. Updated docstring with rationale.
- `algorithms/constraint_trpo.py`: `std_lr` default 1e-4 -> 3e-3 (sync).

### Notes
- Key metrics to watch: noise_std should find equilibrium in 0.3-0.8 range within 100 iter.
  If it oscillates wildly, reduce to 1e-3. If still too slow, increase to 1e-2.
- Encoder recovery depends on sigma coming down: random actions (sigma~1) prevent encoder
  from learning useful z representations.
- Alternative considered: lowering init_std (1.0 -> 0.5). Rejected because it doesn't fix
  the fundamental LR issue -- sigma would still take too long to adjust from any starting point.

## [2026-03-24] Decouple sigma from TRPO + remove yaw_quad_damp from privileged obs

### Context
Analysis of training run `2026-03-23_22-21-42` (2500 iter) revealed two structural
problems causing a mutually-reinforcing death spiral:

1. **noise_std floor lock**: log_std was in TRPO's `_policy_params`, competing with mu
   for KL budget. In 2D action space, sigma's KL consumption was ~33% (vs ~4-8% in 12D
   locomotion papers). TRPO natural gradient preferentially reduced sigma (cheaper in KL
   units), collapsing noise_std to 0.2 floor by iter ~250 and killing exploration for the
   remaining 2250 iterations.

2. **Encoder yaw domination**: z sweep analysis showed 13/13 encoder z dimensions
   dominated by yaw_quad_damp (range 1.37-1.85), a parameter ALBC cannot act on (no yaw
   control authority). Joint stiffness, body mass, main volume all showed range < 0.03
   (effectively flat). The encoder was a 13D yaw-damping classifier, wasting all latent
   capacity on non-actionable information.

Paper analysis (NORAC, Lee2020, Hwangbo2019, Ji2022) confirmed: (a) all 4 papers include
only actionable information in privileged obs, (b) 3/4 use TRPO but with 12-16D actions
where sigma KL impact is much smaller, (c) Ji2022's choice of PPO for concurrent training
indirectly supports sigma decoupling from trust region.

### Changed
- `algorithms/constraint_trpo.py`: Moved `log_std` from `_policy_params` (TRPO natural
  gradient) to new `_std_params` group with separate Adam optimizer (lr=1e-4). Sigma now
  follows the score-function equilibrium `dlogpi/dsigma = ((a-mu)^2 - sigma^2)/sigma^3`
  without consuming KL budget. Post-TRPO baseline re-snapshot ensures IS ratio starts at
  1.0 for sigma update. Barrier term included for constraint feedback to sigma.
  `torch.autograd.grad` used to avoid wasteful actor/encoder gradient computation.
- `agents/rsl_rl_ppo_cfg.py`: Added `std_lr: float = 1e-4` config field. Changed
  `privileged_dim` 28 -> 27.
- `config.py`: `state_space` 28 -> 27. Updated docstrings (privileged 28D -> 27D).
- `mdp/observations.py`: Removed yaw quadratic damping (1D) from privileged obs.
  `compute_privileged_obs()` now returns 27D. Docstring updated.
- `encoder/actor_critic_encoder.py`: `_build_fixed_encoder_normalizer()` updated from
  28D to 27D. Removed yaw_quad_damp mean/std entries (index [26]). Dim guard 28 -> 27.
- `runners/constraint_encoder_runner.py`: Added `std_optimizer.pt` save/load in
  checkpoint methods, following same pattern as encoder_optimizer.

### Notes
- Checkpoint incompatible: both changes break compatibility. Fresh start required.
  `_handle_dim_mismatch` auto-reinitializes encoder on 27D load.
- yaw_vel constraint unaffected: cost function reads angular velocity from sim state
  directly, not from privileged obs. yaw_damping_scale DR still applied to physics model.
- DORAEMON safe: no dimension indexing in doraemon.py (operates at physics level only).
- max_std clamp intentionally omitted: current problem is floor-stuck sigma, not runaway.
- Dry run verified: 10 iter, 64 envs, headless. noise_std=1.00 stable (previously
  collapsed within first iterations). Ruff check/format clean.
- Sigma update NOT gated on ls_success: sigma should adapt independently of TRPO step.
- Key design: re-snapshot post-TRPO log_prob as sigma baseline so ratio=1.0, avoiding
  IS contamination from mu change. Gradient is vanilla PG + barrier for sigma only.

## [2026-03-23] Scale max_kl for 2D action space: 0.01 -> 0.002

### Context
After fixing entropy_coef=0 and barrier_alpha=0.02, training run 2026-03-23_22-03-54
showed a new problem: noise_std collapsed to min_std=0.2 by iter 50 (entropy=-0.38
COLLAPSED). Roll error improved to 7 deg then rebounded to 22 deg -- premature
convergence to a suboptimal policy.

Analysis of Gaussian policy score function: `dlogpi/dsigma = ((a-mu)^2 - sigma^2) / sigma^3`.
This is self-correcting (sigma finds equilibrium where advantage-weighted action
spread matches noise level). However, with 2D action space vs reference paper's 12D:
- KL per dim = max_kl/d. At max_kl=0.01: 0.005/dim (2D) vs 0.00083/dim (12D)
- sigma can change 6x faster per TRPO step in our 2D setup
- Additionally, barrier_alpha=0.02 (now 15x stronger) pushes sigma down when noisy
  actions violate constraints, adding extra downward pressure

Solution: scale max_kl proportionally to action dimensionality. 0.01 * (2/12) = 0.002.
This gives the same per-dimension KL budget as the reference paper's 12D locomotion
setup, preventing premature sigma collapse while maintaining the same per-dim dynamics.

### Changed
- `agents/rsl_rl_ppo_cfg.py`: `max_kl` 0.01 -> 0.002. Scaled for 2D action space
  to match 12D per-dim KL budget from reference paper.
- `agents/rsl_rl_ppo_cfg.py`: `max_encoder_kl` 0.016 -> 0.003. Proportionally
  scaled (max_kl * line_search_kl_margin = 0.002 * 1.5).
- `algorithms/constraint_trpo.py`: `max_kl` default 0.01 -> 0.002 (sync).

### Notes
- Constraint enforcement (barrier_alpha=0.02) confirmed working: joint_torque
  15.31 (under dk=20), joint_vel_limit 9.01 (near dk=10), yaw_vel 41.83
  (under dk=78.5). All 3 previously violated constraints now near/within budget.
- DORAEMON still in backup mode (success=0.22). Will need to rebuild curriculum
  after sigma decay stabilizes.
- mu learning will also slow down (5x) -- this is the trade-off. Monitor whether
  attitude error convergence speed is acceptable.

## [2026-03-23] Fix noise_std runaway: entropy_coef=0, barrier_alpha=0.02

### Context
Deep analysis of constrained ALBC training run 2026-03-23_21-29-11 revealed
noise_std runaway (1.0 -> 3.95 over 664 iters, trend ^^^^). Root cause analysis:

1. **entropy_coef=0.01 drives noise_std up**: Entropy bonus gradient on log_std is
   a constant -0.01 (in minimization), pushing std upward every iteration. During
   reward plateau (since iter ~200), reward advantage ~0, so no counteracting
   downward gradient on log_std. No max_std clamp exists. Result: unbounded growth.

2. **barrier_alpha=0.3 vs paper's 0.02**: Our alpha was 15x larger than NORBC
   Section IV-B-1 recommendation. With alpha=0.3, barrier_base = 0.3*d_k when
   violating, giving weak gradient 1/(50*0.3*d_k). For joint_vel_limit: gradient
   = 0.0067 (0.7% of reward). Paper's alpha=0.02 gives gradient = 0.10 (10% of
   reward). The 15x weaker barrier couldn't correct constraint violations.

Observed effects: 3/6 constraints violated (joint_torque 2.6x, joint_vel_limit
3.1x, yaw_vel 1.4x over budget). DORAEMON in permanent backup mode (success=0.28
< alpha=0.5). Roll error re-increased in Q3-Q4 as noise drowned out policy mean.

### Changed
- `agents/rsl_rl_ppo_cfg.py`: `entropy_coef` 0.01 -> 0.0. min_std=0.2 floor
  provides sufficient exploration maintenance without entropy bonus pressure.
- `agents/rsl_rl_ppo_cfg.py`: `barrier_alpha` 0.3 -> 0.02. Matches NORBC paper
  recommendation. 15x stronger barrier gradient for constraint correction.
- `algorithms/constraint_trpo.py`: `barrier_alpha` default 0.3 -> 0.02 (sync).
- Updated docstrings for barrier_t, barrier_alpha, entropy_coef with correct
  numerical examples and rationale.

### Notes
- barrier formula itself is correct (matches NORBC Section IV-B exactly):
  `d_k^i = max(d_k, J_C_k + alpha*d_k)`, `barrier = -log(margin)/t`
- yaw_vel constraint kept (NORBC stability requirement) despite ALBC having no
  yaw torque authority. Budget may need adjustment if it still diverges.
- DORAEMON state collapsed (entropy=-12.58) during this run. After fix,
  curriculum will rebuild from near-nominal DR distribution.
- Potential secondary issue: cost advantage standardization (lines 508-510)
  equalizes gradient across all 6 constraints, diluting correction signal for
  the 3 violating ones. Monitor after this fix.

## [2026-03-23] Command reward tuning: coefficient form, k_c revert, c=5/7.5

### Context
Iterative tuning of exponential command reward through multiple experiments:

1. **sigma->coefficient** (7028fe4f): Replaced exp(-e^2/sigma^2) with paper's direct
   form exp(-c*e^2). Set c_roll=1.0, c_pitch=1.5, k_c=100, smoothness=-5.0.
   Result: c=1.0 too wide (76% reward at 30 deg error), k_c=100 caused dynamics
   instability (noise_std=4.26 CEILING, act_size=1.40, jnt_vel=3.88). Advantage
   normalization in TRPO means k_c scaling doesn't improve learning signal, only
   breaks term balance.

2. **k_c revert + c=16/24**: Reverted k_c=5, smoothness=-0.5. Set c=16 (peak at
   10 deg). Result: pitch dropped to 3 deg but roll stuck at 43 deg. Root cause:
   sum structure `exp(-cr*e_r^2) + exp(-cp*e_p^2)` allows one-axis optimization --
   agent gets 0.94 from pitch alone, equalizing only gives 0.99 (5% gain, not
   worth moving the arm).

3. **c=5/7.5 (final)**: Lower c gives wider reach. At 43 deg roll, reward=0.06
   (alive). Equalizing to 10 deg/10 deg gives total 1.74 vs 1.04 (67% improvement)
   -- strong incentive to balance both axes. 1.5x pitch ratio maintained.

### Changed
- `mdp/rewards.py`: sigma parametrization removed. Direct coefficient form:
  `command_coeff_roll` (5.0), `command_coeff_pitch` (7.5). command_weight=5.0,
  smoothness_weight=-0.5 (both reverted from 100/-5.0).
- `albc_env.py`: params dict `sigma_roll/pitch` -> `coeff_roll/pitch`.
- `config.py`: Updated defaults to c_roll=5.0, c_pitch=7.5.
- `runners/constraint_encoder_runner.py`: Added `TRPO/surrogate_loss` logging
  (was computed but not logged after dedup cleanup).

### Notes
- Key lesson: k_c scaling is ineffective with advantage normalization (TRPO/PPO).
  Only changes reward term balance, not learning signal strength.
- Gaussian fundamental limit: can't have both reach (low c) and precision (high c).
  Laplacian exp(-c|e|) has both (max gradient at zero, wider reach), but keeping
  exponential per user decision. May revisit if c=5/7.5 insufficient.
- c=5 equivalent to sigma~=0.45. Gradient peak at 14 deg, 50% reward at 21 deg.

## [2026-03-23] Reward architecture: exponential command + torque penalty

### Context
Analyzed reward functions from 4 reference papers to identify gaps in current
constrained ALBC reward design. Paper 1 (base paper) uses a 3-term reward:
command tracking (exponential), joint torque penalty, and action smoothness.
Current code only had 2 terms (command + smoothness) with min_laplacian kernel.
Two changes needed: (1) switch command reward to exponential form (Gaussian kernel)
for better precision near zero, (2) add joint torque penalty for energy efficiency.

### Added
- `mdp/rewards.py`: `joint_torque_penalty()` function -- `mean(tau^2)` for ALBC joints
  using PhysX computed_torque. dt-scaled, default weight=-0.001.
- `mdp/rewards.py`: `"exponential"` command_type -- `exp(-e_r^2/sigma^2) + exp(-e_p^2/sigma^2)`,
  per-axis Gaussian sum. Bounded [0, 2], matches paper 1 Appendix A form.
- `ALBCRewardCfg`: `torque_weight` field (default -0.001)

### Changed
- `config.py`: Default reward config: command_type `"min_laplacian"` -> `"exponential"`,
  smoothness_weight `-0.1` -> `-0.5`, added `torque_weight=-0.001`
- `albc_env.py`: `_build_reward_terms()` registers torque term when weight != 0
- `mdp/__init__.py`: Added `joint_torque_penalty` to exports

### Notes
- Torque weight intentionally small (-0.001): tau^2 scale is large (~90 for 9.5Nm motor).
  Monitor Episode_Reward/torque in WandB and adjust if needed.
- Paper 1's mass scaling (k_tau * m_bar/m) not implemented -- DR body_mass variation
  is only +-10%, effect negligible. Can add if robot mass changes significantly.
- Previous laplacian/min_laplacian types preserved for ablation experiments.

## [2026-03-23] Command reward: sigma -> direct coefficient + k_c scaling

### Context
Training with per-axis sigma (sigma_roll=0.15/0.30, then 0.50/0.50) revealed two
issues: (1) sigma parametrization inherited from Laplacian kernel is poorly suited
for Gaussian -- same sigma produces much narrower effective width (exp(-9) vs exp(-3)
at 3*sigma), causing gradient dead zones at initial errors (30-45 deg). (2) k_c=5.0
too small for bounded exponential output [0,2] -- paper uses k_c=200 (20x quadratic).

Paper analysis showed the exponential form has NO sigma: `exp(-e^2) + exp(-1.5*e^2)`.
Direct coefficient in exponent. The 1.5 coefficient on the harder axis (yaw rate /
pitch) demands tighter precision, not wider reach. With large k_c, gradient is
amplified everywhere so tighter coefficient doesn't cause dead zones.

Failed experiment: sigma_roll=0.15, sigma_pitch=0.30 caused gradient imbalance --
pitch got all learning signal (sigma=0.30 alive at 30 deg), roll starved (sigma=0.15
dead at 30 deg). Roll went to 45 deg while pitch dropped to 2 deg. Root cause: small
k_c couldn't compensate for the gradient reach difference.

### Changed
- `mdp/rewards.py`: Replaced sigma parametrization with direct coefficient form.
  `command_sigma_roll/pitch` -> `command_coeff_roll` (1.0) / `command_coeff_pitch` (1.5).
  Formula: `exp(-cr*e_r^2) + exp(-cp*e_p^2)`. Matches paper exactly.
- `mdp/rewards.py`: `command_weight` 5.0 -> 100.0 (paper pattern: 20x for exponential).
- `mdp/rewards.py`: `smoothness_weight` -0.5 -> -5.0 (scaled 10x to maintain relative
  influence vs 20x command increase).
- `albc_env.py`: params dict updated: `sigma_roll/pitch` -> `coeff_roll/pitch`.
- `config.py`: Default reward config updated with all new values.

### Notes
- torque_weight kept at -0.001: command:torque ratio ~4000:1, similar to paper (~20000:1).
  Torque penalty is soft regularizer; hard limit handled by effort_limit_cost constraint.
- Laplacian/min_laplacian kernels also updated to use coefficient form for consistency.
- coeff_pitch=1.5 means tighter tracking for pitch (harder axis), following paper's
  approach of demanding MORE precision from the harder axis, not less.

## [2026-03-23] WandB logging dedup + train-analyze skill update

### Context
Changelog analysis revealed progress history, then prepared for next training run by
cleaning up WandB logging duplicates and updating the train-analyze skill to match
the current constrained ALBC architecture. Key issues: (1) barrier_penalty logged
twice (Loss/ via loss_dict and Constraint/ via runner), (2) train-analyze skill
referenced dead metrics (Lagrangian, recovery mode, EAPO, energy/smoothness rewards)
and lacked coverage of new features (DORAEMON, TRPO diagnostics, entropy bonus).

### Changed
- `algorithms/constraint_trpo.py`: Removed `barrier_penalty`, `surrogate`, `adv_raw_std`
  from update() return dict. These were logged under Loss/ by parent OnPolicyRunner,
  duplicating Constraint/barrier_penalty and TRPO/ metrics. Final Loss/ section: only
  `value_function`, `kl`, `cost_value` (3 clean items).
- `train-analyze/analyze_training.py` (skill, non-git): Major update to match current
  logging structure:
  - ANOMALY_RULES: `Loss/entropy` -> `Policy/entropy`, removed `lagrangian_penalty`,
    z saturation threshold relaxed -0.95 -> -0.98 (softsign has wider z distribution)
  - TIER 1: entropy source `Policy/entropy`, added `entropy_bonus`
  - TIER 2: Removed Mode Switching section (recovery mode deleted). Added TRPO Step
    Quality (shs, step_norm, grad_norm, backtracks, val_grad, enc_grad) and DORAEMON
    (success_rate, entropy, ess_ratio, mode + quartile trends)
  - TIER 3: Removed dead refs (cost_surrogate, z_bounds, kl_trpo, energy, smoothness,
    penalty_scale). Added DORAEMON sensitivity auto-discovery from TB tags
  - Diagnosis: Removed `_check_mode_cycling`, added `_check_doraemon_ess_low`
  - Constraint margin key: `Constraint/margin_*` -> `Constraint/barrier_margin_*`
  - Plot panels: Removed recovery overlay, added DORAEMON Curriculum + TRPO Line Search
    panels, updated Encoder Learning panel to use TRPO/encoder_grad_norm
- `train-analyze/tslib.py` (skill, non-git): Added `Policy/entropy`, `TRPO/step_norm`,
  `DORAEMON/success_rate` to KEY_METRICS. Added TRPO/DORAEMON lag pairs. Updated HMM
  binary_tags from `Constraint/mode` to `Policy/line_search_success`, `DORAEMON/reverted`.

### Removed
- `algorithms/constraint_trpo.py`: 3 redundant loss_dict entries (barrier_penalty,
  surrogate, adv_raw_std)
- `analyze_training.py`: `_check_mode_cycling()` function, Mode Switching TIER 2 section,
  Plot 1 recovery mode overlay, all Lagrangian/recovery references

### Notes
- Verified with latest run (2026-03-23_19-59-08): all tiers render correctly, DORAEMON
  and TRPO sections populate with real data
- noise_std=8.30 CEILING anomaly detected in latest run -- entropy_coef=0.01 too strong
  for new entropy bonus feature. Needs tuning before next training run

## [2026-03-23] TRPO step quality diagnostic logging

### Context
Analyzed 4 papers on learning rate strategies for constrained RL locomotion
(NORBC, ANYmal, Mini Cheetah, Teacher-Student DAgger). All papers agree:
(1) TRPO actor needs no fixed LR (trust region + natural gradient + line search
provides adaptive step size), (2) critic uses fixed LR (3e-4 to 1e-3) with no
scheduling, (3) curriculum design is more effective than LR tuning for learning
speed control. Current C-TRPO implementation matches all paper consensus -- no LR
changes needed.

However, TRPO step quality and optimizer dynamics had zero diagnostic visibility.
Cannot distinguish "TRPO step accepted but tiny" from "TRPO step accepted and
large", or diagnose encoder/value gradient health. Added 6 passive diagnostic
metrics at zero computational cost (all values already computed but discarded).

### Added
- `algorithms/constraint_trpo.py`: 6 diagnostic attributes (`_last_trpo_shs`,
  `_last_trpo_step_norm`, `_last_trpo_grad_norm`, `_last_line_search_backtracks`,
  `_last_value_grad_norm`, `_last_encoder_grad_norm`). Captures Fisher curvature,
  parameter displacement, gradient norms, and backtrack counts from existing
  computations (clip_grad_norm_ return values were previously discarded).
- `runners/constraint_encoder_runner.py`: 6 `TRPO/*` metrics logged to
  TensorBoard/WandB (shs, step_norm, grad_norm, line_search_backtracks,
  value_grad_norm, encoder_grad_norm).

### Notes
- No config changes, no behavior changes, no additional computation
- Encoder LR decay considered but rejected: KL gating (max_encoder_kl=0.016)
  already provides implicit adaptive scheduling. Need diagnostic data before
  any LR tuning.

## [2026-03-23] C-TRPO entropy bonus: prevent premature noise collapse

### Context
Analyzed 4 reference papers (NORBC, Mini Cheetah, ANYmal locomotion) on noise
policy in RL for robotics. All papers use state-independent learnable std
(nn.Parameter(log_std)) with entropy bonus in the policy loss. The current
C-TRPO implementation had NO entropy coefficient -- the surrogate was purely
reward_surr + barrier with zero upward pressure on action std.

Training run 2026-03-23_18-59-00 at 1387 iterations confirmed the issue:
noise_std hit the min_std floor (0.2) by iteration ~200 and remained there for
1200+ iterations. Despite good reward (20.56) and 100% line search success,
exploration was effectively frozen. 3 constraints (joint_torque, joint_vel_limit,
yaw_vel) exceeded budgets with barrier_penalty negative (-0.19), indicating
adaptive thresholding kept barriers inactive.

Key mathematical insight: in TRPO, entropy bonus changes the gradient direction
g but NOT the Fisher matrix F. The natural gradient F^{-1}g incorporates
entropy-increasing direction while KL trust region controls step size. This
makes entropy_coef less sensitive to tuning than in PPO.

### Added
- `algorithms/constraint_trpo.py`: `entropy_coef` parameter (default 0.0 for
  backward compatibility), `_last_entropy_bonus` monitoring attribute. Entropy
  bonus `-entropy_coef * mean_entropy` added to surrogate objective.
- `agents/rsl_rl_ppo_cfg.py`: `entropy_coef: float = 0.01` config field in
  RslRlConstraintTRPOAlgorithmCfg. Matches standard PPO default.
- `runners/constraint_encoder_runner.py`: `Policy/entropy_bonus` metric logging.

### Notes
- Rejected alternatives: adaptive entropy (SAC-style, too complex for first
  iteration), DORAEMON-linked entropy (ad-hoc coupling), min_std increase
  (no gradient signal), state-dependent noise (overkill for 2D actions).
- Entropy bonus magnitude at std=0.2: ~0.008, vs barrier ~0.2 per constraint,
  vs reward surrogate ~O(1). Constraint safety maintained.
- Existing checkpoints fully compatible (entropy_coef is config, not state_dict).

## [2026-03-23] DORAEMON IS estimator improvements: soft traversability, ESS monitoring, sensitivity

### Context
Compared NORBC paper's particle filter terrain curriculum against current DORAEMON
implementation. Key insight: NORBC uses continuous traversability Tr in [0,1] with a
desirability window [0.5, 0.9], while DORAEMON used binary success {0,1}. Binary
success causes discontinuous constraint surface for the scipy trust-constr optimizer
(10.1 deg failure = 60 deg catastrophic failure). Additionally, IS estimator quality
was never validated -- ESS could silently degrade to 1 (single episode dominates),
and there was no way to diagnose which DR parameters drive training difficulty.

Three targeted improvements applied to constrained_albc DORAEMON only (hero_agent
unchanged -- improvements to be ported after validation).

### Changed
- `constrained_albc/doraemon.py`: Added `traversability_tau_deg` (default 2.0 deg) and
  `min_ess_ratio` (default 0.05) config fields. Added `_compute_ess()` method for
  post-optimization IS quality validation. In `step()`: ESS check reverts distribution
  update if ESS < 5% of buffer size. Per-parameter Pearson sensitivity correlation
  (`sensitivity/{param_name}`) logged for all 7 DORAEMON parameters.
- `constrained_albc/albc_env.py`: Added `import math`. Replaced binary success
  `(err < threshold).float()` with soft sigmoid `sigmoid(-(err - threshold) / tau)`
  for continuous traversability [0, 1]. tau = 2 deg gives transition from ~1.0 at
  threshold-4 deg to ~0.0 at threshold+4 deg.

### Notes
- Desirability window upper bound (Tr <= 0.9) deemed unnecessary -- entropy maximization
  implicitly handles "too easy" by expanding distribution until SR drops toward alpha.
- Particle filter approach rejected for 7D (curse of dimensionality). Beta distributions
  are more sample-efficient but cannot model parameter correlations.
- Sensitivity logging enables future correlation-aware improvements if independent Beta
  proves insufficient (|rho| > 0.3 indicates a dominant parameter).

## [2026-03-23] DORAEMON DR curriculum + privileged obs expansion (23D -> 28D)

### Context
NORBC paper analysis revealed that the paper uses terrain curriculum for progressive
task difficulty, while our constrained ALBC had no equivalent mechanism. Domain
randomization parameters for "environment-like" factors (payload, hydrodynamic
properties) were applied at full range from iteration 0, potentially overwhelming
early policy learning.

Ported DORAEMON (Domain Randomization via Entropy Maximization, ICLR 2024) from
hero_agent to constrained_albc. DORAEMON uses Beta distributions over DR parameters,
maximizing entropy (distribution width) subject to policy success rate >= 50%.
This provides an adaptive curriculum: starts narrow around nominal, expands as
policy improves, contracts if success rate drops.

Additionally, gap analysis of privileged observations identified 5 DR'd parameters
not visible to the encoder. Most impactful: action latency (0-4 steps, affects 40%
of 50Hz control period) and water density (DORAEMON curriculum target that encoder
cannot adapt to if not observed).

### Added
- `constrained_albc/doraemon.py`: DORAEMON scheduler ported from hero_agent with
  7 curriculum-managed parameters (payload_mass, added_mass_scale, linear_damping_scale,
  quadratic_damping_scale, water_density, cog_offset_z, cob_offset_z). Success
  threshold fixed at 10 deg (no annealing). Ranges match current DomainRandomizationCfg.

### Changed
- `config.py`: Added `doraemon: DoraemonCfg` field (enabled by default), updated
  state_space 23 -> 28 for expanded privileged obs.
- `mdp/events.py`: Added `_sample_or_uniform()` and `_apply_xyz_offset_with_doraemon()`
  helpers. `_randomize_hydro_model()`, `randomize_hydrodynamics()`, `randomize_payload()`
  now accept optional `sampled` dict for DORAEMON override of uniform sampling.
- `albc_env.py`: Added `_init_doraemon()` (scheduler + tracking buffers), settling error
  accumulation in `_get_rewards()`, episode recording in `_log_and_reset_rewards()`,
  DORAEMON sampling in `_reset_physics()`.
- `mdp/observations.py`: Extended privileged obs from 23D to 28D with action_latency(1D),
  joint_static_friction(1D), joint_viscous_friction(1D), yaw_quadratic_damping(1D),
  water_density(1D).
- `encoder/actor_critic_encoder.py`: Updated fixed normalizer from 23D to 28D with
  analytical mean/std for all 28 dimensions.
- `runners/constraint_encoder_runner.py`: Added DORAEMON step() call in log(),
  doraemon_state.pt save/load in checkpoint methods.
- `agents/rsl_rl_ppo_cfg.py`: Updated privileged_dim 23 -> 28.

### Notes
- Non-DORAEMON DR parameters (joint gains, effort limits, friction, initial pose) remain
  fixed uniform as before. DORAEMON only manages "environment-like" parameters.
- Encoder dimension change (23->28) breaks checkpoint compatibility. New checkpoints
  will auto-reinitialize encoder via existing dim mismatch handling.
- CoG/CoB xy offsets intentionally not added to privileged obs (+-0.01m, negligible
  vs 6-10Nm effort limits). Only z-components included (dominate roll/pitch dynamics).
- Perturbation forces not in privileged obs (per-step stochastic, not episode-level).

## [2026-03-23] Constraint enforcement: Lagrangian -> Log Barrier (Modified IPO)

### Context
Training analysis at 1136 iterations showed policy plateau: reward unchanged from
iter 253 (16.55 -> 16.60), pitch plateau since ~60% (15.16 deg), noise_std at floor
(0.20) since iter 40. Encoder is healthy (z_std=0.69, no saturation), constraints
not yet violated. All Lagrangian multipliers (lambda_k) were at zero because no
constraint was violated, meaning the policy was optimizing pure reward with zero
constraint gradient -- effectively unconstrained TRPO.

Compared with NORBC paper's Modified IPO which uses log barrier: the key difference
is that the barrier provides always-on, non-zero constraint gradient proportional to
proximity to the budget boundary (coefficient = 1/(t * margin)), unlike Lagrangian
which drops to zero when constraints are satisfied. This provides richer gradient
signal to the policy, potentially helping mu improvement by giving the TRPO natural
gradient more diverse gradient directions beyond pure reward.

Design decisions:
- barrier_t=50: At the closest constraint (joint_vel_limit, margin=1.29),
  barrier gradient is ~1.5% of reward gradient O(1). Conservative start.
- barrier_alpha=0.3: Adaptive threshold expansion ensures log barrier is computable
  even during initial constraint violations (d_k^i = max(d_k, J_C_k + 0.3*d_k)).
- Cost advantage standardization removed: barrier margin requires actual cost units
  (d_k, J_C_k, cost_surrogate all in same scale). Barrier's 1/(t*margin) coefficient
  provides natural per-constraint scaling.
- Reward advantage standardization kept: TRPO step size (max_kl) requires O(1) scale.

### Changed
- `algorithms/constraint_trpo.py`: Replaced Lagrangian dual ascent (lambda_k, lambda_lr,
  lambda_max) with log barrier interior-point method. Removed `_update_lambda()`,
  `_compute_lagrangian_penalty()`, `_compute_cost_surrogates()`. Added
  `_compute_adaptive_thresholds()`. Barrier computed inside `surrogate()` closure:
  margin = (adaptive_d_k - J_C_k_old) - cost_surrs; barrier = -log(margin)/t.
  Gradient flows through autograd: theta -> ratio -> cost_surrs -> margin -> barrier.
- `algorithms/constraint_trpo.py`: Removed cost advantage standardization (lines 308-310).
  Raw cost advantages used for correct barrier margin units.
- `agents/rsl_rl_ppo_cfg.py`: Replaced `lambda_lr=0.035`, `lambda_max=0.5` with
  `barrier_t=50.0`, `barrier_alpha=0.3`. Updated docstrings.
- `runners/constraint_encoder_runner.py`: Checkpoint save/load simplified (barrier is
  stateless -- adaptive thresholds recomputed each iteration, no lambda_k to persist).
  Logging: `lambda_k` -> `barrier_margin`, `lagrangian_penalty` -> `barrier_penalty`.

### Removed
- Lagrangian multiplier state (`_lambda_k`, dual ascent update, `_lambda_lr`, `_lambda_max`)
- Cost advantage standardization (NORBC Sec IV-B) -- unnecessary with barrier's natural scaling
- `constraint_state.pt` checkpoint save/load (barrier has no persistent state)

### Notes
- Checkpoint incompatible: new algorithm has no lambda_k state. Fresh start required.
- Training run with softsign+barrier should show non-zero constraint gradient from iter 1.
- Watch barrier_margin metrics in TB to verify constraint proximity tracking.
- barrier_t and barrier_alpha may need tuning after first run results.

## [2026-03-23] Modified IPO code review: restore standardization, remove Lagrangian remnants

### Context
Systematic comparison of NORBC paper's Modified IPO algorithm (Eq 8-11, Algorithm 1)
against `constraint_trpo.py` implementation. Verified all core elements are correctly
implemented: d_k conversion, adaptive thresholding, log barrier surrogate, TRPO natural
gradient, multi-head cost critic, cost GAE.

Identified three issues in additional elements not from the paper:

1. **Cost advantage standardization incorrectly removed**: Previous session (100e7996)
   removed per-constraint standardization claiming "barrier margin requires actual cost
   units". This was wrong -- barrier_base (d_k - J_C_k) is in raw units regardless of
   advantage standardization. Standardization equalizes gradient magnitude across constraints
   so barrier 1/margin_k provides proximity-based prioritization only. Restored.

2. **B2 Value LR Gating**: Reduced value optimizer LR to 10% when line search failed.
   Comment says "prevent lambda oscillation from cost value drift" -- but log barrier has
   no lambda. This was a Lagrangian remnant. Also affected reward critic unnecessarily
   (shared optimizer). Could create negative feedback loop: LS failure -> slow value
   learning -> poor advantages -> more LS failures. Removed.

3. **Cost critic output ReLU**: Applied F.relu() to cost critic output to enforce
   non-negativity. Zero gradient region at initialization could slow early learning.
   Non-negativity is already handled by cost return clamp(min=0) on training targets.
   Removed. Decision documented in memory for potential revisiting.

### Changed
- `algorithms/constraint_trpo.py`: Restored per-constraint cost advantage standardization
  (NORBC Sec IV-B). `(A_C_k - mean_k) / (std_k + 1e-8)` per constraint k.
- `algorithms/constraint_trpo.py`: Updated module docstring and comments to reflect
  standardization restoration.
- `encoder/actor_critic_encoder_constrained.py`: Removed `F.relu()` from `evaluate_costs()`
  output. Cost critic now returns raw MLP output.

### Removed
- `algorithms/constraint_trpo.py`: B2 value LR gating (`_base_value_lr` field,
  `actor_updated` parameter, LR reduction/restoration logic). Lagrangian remnant
  inapplicable to log barrier.
- `encoder/actor_critic_encoder_constrained.py`: Unused `import torch.nn.functional as F`.

### Notes
- Cost return `clamp(min=0.0)` retained (line 504, 739): theoretically correct defense
  against GAE approximation errors producing negative returns for non-negative quantities.
- Cost critic ReLU removal is unverified experimentally. If cost critic outputs persistently
  negative values during training, consider restoring ReLU or using Softplus.
- Current training run uses old Lagrangian code. New run needed to test all changes.

## [2026-03-23] Encoder activation: tanh -> softsign (gradient vanishing fix)

### Context
Three-run comparison experiment confirmed encoder saturation as root cause of training
plateau. Slower saturation (run3, fixed norm) correlated with: faster pitch improvement
(16.4 vs 28.3 deg @iter100), more stable reward (no regression), active Lagrangian
enforcement (lambda_joint_vel=0.364 vs 0 in saturated runs), and sustained action
magnitude (1.05 vs 0.85 shrinkage in saturated run1).

tanh gradient decays exponentially: at pre-activation x=3, gradient=0.01; at x=5, ~0.
Once saturated, encoder cannot recover. All prior fixes (weight_decay, architecture
reduction, fixed normalization) slowed but did not prevent saturation.

softsign(x) = x/(1+|x|) has same range (-1, 1) but gradient decays algebraically:
1/(1+|x|)^2. At x=3, gradient=0.063 (6x better than tanh). Gradient never reaches 0,
eliminating the self-reinforcing saturation trap.

RSL-RL `resolve_nn_activation` does not support "softsign" (hardcoded dict of 8
activations). Solution: MLP `last_activation=None` + manual `F.softsign()` in `_encode()`.

Also restored weight_decay to 1e-5 (was raised to 1e-4 specifically for tanh saturation
prevention -- unnecessary with softsign) and relaxed encoder grad clip 0.5 -> 1.0
(matching value function clip; current gradients 0.003-0.022 never triggered the 0.5
clip, but softsign may produce larger gradients).

### Changed
- `encoder/actor_critic_encoder.py`: Encoder MLP `last_activation="tanh"` -> `None`.
  `_encode()` now applies `torch.nn.functional.softsign(x)` after MLP forward pass.
  Docstrings updated (tanh -> softsign throughout).
- `algorithms/constraint_trpo.py`: Encoder optimizer weight_decay 1e-4 -> 1e-5
  (restored to pre-tanh-fix level). Encoder grad clip max_norm 0.5 -> 1.0 (matching
  value function clip, relaxed for softsign's healthier gradients).

### Notes
- Checkpoint incompatible: MLP Sequential structure loses Tanh module at end. Fresh start.
- Phase 2 (adapt_tconv) compatible: z range still (-1, 1), same bounded output.
- softsign approaches boundaries slower than tanh (softsign(1.83)=0.65 vs tanh(1.83)=0.95).
  Encoder will use a wider distribution of z values rather than clustering near boundaries.
- Three prior runs this session: run1 (WD=1e-5 emp, instant saturation), run2 (WD=1e-4 emp,
  100-iter saturation), run3 (WD=1e-4 fix, 200-iter saturation but still progressing).
  All showed z_std > 0.87 by iter 200. Softsign eliminates this pattern entirely.

## [2026-03-23] P0 encoder saturation fix: weight_decay + architecture reduction

### Context
Training run `2026-03-23_12-47-29` (306 iter) showed encoder z saturation within 30 iterations:
z_range reached [-1.0, 1.0], z_std=0.97 (bimodal at boundaries), enc_grad=0.0009 (gradient
death). noise_std collapsed to min_std floor (0.2) by iter 60. Attitude errors plateaued at
roll=11.3 deg, pitch=22.4 deg. All Lagrangian lambdas = 0 (constraints not violated but
attitude tracking too poor to matter).

Root cause analysis identified cascading failure: encoder saturation -> noise collapse -> attitude
plateau. Two contributing factors in encoder:
1. weight_decay=1e-5 too weak (hero_agent fix was 1e-4, 10x stronger)
2. Encoder [256,128,64] severely over-parameterized for 23D privileged input (48,141 params,
   2,093x input dimension). ELU hidden layers allow unbounded pre-activation growth through
   3 layers, causing rapid tanh saturation.

Previous z_bounds_loss false-positive analysis (2026-03-21) was for z_std=0.71 with batch
min/max at +-0.97 (tail statistics). Current z_std=0.97 + enc_grad=0.0009 is genuine saturation,
matching the criteria defined in that same entry: "True saturation would show z_std near 1.0
(bimodal at boundaries) and enc_grad < 1e-4."

EAPO removal (2026-03-22 session) also contributed to noise collapse: previous run with EAPO
maintained noise_std=0.33, current run without EAPO collapsed to floor (0.20).

### Changed
- `algorithms/constraint_trpo.py`: Encoder optimizer weight_decay 1e-5 -> 1e-4 (10x increase,
  matching hero_agent fix from 2026-03-03). Stronger L2 regularization constrains encoder weight
  magnitudes, reducing pre-activation scale at tanh output layer.
- `agents/rsl_rl_ppo_cfg.py`: encoder_hidden_dims [256,128,64] -> [128,64] (3-layer to 2-layer,
  48,141 -> 12,173 params, 4x reduction). HORA reference uses 2-layer encoder for similar
  dimensionality. Reduces over-parameterization ratio from 2,093x to 529x.

### Notes
- Checkpoint incompatible: encoder first layer shape changes (256,23) -> (128,23). Must start fresh.
- Constraint analysis: all 6 constraints within budget (lambda=0). Constraint system working
  correctly but irrelevant while attitude tracking is this poor.
- Future consideration: constraint budget relaxation + reward sigma increase (sigma=0.15 gives
  near-zero gradient at 20+ deg error). Deferred until encoder fix is validated.
- noise_std collapse (0.2 floor) may need separate fix (min_std increase or EAPO restoration)
  if weight_decay + architecture changes don't indirectly help.

## [2026-03-23] Replace encoder EmpiricalNormalization with fixed analytical normalization

### Context
Training run `2026-03-23_13-12-59` (P0 fix applied: WD=1e-4, [128,64]) showed encoder
saturation delayed from 30 to ~100 iterations but not prevented (z_std=0.93 at iter 200).
During analysis, questioned the need for EmpiricalNormalization on the encoder input:
1. With 4096 environments, first batch already gives <2% estimation error -- empirical
   tracking adds no value over analytical computation.
2. All 23 privileged dimensions come from DR parameters with known distributions (uniform
   ranges, disk-uniform for payload xy). Mean and std are analytically computable.
3. EmpiricalNormalization adds unnecessary state (running mean/var buffers) and update()
   calls that serve no purpose when the distribution is known a priori.

Decision: replace with fixed normalization using pre-computed mean/std from DR config
parameters and nominal asset values (HeroAgentHydrodynamicsCfg, HeroAgentBuoyHydrodynamicsCfg).

### Added
- `encoder/actor_critic_encoder.py`: `_FixedNormalization` module (registered buffer mean/std,
  `forward = (x - mean) / std`). `_build_fixed_encoder_normalizer()` static method computes
  analytical mean/std for all 23 privileged dimensions from DR config distributions.
  Fallback to EmpiricalNormalization if dim != 23.

### Changed
- `encoder/actor_critic_encoder.py`: Encoder normalizer initialization now calls
  `_build_fixed_encoder_normalizer()` instead of `EmpiricalNormalization(dim)`.
  `update_normalization()` no longer updates encoder normalizer (fixed values, no update needed).

### Notes
- Analytical stats computed from: uniform U(a,b) mean=(a+b)/2, std=(b-a)/sqrt(12);
  disk-uniform (R=0.1) std=R/2=0.05; scaled values base*U(lo,hi).
- Nominal values from HeroAgentHydrodynamicsCfg (main body mass 9.18kg, volume 0.009m^3,
  etc.) and HeroAgentBuoyHydrodynamicsCfg (buoy mass 0.93kg, volume 0.00268m^3).
- This change alone does not fix encoder saturation -- the normalization was already stable
  with 4096 envs. It removes unnecessary complexity and runtime state tracking.

## [2026-03-23] Code restructuring and simplification (2 sessions)

### Context
Previous session (2026-03-22) removed EAPO, z_bounds, EMA smoothing, recovery mode, margins,
entropy_coef and related logic (~268 lines across 5 files). This session continues cleanup:
restructuring stale references, removing dead code, and running code simplifier (3 parallel
review agents: reuse, quality, efficiency). 22 issues found, 8 fixed, rest skipped with rationale
(cross-package dedup out of scope, CUDA race condition false positive, KL measurements at
different time points not redundant).

### Changed
- `__init__.py`: Removed `ConstrainedALBCEncoderEnvCfg` backward-compat alias. Gym registration
  now points directly to `ALBCEnvCfg`.
- `config.py`: Removed `ConstrainedALBCEncoderEnvCfg = ALBCEnvCfg` alias (last line). Updated
  "C-TRPO / IPO" section comment to "C-TRPO".
- `albc_env.py`: Replaced 4 "IPO" comment references with "constraint"/"C-TRPO". Added
  `_constraints_cfg` cache in `__init__` (avoids `getattr` on every `_get_rewards` call).
  Added `_perturb_cycle` property (eliminates duplicate `max(1, interval + duration)` in
  `_init_state_buffers` and `_reset_perturbation_buffers`). Pre-allocated `_env_idx` tensor
  (avoids per-step `torch.arange` GPU allocation in `_get_delayed_actions`).
- `constraints.py`: Updated `ALBCConstraintCfg` docstring "IPO pipeline" -> "C-TRPO pipeline".
- `constraint_trpo.py`: Updated docstring (removed barrier comparison text). Added `Callable`
  import. Fixed `surrogate_fn: object` -> `Callable[[], torch.Tensor]` (2 methods). Merged
  redundant `_cached_lagrangian_penalty`/`_cached_mean_entropy`/`_cached_surrogate_loss` into
  `_last_*` attributes (single canonical name). Pre-allocated `_zero_costs` buffer in
  `init_storage` (avoids per-step `torch.zeros` GPU allocation in `process_env_step`).
  Vectorized finiteness check in `_compute_cost_returns` (K separate GPU reductions -> 1).
- `constraint_encoder_runner.py`: Updated `alg._cached_mean_entropy` -> `alg._last_mean_entropy`.
- `events.py`: Inlined `_rand_uniform` into `_rand_uniform_range` (removed dead helper layer).
- `agents/rsl_rl_ppo_cfg.py`: Updated docstrings (barrier -> Lagrangian references).

### Removed
- `utils/logging.py`: Deleted `unwrap_env()` function (exported but never imported anywhere
  in constrained_albc). Updated module docstring.
- `utils/__init__.py`: Removed `unwrap_env` from imports and `__all__`.
- `config.py`: Deleted `ConstrainedALBCEncoderEnvCfg` backward-compat alias.
- `events.py`: Deleted `_rand_uniform` (only called from `_rand_uniform_range`).

### Notes
- Code reuse review found 3 identical functions between constrained_albc and hero_agent
  (flush_metrics, log_dr_metrics, log_encoder_metrics). Not extracted to shared module
  because hero_agent is out of scope per user instruction.
- Efficiency review found `euler_xyz_from_quat` called 3-4 times per step on same quaternion,
  and duplicate `_get_critic_obs` in `act()`. Both require structural refactoring -- deferred.
- `barrier_state.pt` fallback in `load()` intentionally kept for old checkpoint compatibility.
- All changes pass ruff check + ruff format (13 files).

## [2026-03-22] Remove C-TRPO recovery mode + privileged-only encoder analysis

### Context
Run `2026-03-21_23-58-32` (privileged-only encoder, 23D) analyzed at 1244 and 1930 iter.

At 1244 iter: reward=21.4, roll=8.1 deg, pitch=10.6 deg, cr_vel=4.02, cr_torque=13.17.
Compared to 19D/276D-encoder run at same iter: 3x better reward, 2x better roll, z_std=0.94
(vs 0.65), act_size=0.94 (vs 1.11). joint_vel_limit cost return 3x lower (2.49 vs 7.19 @1200).
Privileged-only encoder confirmed to help encoder quality and cost reduction.

At 1930 iter: reward dropped to 19.89, attitude error regressed (roll=9.68, pitch=11.19).
Cost returns showed "sawtooth" cycling: rise -> recovery mode triggers -> cost drops sharply ->
exits recovery -> reward optimization resumes -> cost rises again. Joint_torque triggered
recovery 179/179 times (100%). act_size shrank from 0.94 to 0.81 as the policy became
increasingly conservative through repeated recovery cycles. Surrogate turned negative (-1.7e-05).

Root cause: C-TRPO recovery mode creates a binary safe/recovery oscillation. In safe mode
the policy optimizes reward (larger actions), increasing costs. When cost exceeds budget,
recovery mode minimizes cost (smaller actions), but then exits recovery when cost drops below
threshold, and the cycle repeats. The policy can never find a stable equilibrium -- it
oscillates between "optimize reward" and "minimize cost" phases indefinitely.

Decision: Remove recovery mode entirely. Keep barrier penalty as the only constraint mechanism.
Barrier provides continuous gradient proportional to 1/margin^2 without binary switching.
Cost monitoring (cost critic, GAE, margin tracking) preserved for logging.

### Changed
- `algorithms/constraint_trpo.py`: Removed recovery mode (73 net lines deleted). Removed
  `_in_recovery` state tracking, `recovery_mask` construction, `recovery_cost` term in
  surrogate, `recovery_cost_fn` in line search (cost non-regression check), recovery
  gradient in encoder update. Surrogate now: `reward_surr + barrier_penalty + entropy_bonus`.
  Line search checks only KL + reward improvement. `_last_mode` always 0 (safe).
  `_handle_critic_dim_mismatch` renamed to `_handle_dim_mismatch` in encoder module.

### Notes
- Barrier penalty with beta=0.02 gives weak gradient at large margins (phi_pp=0.02 at m=7).
  May need to increase beta if constraints are violated without recovery mode as backstop.
- Cost critic and cost GAE computation still present (barrier needs cost advantages).
- `recovery_threshold_frac` parameter no longer used but kept in config for compatibility.
- Previous run checkpoint incompatible (encoder dim change + algorithm structure change).

## [2026-03-22] Replace quadratic barrier with Lagrangian constraint enforcement

### Context
Run `2026-03-22_01-10-42` (barrier-only, no recovery, no Lagrangian) analyzed at 1152 iter.
Confirmed barrier-only approach fails completely: all 3 active constraints OVER budget
(joint_torque=44.69 dk=20, joint_vel_limit=33.43 dk=10, yaw_vel=97.19 dk=78.5).
barrier_penalty=1e-06 (effectively zero due to safe_mask zeroing violated constraints).
eff_sat=0.31 (31% time over torque limit), act_size=1.27 (saturated). Reward=18.82
and pitch=12.85 still improving -- attitude learning works, constraints are completely ignored.

Mathematical analysis confirmed two structural flaws in the quadratic barrier:
1. `safe_mask = margins > 0` zeros out violated constraints (no recovery gradient)
2. Quadratic form: dB/dtheta = 2*S*dS/dtheta -> 0 when S=E[A_cost]~0 (calibrated critic)

Replaced with adaptive Lagrangian multipliers (OmniSafe PPO-Lag standard):
- Linear penalty: lambda_k * E[ratio * A_cost_std] (nonzero per-sample gradient)
- Dual ascent: lambda_k += lr * (J_C_k - d_k), clamped to [0, lambda_max]
- lambda_max=0.5 keeps constraint gradient <= 50% of reward gradient (reward-first)

### Changed
- `algorithms/constraint_trpo.py`: Replaced `_compute_barrier_penalty()` (quadratic,
  safe_mask, beta*phi_pp*surr^2) with `_compute_lagrangian_penalty()` (linear,
  lambda_k*surr_std). Added `_update_lambda()` dual ascent method. New init params:
  `lambda_lr=0.035`, `lambda_max=0.5`. Removed `self.beta` storage. Monitoring renamed:
  `barrier_penalty` -> `lagrangian_penalty`. `set_max_iterations` log updated.
- `agents/rsl_rl_ppo_cfg.py`: Added `lambda_lr=0.035`, `lambda_max=0.5` config fields.
  `beta` and `recovery_threshold_frac` marked deprecated.
- `runners/constraint_encoder_runner.py`: Added lambda_k to save/load state, added
  per-constraint lambda logging (`Constraint/lambda_{name}`), renamed barrier_penalty
  -> lagrangian_penalty in aggregate metrics.
- `analyze_training.py`: Added `lagrangian_penalty` as primary key with `barrier_penalty`
  fallback for backward compat in Tier 2 output.

### Fixed
- `algorithms/constraint_trpo.py`: `set_max_iterations` referenced removed `self.beta`.
- `runners/constraint_encoder_runner.py`: `save()` referenced removed `self.alg._in_recovery`.

## [2026-03-21] Encoder input: privileged-only (HORA Phase 1 style)

### Context
Run `2026-03-21_23-28-54` (23D privileged, enc_epochs=3, z_bounds=0.0) analyzed at 321 iter.
Performance: reward=21.41, roll=7.22 deg, pitch=12.55 deg. Compared to the 19D run
(`17-20-58`) at the same iteration count, the 23D run shows 3x better reward (7.19 -> 21.41)
and 2x better roll (15 -> 7 deg). However, the run plateaued at ~iter 150 (55% mark).

Key findings from analysis:
- C-TRPO barrier penalty is negligible (7.3e-06) and provides no meaningful gradient for cost
  reduction. The barrier is designed to prevent budget VIOLATION, not minimize costs. With
  margins of 3-7, phi_pp = 1/m^2 yields penalty weight ~0.02, 1000x weaker than reward gradient.
- Encoder auxiliary loss was considered but withdrawn: the encoder input already contains
  privileged directly (276D input includes 23D privileged), making reconstruction trivial.
  The fundamental issue is the actor doesn't NEED z badly enough (253D direct input).
- Root cause of weak encoder gradient: encoder input is [policy_obs(13), hist_flat(240),
  privileged(23)] = 276D. hist_flat (240D, 87% of input) dominates the first layer gradient,
  causing the encoder to respond primarily to history (redundant with actor's direct input)
  rather than privileged info (the unique contribution).

Solution: change encoder input to privileged-only (23D), matching HORA Phase 1 architecture.
This ensures z encodes ONLY DR parameters, and the actor must use z to access privileged info.

Also renamed `_handle_critic_dim_mismatch` to `_handle_dim_mismatch` and added encoder
dimension mismatch handling for backward-compatible checkpoint loading.

### Changed
- `encoder/actor_critic_encoder.py`: Encoder input changed from cat([policy_obs, hist_flat,
  privileged]) (276D) to privileged-only (23D). Affects `__init__` (encoder_input_dim),
  `_encode()` (input construction), `update_normalization()` (encoder normalizer input).
  `_handle_critic_dim_mismatch` renamed to `_handle_dim_mismatch` with added encoder prefix
  support. `load_state_dict` now also checks encoder dimension mismatch.
- `encoder/actor_critic_encoder_constrained.py`: Architecture docstring updated (encoder
  input 276D -> 23D). `_handle_critic_dim_mismatch` call updated to `_handle_dim_mismatch`.

### Notes
- Checkpoint incompatible with previous runs (encoder first layer shape changes from
  (256, 276) to (256, 23)). `_handle_dim_mismatch` auto-reinitializes on load.
- Actor/critic/cost_critic architectures unchanged -- only encoder input path modified.
- This change reduces encoder parameter count significantly (first layer: 276*256=70656 ->
  23*256=5888 params, 12x reduction).
- C-TRPO barrier and reward function (min_laplacian) kept unchanged for this experiment.
  Isolating the encoder input change to measure its impact independently.

## [2026-03-21] Expand privileged obs to 23D -- add damping + body mass + constraint cost analysis

### Context
Run `2026-03-21_21-33-37` (enc_epochs=3, grad_clip=0.5) ran to 1924 iter with no improvement
past iter ~500. Reward peaked at 21 then regressed to 19 (Q4 decline). Roll=9.37 deg,
pitch=12.07 deg. Surrogate=1.2e-05 (even smaller than 700-iter measurement of 3.4e-05).
z range saturated at [-1.0, 1.0]. Encoder gradient declining in Q4.

Systematic code review of reward and constraint implementations confirmed both are
theoretically correct: reward math verified numerically (episode reward prediction matches
observed values), constraint budget transformations correct, cost GAE correct, timing of
overshoot detection correct. Reward function (min_laplacian) was also validated -- all three
variants (sum_laplacian, min_laplacian, smooth_min_laplacian) were previously tried and
min_laplacian gives the best roll/pitch balance. Plateau is NOT reward-driven.

Root cause analysis of the plateau identified a critical information gap: joint actuator
parameters (stiffness, damping, effort limit) are domain-randomized with massive ranges
(Kp: 3x, Kd: 10x, effort: 1.4x) but were completely absent from the 19D privileged
observation vector. The encoder literally cannot distinguish Kp=40 from Kp=120 environments.
An action optimal for Kp=40 (aggressive, compensating for sluggish joints) causes overshoot
and torque violation at Kp=120. This directly causes advantage cancellation across DR-diverse
environments (surrogate -> 0), which in turn starves the encoder of gradient signal.

This is inconsistent with standard practice: RMA (Kumar 2021) and HORA (Qi 2023) include
motor strength, joint damping, and friction in their privileged observations.

Also removed CoG x/y for both main and buoy bodies (4D total): +-0.01m offset range creates
negligible torque (0.26Nm vs 6-10Nm effort limit). Only CoG z retained (dominates roll/pitch).

### Changed
- `mdp/observations.py`: Privileged obs expanded in two steps during this session.
  Step 1 (19D->18D): `_hydro_privileged_info()` returns 3D (volume, CoG_z, CoB_z) instead of
  5D (removed CoG_x, CoG_y). Added joint_stiffness/damping/effort_limit (3D).
  Step 2 (18D->23D): Added main body linear damping roll/pitch (2D from
  `hydro.linear_damping[:, 3:5]`), quadratic damping roll/pitch (2D from
  `hydro.quadratic_damping[:, 3:5]`), body mass (1D from `hydro.body_mass`).
  Final privileged: 19D -> 23D (+8D added, -4D removed CoG x/y).
- `config.py`: `state_space` 19 -> 23. Docstring updated.
- `agents/rsl_rl_ppo_cfg.py`: `privileged_dim` 19 -> 23.
- `encoder/actor_critic_encoder.py`: Architecture docstring 272D -> 276D.
- `encoder/actor_critic_encoder_constrained.py`: Architecture docstring 272D -> 276D.

### Notes
- Checkpoint incompatible with previous runs (privileged dim changed). Must start fresh.
- All new values read directly from existing runtime tensors (ArticulationData for joint
  params, HydrodynamicsModel for damping/mass). No additional buffers needed.
- Full DR parameter audit (28 params total): 17 params now in privileged obs, 11 remain
  hidden (water_density 1.03x negligible, joint friction low impact, perturbation/latency
  are per-step events not model params, lateral CoB/CoG +-0.01m negligible).
- Constraint cost divergence analysis: joint_torque/vel_limit worsen because policy learns
  aggressive actions (act_size=1.20) for attitude tracking. At Kp=120 (DR max), action=1.0
  generates 10.08Nm torque > 9.5Nm limit. Policy needs encoder to condition action scale on
  Kp: use action~0.5 for Kp=120, action~1.0 for Kp=40. Without encoder differentiation,
  uniform aggressive action violates torque in high-Kp envs.
- yaw_vel plateaus because aggressive joint actuation creates reaction torques with yaw
  component; equilibrium at ~0.34 rad/s where excitation balances hydrodynamic yaw drag.
- No constraint warm-up/curriculum is implemented in current C-TRPO code.
  `set_max_iterations()` is a log-only no-op. Budget, beta, recovery_threshold all fixed.
- Early results (run `2026-03-21_23-04-17`, 391 iter, 18D version): roll=7.63 deg (improved),
  pitch=15.82 deg (similar), recovery=6% (improved from 14%). Surrogate still near zero.
- Secondary concern: encoder receives 276D input where only 23D (privileged) is unique.
  HORA Phase 1 uses privileged-only encoder input. May need architectural change later.

## [2026-03-21] Encoder update strengthening: multi-step encoder + relaxed grad clip

### Context
Training run `2026-03-21_19-59-33` (z_bounds_coef=0.0) plateaued at roll=8.7 deg, pitch=12.5 deg,
reward=18 after ~1010 iter. Deep analysis revealed:

1. **Plateau root cause is NOT reward function**: Policy uses full KL budget (107%) every iteration
   in 100% safe mode, but reward doesn't improve. The optimization landscape is flat at this point.
2. **Encoder-actor update imbalance**: Critic gets 20 SGD steps/iter, actor gets 1 TRPO step,
   encoder gets only 1 SGD step. Encoder gradient comes ONLY from actor loss (critic is asymmetric,
   uses privileged directly, not z). This creates encoder-actor coordination deadlock at plateau.
3. **smooth_min_laplacian FAILED (reverted)**: Attempted to replace min_laplacian with soft-min
   (LogSumExp alpha=5) to fix gradient oscillation between roll/pitch axes. Result: made axis
   asymmetry WORSE (roll=4.76, pitch=23.46 at 500 iter). At large disparity, smooth_min ~= hard min
   (92.8%/7.2% gradient split at roll=5/pitch=23). The early gradient sharing let the easy axis
   (roll) converge faster, widening the gap. Roll/pitch asymmetry is physical, not reward-driven.
4. **min_std=0.15 FAILED (reverted)**: Lower noise floor (0.2->0.15) accelerated axis asymmetry
   by enabling faster convergence on the easy axis. Noise actually helps maintain axis balance.
5. **eapo_target_entropy=0.3 reverted to 0.5**: Coupled with min_std revert.

### Added
- `mdp/rewards.py`: `smooth_min_laplacian` command_type (LogSumExp soft-min, alpha=5). Kept in
  code for future use but not active (config uses `min_laplacian`).

### Changed
- `agents/rsl_rl_ppo_cfg.py`: `num_encoder_epochs` 1->3 with KL gating safety
  (max_encoder_kl=0.016 reverts encoder if distribution shift exceeds budget). Addresses
  encoder-actor update imbalance (critic 20 steps vs encoder 1 step per iteration).
- `algorithms/constraint_trpo.py`: Encoder gradient clip max_norm 0.2->0.5. Previous 0.2 was
  5x more restrictive than value function clip (1.0). With KL gating as primary safety, clip
  is secondary defense and can be relaxed.

### Notes
- Roll/pitch axis asymmetry is physical/structural, not reward-driven. Both laplacian(sum),
  min_laplacian, and smooth_min_laplacian show the same pattern (roll converges faster).
  The 2-DOF arm may have different authority for roll vs pitch correction.
- min_std=0.2 noise floor serves dual purpose: exploration AND axis balance maintenance.
  Lowering it breaks the balance even though it seems like it should help fine-tuning.
- `num_encoder_epochs` comment "must stay at 1" was outdated -- written before KL gating
  (Fix 2, constraint_trpo.py:925-939) was added. Multi-step is now safe with KL gating.
- hero_agent `rsl_rl_ppo_cfg.py` z_bounds_coef still at 1.0 (desync from constrained_albc 0.0)

## [2026-03-21] Disable z_bounds_loss -- false saturation diagnosis

### Context
Analyzed run `2026-03-21_17-20-58` (z_bounds_coef=0.3, 1388 iter): encoder z_range [-0.97, 0.95]
triggered SAT diagnostic, enc_grad decayed 0.15->0.02, pitch plateaued at 13.38 deg. Initial
hypothesis was encoder gradient death from tanh saturation.

Attempted fix: z_bounds_coef 0.3->1.0 (run `2026-03-21_19-33-03`). Result: z_std dropped from
0.71 to 0.34 (encoder using only 34% of latent range), z_range still triggered SAT at [-0.95, 0.94].
Pitch improved to 12.15 deg at 537 iter but at the cost of severely constrained encoder expressiveness.

Key realization: the SAT diagnostic is a **false positive**. With 4096 envs x 13 dims = 53,248 tanh
outputs per step, the batch min/max naturally reaches +-0.95 even with a healthy z_std=0.34 distribution.
This is normal tail statistics, not saturation. True saturation would show z_std near 1.0 (bimodal at
boundaries) and enc_grad < 1e-4. Neither condition holds here.

z_bounds_loss is borrowed from HORA (which used sigmoid activation where saturation is more severe).
With tanh (gradient=0.75 at z=0.5, vs sigmoid gradient=0.25), the anti-saturation penalty is
unnecessary and actively harms encoder expressiveness by compressing the latent distribution.

Also identified cost_return oscillation as structural recovery mode cycling (mean_cycle=159 iter),
not DR randomness. The safe->recovery->safe hysteresis (recovery_threshold_frac=0.4, joint_torque
entering recovery at cost=20, exiting at cost<8) creates deterministic ~159 iter oscillation periods.

### Changed
- `constrained_albc/agents/rsl_rl_ppo_cfg.py`: z_bounds_coef 0.3 -> 0.0 (policy and algorithm cfg)
- `constrained_albc/agents/rsl_rl_ppo_cfg.py`: z_bounds_soft_bound 0.9 -> 0.85 (no effect when coef=0)
- `constrained_albc/algorithms/constraint_trpo.py`: z_bounds_coef default 0.3 -> 0.0
- `constrained_albc/encoder/actor_critic_encoder.py`: z_bounds_coef default 0.1 -> 0.0, soft_bound 0.9 -> 0.85
- `hero_agent/agents/rsl_rl_ppo_cfg.py`: z_bounds_coef 0.3 -> 1.0, soft_bound 0.9 -> 0.85 (synced but hero_agent not actively used)

### Notes
- hero_agent z_bounds_coef set to 1.0 (not 0.0) -- was changed first before the false-positive analysis. Will sync to 0.0 if this approach works.
- Improved SAT diagnostic should use "fraction of z values with |z| > 0.9" instead of batch min/max
- Cost return cycling is structural (recovery hysteresis), not a bug. May need recovery_threshold_frac tuning (currently 0.4) or budget adjustment

## [2026-03-21] Recovery cycling fix: smooth barrier gradient at mode transitions

### Context
Run `2026-03-21_16-45-28` (1063+ iter, barrier fix applied) exhibited periodic oscillation in
all metrics with ~64 iter period. Attitude error oscillated 6-19 deg (roll) and 12-25 deg (pitch).

Root cause: joint_vel_limit constraint budget d_k=5.0 was too tight. Cost return oscillated
2.0-5.3, hitting d_k=5.0 and entering recovery mode every ~64 iterations. During recovery,
barrier penalty was excluded (`safe_mask = (margins > 0) & ~recovery`), causing a 500:1 gradient
magnitude discontinuity at mode transitions: barrier gradient phi_pp=10000 * beta=0.05 * surr^2
= 500*surr^2 in safe mode vs 1*surr in recovery. This binary switching created a deterministic
limit cycle.

Fix: remove recovery exclusion from barrier mask so barrier stays active through mode transitions.
Combined with margin_min increase (0.01->0.1 to cap phi_pp at 100), beta reduction (0.05->0.02),
and budget doubling (d_k 5->10), the gradient discontinuity at mode boundaries is reduced from
500:1 to ~2:1 (additive recovery cost only).

### Changed
- `constraint_trpo.py`: Changed `safe_mask = (margins > 0) & ~recovery` to `safe_mask = margins > 0`
  -- barrier stays active for all positive-margin constraints regardless of recovery mode
- `constraint_trpo.py`: Increased margin_min clamp from 0.01 to 0.1 (phi_pp max: 10000 -> 100)
- `config.py`: Doubled joint_vel_limit budget from 0.05 to 0.10 (d_k: 5.0 -> 10.0)
- `rsl_rl_ppo_cfg.py`: Reduced beta from 0.05 to 0.02 (max barrier gradient: 2*surr^2)
- `rsl_rl_ppo_cfg.py`: Reduced recovery_threshold_frac from 0.6 to 0.4 (faster recovery exit)

### Notes
- Analysis confirmed joint_vel_limit triggered 105/122 recovery entries (86%)
- barrier_penalty max=0.656 (spikes at iter 222, 296) overwhelmed reward surrogate (~0.1)
- train-analyze skill updated (outside isaaclab repo) with oscillation detection,
  mode switching summary, and barrier penalty display
- Dry run needed: 50 iter, 64 envs to verify smooth barrier behavior before full training

## [2026-03-21] Barrier penalty bug fix: structurally zero gradient

### Context
Run `2026-03-20_18-15-06` (2500 iter, EAPO + C-TRPO) showed attitude error stagnation at 10+
deg. Root cause analysis revealed barrier penalty was structurally zero throughout ALL training.

The barrier mechanism uses `cost_surr^2` (linearized Bregman divergence). But the code applied
per-constraint cost advantage standardization (NORBC Sec IV-B) which sets `E[A_cost] = 0` by
construction. At ratio=1: `cost_surr = mean(1 * A_cost) = 0`. Therefore `d(0^2)/d(theta) = 0`
-- barrier gradient is identically zero regardless of constraint proximity.

This means the TRPO step has been entirely reward-driven with no barrier repulsion from
constraint boundaries for ALL prior constrained training runs. Constraints only received
enforcement from recovery mode (after violation), never from the barrier (before violation).

Fix: store raw (unstandardized) cost advantages separately. Use raw for barrier (needs actual
cost change signal), standardized for recovery (needs balanced gradients). Scale-normalize raw
by per-constraint std to prevent binary vs continuous cost scale domination.

### Fixed
- `constraint_trpo.py`: Added `cost_advantages_raw` buffer in `init_storage()` to store
  unstandardized cost advantages
- `constraint_trpo.py`: In `_compute_cost_returns()`, clone raw advantages before
  standardization step
- `constraint_trpo.py`: In `update()`, flatten raw advantages and compute `raw_cost_std`
  for scale normalization
- `constraint_trpo.py`: In `surrogate()`, barrier penalty now uses raw/scale-normalized
  cost surrogates; recovery cost remains on standardized advantages

### Notes
- `_compute_barrier_penalty()` itself was correct -- it was just always fed zero input
- No checkpoint format changes (raw buffer is compute-only, not saved)
- Constraint budgets left unchanged initially -- barrier should naturally maintain margin
- If barrier causes excessive conservatism, budgets can be increased (Task 3 in plan)
- Also updated train-analyze skill with config reading (non-git, in `.claude/skills/`)

## [2026-03-20] EAPO: Entropy Advantage Policy Optimization for C-TRPO

### Context
Implemented EAPO (arXiv:2407.18143) to solve C-TRPO entropy collapse. Previous attempt
(min_std=0.2 floor, see entry below) failed: std monotonically collapsed to floor and
stayed there permanently. Root cause: surrogate gradient w.r.t. log_std is almost always
negative (high-advantage actions cluster near mean), and entropy_coef=0.0 provides no
counterforce. Naive entropy_coef is structurally unstable in TRPO (competes with reward
for single KL budget step).

EAPO solution: per-sample entropy advantage `A_H = normalize(-log_prob)` added to task
advantage as `soft_adv = A_task + tau * A_H`. Actions far from mean get positive entropy
advantage, creating std-increase gradient. With state-independent log_std, discounted
return reduces to simple batch normalization (entropy V(s) is constant). Adaptive tau
via SAC v2 dual gradient: tau increases when entropy < target, decreases otherwise.
Applied outside TRPO step, so KL budget is not consumed.

Dry run verification (10 iter, 64 envs, headless): entropy_tau logged correctly (0.001,
at tau_min since init_std=1.0 gives entropy >> target). noise_std=0.95 after 10 iter
(gradual, not collapsing). Surrogate finite, training stable.

### Added
- `agents/rsl_rl_ppo_cfg.py`: 6 EAPO config fields (`eapo_enabled=True` default,
  `eapo_tau_init=0.01`, `eapo_target_entropy=0.5`, `eapo_tau_lr=0.001`,
  `eapo_tau_min=0.001`, `eapo_tau_max=0.5`)
- `algorithms/constraint_trpo.py`: `_compute_entropy_advantages()` method (batch-normalized
  `-log_prob`), entropy_advantages storage allocation in `init_storage()`
- `runners/constraint_encoder_runner.py`: `Policy/entropy_tau` metric logging, EAPO tau
  checkpoint save/load (`eapo_state.pt`)

### Changed
- `algorithms/constraint_trpo.py`: `__init__` accepts 6 EAPO kwargs and stores state.
  `compute_returns()` calls `_compute_entropy_advantages()` when enabled. `update()`
  computes `soft_adv = adv + tau * entropy_adv` for surrogate closure. Adaptive tau
  update after noise floor clamp (before encoder update). `loss_dict` includes
  `entropy_tau` when enabled.

### Notes
- EAPO is actor-only: encoder update uses task-only advantages (no entropy pressure on
  encoder, which must optimize for task performance)
- tau_min=0.001 prevents zero entropy pressure; tau_max=0.5 prevents entropy domination
- When entropy is already above target (e.g., init_std=1.0), tau naturally decreases to
  floor -- EAPO only activates when exploration is actually needed
- Pending: 200+ iter training run to verify entropy stabilizes near 0.5, std stays above
  0.2 floor, and attitude convergence improves

## [2026-03-20] C-TRPO noise floor fix: min_std=0.2 -- FAILED to solve entropy collapse

### Context
Replaced entropy_coef (structurally unstable in TRPO surrogate) with min_std=0.2 hard
floor applied after TRPO step. Rationale: PPO uses min_std=0.18 floor in base_runner.py.

Run `2026-03-20_17-41-26` (229 iter, 4096 envs) showed the fix is **insufficient**:
- noise_std: 0.99 -> 0.52 (iter20) -> 0.27 (iter40) -> 0.20 (iter80+). Monotonic
  decrease to floor regardless of error magnitude. Floor prevents going below 0.2 but
  does not prevent the collapse itself.
- Performance: roll 7 deg / pitch 14 deg at iter 150 (best), then pitch collapsed back
  to 45 deg by iter 225. Same "converge then collapse" pattern as previous runs.
- Entropy: stabilized at -0.39 (theoretical minimum for std=0.2 Gaussian). Permanently
  locked at floor -- no adaptive response to high error.
- Encoder z: saturated [-1.00, 0.99] (anomaly flagged).
- joint_vel_limit: diverging (cr=3.76, dk=5.0).

**Why min_std floor alone fails**: PPO's floor is a safety net, not the primary mechanism.
PPO's entropy_coef works because multi-step mini-batch updates don't have KL budget
competition. In C-TRPO, with entropy_coef=0.0 and only a passive floor, there is NO
mechanism to resist std reduction. The surrogate gradient w.r.t. log_std is almost always
negative (high-advantage actions cluster near mean), so std decreases every iteration
until hitting the floor and staying there permanently.

### Changed
- `agents/rsl_rl_ppo_cfg.py`: Added `min_std: float = 0.2`. Changed `entropy_coef` 0.001 -> 0.0.
- `algorithms/constraint_trpo.py`: Added `min_std` param, changed floor from `log(0.01)` to
  `log(self.min_std)`. Entropy code preserved (disabled at coef=0.0).

### Notes
- The fix is committed but does not solve the problem. An active exploration mechanism is needed.
- Candidates: (1) adaptive floor via PI controller, (2) state-dependent std, (3) entropy with
  separate KL budget outside TRPO step. All require further analysis before implementation.

## [2026-03-20] Min-axis Laplacian reward + entropy/logging fixes

### Context
Analysis of run `2026-03-20_16-40-57` (650+ iter, post-stability-fix) revealed:
1. **Roll/pitch asymmetry**: roll converged to 3-5 deg but pitch stuck at 25-30 deg. Root cause:
   Laplacian reward `exp(-|e|/sigma).sum()` with sigma=0.15 makes roll (3 deg, reward=0.707)
   dominate gradient over pitch (25 deg, reward=0.054) -- 93:7 ratio. Advantage function sees
   total reward, so pitch-improving actions produce negligible advantage signal.
2. **entropy_coef=0.005 too strong**: noise_std climbed to 1.62 (above init 1.0), entropy=3.80
   at ceiling. TRPO's single natural gradient step amplifies entropy bonus vs PPO's ~20 mini-batch
   steps. Reduced to 0.001.
3. **Duplicate logging**: `entropy` and `pre_encoder_kl` appeared in both `Loss/` (via loss_dict)
   and `Policy/` (via runner metrics). Removed from loss_dict, kept in Policy/ only.

### Changed
- `mdp/rewards.py`: Added `"min_laplacian"` command_type to `command_reward()`. Uses
  `per_axis.min(dim=-1).values` instead of `.sum()`. Worst axis determines reward, preventing
  the better axis from dominating gradient signal. Updated `ALBCRewardCfg` docstring.
- `config.py`: `command_type="laplacian"` -> `"min_laplacian"`.
- `algorithms/constraint_trpo.py`: `entropy_coef` default 0.005 -> 0.001. Removed `entropy`
  and `pre_encoder_kl` from `loss_dict` (duplicate with Policy/ metrics in runner).
- `agents/rsl_rl_ppo_cfg.py`: `entropy_coef` default 0.005 -> 0.001. Updated docstring with
  rationale (TRPO amplification of entropy bonus).

### Notes
- min_laplacian preserves Laplacian's near-zero gradient advantage while fixing far-error blindness
- At roll 3 deg / pitch 25 deg: min-axis reward = 0.054 (pitch determines), gradient 100% on pitch
- When both axes reach ~5 deg, rewards become balanced and fine convergence activates naturally
- joint_torque cost still diverging (cr=18.85, dk=20.0) -- monitor but not yet critical

## [2026-03-20 Summary] Code review marathon + stabilization fixes (11 sessions)

### Context
Intensive code review and stabilization day. Started with comprehensive simplification (forked
constrained_albc package from hero_agent, reduced from ~7000 to ~4900 lines by removing unused
features). Then 6 targeted code reviews found 3 critical bugs (prev_actions_obs causal violation,
effort_limit_cost per-joint masking, joint DR running in debug mode) plus encoder optimizer resume
bug, barrier singularity, and 5 theoretical fixes. Stabilization work: Laplacian reward (replaces
quadratic for near-zero gradient), min_laplacian (worst-axis focus), noise floor (0.25->0.01),
EAPO entropy advantages, EMA cost smoothing (B1), value LR gating (B2), per-constraint blend
mode, entropy bonus, encoder KL gating, yaw_vel budget relaxation. Removed PBRS progress reward
(redundant). Removed constrained_encoder_base from hero_agent (4 files deleted, 10 edited).

### Changed
- `algorithms/constraint_trpo.py`: Full evolution through 11 sessions. Final state: barrier-based
  C-TRPO with unified surrogate (reward + barrier + recovery cost). EMA smoothing on cost returns,
  value LR gating on actor freeze, encoder KL gating (max_encoder_kl=0.016), min_std noise floor
  (0.01), entropy_coef (0.005->0.001), EAPO soft advantages, isfinite guard on encoder loss,
  barrier margin clamp (min=0.01), **_kwargs for RSL-RL compat, cost GAE dones shape fix.
- `config.py`: Merged 4-class hierarchy to single ALBCEnvCfg. Laplacian reward (sigma=0.15).
  Overshoot threshold 0.035->0.087, budget 0.10->0.20. yaw_vel budget 0.35->0.785.
  Removed dead enable_payload field. Budget D_k vs d_k documentation added.
- `albc_env.py`: Simplified from ~1200 to ~1000 lines. Key fixes: _prev_actions_obs causal
  violation, _prev_joint_pos timing, control_dt (physics_dt -> step_dt). Extracted helper
  methods. Joint DR gated on rand_cfg.enable. Payload always computed.
- `mdp/rewards.py`: 3-term -> 2-term (removed PBRS). Added laplacian/min_laplacian/smooth_min
  command types. ALBCRewardCfg reduced from 10 to 5 fields.
- `mdp/constraints.py`: effort_limit per-joint comparison, overshoot per-axis conjunction,
  removed dead cost_type field. Removed 5 unused cost functions.
- `mdp/events.py`: DRSampler simplified, DORAEMON removed, payload xy-norm clamp, inertia
  fallback warning.
- `encoder/`: DRY fix (_encode delegation), no-grad normalization update (262K pass savings),
  softplus->ReLU on cost critic, restored load_state_dict backward compat, z_bounds_loss device
  fix. Removed no-history mode, symmetric critic, sigmoid activation. 465+131 -> 295+78 lines.
- `runners/`: Flattened 3-level to single ConstraintEncoderRunner. Encoder optimizer checkpoint.
  EMA state persistence. ALBC-prefixed namespace registration. Dict-based auto-sync fix.
- `agents/rsl_rl_ppo_cfg.py`: Config hierarchy flattened. Added EAPO, EMA, entropy, KL gating
  params. ALBC-prefixed runner module names.

### Removed
- `doraemon.py` (728 lines), `runners/base_runner.py`, `runners/encoder_runner.py`,
  `encoder/history_tcn.py`, hero_agent constrained files (4 files, ~65KB total).
- Dead code: penalty curriculum, settling/energy rewards, TDE obs, buoy perturbation,
  backward-compat checkpoint loading, DORAEMON integration, PBRS progress reward.

### Notes
- ruff check + ruff format clean across entire package (13 files)
- All public API preserved: ALBCEnv, ALBCEnvCfg, gym registration, encoder architecture
- No checkpoint breaking: state_dict keys unchanged (removed only backward-compat loading logic)
- Pre-existing issue: runner namespace collision between hero_agent and constrained_albc (not addressed)

## [2026-03-18] C-TRPO encoder fix experiments: multi-step caused KL instability

### Context
The encoder gradient fix (2026-03-17) restored enc_grad from 8.3e-4 to 0.04, but introduced
new problems. Three experimental runs were conducted:

1. **Post-mod baseline** (5-epoch, lr=1e-3, torque=15, min_std=0.25, 514it): Best error 5.3-5.5 deg
   at iter 500 but plateaued. kl=0.13~2.87 (vs pre-mod 0.013). Entropy crashed to floor by iter 100.
2. **entropy_coef=0.005** (178it): Performance degraded (r_cmd=-0.97 vs -0.37). Entropy bonus too
   strong for TRPO single-step; slowed convergence without preventing collapse. Reduced to 0.001.
3. **entropy_coef=0.001, torque=20, min_std=0.18** (2500it): Roll 6.2 deg, pitch 6.1 deg. Slower
   convergence than baseline (~5x). kl still unstable (0.04~5.6). min_std had no effect (noise
   never reached 0.18 floor). mode oscillation: 146 transitions, yaw_vel in recovery 23% of time.

Root cause identified: **5-epoch encoder update causes uncontrolled distribution shift**. TRPO
trust region constrains actor KL to 0.01, but encoder changes z (actor input), causing indirect
distribution shift of kl=0.08~5.6 (up to 560x larger). Pre-mod code had kl=0.013 because encoder
barely changed (recovery bug suppressed updates + single cached gradient).

### Changed
- `agents/rsl_rl_ppo_cfg.py`: Reverted `num_encoder_epochs: 5 -> 1`. Multi-step encoder
  update incompatible with TRPO trust region (indirect KL not bounded).
- `agents/rsl_rl_ppo_cfg.py`: Reverted `encoder_lr: 1e-3 -> 3e-4`. Even with 1 epoch,
  3.3x higher LR caused excessive distribution shift vs pre-mod baseline (kl=0.013).
- `algorithms/constraint_trpo.py`: Matched default `encoder_lr` to 3e-4.
- `agents/rsl_rl_ppo_cfg.py`: `joint_torque` budget 0.15 -> 0.20 (reduces constraint floor).
- `config.py`: Synced `joint_torque` budget to 0.20.

## [2026-03-17 Summary] C-TRPO migration, encoder architecture, eval fairness

### Context
Major architecture day: migrated from Lagrangian to C-TRPO (Muller et al., ICML 2025), fixed
encoder gradient death (50x drop), designed and implemented history-augmented encoder (TCN -> raw
flatten), unified eval_dr conditions for fair TDC vs Encoder comparison. Seven Lagrangian tuning
sessions concluded the approach was fundamentally flawed (lambda hysteresis, entropy dilemma).

### Changed
- `algorithms/constraint_trpo.py`: Full rewrite from Lagrangian to C-TRPO barrier-based.
  Removed lambda_k, log_alpha, alpha_optimizer. Added barrier penalty, margin tracking,
  safe/recovery modes. Fixed encoder gradient death: multi-step fresh forward passes
  (num_encoder_epochs=5, later reverted to 1). Removed recovery mode encoder blocking.
- `encoder/actor_critic_encoder.py`: Replaced HistoryTCN with raw flatten concat
  (TRPO OOM fix: 960MB -> 260MB). hist(N,30,8) -> flatten(N,240) -> concat to MLP.
- `agents/rsl_rl_ppo_cfg.py`: Lagrangian params replaced with barrier params (beta=0.01,
  recovery_threshold_frac=0.8). num_encoder_epochs and encoder_lr added.
- `runners/constraint_encoder_runner.py`: lambda_state.pt -> barrier_state.pt checkpoint.
- `eval_dr.py`: Unified DR conditions (removed is_tdc override). Pre-allocate latency buffers.
- `tdc_env.py`: Added TDC output latency buffer for fair evaluation.
- `config.py`: Constraints finalized at 6 terms. Budgets tuned (joint_torque 0.05->0.20,
  yaw_vel 0.15->0.35). PBRS progress_weight=2.0, quadratic command.

### Removed
- `encoder/history_tcn.py`: Deleted (OOM on TRPO full-batch).
- Lagrangian mechanism: lambda_k, alpha_entropy, dual update, warmup logic.
- `_cache_encoder_grads()` method (replaced by multi-step fresh passes).

### Notes
- Encoder z sweep (2500 iter run): 10/13 z dims near-constant, cosine sim 0.9482.
  Encoder not learning useful DR representations from privileged info alone.
- History-augmented architecture provides command-response relationship needed for informative z.
- TRPO entropy dilemma unsolvable in Lagrangian framework: alpha>0 -> unbounded growth, alpha=0 -> collapse.

## [2026-03-16 Summary] Lagrangian migration + entropy stabilization (10 sessions)

### Context
Intensive debugging day (10 sessions) focused on migrating constraint enforcement from
IPO log-barrier to Lagrangian primal-dual, and fixing the entropy collapse/explosion cycle.

**Core problem**: IPO log-barrier assumed feasible start, but random policy starts infeasible.
Barrier gradient's easiest path was reducing action variance (kills exploration). Multiple
fixes attempted (barrier_t tuning, noise floors/ceilings, adaptive thresholds, n_active
normalization) were all band-aids. Fundamental fix: Lagrangian primal-dual where lambda
starts at 0 and grows with violations.

### Changed
- `algorithms/constraint_trpo.py`: Replaced IPO log-barrier with Lagrangian primal-dual.
  Detached std from cost gradient, detached encoder z from cost surrogate. Added reward
  advantage normalization, LS-gated updates, d_k-normalized dual update.
- `agents/rsl_rl_ppo_cfg.py`: Barrier params -> Lagrangian params. Budgets recalibrated.
- `runners/constraint_encoder_runner.py`: Lambda checkpoint persistence.
- `config.py`: energy_weight=0.0, smoothness_weight=0.0. effort_limit budget 0.05->0.25.
- `mdp/constraints.py`: effort_limit_cost uses per-env DR'd limits. Removed dead barrier fields.

### Fixed
- `algorithms/constraint_trpo.py`: Cost value loss clamps targets >=0. NaN guard on cost
  advantage normalization.

## [2026-03-05/10 Summary] ConstraintTRPO build-out, stabilization, NORBC conformance

### Context
Built NORBC-style constrained RL (IPO -> Lagrangian TRPO) from scratch. Key milestones:
initial 3-constraint IPO implementation (3/5-6), constraint expansion to 8 terms with
cost critic softplus fix (3/8), NORBC conformance audit with asymmetric critic and actuator
DR unification (3/9), barrier_t fix 10->50/100 for arm freeze (3/9), noise floor unification
0.1/0.15 -> 0.25 + effort_limit DR conflict removal (3/10).

### Changed
- `algorithms/constraint_trpo.py`: Full TRPO + IPO (~600 lines). barrier_t tuning (1->10->50->10). min_log_std unified to log(0.25).
- `encoder/actor_critic_encoder.py`: Asymmetric critic (raw privileged input)
- `encoder/actor_critic_encoder_constrained.py`: Multi-head cost critic, F.softplus() output
- `runners/constraint_encoder_runner.py`: Named constraint logging, auto-sync K, lambda checkpoint
- `mdp/constraints.py`: 3->8 cost functions, effort_limit uses per-env DR'd limits
- `config.py`: 8-term constraints, unified actuator DR (Kp 40-120, Kd 0.5-5.0, effort 0.7-1.0)
- `hero_agent.py`: velocity_limit_sim 6.28->4.19 rad/s (datasheet 40 rpm)
- `base_env.py`: Accumulated rotation tracking, cost computation, constraint buffers
