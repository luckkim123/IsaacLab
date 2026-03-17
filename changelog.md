# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [2026-03-17] Training analysis: entropy hysteresis diagnosis

### Context
Analyzed run `2026-03-17_13-13-31` (encoder LR=3e-4, 6 constraints, history-augmented encoder).
KL desync fix confirmed: kl_trpo=0.01, kl=0.012 (was 0.38 with LR=3e-3). Encoder stable:
z_std=0.55, z_mean~0, grad healthy. BUT:

1. **Roll error spiked iter 220-240** (7 -> 32 deg) then V-shaped recovery. NOT caused by
   encoder z saturation (z_mean stable +-0.05). Caused by lambda_max crossing 0.10 ->
   policy shift (kl spike to 0.10 at iter 230) -> temporary destabilization.

2. **Entropy collapsed by iter 400**: 2.84 -> 0.03 -> -0.38. noise_std hit floor 0.20 by
   iter ~900. Root cause: alpha_entropy=0 + 6 constraints ALL push noise down (joint_torque,
   yaw_vel, joint_vel_limit penalize large/fast actions -> reducing noise is easiest compliance
   path). Dual downward pressure with zero counterforce.

3. **Action magnitude halved at iter 650-700**: 1.05 -> 0.55. Coincided with lambda_joint_torque
   peak (0.22 at iter 673). Policy learned "do less = safe" instead of "do precise = efficient".

4. **Lambda hysteresis (key finding)**: lambda_joint_torque peaked at 0.22 (iter 673) then
   dropped to 0.00 (iter 1022, constraint satisfied). BUT action_size stayed at 0.55 and
   noise_std stayed at floor. Transient constraint pressure permanently collapsed exploration
   with no recovery mechanism. Constraints acted as one-way ratchet on entropy.

5. **Grad norm escalation**: grad_norm_reward 0.015 -> 0.61 (40x). Caused by 1/sigma^2
   amplification as noise_std decreases. TRPO protects actor (KL step size), but encoder
   uses Adam -> effective LR increases. encoder grad 0.002 -> 0.08.

6. **Latest (iter 1597)**: roll=5.98, pitch=9.89, reward=-6.13. Slowly improving despite
   zero exploration. Approaching asymptotic limit.

### Notes
- Encoder is NOT the problem. z_mean +-0.05, z_std stable at 0.53-0.57, kl synced.
- z_min/z_max hitting [-1,1] is min/max across 4096 envs x 13 dims -- outlier, not systemic.
- Root cause is purely exploration death: alpha_entropy=0 provides no entropy recovery force.
- noise_floor=0.20 only sets lower bound; doesn't push entropy UP when constraint pressure eases.
- Planned fix: alpha_entropy=0.005 (fixed, not adaptive), noise_floor 0.20->0.25 (moderate),
  encoder grad clip max_norm=0.1 (safety). Avoid floor=0.30 (prevents fine-grained convergence).

## [2026-03-17] Reduce encoder LR for 272D input (actor-encoder desync fix)

### Context
Run `2026-03-17_12-51-10` (286 iters, raw flatten history concat) showed:
- z_std=0.70 (up from 0.12 pre-history) -- history concat is providing useful signal
- BUT reward stuck at -18.86, error 15-17deg (no improvement over random)
- kl_trpo=0.01 (TRPO step fine) vs kl=0.31 with peaks >1.0 (post-encoder update)
- entropy=-0.14, noise_std=0.22 (rapidly dropped to floor)
- 3 constraints OVER budget: joint_torque, joint_vel_limit, yaw_vel

Root cause: encoder LR=3e-3 was tuned for 19D input (privileged only). With 272D input
(13 policy + 240 hist_flat + 19 privileged), the encoder first layer has 70K params
(was 5K). Each Adam step shifts z significantly, invalidating the TRPO KL guarantee.
The actor optimizes under one z-mapping, then encoder changes it -- actor-encoder desync.

Failure chain: encoder LR too high -> large z-shift per step -> kl jumps 0.01->0.31+
-> rollout advantages become stale -> no learning signal -> noise drops to floor
(reducing noise is the only "stable" strategy) -> error plateaus at random level.

### Changed
- `algorithms/constraint_trpo.py`: `encoder_lr` 3e-3 -> 3e-4 (10x reduction).
  272D input has 14x more first-layer params than 19D; reducing LR by 10x keeps
  per-step z-shift comparable to pre-history encoder.

### Notes
- z_bounds_loss=0.003 (not 0 as initially reported -- script rounding). z_bounds is
  working correctly; z_range [-1,1] is expected min/max across 4096 envs.
- ConstraintEncoderRunner overrides _update_encoder_lr to no-op; encoder_lr is solely
  managed by ConstraintTRPO. No other files need changes.
- Success metric: kl (post-encoder) should drop from 0.31 to <0.05
- If kl still too high, next step is dimension reduction (linear projection or
  history subsampling), not further LR reduction.

## [2026-03-17] Replace HistoryTCN with raw flatten concat (TRPO OOM fix)

### Context
Previous session designed a history-augmented encoder using a shared HistoryTCN
(temporal convolution) to produce h_embed (32D) from proprioception history (30, 8).
Three consecutive CUDA OOM errors occurred during constrained encoder training:

1. TCN called 3x per step -> added h_embed caching -> OOM (960MB backward)
2. Added gradient checkpointing -> OOM (recomputation needs same memory)
3. Fundamental: Conv1d intermediate activations on TRPO full-batch (4096x64=262K samples
   x 30 timesteps) require ~960MB, exceeding RTX 4070 available VRAM (~0.6GB free after
   model + rollout buffer)

User questioned whether TCN and embedding are necessary at all. Decision: eliminate TCN
entirely, flatten proprio_hist (N, 30, 8) -> (N, 240) and concatenate directly to MLP
inputs. Adds only ~260MB input tensor (vs ~960MB TCN intermediates). The encoder/actor/
critic MLPs learn from raw history without any preprocessing.

Phase 2 adaptation (ProprioAdaptTConv) retains its TCN because it runs in PPO minibatch
mode via AdaptRunner (not TRPO full-batch), so Conv1d memory is manageable.

Post-fix: training crashed with `IndexError: too many indices for tensor of dimension 2`
in `log_encoder_metrics()` because `_encode()` now expects TensorDict, not raw tensor.
Fixed by passing full obs TensorDict.

### Changed
- `encoder/actor_critic_encoder.py`: Complete rewrite. Removed HistoryTCN, h_embed caching,
  gradient checkpointing, shared_tcn, hist_normalizer, h_embed_dim. Added `_get_hist_flat()`
  (simple flatten). New dimensions: Encoder 272D, Actor 266D, Critic 272D (with history)
- `encoder/adaptation.py`: Removed dual-mode (adapt_head vs adapt_tconv). Always uses
  ProprioAdaptTConv for Phase 2. `_get_combined_obs`/`evaluate` use hist_flat from parent
- `encoder/__init__.py`: Removed HistoryTCN export and docstring reference
- `algorithms/constraint_trpo.py`: Removed `shared_tcn`/`hist_normalizer` from
  `encoder_prefixes`. Removed 2x `clear_h_embed_cache()` calls
- `agents/rsl_rl_ppo_cfg.py`: Removed `h_embed_dim: int = 32` from `_RslRlPpoEncoderBaseCfg`
- `adapt_base_env.py`: Updated docstring (shared_tcn/h_embed -> adapt_tconv/hist_flat)

### Fixed
- `utils/logging.py`: `log_encoder_metrics()` called `policy._encode(privileged)` with raw
  tensor, but `_encode()` now expects TensorDict. Changed to `policy._encode(obs)`

### Removed
- `encoder/history_tcn.py`: Deleted (no longer imported by any module)

### Notes
- Memory budget: TCN backward ~960MB -> raw flatten input ~260MB (fits in RTX 4070)
- ProprioAdaptTConv (Phase 2) kept: runs in PPO minibatch mode, not TRPO full-batch
- Architecture now: hist(N,30,8) -> flatten(N,240) -> concat to MLP inputs directly
- No embedding, no normalization module for history -- MLPs learn from raw data

## [2026-03-17] Encoder z sweep analysis + history-augmented encoder design

### Context
Final constrained encoder run `2026-03-17_10-01-14` (2500 iters) achieved roll=5.46 deg,
pitch=6.74 deg -- a plateau comparable to TDC baseline (6 deg avg under hard DR). Gradient
ratio healthy (0.08-0.15), constraints mostly satisfied, training stable. But encoder z
analysis revealed the encoder is NOT learning useful DR representations.

Encoder z sweep analysis across 8 DR conditions:
- 10/13 z dimensions near-constant (std < 0.12)
- Cosine similarity = 0.9482 (z vectors nearly identical regardless of DR)
- Max Pearson correlation with physics params: |r| = 0.239 (very weak)
- added_mass_surge: no z dimension had |r| > 0.1

Root cause: privileged info alone (static hydrodynamic parameters) does not reveal dynamic
response characteristics. The encoder cannot distinguish DR conditions without observing
how the system responds to commands. The same physical parameters produce different
dynamics depending on operating point, coupling, and transient behavior.

Designed a new history-augmented encoder architecture that adds shared proprioception
history (TCN) to all modules. Key principle: proprioception history provides command-response
relationship (same action + different physics -> different angular velocity) that the encoder
needs to produce informative z.

### Added
- `docs/plans/2026-03-17-history-encoder-architecture.md`: Implementation plan for
  history-augmented encoder architecture. 13 tasks across 4 chunks.

### Notes
- New architecture:
  - Proprio History (30, 8D) -> shared HistoryTCN -> h_embed (32D)
  - Encoder: [policy(13D), h_embed(32D), privileged(19D)] -> z(13D)
  - Actor: [policy(13D), h_embed(32D), z(13D)] -> actions(2D) (no privileged)
  - Critics: [policy(13D), h_embed(32D), privileged(19D)] -> value/cost
- Actor alone cannot see privileged; accesses DR info only via z
- Phase 2 simplified: shared_tcn(frozen) + adapt_head(h_embed->z_hat) replaces full TCN
- No code changes in this session -- design and planning only
- TDC baseline comparison: classic controller achieves ~6 deg under hard DR, matching
  current RL+encoder (5.5-6.7 deg). Encoder contributes negligible improvement.

## [2026-03-17] Reward restructure + quadratic command reward

### Context
Analysis of 9-constraint run `2026-03-17_08-21-47` (762 iters) showed plateau at roll 8 deg,
pitch 10 deg. Four constraints OVER budget (attitude_err, singularity, yaw_vel, joint_osc).
joint_osc replaced with smoothness reward, PBRS removed, command_sigma tightened 0.35->0.20.

Noise floor 0.15 caused entropy collapse: run `08-48-06` (269 iters) entropy=-0.96, error +80%.
Reverted to 0.20. But even with floor=0.20, run `08-57-07` showed noise_std reaching floor by
iter 70 with entropy=-0.38 (COLLAPSED). Root cause: Laplacian reward `exp(-e/sigma)` with
sigma=0.20 creates gradient=5.0*exp(-e/0.20) that stays strong near zero, driving continuous
noise reduction with no counterforce (alpha_entropy=0).

Switched to quadratic command reward `r_c = -k_c*(roll_err^2 + pitch_err^2)` per reference
paper. Quadratic gradient = -2*k*error weakens near zero, providing natural entropy-friendly
structure. Policy stops compressing noise once error reduction slows. This matches the paper's
3-term reward design (command quadratic + torque penalty + smoothness penalty).

### Added
- `mdp/rewards.py`: `command_type` parameter in `command_reward()` supporting "quadratic"
  and "laplacian" modes. Quadratic: `-(roll_err^2 + pitch_err^2)`, uses per-axis error
  (not L2 norm). Laplacian: existing composite exp + linear ramp (unchanged).
- `mdp/rewards.py`: `command_type` field in `ALBCRewardCfg` (default: "laplacian" for
  backward compatibility with non-constrained envs)

### Changed
- `config.py`: Constrained encoder `command_type` set to "quadratic" (was implicit "laplacian")
- `config.py`: Removed `command_sigma=0.20` override (irrelevant for quadratic mode)
- `config.py`: `smoothness_weight` 0.0 -> -0.5, `progress_weight` 2.0 -> 0.0,
  `settling_weight` -> 0.0, joint_osc constraint removed (9->8 constraints)
- `agents/rsl_rl_ppo_cfg.py`: `num_constraints` 9->8, budgets updated
- `base_env.py`: Pass `reward_type` from config to `command_reward` function

### Fixed
- `runners/base_runner.py`, `algorithms/constraint_trpo.py`: Reverted noise floor 0.15 -> 0.20.
  0.15 caused entropy collapse within 269 iters. Even 0.20 hits floor by iter 70 with Laplacian
  sigma=0.20. Quadratic reward should resolve the underlying pressure.

### Notes
- Noise floor tested: 0.10 (immediate collapse), 0.15 (collapse in 269 iters), 0.20 (floor
  reached by iter 70 with Laplacian). Quadratic should allow noise_std to stay above floor.
- Quadratic gradient at 3deg: 0.52 vs Laplacian(sigma=0.20): 3.85. Weaker gradient is
  actually desirable: constraint system (attitude_err budget=7deg) handles fine control,
  reward provides coarse tracking signal.
- Other envs (Base, Encoder-Base, etc.) unaffected: default command_type="laplacian"
- Quadratic alone did NOT fix entropy collapse: run `09-02-53` still showed entropy=-0.38,
  noise_std=0.20 (floor) by early iterations. Root cause identified as `smoothness_weight=-0.5`:
  E[da^2] contains 2*sigma^2 term, so reducing noise directly reduces smoothness penalty.
  With alpha_entropy=0, this constant downward pressure is uncontested.
  Reduced smoothness_weight -0.5 -> -0.1 (1/5 pressure) to test hypothesis.

## [2026-03-17] Constraint reduction 8→6 + budget relaxation

### Context
Analysis of run `2026-03-17_09-10-27` (124 iters, smoothness=-0.1 + quadratic command)
showed 4/8 constraints simultaneously OVER budget: attitude_err (3.02x), yaw_vel (2.31x),
joint_torque (1.67x), singularity (1.35x). Excessive simultaneous constraint violations
cause cost gradient to dominate reward gradient, suppressing learning.

Two constraints identified as redundant:
- `singularity`: DLS IK already handles singularity via damping (no safety benefit in sim)
- `attitude_err`: duplicates quadratic command reward's tracking incentive (double-penalizing
  error reduction through both reward and constraint). Budget 0.122 rad (7 deg) too tight
  for early training, generating largest lambda and dominating policy gradient.

Remaining OVER constraints (joint_torque, yaw_vel) kept but budgets relaxed to reduce
initial constraint pressure while maintaining eventual compliance.

Noise floor kept at 0.20: with alpha_entropy=0, quadratic reward's E[-(e+noise)^2] = -(E[e^2] + sigma^2)
structurally drives noise to floor regardless of smoothness weight. Floor is the intended defense.

### Changed
- `config.py`: Disabled `singularity` constraint (DLS IK handles it mechanically)
- `config.py`: Disabled `attitude_err` constraint (quadratic command reward covers tracking)
- `config.py`: Relaxed `joint_torque` budget 0.05 -> 0.10 (was 1.67x OVER)
- `config.py`: Relaxed `yaw_vel` budget 0.15 -> 0.35 (was 2.31x OVER)
- `config.py`: Removed unused imports (`attitude_error_cost`, `singularity_cost`)
- `agents/rsl_rl_ppo_cfg.py`: `num_constraints` 8 -> 6, budgets synced to
  (0.02, 0.01, 0.10, 0.05, 0.10, 0.35) matching: accum_rot, attitude_abs,
  joint_torque, joint_vel_limit, overshoot, yaw_vel

### Notes
- Remaining 6 constraints: accum_rot, attitude_abs, joint_torque, joint_vel_limit,
  overshoot, yaw_vel
- Per-constraint advantage normalization handles cost_return scale differences
  (absolute scale irrelevant, violation ratio matters)
- With 2 fewer constraints, total constraint pressure reduced -- remaining OVER
  constraints should converge faster

## [2026-03-17] Lambda warmup extension + lr reduction

### Context
Analysis of run `2026-03-17_09-20-21` (6 constraints, post-reduction) revealed that
cost gradient overtakes reward gradient by iter 150+ (ratio cost/reward > 1.0). Direct
comparison of `Loss/grad_norm_reward` vs `Loss/grad_norm_cost` from TensorBoard confirmed:
  - iter 0-30: reward dominates (ratio 0.01), error drops rapidly
  - iter 60-120: gradients equalize (ratio 0.5-1.0), error plateaus
  - iter 150+: cost dominates (ratio 1.0-2.0), policy minimizes movement instead of error

Root cause: lambda warmup uses a linear ramp (not hard cutoff), so lambda grows from
iter 0. With `lambda_warmup_frac=0.3` (warmup_end=300) and `lr_lambda=0.01`, effective
lr at iter 150 was already 0.005 (50% of full). Combined with OVER-budget constraints
(joint_torque 1.29x, yaw_vel 1.11x), lambda grew fast enough to dominate by mid-training.

Applied both: longer warmup (more reward-dominant iterations) and slower lambda growth
(reduced ceiling on cost gradient pressure).

### Changed
- `agents/rsl_rl_ppo_cfg.py`: `lr_lambda` 0.01 -> 0.005 (halve lambda growth rate)
- `agents/rsl_rl_ppo_cfg.py`: `lambda_warmup_frac` 0.3 -> 0.5 (warmup 300 -> 500 iters)

### Notes
- At iter 150 now: eff_lr = 0.005 * (150/500) = 0.0015 (was 0.005, 3.3x reduction)
- Full lr reached at iter 500 instead of 300, giving 200 more reward-dominant iters
- Lambda values at iter 236 in previous run: joint_torque=0.59, joint_vel_limit=0.35,
  yaw_vel=0.27. These should grow ~3x slower with new settings.

## [2026-03-17] Constraint expansion 3→9 + PBRS progress reward

### Context
With alpha_entropy=0, noise_floor=0.2, lambda_warmup=0.3 stabilized (previous session),
expanded constraints for behavioral quality improvement. The prior 8-constraint failure
was caused by target_entropy + continuous constraint interaction (now resolved: alpha=0 +
noise floor). Added PBRS progress reward to accelerate rise time (replacing settling reward).

### Added
- `mdp/constraints.py`: 4 new cost functions:
  - `joint_torque_cost` (alias of effort_limit_cost, clearer name)
  - `joint_velocity_limit_cost` (binary: joint_vel > 4.189 rad/s = 40 RPM)
  - `overshoot_cost` (binary: error sign flip + magnitude > 2 deg threshold)
  - `attitude_error_cost` (continuous: reuses env._potentials L2 norm)
- `mdp/rewards.py`: `progress_reward` PBRS function (prev_potential - gamma * potential),
  `ALBCRewardCfg.progress_weight` and `progress_gamma` fields
- `base_env.py`: `_prev_attitude_error_rp` buffer for overshoot detection (initialized to
  initial error in `_reset_task_and_state` to prevent false positives on first step)

### Changed
- `config.py`: `HeroAgentConstrainedEncoderEnvCfg.constraints` expanded from 3 to 9 terms:
  binary(6): accum_rot(0.02), attitude_abs(0.01), singularity(0.15),
  joint_torque(0.05), joint_vel_limit(0.05), overshoot(0.10);
  continuous(3): attitude_err(0.122=7deg), joint_osc(0.30), yaw_vel(0.15)
- `config.py`: Reward updated: settling_weight=0.0 (replaced by attitude_err constraint),
  progress_weight=2.0 (PBRS, scale_by_dt=False)
- `agents/rsl_rl_ppo_cfg.py`: `num_constraints` 3→9, `constraint_budgets` synced to 9-tuple
  in both `RslRlConstraintTRPOAlgorithmCfg` and `RslRlPpoActorCriticEncoderConstrainedCfg`
- `base_env.py`: `_build_reward_terms()` adds progress term when weight != 0;
  `_get_rewards()` updates overshoot buffer after constraint computation
- `mdp/__init__.py`: Exports updated for all new functions

### Notes
- PBRS is theoretically safe (Ng et al. 1999): does not change optimal policy
- attitude_err budget=0.122 rad (7 deg) is moderately strict; if lambda saturates, relax to 10 deg
- Overshoot false positive prevention: `_prev_attitude_error_rp` set to initial error at reset
- joint_torque_cost is a pure alias of effort_limit_cost (no code duplication)

## [2026-03-17] 3-constraint Lagrangian baseline + disable entropy bonus for TRPO

### Context
Previous run `2026-03-16_15-09-42` (999 iters, Lagrangian, 8 constraints, target_entropy=2.0)
failed with 17-20 deg attitude error, noise_std stuck at 1.0. Compared with successful run
`2026-03-06_18-26-36` (IPO, 3 constraints, entropy_coef=0.005): 3.7 deg error, noise_std 0.2.

**Root cause 1 (8→3 constraints)**: 5 continuous constraints (joint_vel, oscillation, yaw_vel,
cob_cog, effort_limit) produce costs proportional to noise_std. As noise increased, continuous
costs grew, lambda grew, cost gradient dominated reward gradient, creating a vicious cycle.
The 3/6 run used only 3 binary constraints (noise-insensitive).

**Root cause 2 (target_entropy)**: SAC-style alpha kept noise_std at 1.0, preventing the
natural reward-driven noise reduction that the 3/6 run exhibited (converged to 0.2).

**Fix applied**: Reduced to 3 binary constraints (accum_rot, attitude_abs, singularity) +
fixed alpha_entropy_init=0.005. Restored velocity_limit_sim=6.28 (was 4.19).

**Run `2026-03-17_07-15-39` (227 iters)**: Error improved to 5-8 deg (good!), reward peaked
at 69.2 (iter 150). BUT noise_std grew unboundedly: 1.02→4.45, entropy 2.84→5.75. Reward
started declining after iter 150.

**Root cause 3 (entropy bonus in TRPO)**: With all constraints satisfied (lambda=0), the
fixed alpha=0.005 entropy bonus has no counterbalancing force. PPO has clip ratio + adaptive
LR to resist noise growth; TRPO takes max-KL steps every iteration, so any alpha > 0
consistently pushes noise_std up. TRPO's KL constraint alone provides sufficient exploration.

**Fix**: Set alpha_entropy_init=0.0. Also fixed math.log(0) crash in constraint_trpo.py
(added guard: log(max(init, 1e-8))).

### Changed
- `config.py`: Reduced `HeroAgentConstrainedEncoderEnvCfg.constraints.terms` from 8 to 3
  (kept: accum_rot budget=0.02, attitude_abs budget=0.01, singularity budget=0.15;
  removed: effort_limit, joint_vel, oscillation, yaw_vel, cob_cog)
- `agents/rsl_rl_ppo_cfg.py`: `num_constraints` 8→3, `constraint_budgets` updated to
  (0.02, 0.01, 0.15), `alpha_entropy_lr` 0.01→0.0, `alpha_entropy_init` 0.005→0.0
  (TRPO KL constraint provides exploration; entropy bonus causes unbounded noise growth)
- `agents/rsl_rl_ppo_cfg.py`: `RslRlPpoActorCriticEncoderConstrainedCfg.num_constraints` 8→3
- `hero_agent.py`: `velocity_limit_sim` 4.19→6.28 rad/s (restored 3/6 value)

### Fixed
- `algorithms/constraint_trpo.py`: `math.log(alpha_entropy_init)` crashes when init=0.0.
  Added guard: `log(max(init, 1e-8))` so alpha initializes to ~1e-8 (effectively zero).

### Added
- `docs/plans/2026-03-17-lagrangian-baseline-3constraint.md`: Design document for experiment

### Notes
- All Lagrangian code improvements retained: std detach, reward adv normalization,
  lambda warmup, d_k normalization, LS-gated updates, asymmetric critic, z detach from cost
- Key insight: entropy bonus interacts fundamentally differently with TRPO vs PPO. In PPO,
  clip + adaptive LR naturally resist noise growth. In TRPO, max_kl step has no such mechanism.

## [2026-03-17] Raise noise floor 0.1 -> 0.2 + alpha=0 run analysis

### Context
Run `2026-03-17_07-25-13` (alpha=0, 3 constraints, 454 iters) confirmed noise_std fix works:
noise decreased naturally 1.0 -> 0.17, reward 60-66, roll 4-7 deg, pitch 5-9 deg.
All constraints satisfied (lambda=0), line search 100%.

**However, noise_std kept falling without bound** (0.17 at iter 454, still declining).
Entropy went negative (-0.73) -- exploration effectively dead. Reward plateaued at 60-66
(vs 74.6 in 3/6 run) because policy stopped exploring for better strategies.

TRPO entropy bonus dilemma:
- alpha=0.005: noise grows unboundedly (1.0 -> 7.44 in 316 iters)
- alpha=0.0: noise shrinks to floor (1.0 -> 0.17 in 454 iters)

Simplest fix: raise noise floor from 0.1 to 0.2. This matches the 3/6 run's converged
noise_std (0.20-0.24) and guarantees minimum exploration without any entropy bonus tuning.
The floor is a hard clamp -- no interaction with TRPO step dynamics.

### Changed
- `runners/base_runner.py`: `min_std` 0.1 -> 0.2 in `_apply_noise_floor()` -- ensures
  exploration persists at convergence, matching 3/6 run's natural noise level
- `algorithms/constraint_trpo.py`: `min_log_std` from `log(0.1)` to `log(0.2)` -- unified
  with base_runner floor

### Notes
- Next step: verify noise stabilizes at 0.2, then add constraints back
- The 3/6 run (IPO, entropy_coef=0.005) had implicit exploration from barrier pressure;
  Lagrangian with lambda=0 has no equivalent, so the floor is essential

## [2026-03-16 Summary] Lagrangian migration + entropy stabilization (10 sessions)

### Context
Intensive debugging day (10 sessions) focused on migrating constraint enforcement from
IPO log-barrier to Lagrangian primal-dual, and fixing the entropy collapse/explosion cycle.

**Core problem**: IPO log-barrier assumed feasible start, but random policy starts infeasible.
Barrier gradient's easiest path was reducing action variance (kills exploration). Multiple
fixes attempted (barrier_t tuning, noise floors/ceilings, adaptive thresholds, n_active
normalization) were all band-aids. Fundamental fix: Lagrangian primal-dual where lambda
starts at 0 and grows with violations.

**Entropy debugging arc**: (1) Initial 6-bug fix session -> entropy still collapsed.
(2) barrier_t 10->50 + n_active normalization + entropy_coef 0.04 -> explosion (std=1678).
(3) Revert entropy_coef to 0.02 + add noise ceiling -> balanced (std=0.48, best run).
(4) Replace IPO with Lagrangian -> entropy collapsed via std cost gradient path.
(5) Detach std from cost gradient -> entropy locked at ceiling (entropy_coef push only).
(6) Remove entropy_coef entirely -> natural reward-driven std decrease. (7) Lambda warmup
+ d_k^2 cost value normalization + budget recalibration from unconstrained baselines.
(8) Reward advantage normalization + LS-gated lambda/encoder updates. (9) SAC-style target
entropy -> nearly worked but alpha_lr too slow initially. (10) Encoder z detach from cost
surrogate -> LS 100% success, BUT error stuck at 17-20 deg (effort_limit budget too tight).

### Changed
- `algorithms/constraint_trpo.py`: Replaced IPO log-barrier with Lagrangian primal-dual
  (lambda_k dual variables, lr_lambda=0.01, lambda_max=20, lambda_warmup_frac=0.3).
  Detached std from cost gradient (`_log_prob_mean_only()`). Detached encoder z from cost
  surrogate. Added reward advantage normalization. LS-gated lambda/encoder/z_bounds updates.
  d_k-normalized dual update. d_k^2-normalized cost value loss. SAC-style alpha_entropy
  (target_entropy, alpha_entropy_lr, alpha_entropy_init). Removed: barrier params, adaptive
  threshold, n_active normalization, line_search_cost_margin, encoder_value_grad_scale.
- `agents/rsl_rl_ppo_cfg.py`: Replaced barrier params with Lagrangian params (lr_lambda,
  lambda_max, lambda_init, lambda_warmup_frac). Constraint budgets recalibrated from
  unconstrained baseline (oscillation 0.3->0.4, yaw_vel 0.15->0.4). entropy_coef replaced
  by target_entropy/alpha_entropy_lr/alpha_entropy_init (final: all 0.0).
- `runners/constraint_encoder_runner.py`: Lambda checkpoint persistence (lambda_state.pt).
  Logging: barrier metrics -> lambda/violation metrics. log_alpha state in checkpoints.
- `config.py`: energy_weight=0.0, smoothness_weight=0.0 (double-counting with constraints).
  effort_limit budget 0.05->0.25. Removed joint_effort_limit_range override.
- `mdp/constraints.py`: effort_limit_cost uses per-env DR'd limits. Removed dead fields
  (barrier_t, barrier_t_final, barrier_t_schedule_frac, adaptive_threshold_alpha).
- `runners/base_runner.py`: Noise floor 0.15->0.25->0.1, ceiling added then removed.

### Fixed
- `algorithms/constraint_trpo.py`: Cost value loss clamps targets >=0. NaN guard on cost
  advantage normalization. Log_std clamp before KL measurement.
- `mdp/constraints.py`: effort_limit_cost uses per-env DR'd limits (was cached scalar).

### Notes
- Key architectural decisions: (1) std detach from cost = permanent fix for entropy collapse,
  (2) Lagrangian vs IPO = handles infeasible start, (3) LS-gated updates = prevents death
  spiral during line search failures, (4) z detach from cost = prevents encoder instability
- Best run of the day: `04-27-48` (reward 42, roll 11.6 deg, pitch 13.4 deg, std=0.48)
  but that used 8 constraints which were later reduced to 3 for stability

## [2026-03-05/10 Summary] ConstraintTRPO build-out, stabilization, NORBC conformance

### Context
Built NORBC-style constrained RL (IPO -> Lagrangian TRPO) from scratch. Key milestones:
initial 3-constraint IPO implementation (3/5-6), constraint expansion to 8 terms with
cost critic softplus fix (3/8), NORBC conformance audit with asymmetric critic and actuator
DR unification (3/9), barrier_t fix 10->50/100 for arm freeze (3/9), noise floor unification
0.1/0.15 -> 0.25 + effort_limit DR conflict removal (3/10).

### Changed
- `algorithms/constraint_trpo.py`: Full TRPO + IPO (~600 lines). barrier_t tuning (1->10->50->10, tested range). min_log_std unified to log(0.25). Update order, per-constraint cost advantage standardization.
- `encoder/actor_critic_encoder.py`: Asymmetric critic (raw privileged 32D input)
- `encoder/actor_critic_encoder_constrained.py`: Multi-head cost critic, F.softplus() output, K mismatch detection
- `runners/constraint_encoder_runner.py`: Named constraint logging, auto-sync K, lambda checkpoint
- `runners/base_runner.py`: min_std 0.15->0.25 (unified with constraint_trpo floor)
- `mdp/constraints.py`: 3->8 cost functions, effort_limit uses per-env DR'd limits
- `config.py`: 8-term constraints, unified actuator DR (Kp 40-120, Kd 0.5-5.0, effort 0.7-1.0). Removed joint_effort_limit_range override.
- `hero_agent.py`: velocity_limit_sim 6.28->4.19 rad/s (datasheet 40 rpm)
- `base_env.py`: Accumulated rotation tracking, cost computation, constraint buffers
