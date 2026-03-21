# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

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

## [2026-03-20] C-TRPO training stability: entropy bonus, encoder KL gating, yaw_vel budget, surrogate logging

### Context
Analysis of run `2026-03-20_14-01-34` (1000+ iter) identified 4 linked stability issues:
(1) Noise std monotonically decayed 1.0->0.15 because TRPO has no entropy bonus -- the natural
gradient always reduces std (tighter distribution = higher expected reward for current mean).
This caused roll/pitch "axis alternation" (one axis converges, other collapses). (2) Encoder
updates change z (actor input), shifting the conditional distribution `pi(a|s,z)` beyond the
trust region. Logged KL reached 0.02-0.10 (2-10x the 0.015 target), with step 375-384 KL
spike directly draining yaw_vel margin. (3) yaw_vel budget=0.35 (d_k=35, ~20 deg/s) too tight
for passive yaw dynamics. (4) surrogate loss not logged, hindering gradient debugging.

### Changed
- `algorithms/constraint_trpo.py`: Fix 1 -- entropy bonus `-entropy_coef * H(pi)` added to
  `surrogate()` closure. Counteracts TRPO's inherent std-reduction bias. `entropy_coef=0.005`
  (matching RSL-RL PPO default). Only in surrogate, not encoder update (log_std not encoder param).
- `algorithms/constraint_trpo.py`: Fix 2 -- Pre-encoder KL measured after noise floor clamp.
  `_update_encoder()` gains KL gating: after each encoder step, if KL exceeds
  `pre_encoder_kl + max_encoder_kl` (default 0.016), encoder params are reverted and epoch stops.
  Prevents encoder-induced distribution shift from violating trust region.
- `algorithms/constraint_trpo.py`: Fix 4 -- `_trpo_step()` caches `loss.item()` for surrogate
  logging. Added `entropy`, `surrogate`, `pre_encoder_kl` to `loss_dict` (auto-logged as
  `Loss/entropy`, `Loss/surrogate`, `Loss/pre_encoder_kl` by OnPolicyRunner).
- `config.py`: Fix 3 -- yaw_vel budget 0.35 -> 0.785 (d_k=78.5, ~45 deg/s).
- `agents/rsl_rl_ppo_cfg.py`: Added `entropy_coef=0.005` and `max_encoder_kl=0.016` to
  `RslRlConstraintTRPOAlgorithmCfg`.
- `runners/constraint_encoder_runner.py`: Added `Policy/entropy` and `Policy/pre_encoder_kl`
  to constraint metrics logging.

## [2026-03-20] C-TRPO per-constraint blend: fix recovery damage to attitude performance

### Context
Post-B1/B2 analysis of run 2026-03-20_13-21-52 (571 iter) revealed that while phantom mode
oscillation was resolved, recovery mode caused irreversible step-wise attitude degradation.
When yaw_vel triggered recovery, ALL reward optimization halted (reward gradient = 0) for
44+ iterations. Since yaw_vel and attitude control are orthogonal, cost reduction during
recovery provided zero benefit to attitude -- but removing reward gradient caused pitch error
to jump from 5-7 deg to 13-18 deg in staircase pattern, never fully recovering.

Additionally, encoder was updated with reward-only objective during recovery while actor used
cost-only -- directional mismatch caused z drift (z_std +39%), z_bounds_loss 15x spike, and
KL explosion to 0.13 (vs normal 0.01) from encoder-driven distribution shift.

Solution: per-constraint blend replaces binary if/else with unified surrogate
`reward + barrier(safe) + cost(violated)`. Reward gradient never turns off. Line search
gains cost non-regression check to prevent reward improvement from masking cost worsening.
Encoder objective aligned with actor (includes recovery cost when in blend mode).

### Changed
- `algorithms/constraint_trpo.py`: Replaced binary `if any_recovery / else` (lines 599-623)
  with unified surrogate: `reward_surr + barrier_penalty + recovery_cost`. Reward always active;
  recovery constraints add cost minimization gradient without killing reward optimization.
- `algorithms/constraint_trpo.py`: `_line_search()` gains `recovery_cost_fn` parameter. When
  provided, steps where recovery cost worsens (new_rc > old_rc + 1e-6) are rejected, preventing
  reward improvement from masking cost regression in the total surrogate.
- `algorithms/constraint_trpo.py`: `_trpo_step()` passes `recovery_cost_fn` through to
  `_line_search()`. Mode name changes from "recovery" to "blend".
- `algorithms/constraint_trpo.py`: `_update_encoder()` gains `recovery_mask` and
  `cost_advantages_flat` parameters. During blend mode, encoder objective includes recovery cost
  term, aligning encoder and actor gradient directions (fixes z drift root cause).

### Notes
- `_compute_barrier_penalty()` unchanged: already uses `safe_mask = (margins>0) & ~recovery`
- `_compute_margins()`, `_in_recovery` hysteresis logic preserved
- Noise floor (0.01) unchanged -- already fixed in prior session
- No new config parameters needed
- Verification criteria: pitch error recovery < 2 deg after blend episodes, KL < 0.05,
  z_std expansion < 10% during blend, cost_return_yaw_vel decreasing during blend

## [2026-03-20] 5-7 deg convergence plateau fix: Laplacian reward + noise floor + overshoot relaxation

### Context
`Isaac-Constrained-ALBC-Encoder-v0` training plateaued at 5-7 deg attitude error (target: 3 deg).
Root cause analysis identified 3 primary causes working together:

1. **Quadratic reward gradient dies near zero** (CRITICAL): gradient = -2e, so at 5 deg the
   reward improvement for 5->3 deg is only 0.000243/step -- barely above smoothness penalty noise.
2. **Action noise floor std=0.25 blocks precision** (CRITICAL): `min_log_std = log(0.25)` gave
   1.2 deg/step noise (40% of 3 deg target). Policy reached the floor at ~100 iterations and
   couldn't reduce std further despite 2400 more iterations of training.
3. **Overshoot constraint too tight for fine convergence** (HIGH): threshold=0.035 rad (2 deg)
   triggered recovery mode during normal fine-correction oscillation at 5-7 deg, halting all
   reward optimization.

Implemented Phase 1 (Exp 3A + 1A) and Phase 2 (Exp 2) simultaneously:
- Laplacian reward: gradient = (1/sigma)*exp(-|e|/sigma), INCREASES near zero (4.71/rad at 3 deg
  vs quadratic's 0.10/rad). Per-axis kernel with sigma=0.15.
- Noise floor 0.25 -> 0.01: per-step noise drops from 1.2 deg to 0.048 deg. KL constraint
  (max_kl=0.01) already prevents std collapse, making the floor redundant.
- Overshoot threshold 2 deg -> 5 deg, budget 0.10 -> 0.20: allows fine correction without
  triggering recovery mode.

Phase 3 (perturbation torque 0.4->0.2 Nm) and Phase 4 (observation noise halving) deferred --
apply only if Phase 1+2 insufficient.

### Changed
- `mdp/rewards.py`: `command_reward()` accepts `command_type` ("quadratic" or "laplacian") and
  `sigma` params. Laplacian mode: `exp(-|e_i|/sigma)` per axis, summed. `ALBCRewardCfg` gained
  `command_type` and `command_sigma` fields (defaults: "quadratic", 0.15).
- `config.py`: Reward default changed to `command_type="laplacian"`, `command_sigma=0.15`.
  Overshoot constraint: threshold 0.035 -> 0.087 rad (~5 deg), budget 0.10 -> 0.20.
- `algorithms/constraint_trpo.py`: `min_log_std` changed from `log(0.25)` to `log(0.01)`.
  std floor from 0.25 to 0.01 (per-step noise 1.2 deg -> 0.048 deg).
- `albc_env.py`: `_build_reward_terms()` passes `command_type` and `sigma` from config to
  `command_reward` via `RewardTermCfg.params`.

### Notes
- Deferred experiments (config-only changes, no code needed):
  - Exp 4: `perturbation_torque_range (0.0, 0.2)` -- if 3 deg infeasible under max perturbation
  - Exp 5: observation noise std halved -- if SNR too low at 3 deg (sim2real impact to consider)
- Verification: `Episode/attitude_error_mean < 0.052 rad`, `Policy/action_std` freely decreasing
  below 0.25, `Constraint/mode` safe(0) ratio > 70%.

## [2026-03-20] Remove constrained_encoder_base code from hero_agent

### Context
All constrained RL code has been fully migrated to `constrained_albc/` package.
This session removes the remaining constrained_encoder_base references from
hero_agent to eliminate dead imports and prevent runtime confusion between the
two packages. 4 files deleted, 10 files edited, ruff check clean.

### Changed
- `base_env.py`: Removed `compute_all_costs` import, constraint cost computation block,
  `_prev_attitude_error_rp` buffer (init + 3 reset/update sites), `_check_dr_infeasibility()`
  method and its call site, `log_dr_infeasibility` import.
- `config.py`: Removed constraint imports (ALBCConstraintCfg, ConstraintTermCfg, 6 cost functions).
  Deleted `HeroAgentConstrainedEncoderEnvCfg` class (~70 lines).
- `agents/rsl_rl_ppo_cfg.py`: Removed ConstraintTRPO/ConstraintEncoderRunner/
  ActorCriticEncoderConstrained imports and module registrations. Deleted 3 config classes
  (RslRlConstraintTRPOAlgorithmCfg, RslRlPpoActorCriticEncoderConstrainedCfg,
  HeroAgentConstrainedEncoderRunnerCfg, ~95 lines).
- `__init__.py`: Removed gym.register for Isaac-HeroAgent-Constrained-Encoder-Base-v0,
  config import, and __all__ entry.
- `encoder/__init__.py`, `agents/__init__.py`, `runners/__init__.py`,
  `algorithms/__init__.py`, `mdp/__init__.py`: Removed all constrained-related imports/exports.
- `utils/logging.py`: Deleted `log_dr_infeasibility()` and `_get_dr_infeasibility_logger()`.
- `utils/__init__.py`: Removed `log_dr_infeasibility` from imports and __all__.
- `runners/base_runner.py`: Removed ConstraintTRPO reference in docstring comment.

### Removed
- `encoder/actor_critic_encoder_constrained.py`: Constrained encoder network (4.5KB)
- `algorithms/constraint_trpo.py`: C-TRPO algorithm (42KB)
- `runners/constraint_encoder_runner.py`: Constrained encoder runner (6.4KB)
- `mdp/constraints.py`: Constraint cost functions (12.5KB)

## [2026-03-20] C-TRPO mode oscillation fix: EMA smoothing + cost critic LR gating

### Context
C-TRPO training exhibited rapid mode oscillation (loss/mode flipping 0<->1 every
5-10 iterations in later training). Root cause analysis identified 5 layers:
- RC1 (HIGH): Cost critic decoupling -- `_update_values()` runs 20 gradient steps
  regardless of actor update success. When actor is frozen (ls_success=False), cost
  critic drifts, changing margin without policy change -> "phantom" mode switches.
- RC2 (HIGH): Weak barrier beta=0.01 -- barrier penalty negligible until margin < 1.
- RC3 (MEDIUM): Hard binary mode switch with no continuous interpolation.
- RC4 (MEDIUM): Narrow hysteresis band (0.8) allowing rapid safe<->recovery cycling.
- RC5 (LOW): Binary constraint volatility (minor with 4096-env averaging).

Implemented Approach B (B1 + B2, ~30 lines) targeting RC1, plus Approach A tuning
targeting RC2 + RC4. Approach C (soft mode transition, ~120 lines) reserved as
follow-up if oscillation persists.

### Changed
- `algorithms/constraint_trpo.py`: B1 -- EMA smoothing (alpha=0.3) on mean_cost_returns
  before margin computation. `_compute_margins()` now receives smoothed values instead
  of raw per-iteration cost returns. ~3-iteration lag, sufficient for real violation
  detection while filtering single-iteration cost value jumps.
- `algorithms/constraint_trpo.py`: B2 -- Cost critic LR gated on actor update success.
  When `ls_success=False`, value optimizer LR reduced to 10% of base LR. Prevents
  cost critic from drifting while actor is frozen, eliminating the primary source of
  phantom mode switches. LR restored after `_update_values()` completes.
- `agents/rsl_rl_ppo_cfg.py`: Added `ema_cost_alpha=0.3` parameter. Updated `beta`
  default 0.01->0.05 (5x barrier strengthening). Updated `recovery_threshold_frac`
  default 0.8->0.6 (wider hysteresis band).
- `runners/constraint_encoder_runner.py`: EMA state (ema_cost_returns, ema_initialized)
  persisted in barrier_state.pt checkpoint. Backward-compatible with old checkpoints
  (missing EMA keys handled gracefully). Added per-constraint `ema_cost_return` metric
  to WandB/TensorBoard logging for monitoring smoothed vs raw cost returns.

### Notes
- B3 (adaptive beta based on min margin) is designed but not implemented -- add if
  beta=0.05 proves insufficient after B1+B2 stabilize mode switching.
- Approach C (soft mode transition with sigmoid alpha_k blending) is the structural
  solution if oscillation fundamentally persists. ~120 lines, replaces binary
  safe/recovery with continuous interpolation. Reserved as follow-up.
- Verification: run with same config, check Loss/mode switch period > 30 iters in
  step 100-300 range, Constraint/cost_return stable near d_k, Policy/line_search_success > 70%.

## [2026-03-20] Remove PBRS progress reward (redundant with quadratic command)

### Context
Analysis of reward structure revealed that `progress_reward` (PBRS: prev_potential -
gamma * potential) is redundant with `command_reward` (quadratic: -(roll_err^2 + pitch_err^2)).
Both derive from the same attitude_error variable, producing identical gradient directions.

Key findings from mathematical analysis:
- PBRS telescopes to initial minus discounted final error (path-independent),
  while command integrates error over entire trajectory (path-dependent).
  However, for stabilization tasks with monotonic error decrease, both produce
  mirror-image WandB curves with no independent learning signal.
- PBRS theorem (Ng et al. 1999) guarantees progress does not change optimal policy.
- Progress was NOT dt-scaled, making it ~2x stronger per step than command --
  effectively dominating the reward signal while adding no new information.
- C-TRPO uses full-batch natural gradient, so the "early training value function
  noise" argument for PBRS is weaker than in PPO.

Dry run verified: 2 iterations, 4 envs, headless -- clean execution with only
command + smoothness reward terms logged.

### Removed
- `mdp/rewards.py`: Deleted `progress_reward()` function and `progress_weight`/`progress_gamma`
  fields from `ALBCRewardCfg`. Reward architecture simplified from 3-term to 2-term.
- `albc_env.py`: Removed `_potentials` and `_prev_potentials` buffers (only used by progress).
  Renamed `_update_potentials()` to `_update_attitude_error()` (simpler, reflects actual purpose).
  Removed progress term construction in `_build_reward_terms()`.
  Removed potential initialization/reset in `_reset_task_and_state()`.
- `config.py`: Removed `progress_weight=2.0` from `ALBCRewardCfg` instantiation.
- `mdp/__init__.py`: Removed `progress_reward` from imports and `__all__`.

## [2026-03-20] config.py code review: dead fields, barrier singularity, documentation

### Context
Post-simplification code review of `config.py` plus cross-referenced issues in
`constraint_trpo.py`. 8 verified issues found (3 exploration agents, false positives
filtered). Issues 1-4 in config.py (dead field, undocumented budget scaling, magic numbers,
missing validation); Issues 5-8 in constraint_trpo.py (barrier near-singularity, any-recovery
design, kwargs absorption, dead clamp guard). Issue 4 (DR range validation) skipped as low ROI.

Key finding: barrier penalty has a singularity gap -- when margin is small positive (0, ~0.01)
but recovery mode hasn't triggered (threshold is margin <= 0), phi_pp = 1/m^2 can reach 1e6,
causing explosive gradients. Clamping margin to min=0.01 caps phi_pp at 1e4 (barrier penalty
~100 with beta=0.01 and cost_surrogate=0.1).

### Changed
- `config.py`: Added budget D_k vs d_k documentation. Per-step budget D_k is scaled to
  discounted d_k = D_k / (1 - cost_gamma) = D_k * 100 by the algorithm. This relationship
  was undocumented, making budget tuning non-obvious.
- `config.py`: Added inline unit comments for constraint magic numbers: `1.396` -> `# ~80 deg`,
  `4.189` -> `# 40 RPM (Dynamixel XW540 no-load)`.
- `algorithms/constraint_trpo.py`: Added design note comment on "any-recovery" policy trade-off
  at lines 569-578. Documents that per-constraint blend is a known alternative but adds
  complexity with shared trust region interaction.

### Fixed
- `algorithms/constraint_trpo.py`: Barrier margin clamped to min=0.01 in
  `_compute_barrier_penalty()`. Prevents phi_pp explosion when margin is small positive but
  recovery hasn't triggered. Old: `1/(m^2 + 1e-8)` -> New: `1/max(m, 0.01)^2`.
- `algorithms/constraint_trpo.py`: `**_kwargs` now logs ignored kwargs at debug level instead
  of silently absorbing. Aids diagnosis when RSL-RL passes unexpected parameters.
- `algorithms/constraint_trpo.py`: Added comment explaining d_k^2 clamp (min=0.01) is a
  defensive guard that never activates with default cost_gamma=0.99 (min d_k=1.0).

### Removed
- `config.py`: Deleted dead `enable_payload: bool = True` field. Payload is always initialized
  and computed unconditionally since the simplification removed its conditional logic.

## [2026-03-20] MDP code review: 3 critical bugs + 5 theoretical fixes

### Context
Systematic code review of Constrained ALBC MDP modules (rewards.py, constraints.py,
observations.py, events.py, albc_env.py) using 3 parallel code-explorer agents.
Identified 3 critical bugs, 7 theoretical issues, 7 design items, and 6 minor items.
Fixed all critical bugs and 5 of 7 theoretical issues in this session.

BUG-1 root cause: `_update_action_buffers()` stored `self._actions` (current step a_t)
into `_prev_actions_obs` instead of `self._prev_actions` (a_{t-1}). This caused a causal
violation -- policy obs[11:13] contained the current action, while encoder proprio history
correctly used the previous action. Temporal inconsistency between policy and encoder.

BUG-2: `effort_limit_cost` compared `max(torques)` against `max(limits)` instead of
per-joint comparison. When joints have different DR'd limits, a violation on the weaker
joint could be masked by the stronger joint's higher limit.

BUG-3: Joint gain/friction randomization ran unconditionally even with `rand_cfg.enable=False`
(debug/eval mode). Comment claimed "ranges collapse to defaults" but actual DR ranges were
wide (Kp 40-120, Kd 0.5-5.0), so debug envs had randomized actuator properties.

### Fixed
- `albc_env.py`: BUG-1 -- `_prev_actions_obs` now stores `_prev_actions` (a_{t-1}) instead
  of `_actions` (a_t). Both sliced and full-clone paths corrected.
- `mdp/constraints.py`: BUG-2 -- `effort_limit_cost` uses per-joint comparison
  `(computed.abs() > limits).any(dim=-1)` instead of `max(dim=-1)` reduction on both sides.
- `albc_env.py`: BUG-3 -- Joint actuator DR (gains, effort limits, friction) wrapped in
  `if rand_cfg.enable:` guard. Debug/eval envs now keep default actuator properties.
- `mdp/constraints.py`: THEO-1 -- `overshoot_cost` checks `prev.abs() > threshold` (departure
  magnitude) instead of `curr.abs() > threshold` (landing magnitude). Catches small overshoots
  where zero crossing lands below threshold (e.g., prev=+0.04rad -> curr=-0.01rad).

### Changed
- `albc_env.py`: THEO-3 -- `_get_attitude_error()` returns cached `self._attitude_error` instead
  of recomputing. Safe because `_get_rewards()` -> `_update_potentials()` always runs first in
  Isaac Lab's step order (line 393 before 410 in direct_rl_env.py). Eliminates duplicate
  `compute_attitude_error()` call per step.
- `mdp/events.py`: THEO-6 -- Payload restoring moment clamp uses horizontal (xy) norm instead
  of 3D norm. Roll/pitch restoring moment depends only on horizontal offset; Z-component payload
  was being over-constrained.
- `mdp/events.py`: THEO-7 -- `_HydroBaseCache.inertia` fallback (0.5 * added_mass[3:6]) now
  emits `logger.warning()` when `rigid_body_inertia` is None. Added `import logging`.

### Removed
- `mdp/constraints.py`: THEO-2 -- Removed dead `cost_type` field from `ConstraintTermCfg`.
  Never read by `compute_all_costs()` or `constraint_trpo.py`. Removed `cost_type="average"`
  from `yaw_velocity_cost` term in `config.py`. Updated module docstring.

### Notes
- THEO-4 (PBRS L2 norm vs quadratic gradient mismatch): Not fixed -- reward landscape change
  would require full retraining. Documented only.
- THEO-5 (yaw in obs but not in reward): Not fixed -- obs dimension change affects encoder/actor
  architecture. Separate task.
- DES-1~7 and M-1~6: Out of scope (no functional impact). Documented in review plan.

## [2026-03-20] Full package code review: encoder optimizer resume + NaN guard

### Context
Systematic code review of entire constrained_albc package (6 review areas: agents,
encoder, algorithms, mdp, config+env, runners+utils). Found 1 confirmed bug
(encoder optimizer state not saved on checkpoint), 2 theoretical concerns (recovery
drops reward globally, standardization-barrier inverse variance), and 2 design issues
(unused cost_type field, missing isfinite guard on encoder loss).

BUG-1 root cause: `self.optimizer = self.value_optimizer` alias (constraint_trpo.py:210)
means OnPolicyRunner.save() only persists value_optimizer state_dict. The separate
`encoder_optimizer` (Adam with lr=3e-4, wd=1e-5) loses momentum (exp_avg, exp_avg_sq)
on resume, causing a transient gradient magnitude spike as Adam re-estimates statistics.

### Fixed
- `runners/constraint_encoder_runner.py`: Save/load `encoder_optimizer.pt` alongside
  `barrier_state.pt` in checkpoint. Uses existing `_save_aux_state`/`_load_aux_state`
  helpers. Load respects `load_optimizer` flag (skip during eval/play).
- `algorithms/constraint_trpo.py`: Added `torch.isfinite(total_loss)` guard before
  `.backward()` in `_update_encoder()`. NaN/Inf loss skips the epoch with warning
  instead of corrupting encoder parameters irreversibly.

### Notes
- THEORY-1 (recovery drops reward surrogate globally): conservative valid choice per
  C-TRPO paper (Muller et al. Sec 4.1). Monitor reward stalls during recovery.
- THEORY-2 (standardization-barrier inverse variance): tight constraints get stronger
  barrier (arguably correct). Undocumented interaction, monitor per-constraint magnitude.
- DESIGN-1 (cost_type field in ConstraintTermCfg): dead field, never consumed by
  algorithm. Deferred cleanup.

## [2026-03-20] Constrained ALBC encoder code review: DRY, perf, backward compat

### Context
Code review of constrained_albc encoder directory (`actor_critic_encoder.py`,
`actor_critic_encoder_constrained.py`) after the simplification session. Found 7 issues
(5 required code changes, 2 already fixed). Key findings: (1) `_encode()` defined but
never called -- `_get_combined_obs()` duplicated its logic inline (DRY violation from
simplification that inlined `_build_encoder_input()` but forgot to delegate). (2)
`update_normalization()` ran encoder forward pass with grad tracking on every env step
(262K unnecessary grad-enabled passes per iteration). (3) `load_state_dict()` had all
backward compatibility stripped, causing silent partial loads on architecture mismatch.
(4) softplus on cost critic biased gradient for near-zero costs. (5) z_bounds_loss
returned CPU tensor on fallback path.

### Changed
- `encoder/actor_critic_encoder.py`: `_get_combined_obs()` now delegates to `_encode()`
  instead of duplicating encoder forward-pass logic inline (DRY fix).
- `encoder/actor_critic_encoder.py`: `update_normalization()` wraps encoder call in
  `torch.no_grad()` and only runs when `actor_obs_normalization=True`. Saves 262K
  unnecessary grad-tracked encoder passes per iteration (4096 envs x 64 steps).
- `encoder/actor_critic_encoder_constrained.py`: Cost critic activation `F.softplus()`
  -> `F.relu()`. softplus required x -> -inf for zero output (gradient vanishing for
  healthy constraints); ReLU allows exact zero with finite MLP values.

### Fixed
- `encoder/actor_critic_encoder.py`: `z_bounds_loss()` device fallback uses
  `next(self.parameters()).device` instead of hardcoded "cpu" when `_last_z is None`.
- `encoder/actor_critic_encoder.py`: Restored `_handle_critic_dim_mismatch()` and full
  `load_state_dict()` with backward compatibility: encoder_obs_normalizer injection for
  old checkpoints, critic input dim mismatch detection + reinitialization, unknown key
  filtering with logging, missing essential key warnings.
- `encoder/actor_critic_encoder_constrained.py`: Restored `load_state_dict()` override
  with cost_critic handling: K mismatch detection (different num_constraints), input dim
  mismatch via parent `_handle_critic_dim_mismatch()`, missing cost_critic key injection.

### Notes
- MEDIUM-3 (num_encoder_epochs default=5) and LOW-2 (dead encoder_output_activation
  config field) already fixed in prior sessions
- Agent-reported "missing encoder gradient from cost surrogate" verified as by-design
  (encoder role is information compression, cost avoidance is actor's responsibility)

## [2026-03-20] albc_env.py code review: _prev_joint_pos timing + control_dt fix

### Context
Code review of `albc_env.py` (1016 lines) identified 2 bugs and 1 question requiring
user confirmation. BUG-1: `_prev_joint_pos` was set in `_reset_action_buffers()` (during
`_reset_framework` phase) but joint positions are subsequently changed by
`_reset_task_and_state()` (equilibrium/random init). This injected a false delta (~0.5 rad)
into `_accumulated_rotation` on the first step after reset, skewing IPO constraint budget.
BUG-2: `control_dt` used `physics_dt` instead of `step_dt` -- latent bug currently masked
by `decimation=1` but would cause position_delta underestimation if decimation changed.
BUG-3: encoder weight_decay 1e-5 vs hero_agent's 1e-4 -- deferred to user confirmation.

### Fixed
- `albc_env.py`: Added `_prev_joint_pos` re-sync at end of `_reset_task_and_state()` after
  joint positions are set to equilibrium/random. Prevents false delta in
  `_accumulated_rotation` (IPO constraint) on first post-reset step.
- `albc_env.py`: Changed `control_dt = self.physics_dt * ...` to `self.step_dt * ...`.
  Latent bug -- no behavioral change at current `decimation=1`, but correct for any value.

### Notes
- BUG-3 (encoder weight_decay 1e-5 in constraint_trpo.py vs 1e-4 in hero_agent) pending
  user confirmation on whether the difference is intentional.
- 10 additional items verified correct (VERIFY-1~6, DESIGN-1~3, MINOR-1~2).

## [2026-03-20] Constrained ALBC algorithms code review + runtime integration fixes

### Context
Executed constrained ALBC algorithms code review plan targeting mathematical correctness
and latent bugs. Plan identified 3 fixes (overshoot_cost cross-axis false positive,
num_encoder_epochs default mismatch, barrier beta docstring). During dry run verification,
5 additional runtime integration bugs were discovered that prevented the constrained_albc
task from running at all -- the previous simplification session removed compatibility shims
that were actually load-bearing, and the auto-sync mechanism was silently broken.

Root cause of runtime failures: (1) hero_agent and constrained_albc registered identical
class names into `_runner_module` namespace; alphabetical import order caused hero_agent to
overwrite constrained_albc's classes. (2) `train.py` `_RUNNER_MAP` only had hero_agent
paths. (3) Runner auto-sync used `hasattr()` on plain dicts (from `to_dict()`), silently
skipping the `num_constraints` sync. (4) RSL-RL `OnPolicyRunner` expected `rnd` and
`multi_gpu_cfg` attributes. (5) `storage.dones` shape was `(T,N,1)` not `(T,N)`.

Dry run verified: 5 iterations of `Isaac-Constrained-ALBC-Encoder-v0` with 64 envs
completed successfully after all fixes.

### Fixed
- `mdp/constraints.py`: `overshoot_cost` per-axis conjunction -- sign flip and magnitude
  now checked on the SAME axis. Previously `any(dim=-1)` for sign flip and `max(dim=-1)`
  for magnitude could match different axes, triggering false positive overshoot cost.
- `algorithms/constraint_trpo.py`: `num_encoder_epochs` default 5 -> 1 to match config.
  Default=5 would cause stale importance sampling ratio when instantiated without config.
- `algorithms/constraint_trpo.py`: Barrier beta docstring now documents re-parametrization
  `beta_code = beta_paper / (2*t)` absorbing the 1/2 and 1/t factors.
- `algorithms/constraint_trpo.py`: Added `**_kwargs` to `__init__()` for RSL-RL
  `multi_gpu_cfg` compatibility, and `self.rnd = None` for `OnPolicyRunner.learn()` line 84.
- `algorithms/constraint_trpo.py`: Fixed `_compute_cost_returns` dones shape -- added
  `.squeeze(-1)` before `unsqueeze(-1)` to handle `(T,N,1)` storage dones.
- `agents/rsl_rl_ppo_cfg.py`: ALBC-prefixed `_runner_module` registration names
  (`ALBCConstraintEncoderRunner`, `ALBCConstraintTRPO`, `ALBCActorCriticEncoderConstrained`)
  to avoid namespace collision with hero_agent's identically-named registrations.
- `runners/constraint_encoder_runner.py`: Changed `num_constraints` auto-sync from
  `hasattr()`/attribute access to dict key access (`in`/`[]`). `train_cfg` is a plain dict
  from `agent_cfg.to_dict()`, so `hasattr()` always returned False, silently skipping sync.
- `scripts/.../train.py`: Added `ALBCConstraintEncoderRunner` to `_RUNNER_MAP` pointing
  to `constrained_albc.runners.ConstraintEncoderRunner`.

### Notes
- Previous simplification session removed `**kwargs`, `self.rnd = None`, and `hasattr`
  guards from `ConstraintTRPO` (commit cbd2dd24) -- these were actually required for
  RSL-RL `OnPolicyRunner` compatibility when not running through hero_agent's BaseRunner.
- hero_agent's `ConstraintEncoderRunner` has the same `hasattr()` auto-sync bug but was
  masked because hero_agent's `BaseRunner` chain handles the initialization differently.
- The `num_encoder_epochs` default mismatch existed since the C-TRPO migration (2026-03-17)
  but was never triggered because config always provided the value explicitly.

## [2026-03-20] Constrained ALBC code review fixes

### Context
Post-simplification code review of constrained ALBC found 8 issues. Three required
code changes (Fix 1, 5, 8); five are design issues documented for future work.
Fix 8 (num_encoder_epochs default) was already applied in the simplification session.

### Fixed
- `algorithms/constraint_trpo.py`: Initialize 8 `_last_*` / `_cached_*` monitoring
  attributes in `__init__()`. Without initialization, `ConstraintEncoderRunner` calling
  `_log_constraint_metrics()` before first `update()` would raise `AttributeError`.
  Also fixes `_cached_barrier_penalty` missing when first iteration enters recovery mode.
- `utils/logging.py`: Fixed `log_encoder_metrics` docstring claiming "Metrics kept (3)"
  when the function actually logs 5 metrics (z_mean, z_std, z_min, z_max, grad_norm).

### Notes
- Issues 2 (any-recovery mode), 4 (KL distribution overwrite), 6 (encode/get_combined_obs
  duplication), 7 (log_encoder_metrics env.get_observations() cost) documented as design
  issues for future work. No code changes needed -- current behavior is correct.

## [2026-03-20] Constrained ALBC algorithm simplification (10-step plan)

### Context
Comprehensive simplification of the `constrained_albc/` package across 10 sessions on 2026-03-20.
The codebase (23 files, ~7,000 lines) had accumulated complexity from being forked from hero_agent
and supporting multiple unused features. Only 1 task is registered (`Isaac-Constrained-ALBC-Encoder-v0`),
but the code supported 4 config levels, optional DORAEMON DR, TDE observation, legacy encoder modes,
and backward checkpoint compatibility. Goal: remove all unused code paths so root cause analysis is
straightforward when problems occur. Final package: ~4,900 lines (~2,100 lines removed, ~30% reduction).

### Added
- `constrained_albc/` package (23 files): Extracted from hero_agent as standalone C-TRPO + encoder
  constrained RL. Registered as `Isaac-Constrained-ALBC-Encoder-v0`. Zero runtime dependency on hero_agent.

### Changed
- `config.py`: Merged 4-class hierarchy (`ALBCEnvCfg` + `ALBCTrainEnvCfg` + `ALBCEncoderTrainEnvCfg`
  + `ConstrainedALBCEncoderEnvCfg`) into single `ALBCEnvCfg`. All fields at final production values.
  `DomainRandomizationCfg` removed `fixed_pose()`/`half_strength()` classmethods and buoy perturbation fields.
  Backward-compat alias `ConstrainedALBCEncoderEnvCfg = ALBCEnvCfg` for gym registration.
- `albc_env.py`: Removed `enable_payload` conditional (always True), `_payload_enabled` property,
  `state_space` vs `enable_payload` validation. Payload wrench always computed (no None checks).
  Extracted `_collect_termination_metrics()`, `_collect_dynamics_metrics()`, `_reset_action_buffers()`,
  `_reset_perturbation_buffers()` from monolithic methods. Removed DR infeasibility logging.
- `mdp/observations.py`: Removed `state_space >= 18/19` guards (always 19). Payload and added mass
  always included in privileged obs.
- `mdp/rewards.py`: Reduced `ALBCRewardCfg` from 10 to 5 fields. Removed penalty curriculum, settling/
  energy rewards, laplacian command branch. `RewardManager` simplified.
- `mdp/events.py`: `DRSampler.get()` simplified: removed `_key` string parameter and `**_kwargs`.
  All 14 call sites updated. Removed DORAEMON integration from `_DRSampler`.
- `mdp/constraints.py`: Removed `joint_torque_cost` alias and 5 unused cost functions.
- `encoder/actor_critic_encoder.py`: Removed no-history mode, symmetric critic, sigmoid activation,
  backward-compat `load_state_dict()` (key filtering, dim mismatch handling). 465 -> 295 lines.
- `encoder/actor_critic_encoder_constrained.py`: Removed `load_state_dict()` override with K-mismatch
  and input-dim-mismatch handling. 131 -> 78 lines.
- `algorithms/constraint_trpo.py`: Removed `**kwargs` catch-all, `self.rnd = None`, 3 `hasattr`
  guards around `evaluate_costs()`, 3 standalone surrogate methods (inlined as closures), entropy_coef
  (dead code). Vectorized cost GAE. 1043 -> 780 lines.
- `runners/`: Flattened 3-level hierarchy (`BaseRunner` -> `EncoderRunner` -> `ConstraintEncoderRunner`)
  to single `ConstraintEncoderRunner(OnPolicyRunner)`. Removed DORAEMON scheduling, noise floor/LR
  methods. Deleted `base_runner.py` and `encoder_runner.py`.
- `agents/rsl_rl_ppo_cfg.py`: Removed `asymmetric_critic`, `encoder_output_activation` from policy
  config (unused by encoder). Flattened 3-level config hierarchy to 2-level.
- `utils/logging.py`: Removed `log_dr_infeasibility()`, `_get_dr_infeasibility_logger()`,
  `connect_encoder_to_env()`.

### Removed
- `doraemon.py`: Deleted entirely (728 lines, unused adaptive DR scheduler).
- `runners/base_runner.py`, `runners/encoder_runner.py`: Deleted (merged into ConstraintEncoderRunner).
- `encoder/history_tcn.py`: Deleted (HistoryTCN replaced by raw flatten concat for TRPO OOM fix).
- Dead code: penalty curriculum, settling/energy rewards, laplacian command, TDE obs, buoy perturbation,
  symmetric critic, sigmoid activation, backward-compat checkpoint loading, DORAEMON integration.

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
