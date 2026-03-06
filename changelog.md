# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [2026-03-06] Revert equilibrium joint init default to random

### Context
Code review of equilibrium joint initialization (commit ac6b2112) identified a
sim-to-real coverage gap: with equilibrium init as default, the policy never trains
from large joint-attitude mismatches at episode start. Real deployment will not
guarantee equilibrium starting conditions (e.g., robot placed into water at arbitrary
joint config). Per-step perturbations provide mid-episode recovery training but do not
replicate the specific pattern of starting with joints misconfigured relative to attitude.

Physics and math were verified correct (x_eq = -h*tan(pitch)/cos(roll), y_eq = h*tan(roll),
F_bu cancels, analytical IK matches). No code bugs found. The feature itself is sound
but premature as the default -- better suited as an opt-in mode or part of a mixed-init
curriculum when sim-to-real transfer is attempted.

### Changed
- `config.py`: Reverted `joint_init_mode` default from "equilibrium" to "random"
  (equilibrium code and config toggle preserved for future use)

## [2026-03-06] Revert value grad accumulation + encoder sensitivity analysis

### Context
Post-run review of 5 ConstraintTRPO encoder runs revealed the value gradient
accumulation (commit 68b4483f) was harmful: it accumulated encoder grads across
20 mini-batch backward passes creating a 20:1 ratio imbalance vs the single
full-batch policy gradient. At scale=1.0, the encoder optimized for "minimize
value prediction error" by collapsing z variance (z_std 0.8 -> 0.2).

Fix: gated all value gradient code behind `encoder_value_grad_scale` (default=0.0,
disabled). When disabled, encoder receives only policy loss gradient (baseline
behavior). Scale parameter preserved for future experimentation.

Validation run (17-16-19, 700 iter): NO catastrophic instability, attitude error
~7-8 deg (baseline-equivalent), constraints satisfied (feasibility_rate > 0.98).
However, encoder sensitivity analysis (z_sweep heatmap) showed max z_range 0.29
vs PPO baseline's 1.24 -- ~4x weaker.

Deep gradient flow analysis identified root cause: PPO encoder gets gradient from
surrogate loss (20x/iter) + value loss (20x/iter) + z_bounds (20x/iter) = 60 updates.
TRPO encoder gets only policy loss (1x/iter) + z_bounds (1x/iter) = 2 updates.
The value loss through the symmetric critic (encoder -> z -> critic -> value) is the
strongest signal and is completely absent in TRPO.

Proposed next step: replace mini-batch accumulation with a single full-batch value
forward pass after critic training, providing value gradient at 1:1 ratio with policy
gradient. This avoids the 20:1 imbalance while restoring the missing signal.

### Changed
- `constraint_trpo.py`: Added `encoder_value_grad_scale` parameter (default=0.0) to
  `__init__`. Gated value gradient accumulation reset, capture, and merge on `scale > 0`.
  When scale=0.0 (default), encoder receives only policy loss gradient -- identical
  to original baseline behavior.
- `rsl_rl_ppo_cfg.py`: Added `encoder_value_grad_scale: float = 0.0` to
  `RslRlConstraintTRPOAlgorithmCfg` with documentation comment.

### Fixed
- `mdp/events.py`: Fixed `IndexError: too many indices for tensor of dimension 1` in
  `compute_equilibrium_joint_positions()`. `_joint_limits_lower/upper` are shape `(2,)`
  (1D), but indexing used `[0, 0]`/`[0, 1]` (2D). Changed to `[0]`/`[1]`.
- `base_runner.py`: Removed `_save_best_model()` and `_best_mean_reward` tracking
  (dead code after OnPolicyRunner upstream added its own best model saving).
- `eval_dr_comparison.py`: Output directory now uses training run timestamp from
  checkpoint path instead of current time (easier to correlate eval with training).

### Notes
- Encoder sensitivity at iter 650: Main Volume z_range=0.29, Payload Mass=0.06 (near dead)
  vs PPO baseline Main Volume=1.24, Payload Mass=0.28
- Full-batch value gradient plan written (see plan file). Conservative start: scale=0.1
- Run may improve with longer training (currently 700/2500 iter) but structural gradient
  deficit suggests sensitivity plateau without value signal

---

## [2026-03-06] Equilibrium-consistent joint initialization

### Context
Hero Agent episodes started with large initial attitude errors because ALBC joints
were initialized uniformly in [-pi, pi], placing the buoy far from the gravity-buoyancy
equilibrium. This caused large unbalanced torques and violent initial transients that
the controller had to fight through before meaningful learning could begin.

Solution: compute the zero-torque equilibrium EE position from the current attitude
(roll, pitch) using the analytical relation tau = Lambda @ p_EE + T_b = 0, then solve
2-link analytical IK for joint angles, and add small noise (+-0.3 rad). This ensures
the buoy starts near its natural resting position for the given attitude.

Critical implementation detail: robot pose must be set BEFORE joint initialization
(previously joints were set first), because equilibrium computation reads roll/pitch
from the already-written root quaternion.

### Added
- `mdp/events.py`: `compute_equilibrium_joint_positions()` -- computes equilibrium
  EE from roll/pitch (x_eq = -h*tan(pitch)/cos(roll), y_eq = h*tan(roll)), solves
  2-link analytical IK, adds per-joint noise, clamps to limits
- `config.py`: `joint_init_mode` ("equilibrium" default) and `equilibrium_joint_noise`
  (+-0.3 rad) parameters in HeroAgentEnvCfg
- `mdp/__init__.py`: Export `compute_equilibrium_joint_positions`

### Changed
- `base_env.py`: Reordered `_reset_task_and_state()` -- pose reset now happens BEFORE
  joint initialization. Added 3-way dispatch: equilibrium (default) / random (legacy) /
  default (no DR). Backward compatible via `joint_init_mode="random"`.

## [2026-03-06] Fix encoder gradient starvation in ConstraintTRPO

### Context
Deep analysis of 3-run comparison (baseline 15-30-39, entropy-fix 15-54-09, round-2
16-17-25) revealed a structural bug: encoder gets NO gradient from value/cost_value
loss in ConstraintTRPO, unlike PPO where a shared optimizer propagates value gradients
to the encoder naturally. In ConstraintTRPO, `value_optimizer` only contains
critic/cost_critic params, so encoder `.grad` from `total_value_loss.backward()` is
populated (via symmetric critic computation graph through z) but never applied.

Fix: accumulate encoder gradients during the value update mini-batch loop, then merge
(averaged) with the policy-loss gradients in the deferred encoder Adam step. Also
reverted entropy_coef 0.02 -> 0.005 (analysis showed entropy collapse is desirable
for 2-DOF control; high entropy injects actuator noise).

Run 16-46-07 results:
- Encoder grad_norm: **0.005-0.015** in first 100 iter (3-10x increase vs ~0.001 broken runs).
  Confirms value gradient path is mechanically working.
- However, catastrophic instability at iter ~150: z_std collapsed 0.8 -> 0.2, cost_return_0
  spiked to 40 (d_k=15), feasibility_rate_0 dropped to 0.3, mean_reward dropped to 20.
- Recovery after iter 200 to near-baseline performance, but with z_std=0.2 (dead encoder).
- Root cause of crisis: value MSE gradient magnitude far exceeds policy surrogate gradient
  (~0 for TRPO at ratio=1), so encoder optimizes for "minimize value prediction error"
  rather than "improve policy". z converges to a constant (low z variance = easier prediction).
- Next step: add `encoder_value_grad_scale` (e.g., 0.1) to balance the two gradient sources.

### Changed
- `constraint_trpo.py`: Added encoder value gradient accumulation in value update loop --
  captures encoder `.grad` after `total_value_loss.backward()`, accumulates across
  mini-batches, merges (averaged) with policy-loss gradient in deferred encoder step
- `constraint_trpo.py`: Reverted entropy_coef default 0.02 -> 0.005
- `rsl_rl_ppo_cfg.py`: Reverted entropy_coef 0.02 -> 0.005 in RslRlConstraintTRPOAlgorithmCfg

### Notes
- Value gradient to encoder is too strong without scaling -- causes z collapse (z_std 0.8 -> 0.2)
- Need `encoder_value_grad_scale` parameter to attenuate value-path gradients relative to policy-path
- Entropy revert confirmed correct: noise_std follows baseline collapse pattern (0.95 -> 0.2)
- First time constraints were genuinely active (feasibility_rate_0 dropped to 0.3 during crisis)

---

## [2026-03-06] ConstraintTRPO Round 2: entropy, budget, diagnostic logging

### Context
Round 2 tuning based on comparison of runs 15-30-39 (baseline) and 15-54-09 (entropy
fix). Previous round applied entropy_coef 0.005->0.01 and min_std floor -- slowed
entropy collapse but didn't stop it. Also d_k_adaptive_0 anomalously dropped to 0
(should be impossible by formula).

Applied 4 changes: entropy_coef 0.01->0.02, constraint_budgets halved (0.3,0.05,0.3)
->(0.15,0.02,0.15), d_k diagnostic warning, and d_k base value logging.

Run 16-17-25 results vs previous runs:
- Entropy: STABILIZED at 1.0-1.2 (no longer collapsing). entropy_coef=0.02 worked.
- d_k anomaly RESOLVED: d_k_0=15, d_k_adaptive_0=15, both flat. Previous anomaly
  was likely a logging/comparison artifact (no d_k base metric to compare against).
- noise_std: 0.40 (vs 0.25 baseline) -- much more exploration maintained.

However, Dynamics metrics revealed the entropy fix HURT overall quality:
- Torque: 3-5 Nm (vs 2-3 Nm baseline) -- 2x higher actuator effort
- Joint velocity: 1.5-2.0 (vs 1.0 baseline) -- faster, less controlled
- Oscillation HF RMS: 0.5-0.8 (vs 0.3-0.4 baseline) -- more vibration
- Attitude error: 10-15 deg (vs 8-10 deg baseline) -- worse tracking
- Action rate: 0.6-0.8 (vs 0.4 baseline) -- jerkier commands

Key insight: entropy collapse is NOT a bug for 2-DOF attitude control. The baseline's
entropy collapse produced a smooth, conservative policy (low torque, low oscillation)
which is actually desirable. Forcing high entropy injects noise into actuators.
ConstraintTRPO's value may lie in constraint enforcement, not entropy maintenance.

### Changed
- `rsl_rl_ppo_cfg.py`: entropy_coef 0.01 -> 0.02 in RslRlConstraintTRPOAlgorithmCfg
- `rsl_rl_ppo_cfg.py`: constraint_budgets (0.3, 0.05, 0.3) -> (0.15, 0.02, 0.15)
- `constraint_trpo.py`: Aligned __init__ defaults with config (budgets + entropy_coef)

### Added
- `constraint_trpo.py`: Diagnostic warning when d_k_adaptive drops below d_k (invariant violation)
- `constraint_encoder_runner.py`: Log `Constraint/d_k_{k}` base budget values alongside d_k_adaptive

### Notes
- Baseline run (15-30-39) remains best overall: similar attitude error with half the torque/velocity
- Next direction: either revert entropy_coef to 0.005 and focus on constraint tightening,
  or conclude ConstraintTRPO experiment and return to PPO Encoder-Base for Phase 2

---

## [2026-03-06] ConstraintEncoderRunner support + ConstraintTRPO stabilization

### Context
Added ConstraintEncoderRunner support to play.py and eval_dr_comparison.py (were
missing from runner dispatch maps). Then stabilized ConstraintTRPO through 5 rounds
of debugging -- the algorithm went from 0% line search success to 100%.

Root causes found (in discovery order):
1. Cost advantage normalization amplified noise 1000x when constraints satisfied
2. z_bounds_loss updated encoder 20x/iter during value loop, violating TRPO old-policy assumption
3. Barrier margin floor 1e-6 caused gradient explosion (fixed to 0.1*d_k)
4. Gradient/line-search objective mismatch (gradient used combined IPO, LS checked reward-only)
5. Missing 1/(1-gamma) factor made cost gradient 100x too weak
6. **ROOT CAUSE**: TRPO step direction was +F^{-1}g (ascent) instead of -F^{-1}g (descent)

Final run (2026-03-06_15-30-39, 4096 envs): line_search_success 100%, mean_reward
1->80, tracking 0.5->3.5, noise_std 0.95->0.20, cost_surrogate 8->0.

Training analysis of run 15-30-39 identified entropy collapse as primary bottleneck
(entropy 2.8->0.05 in ~100 steps, near-deterministic policy). Applied entropy_coef
2x increase + min noise_std floor. Also added pre-encoder KL metric to distinguish
TRPO trust region compliance from encoder-induced distribution shift.

Follow-up run (2026-03-06_15-54-09): entropy collapse slowed significantly (0.8->0.5
at step 250 vs instant collapse). `kl_trpo` ~0.008 (well within 0.015 limit). However,
encoder z_std lower (0.5 vs 0.8) and grad_norm 10x weaker. Suspicious: d_k_adaptive_0
dropping to 0 despite target >= d_k[0] guaranteed by formula -- needs investigation.

### Added
- `play.py`: `ConstraintEncoderRunner` in `_RUNNER_MAP`
- `eval_dr_comparison.py`: ConstraintEncoderRunner + ActorCriticEncoderConstrained support

### Changed
- `constraint_trpo.py`: Negated TRPO step direction (-F^{-1}g for loss minimization)
- `constraint_trpo.py`: Replaced `_log_barrier_objective` with `_linearized_surrogate` matching gradient
- `constraint_trpo.py`: Added 1/(1-cost_gamma) scaling factor (NORBC Eq. 10)
- `constraint_trpo.py`: Removed cost advantage normalization (barrier handles scaling)
- `constraint_trpo.py`: Deferred encoder update (1 policy + 1 z_bounds per iter, was 21)
- `constraint_trpo.py`: Margin floor 1e-6 -> 0.1*d_k in barrier loss and cost surrogate
- `constraint_trpo.py`: Added min log_std floor (log(0.1)) to prevent entropy collapse
- `constraint_trpo.py`: Added `kl_trpo` metric (post-TRPO-step, pre-encoder KL)
- `rsl_rl_ppo_cfg.py`: entropy_coef 0.005 -> 0.01
- `config.py`: Set `linear_error_weight=0.0` in constrained env (penalties belong in constraints)

### Removed
- `constraint_trpo.py`: `_log_barrier_objective()`, `_surrogate_loss()`, `_full_surrogate_loss()`, diagnostic print()

### Notes
- Constraints 1,2 remain trivially satisfied (feasibility_rate=1.0 constant). Budget tightening deferred to next run.
- Encoder grad_norm weakened in new run -- higher entropy may cause noisier signals that cancel in expectation.
- `d_k_adaptive_0` anomaly: formula guarantees target >= d_k[0], but metric drops to 0. Config forwarding issue suspected.
- Value function LR is fixed 3e-4 (Adam). TRPO policy has no LR (natural gradient + line search).

---

## [2026-03-05] ConstraintTRPO implementation + code cleanup

### Context
Full implementation of NORBC-style constrained RL (IPO + TRPO) for Hero Agent. Separates
physical constraints (joint velocity, rotation, oscillation) from rewards using explicit cost
budgets and log-barrier penalties. Architecture: actor TRPO natural gradient, encoder Adam
(lr=3e-3), value/cost_critic Adam. Two rounds of code review found 15 issues including
missing cost gradient path, barrier loss without grad_fn, and encoder grad accumulation.

Also performed codebase cleanup (~7,700 lines), theoretical audit against TDE/HORA theory,
and applied monkey-patch for encoder optimizer persistence.

### Added
- `algorithms/constraint_trpo.py`: Full TRPO + IPO (~600 lines) -- CG solver, Fisher-vector product, line search, log-barrier, adaptive thresholds
- `encoder/actor_critic_encoder_constrained.py`: Multi-head cost critic (K outputs)
- `runners/constraint_encoder_runner.py`: Barrier schedule + constraint metrics logging
- `mdp/constraints.py`: 3 binary cost functions (joint_velocity, accumulated_rotation, joint_oscillation)
- `algorithms/ppo_patch.py`: Monkey-patch for RSL-RL PPO encoder optimizer (WD=1e-5)
- `docs/THEORETICAL_ANALYSIS.md`: TDC, rewards, NORBC pipeline analysis

### Changed
- `base_env.py`: Added accumulated rotation tracking, cost computation, constraint buffers; consolidated perturbation/noise/termination helpers
- `config.py`: Added `HeroAgentConstrainedEncoderEnvCfg`; relaxed constraint budgets to (0.3, 0.05, 0.3)
- `__init__.py`: Registered `Isaac-HeroAgent-Constrained-Encoder-Base-v0`
- `base_env.py`: TDE observation uses previous-step Lambda*p_EE and T_b (matching TDC TDE pattern)
- `encoder/adaptation.py`: Phase 2 critic evaluate() uses z_hat (consistent with actor)
- `controllers/tdc.py`: F_bu accepts per-env tensor; extracted `_set_param()` helper

### Removed
- `base_env.py`: `_cumulative_effort` buffer, `HeroAgentEnvWindow` class
- `mdp/rewards.py`: `action_rate_penalty()`, `angular_velocity_penalty()` (weight=0 everywhere)
- MPC docstring references from controllers/encoder/runners `__init__.py`

### Fixed
- Barrier loss gradient path, encoder grad isolation, NaN guard (shs <= 0)
- Line search margin floor, squeeze() -> squeeze(-1) for B=1 safety
- `mdp/rewards.py`: Restored `termination_penalty` accidentally removed during cleanup
