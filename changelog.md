# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

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
