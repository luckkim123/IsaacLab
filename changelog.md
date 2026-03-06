# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [2026-03-06] Fix ConstraintTRPO Line Search 0% -- Gradient/LineSearch Objective Mismatch

### Context
Previous fixes (cost adv normalization, encoder deferral, margin floor) stabilized
cost_surrogate (no longer explodes) but line_search_success remained flat 0% over
200 iterations. Policy never updated; all loss signals collapsed to ~0.

Root cause: gradient and line search used DIFFERENT objectives. Gradient (line 657)
was computed from `policy_loss = reward_surr + cost_surr - entropy` (combined IPO
objective). But line search (lines 460, 476) checked improvement on `_surrogate_loss()`
which was reward surrogate ONLY. The CG direction optimized the combined objective,
rotating the step away from pure reward improvement by the cost gradient (~25% of total
at barrier_t=1). Result: `improvement ~ 0` every iteration -- an IPO gradient with a
CPO-style line search that doesn't work together.

### Changed
- `algorithms/constraint_trpo.py`: Added `_full_surrogate_loss()` method computing the
  same reward + cost barrier - entropy objective used for gradient computation
- `algorithms/constraint_trpo.py`: Updated `_line_search()` to use `_full_surrogate_loss()`
  for both `old_loss` and `new_loss` evaluation. Removed separate cost feasibility check
  (3rd condition) -- barrier term in the full surrogate handles cost feasibility implicitly.
  Acceptance simplified to 2 conditions: improvement > 0 and KL <= max_kl * margin
- `algorithms/constraint_trpo.py`: Updated `old_loss` computation in `update()` call site
  to use `_full_surrogate_loss()` with all required args (cost_advantages, mean_cost_returns)

### Notes
- `_surrogate_loss()` (reward-only) is now dead code but retained for potential future use
- `line_search_cost_margin` parameter is now unused (barrier handles cost implicitly)
- Verification targets: line_search_success > 30%, non-zero sustained Loss/surrogate
- Fallback if still failing: abandon ConstraintTRPO, switch to Lagrangian PPO

## [2026-03-06] Fix ConstraintTRPO Training Failure -- NORBC Discrepancies

### Context
ConstraintTRPO training (1900 iterations, 2026-03-05 run) showed no learning: line search
success ~5%, attitude error 20-50 deg flat, encoder grad_norm collapsing to 0 after step
1200, cost_surrogate exploding to 30,000+ around step 1000. Previous margin floor fix was
insufficient. Analysis revealed 3 fundamental discrepancies with the NORBC paper.

RC1: RSL-RL normalizes reward advantages `(adv - mean) / std`, but cost advantages were
stored raw. Unpredictable cost scale dominated the natural gradient direction, making reward
improvement impossible.

RC2: z_bounds_loss updated the encoder 20 times per iteration (5 epochs x 4 minibatches)
during the value loop. This shifted the actor distribution before the TRPO step, violating
TRPO's assumption that the gradient is computed at the old policy. Result: ratio != 1.0
before TRPO starts, KL budget partially consumed by uncontrolled drift.

RC3: Barrier loss and cost surrogate still used `clamp(min=1e-6)` margin floor (only line
search was fixed previously). Near-zero margins caused gradient explosion.

### Changed
- `algorithms/constraint_trpo.py`: Added per-constraint cost advantage normalization
  `(adv - mean) / (std + 1e-8)` after `_compute_cost_returns()`, matching reward advantage
  normalization (NORBC Eq. 10 notes require standardization before policy gradient)
- `algorithms/constraint_trpo.py`: Replaced 20 per-minibatch z_bounds encoder updates in
  value loop with no-grad logging only. Added single full-batch z_bounds encoder update
  after the deferred policy gradient step. Encoder now gets exactly 2 updates per iteration
  (1 policy grads + 1 z_bounds) instead of 21
- `algorithms/constraint_trpo.py`: Changed margin floor from `clamp(min=1e-6)` to
  `clamp(min=0.1 * d_k[k])` in barrier loss and cost surrogate, consistent with the
  line search fix already applied

### Notes
- Verification targets: line_search_success >30%, cost_surrogate <100, encoder grad_norm
  sustained >0.01, attitude error downward trend, total reward increasing over 300 iterations

---

## [2026-03-05] ConstraintTRPO -- Implementation, Reviews, and Stabilization

### Context
Full implementation of NORBC-style constrained RL (IPO + TRPO), followed by 3 rounds of
code review and bug fixes. The algorithm separates physical constraints (joint velocity,
rotation, oscillation) from rewards using explicit cost budgets and log-barrier penalties.
Architecture: actor uses TRPO natural gradient, encoder uses separate Adam (lr=3e-3),
value/cost_critic use Adam.

Implementation uncovered multiple runtime issues: inference tensors from rollout storage
needed `.clone()` for autograd, encoder Adam step had to be deferred until after TRPO line
search, and z_bounds_loss required fresh forward pass for grad_fn.

Code review (2 rounds, 15 total issues) found critical problems: (1) cost constraints
computed via GAE but never in policy gradient -- policy was unconstrained TRPO, (2) barrier
loss had no gradient path (detached leaf tensor), (3) encoder grad accumulation across
mini-batches, (4) squeeze() shape collapse risk at B=1.

Stabilization: NaN crash from negative `shs` (CG approximation), line search deadlock from
`1e-6` margin floor when constraints violated, and parameter refinement (line_search_kl/cost
margins, split adaptive_alpha into threshold_scale + ema_alpha).

### Added
- `algorithms/constraint_trpo.py`: Full TRPO + IPO (~600 lines). CG solver, Fisher-vector
  product, line search with KL + cost feasibility, log-barrier, adaptive thresholds
- `encoder/actor_critic_encoder_constrained.py`: Multi-head cost critic (K outputs)
- `runners/constraint_encoder_runner.py`: Barrier schedule, constraint metrics logging
- `mdp/constraints.py`: 3 binary cost functions (joint_velocity, accumulated_rotation,
  joint_oscillation)
- `algorithms/ppo_patch.py`: Monkey-patch for RSL-RL PPO encoder optimizer (WD=1e-5)
- `docs/THEORETICAL_ANALYSIS.md`: Full theoretical analysis of TDC, rewards, NORBC pipeline

### Changed
- `base_env.py`: Added accumulated rotation tracking, cost computation, constraint buffers
- `config.py`: Added `HeroAgentConstrainedEncoderEnvCfg`, zeros constraint reward weights
- `agents/rsl_rl_ppo_cfg.py`: Added constrained algorithm/policy/runner configs, constraint
  budgets relaxed from (0.1, 0.05, 0.1) to (0.3, 0.05, 0.3)
- `__init__.py`: Registered `Isaac-HeroAgent-Constrained-Encoder-Base-v0`

### Fixed
- Barrier loss gradient path (detached → differentiable cost_value_pred)
- Encoder grad isolation (scoped clip_grad_norm_, added zero_grad before value backward)
- NaN crash guard (shs <= 0, isfinite checks on step_dir)
- Line search margin floor (1e-6 → 0.1*d_k in line search acceptance)
- squeeze() → squeeze(-1) for B=1 safety
- Barrier schedule moved from runner to algorithm (eliminated 1-iteration lag)
- cost_gamma >= 1.0 validation

---

## [2026-03-05] Theoretical/Logical Error Fixes

### Context
Systematic theoretical audit against TDE/HORA theory. 3 critical fixes, plus monkey-patch
to make encoder optimizer/z_bounds patches persistent (previously only in site-packages).

### Changed
- `base_env.py`: TDE observation now uses previous-step Lambda*p_EE and T_b buffers,
  matching TDC controller's TDE pattern (was using current-step, violating H_t ~ H_{t-L})
- `encoder/adaptation.py`: Phase 2 critic evaluate() uses z_hat instead of encoder z,
  ensuring actor and critic see the same state representation
- `controllers/tdc.py`: F_bu parameter accepts per-env tensor (was scalar mean)
- `tdc_env.py`: Passes full per-env F_bu tensor to TDCController

### Notes
- 4 reported issues verified as false positives (pitch T_b, action latency indexing,
  z activation, Lambda_inv DLS math)

---

## [2026-03-05] Code Simplification

### Context
Cleanup of hero_agent codebase (~7,700 lines). Dead code removal, duplicate consolidation,
unused reward cleanup. Accidentally removed active `termination_penalty = -10.0` during
cleanup -- restored immediately.

### Changed
- `base_env.py`: Consolidated perturbation logic (`_apply_perturbation_cycle`), noise config
  iteration (`_iter_noise_params`), termination logging (`_term_rate`)
- `controllers/tdc.py`: Extracted `_set_param()` helper, consolidated reset buffers to loop
- `mdp/events.py`: Merged CoB/CoG DORAEMON branches into `_apply_xyz_offset_with_doraemon`

### Removed
- `base_env.py`: `_cumulative_effort` buffer, `HeroAgentEnvWindow` class
- `mdp/rewards.py`: `action_rate_penalty()`, `angular_velocity_penalty()` (weight=0 everywhere)
- MPC docstring references from controllers/encoder/runners `__init__.py`

### Fixed
- `mdp/rewards.py`: Restored `termination_penalty` field incorrectly removed during cleanup
