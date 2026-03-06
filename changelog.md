# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [2026-03-06] Fix play.py: Register ConstraintEncoderRunner

### Context
Running `play.py` with a trained ConstraintEncoderRunner checkpoint failed with
`ValueError: Unsupported runner class: ConstraintEncoderRunner`. The runner was
already registered in `train.py`'s `_RUNNER_MAP` but was missing from `play.py`'s
identical map -- a sync issue from when ConstraintEncoderRunner was added.

### Fixed
- `scripts/reinforcement_learning/rsl_rl/play.py`: Added `ConstraintEncoderRunner` entry to `_RUNNER_MAP`, matching `train.py`

## [2026-03-06] Fix ConstraintTRPO: Negate Step Direction (Gradient Descent, Not Ascent)

### Context
After removing cost normalization and aligning line search objective, diagnostic logging
revealed the ROOT CAUSE of all previous line search failures: the TRPO step direction
was inverted. The natural gradient step was doing gradient ASCENT on the loss instead
of descent.

Diagnostic output showed:
```
old=1.6667  new=1.6993  impr=-0.0326  FAIL:impr
```
Loss INCREASED at every backtrack step -- even the smallest step (1/1024) made it worse.
KL was fine (8.2e-3 < 1.5e-2 limit). The issue: `step_dir = +step_scale * nat_grad`
steps in the +gradient direction (ascent), but policy_loss should be MINIMIZED.

Standard TRPO maximizes `surrogate = +(ratio * adv).mean()` so `+F^{-1}g` is correct
(ascent on thing to maximize). Our code minimizes `policy_loss = -(adv*ratio).mean() +
cost_surr - entropy`, so the step must be `-F^{-1}g` (descent on thing to minimize).

This sign error has been present since initial implementation. All 4 previous fix rounds
(margin floor, cost normalization, objective alignment, 1/(1-gamma) scaling) were
addressing real issues but could never work because the step always went the wrong way.

Also changed diagnostic logging from `logger.info()` to `print()` since Python logging
default level is WARNING, making info-level messages invisible.

### Fixed
- `algorithms/constraint_trpo.py`: Negated TRPO step direction from
  `step_dir = step_scale * nat_grad` to `step_dir = -step_scale * nat_grad`.
  Gradient of loss-to-minimize requires descent (-F^{-1}g), not ascent (+F^{-1}g)

### Changed
- `algorithms/constraint_trpo.py`: Removed all diagnostic print() statements after
  verifying fix (were temporary: TRPO gradient norms, per-backtrack LS diagnostics)

### Verified
Run 2026-03-06_15-30-39 (40+ iterations, 4096 envs):
- `line_search_success`: **100%** (was 0% in all previous runs)
- `mean_noise_std`: 0.95 -> 0.15 (policy specializing, was flat 1.0)
- `Loss/entropy`: 2.8 -> 0.5 (was flat 3.0)
- `Loss/kl`: ~0.01 active (was 0.03 from encoder-only drift)
- `Train/mean_reward`: 1 -> 16 (genuine learning, not just curriculum)
- `cost_surrogate`: 8 -> 0 (policy actively reducing violations)
- `cost_value`: 2.5 -> 0.3 (cost critic learning)

---

## [2026-03-06] Fix ConstraintTRPO: Remove Cost Normalization + Align Line Search Objective

### Context
Line_search_success remained 0% after 4 rounds of fixes. Code audit identified two
interacting bugs creating a fatal combination:

Bug 1 (PRIMARY): Cost advantage normalization (added in RC1 as band-aid for 1e-6 margin
floor) amplified noise 1000x when constraints were satisfied. With raw cost std~0.001,
normalization blew it to std=1.0. Combined with 1/(1-gamma)=100 and 1/(t*margin)=1/30,
each constraint contributed ~3.3x reward gradient scale. Total cost gradient was ~10x
reward, but filled with normalized noise. Natural gradient direction became random.

Bug 2 (SECONDARY): Gradient used linearized cost surrogate but line search used nonlinear
log-barrier (_log_barrier_objective). TRPO guarantees improvement on the objective whose
gradient was used for CG -- using a different objective breaks this guarantee.

The margin floor fix (RC3: 0.1*d_k~3.0) already solved the original explosion that
motivated normalization. With proper margin floor, raw cost advantages naturally scale:
p=0.1 violation rate -> cost_gradient~10 (strong), p=0.001 -> ~1.0 (balanced with reward).

Run result: line_search_success still 0%. Reward increase (2->20) is purely curriculum +
encoder learning; actor remains frozen (surrogate~0, entropy~3 constant, noise_std=1.0).
Added diagnostic logging (gradient norms, per-backtrack improvement/KL) to identify
remaining failure mode.

### Changed
- `algorithms/constraint_trpo.py`: Removed cost advantage normalization loop in
  `_compute_cost_returns()`. Raw GAE cost advantages used directly -- barrier weighting
  `1/(t * margin * (1-gamma))` handles cost-vs-reward scaling automatically
- `algorithms/constraint_trpo.py`: Replaced `_log_barrier_objective()` (nonlinear
  log-barrier) with `_linearized_surrogate()` using same formula as gradient computation
  (reward_surr + barrier-weighted cost_surr - entropy). Ensures natural gradient direction
  is guaranteed to improve line search objective
- `algorithms/constraint_trpo.py`: Updated `_line_search()` and `update()` to call
  `_linearized_surrogate()` instead of `_log_barrier_objective()`
- `algorithms/constraint_trpo.py`: Added diagnostic logging -- TRPO gradient norms
  (|g|, |nat_grad|, shs, rew_surr, cost_surr) and per-backtrack line search diagnostics
  (old_loss, new_loss, improvement, kl, failure reason: FAIL:impr vs FAIL:kl)

### Removed
- `algorithms/constraint_trpo.py`: Deleted `_log_barrier_objective()` method entirely

### Notes
- NORBC paper does NOT normalize cost advantages -- barrier function handles scaling
- Reward advantages ARE normalized (standard TRPO practice); cost advantages are NOT
  (deliberate asymmetry, correct per barrier theory)
- Next step: run 10 iterations with diagnostic logging to determine if failure is
  improvement<=0 (gradient direction issue) or KL>threshold (step size issue)

---

## [2026-03-06] Fix ConstraintTRPO 3 NORBC Paper Bugs -- Cost Scaling, Log-Barrier, Reward Contamination

### Context
Line_search_success remained 0% after the previous fix (gradient/line-search objective
alignment). The previous fix was insufficient because it aligned to the wrong (linearized)
objective and missed a critical scaling factor. Three bugs identified by comparing current
code against NORBC paper (arXiv:2308.12517v4, Eq. 10):

1. Missing `1/(1-gamma)` factor in cost gradient: NORBC Eq.(10) relates advantage to
   discounted return via `1/(1-gamma)`. With cost_gamma=0.99, cost gradient was 100x too
   weak, making constraints invisible to the optimizer.
2. Line search used linearized surrogate with FIXED margin (old policy): old_loss and
   new_loss saw identical barrier values, so barrier improvement was always ~0. Replaced
   with actual log-barrier where margin depends on new policy via ratio.
3. `linear_error_weight=-3.0` in constrained reward produced ~-2.0/step, dominating total
   reward (~-0.6) and compressing advantages to near-zero. NORBC philosophy: penalties
   belong in constraints, not reward.

### Changed
- `algorithms/constraint_trpo.py`: Added `1/(1-cost_gamma)` factor to cost gradient
  computation (100x scaling with gamma=0.99), matching NORBC Eq.(10)
- `algorithms/constraint_trpo.py`: Replaced `_full_surrogate_loss()` with
  `_log_barrier_objective()` -- barrier margin now depends on new policy via
  `new_J_Ck = mean_cost_returns[k] + cost_change(ratio)`, enabling actual improvement
  detection in line search
- `algorithms/constraint_trpo.py`: Applied same `1/(1-cost_gamma)` factor inside
  `_log_barrier_objective()` for consistency
- `config.py`: Set `linear_error_weight=0.0` in `HeroAgentConstrainedEncoderEnvCfg`,
  removing reward contamination. Reward now = tracking(+5.0) + settling(+3.0) + progress(+0.3)

### Removed
- `algorithms/constraint_trpo.py`: Deleted `_surrogate_loss()` (reward-only, unused since
  previous fix) and `_full_surrogate_loss()` (replaced by `_log_barrier_objective()`)

### Notes
- Cost gradient now 100x stronger; mitigated by barrier schedule (t: 1->50 reduces weight)
- Log numerical stability safe: margin >= 0.1*d_k >= 3.0, so log(3.0)=1.1
- Verification targets: line_search_success > 30%, Loss/surrogate non-zero sustained,
  Episode_Reward/total positive range, Loss/cost_surrogate 100x larger than before

---

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
