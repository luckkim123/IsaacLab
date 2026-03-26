# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

For entries before 2026-03-27, see [changelog_legacy.md](changelog_legacy.md).

## [2026-03-27] Increase barrier_alpha to reduce barrier-to-reward gradient imbalance

### Context
Analysis of 3 consecutive runs revealed that per-constraint cost advantage standardization
(restored in previous commit) combined with `1/(1-gamma)=100` and `barrier_t=100` creates
a 9.2:1 barrier-to-reward gradient ratio (sum of `1/margin_k` across 4 constraints at floor).

**3-run comparison:**
| Run | Changes | reward | noise | entropy | enc_grad max | action_rate |
|-----|---------|--------|-------|---------|-------------|-------------|
| 00-09-23 | baseline | -78.80 | 0.64 | 1.41 | 1.0 | - |
| 01-15-43 | +1/(1-γ)+enc_TRPO | -38.77 | 0.60 | 1.29 | 322 | 1.02 |
| 01-38-08 | +standardization | -37.36 | 0.44 | 0.82 | 14097 | 2.00 |

Run 3 showed exploration collapse (noise 0.44, entropy 0.82), action oscillation (rate 2.0,
smoothness reward 4x worse), and encoder grad norm spike to 14097. Root cause: effective barrier
weight = `[1/(1-γ)] / barrier_t / margin_k = 100/100/margin_k = 1/margin_k`. With deeply
infeasible constraints at floor margins (0.20-1.57), total barrier weight = 9.2 vs reward = 1.

**Fix: increase barrier_alpha** from 0.02 to 0.05. This enlarges the adaptive threshold floor
margin (`alpha * d_k`), directly reducing `1/margin_k`:
- torque: 0.40 -> 1.0, velocity: 0.20 -> 0.50, yaw_vel: 1.57 -> 3.93
- Total barrier weight: 9.19 -> 2.26 (barrier:reward ≈ 2.3:1)

Chosen over increasing barrier_t because alpha only affects deeply infeasible constraints
(margin = alpha*d_k floor). When constraints become feasible (margin > alpha*d_k), the
alpha value becomes irrelevant -- a self-deactivating mechanism.

### Changed
- `agents/rsl_rl_ppo_cfg.py`: `barrier_alpha` 0.02 -> 0.05 (adaptive threshold floor margin)

### Notes
- Effective gradient balance: barrier:reward ≈ 2.3:1 (was 9.2:1)
- If still too strong, alpha=0.10 gives 1.6:1 ratio
- Classical IPM interpretation: larger alpha = larger trust region around current infeasible point,
  allowing more reward optimization while maintaining directional constraint pressure

## [2026-03-27] Restore per-constraint cost advantage standardization (NORBC Sec IV-B)

### Context
Analysis of run `2026-03-27_01-15-43` (197 iters, post-1/(1-gamma) fix) revealed that while the
barrier fix improved reward (2x), attitude error (38-55% better), and eliminated z saturation,
3/4 constraints (torque, velocity, yaw_vel) remained deeply infeasible with margins stuck at the
adaptive threshold floor (alpha*d_k).

**Root cause:** Per-constraint cost advantage standardization was removed during the paper-aligned
architecture overhaul (`8ba1827c`). Without it, constraints with different physical scales
(binary 0/1 costs vs continuous |omega_z| costs) have vastly different gradient magnitudes.
When deeply infeasible (e.g., 96% torque violation), the cost value function accurately predicts
high costs, making raw cost advantages near-zero (A_Ck ≈ 0.04 for violating steps). This leaves
the barrier gradient direction dominated by noise.

**Paper reference (NORBC Sec IV-B):** The paper explicitly standardizes per-constraint cost
advantages: `A_hat_Ck = (A_Ck - mu) / sigma` per constraint k. This equalizes gradient scale
across constraints so barrier weight 1/(t*margin_k) provides proximity-based prioritization only.

**Synergy with adaptive threshold:** Zero-mean standardization means positive A_Ck = worse than
average cost, negative = better. Combined with adaptive d_k^i ensuring positive margin at ratio=1
(since standardized mean=0 gives cost_surrs=0), the barrier remains well-defined while providing
balanced gradient direction across all constraints.

### Fixed
- `algorithms/constraint_trpo.py`: Restored per-constraint cost advantage standardization in
  `update()`. Was: raw `cost_advantages_flat`. Now: standardized per constraint
  `(A_Ck - mean) / (std + 1e-8)`. Originally added in `332eff85`, removed in `8ba1827c`.

### Notes
- The 1/(1-gamma) factor (previous fix) and standardization serve complementary roles:
  1/(1-gamma) provides correct barrier sensitivity to ratio changes; standardization
  equalizes gradient magnitude across constraints
- Run comparison (pre-fix vs post-fix): reward -78.80 -> -38.77, roll 29.20 -> 17.96 deg,
  pitch 26.45 -> 11.91 deg, z_range [-0.99,0.99] -> [-0.53,0.40] (no saturation)
- Encoder grad_norm 20-300 is expected: barrier amplifies by 1/(1-gamma)=100, encoder is
  ~50% of policy params. Scalar gradient clipping preserves direction in TRPO.
- z_std 0.08 -> 0.21 (steadily increasing) indicates genuine encoder learning

## [2026-03-27] Add missing 1/(1-gamma) factor to IPO barrier cost surrogate

### Context
Systematic comparison of the NORBC paper's Equation 10 against the current ConstraintTRPO
implementation revealed the log-barrier's cost surrogate was missing the `1/(1-gamma)` factor
from the performance difference lemma.

**Paper's formula (Eq. 10):**
```
margin_k = d_k^i - J_Ck(pi_i) - [1/(1-gamma)] * E[ratio * A_Ck]
                                  ^^^^^^^^^^^^
                                  MISSING in code
```

**Impact:** With `cost_gamma=0.99`, the factor `1/(1-gamma) = 100`. The barrier was estimating
the constraint margin change as 100x smaller than reality, making it effectively inactive.
The barrier could not detect that a proposed policy step would violate constraints.

**Example (attitude, d_k=1.0, barrier_base=0.5):**
- Paper: margin = 0.5 - 100*0.003 = 0.2 (barrier detects shrinking margin)
- Code:  margin = 0.5 - 0.003 = 0.497 (barrier sees almost no change)

**barrier_t analysis:** Verified that `barrier_t=100` (paper default) remains correct after
the fix. At margin floor (alpha*d_k), effective barrier weight = 50/d_k for attitude (strong
enforcement when infeasible), dropping to ~2 when feasible (reward takes over). This is the
intended log-barrier behavior. No adjustment needed.

**Reward term asymmetry (intentional):** The paper omits `1/(1-gamma)` from the reward
surrogate because it's a constant scale factor that doesn't affect the TRPO optimization
direction. But for the cost term INSIDE the log(), the factor changes the argument, not just
the scale -- it determines when the barrier approaches -inf.

### Fixed
- `algorithms/constraint_trpo.py`: Added `inv_one_minus_gamma = 1/(1-cost_gamma)` factor
  to `cost_surrs` in the IPO barrier surrogate function. Was: `E[ratio * A_Ck]`.
  Now: `[1/(1-gamma)] * E[ratio * A_Ck]` (matching NORBC Eq. 10).

### Notes
- Combined with the encoder TRPO integration (previous entry), this completes alignment
  with the NORBC paper's TRPO+IPO formulation
- The `1/(1-gamma)` was NOT needed in the reward surrogate (TRPO standard: direction-only,
  step size from KL constraint)
- Code reviewer identified 3 secondary items for future monitoring:
  1. `margin.clamp(min=1e-8)` kills gradient when margin <= 0 (OK at ratio=1, may need
     smooth barrier if value function accuracy is poor)
  2. `mean_cost_returns.clamp(min=0)` slightly inflates margin (acceptable for non-negative costs)
  3. Gradient clipping before CG (practical stabilization, not in paper)

## [2026-03-27] Integrate encoder into TRPO trust region (joint natural gradient + line search)

### Context
Deep analysis of run `2026-03-27_00-09-23` (280 iters) revealed the root cause of training
stagnation: the separate Adam-based encoder update was destroying the TRPO trust region.

**Evidence from TensorBoard data:**
- TRPO pre_encoder_kl: 0.0035 avg (within max_kl=0.005 budget)
- Post-encoder KL: 0.138 avg (**27.6x budget**, median 32.1x, max 1153.4x)
- Encoder added 26.9x the TRPO KL budget per iteration on average
- 11.4% of iterations had barrier_penalty = -inf (numerical degeneration from ratio overflow)
- Reward flat at -67, all constraints 2-5x over budget, no convergence

**Root cause:** The encoder update ran 5 Adam epochs (lr=3e-4) after each TRPO step,
changing z which shifts the actor input distribution without any KL constraint. This
directly contradicts the NORBC paper's rationale for choosing TRPO+IPO:
1. TRPO's line search verifies barrier feasibility -- encoder bypassed it entirely
2. TRPO's KL constraint limits policy change -- encoder added 32x the budget
3. TRPO protects log-barrier from numerical explosion -- encoder caused -inf values

The NORBC paper trains encoder jointly with actor (same optimizer, same KL constraint).
The separate encoder update was an implementation deviation that nullified the trust region.

### Changed
- `algorithms/constraint_trpo.py`: Moved encoder params from separate Adam optimizer into
  `_policy_params` (TRPO natural gradient group). CG + line search now jointly optimize
  actor and encoder. KL constraint covers the combined distribution shift. Line search
  verifies barrier feasibility for the joint actor+encoder update.
- `algorithms/constraint_trpo.py`: Added encoder gradient norm extraction from TRPO flat
  gradient vector (`_encoder_param_offset`, `_encoder_param_count`) for monitoring.
- `utils/logging.py`: `log_encoder_metrics()` now accepts `alg` parameter to read
  `_last_encoder_grad_norm` from the TRPO gradient (no `.grad` available after `autograd.grad`).
- `runners/constraint_encoder_runner.py`: Removed encoder optimizer save/load. Replaced
  `pre_encoder_kl` logging with `encoder_grad_norm`. Passes `alg` to `log_encoder_metrics`.
- `agents/rsl_rl_ppo_cfg.py`: Removed `num_encoder_epochs` and `encoder_lr` config fields.

### Removed
- `algorithms/constraint_trpo.py`: Deleted `_update_encoder()` method (22 lines).
  Encoder no longer has a separate update loop.
- `algorithms/constraint_trpo.py`: Removed `encoder_optimizer` (Adam), `_encoder_params`,
  `_has_encoder_params`, `_last_pre_encoder_kl` fields.
- `runners/constraint_encoder_runner.py`: Removed `encoder_optimizer.pt` checkpoint
  save/load (no separate optimizer to persist).

### Notes
- CG Fisher matrix automatically captures encoder's KL contribution: params that strongly
  affect the distribution get smaller steps via natural gradient curvature
- Encoder weight_decay was 1e-5 in Adam; now omitted (TRPO has no optimizer). If needed,
  L2 penalty can be added to the surrogate as future work
- Previous encoder grad_norm=1.0 (always clipped) was post-clip from separate Adam update.
  New metric reports pre-clip norm from the TRPO surrogate gradient, which is more informative
- Backward compatible: old configs with `num_encoder_epochs`/`encoder_lr` are silently
  ignored via `**_kwargs`

## [2026-03-27] Remove cost critic d_k^2 normalization and encoder line-search gating

### Context
Training analysis of run `2026-03-26_23-45-24` (254 iters) revealed two structural issues
in ConstraintTRPO:

**Issue A -- Cost critic `d_k^2` normalization**: The cost value loss divided per-constraint
MSE by `d_k^2`, intended to prevent large-budget constraints from dominating. Numerical
analysis showed the opposite effect: yaw_vel (`d_k=78.5`, `d_k^2=6162`) contributed 98.6%
of cost critic loss, while attitude (`d_k=1.0`) contributed ~0%. The normalization was
ineffective because raw MSE scales as `O(d_k^2)`, so dividing by `d_k^2` merely cancels
the scaling rather than equalizing gradient contributions. Literature survey confirmed
`d_k^2` normalization is non-standard -- OmniSafe, CPO, FOCOPS, and IPO all use plain MSE
for cost value functions.

**Issue B -- Encoder gated on `ls_success`**: Encoder update was hard-gated on line search
success, meaning encoder received zero gradient when line search failed. Survey of all
reference implementations (HORA, RMA, Extreme Parkour, RSL-RL PPO, PPG) found **no
precedent** for gating encoder updates on policy step acceptance. HORA/RMA train encoder
jointly with actor via PPO (no gating possible). Extreme Parkour uses periodic DAgger
(fixed frequency, not conditioned on step success). The gating creates a starvation risk:
encoder doesn't learn -> bad z -> bad actions -> constraint violation -> barrier dominates
-> line search fails -> encoder frozen (positive feedback loop). Current run showed 96.3%
success rate but longest freeze streak was 8 iterations (iter 142-149), during which reward
dropped 4.3x faster and z_std stagnated.

### Changed
- `algorithms/constraint_trpo.py`: Removed `d_k^2` normalization from cost value loss.
  Was: `(per_k_mse / self.d_k.pow(2).clamp(min=0.01)).mean()`.
  Now: `per_k_mse.mean()` (standard MSE, matching OmniSafe/CPO convention)
- `algorithms/constraint_trpo.py`: Removed `ls_success` gate on encoder update.
  Was: `if self.encoder_optimizer is not None and ls_success:`.
  Now: `if self.encoder_optimizer is not None:` (encoder always updates, matching HORA/RMA)

### Notes
- Cost critic uses shared backbone with reward critic (multi-head). If gradient conflict
  persists, consider separating into independent networks (OmniSafe standard)
- Encoder update uses post-TRPO log_prob as baseline. When ls fails, params are reverted,
  so post_trpo_lp equals old_log_prob, making ratio=1 and encoder gradient signal weak but
  not harmful (centering only, no policy shift)
- Barrier margins are narrowing (torque: 2.12->0.40, yaw_vel: 39.4->1.57), meaning line
  search failure rate may increase in later training. The encoder ungating prevents the
  starvation spiral in that scenario

## [2026-03-27] Fix line search metric spike logging artifact in ConstraintTRPO

### Context
During constrained ALBC training, `barrier_penalty` and `entropy` metrics spiked
sharply whenever line search failed. Investigation revealed this was a logging artifact,
not a real policy instability. The `surrogate()` closure sets `_last_barrier_penalty` and
`_last_mean_entropy` on every call. During backtracking line search (up to 10 attempts),
each call to `surrogate()` with rejected candidate parameters overwrites these monitoring
variables. On failure, `_line_search()` reverts policy params to `old_params`, but the
monitoring vars retain the last rejected candidate's values -- often with inflated barrier
penalty from near-constraint-boundary proposals. Literature confirms this is a known issue
with interior point methods: log(margin) diverges as margin approaches zero (Boyd &
Vandenberghe; Nocedal et al., SIAM 2008).

### Fixed
- `algorithms/constraint_trpo.py`: After line search failure, recalculate `surrogate()`
  with reverted parameters so `_last_barrier_penalty` and `_last_mean_entropy` reflect
  actual policy state, not rejected candidates

### Notes
- Only affects monitoring/logging -- actual policy update logic was already correct
  (params properly reverted on failure, encoder update correctly gated on `ls_success`)
- The structural causes of line search failure itself (adaptive threshold constant gradient,
  TRPO scale invariance, barrier landscape ill-conditioning) remain separate issues
