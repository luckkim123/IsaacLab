# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

For entries before 2026-03-27, see [changelog_legacy.md](changelog_legacy.md).

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
