# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

For entries before 2026-03-27, see [changelog_legacy.md](changelog_legacy.md).

## [2026-03-27] Tune delta_scale and reward weights after delta action analysis

### Context
Analysis of run `2026-03-27_02-40-36` (139 iters, first delta action run) showed dramatic
improvements in actuator dynamics but attitude control regression:

**Dynamics success (delta action working):**
- effort_saturation: 91% -> 2.2% (PD controller no longer saturated)
- applied_torque_max: 12.3 -> 6.5 Nm (within 9.5 Nm constraint limit)
- joint_vel_max: 6.0 -> 2.1 rad/s (within 4.189 constraint limit)
- Torque cost_return: 92 -> 4.5 (within budget 20 for the first time!)
- Velocity cost_return: 91 -> 0.02 (within budget 10, essentially zero violation)

**Attitude regression:**
- Roll error: 17 -> 21.6 deg, Pitch error: 13 -> 18.8 deg (worse than absolute action)
- Per-step reward breakdown: command=-2.92 (97.3%), torque=-0.014 (0.5%), smoothness=-0.068 (2.3%)
- The 160:1 ratio between tracking and smoothness means the policy has almost no incentive
  to be smooth or energy-efficient -- only attitude matters in the reward landscape.

**Two issues identified:**
1. delta_scale=0.05 limits arm bandwidth: 2.9 deg/step means reaching 90 deg offset takes
   0.62 seconds. Arm may be too slow to compensate for disturbances.
2. Reward weight imbalance: k_tau and k_s contribute <3% combined to the total reward,
   making torque efficiency and smoothness invisible to the optimizer.

### Changed
- `config.py`: `delta_scale` 0.05 -> 0.08 (arm bandwidth +60%, max 4.6 deg/step, max PD
  torque = 8.0 Nm still within 9.5 limit). Time to reach 90 deg offset: 0.39s (was 0.62s).
- `config.py`: `k_tau` -0.001 -> -0.01 (10x increase, torque penalty ~5% of reward).
  Encourages energy efficiency now that constraints handle hard limits.
- `config.py`: `k_s` -0.05 -> -0.2 (4x increase, smoothness penalty ~10% of reward).
  Discourages jerky acceleration in delta action space.

### Notes
- Reward contribution targets: command ~85%, smoothness ~10%, torque ~5%.
  Previous: command 97.3%, smoothness 2.3%, torque 0.5%.
- delta_scale=0.10 was considered but rejected: PD torque = 10 Nm exceeds 9.5 limit.
- k_c=-8.0 intentionally unchanged: attitude tracking remains the primary objective.

## [2026-03-27] Switch from absolute to delta action parameterization

### Context
Analysis of runs 01-51-47 (computed_torque) and 02-09-08 (applied_torque fix) revealed that
while the torque constraint fix dramatically improved gradient stability (enc_grad max 19680->218,
entropy 0.93->1.60), reward and constraint cost_returns showed no improvement. Per-step reward
components were actually worsening: command -0.48->-1.74, smoothness -0.04->-0.35.

**Root cause: Gaussian policy noise creates high-frequency jitter in joint targets.**

The policy samples `a_t ~ N(mean, std)` independently each step. With `action_scale = pi` and
`noise_std = 0.64`, this creates per-step target jumps of `0.64 * pi = 2.0 rad = 115 deg` from
noise alone (even if the mean is perfectly stable). The PD controller (Kp=100) cannot track these
rapid target changes, resulting in 91% effort saturation and permanent torque/velocity constraint
violation.

**Key calculation:** For `applied_torque < 9.5 Nm` with Kp=100, position error must be < 0.095 rad
(5.4 deg). Even at min_std=0.2, noise amplitude = `0.2 * pi = 0.63 rad = 36 deg` -- 7x the
constraint-feasible range.

**Reference comparison:** TDC controller achieves 0.2 deg (no DR) to 6 deg (max DR) attitude error
on the same system, using small incremental IK-computed joint deltas. The RL policy at 17-18 deg
is worse than the classical controller because of action jitter.

**Paper reference (NORBC):** Uses `sigma_a = 0.4` (8x smaller than our pi) for legged robots.
However, absolute scaling doesn't suit ALBC's continuous-rotation arm because +-23 deg range may
be insufficient. Delta action is the right approach: limits per-step change while allowing any
absolute position via accumulation.

### Changed
- `config.py`: Replaced `action_scale: float = pi` with `delta_scale: float = 0.05`.
  At 50Hz, max joint velocity = 0.05 * 50 = 2.5 rad/s (within 4.189 constraint).
  With min_std=0.2, noise position change = 0.65 deg/step (within PD tracking range).
- `albc_env.py`: `_apply_joint_pd_action()` changed from absolute
  (`q_des = q_nominal + scale * a_t`) to delta accumulation
  (`q_des += delta_scale * a_t`). Joint limits still enforced via clamp.

### Notes
- Reward weights (k_c=-8.0, k_s=-0.05, k_tau=-0.001) intentionally unchanged to isolate
  the effect of delta action. The 160:1 tracking/smoothness ratio may need adjustment later.
- Reset behavior unchanged: on episode reset, q_des initializes to current joint position
  (already the case in `_reset_action_buffers`).
- Smoothness reward now penalizes acceleration (change in velocity command) rather than
  change in absolute position target -- a more physically meaningful quantity with delta actions.

## [2026-03-27] Fix torque constraint: computed_torque -> applied_torque

### Context
Analysis of run `2026-03-27_01-51-47` (200 iters, post-standardization + alpha=0.05) revealed that
torque and velocity cost_returns were not improving -- in fact worsening (torque: 30.7 -> 98.7,
velocity: 28.3 -> 88.3) while reward also degraded (-9.2 -> -30.6).

**Root cause:** `torque_limit_cost()` checked `_robot.data.computed_torque` (PD controller output
BEFORE actuator clamping) against limit=9.5 Nm. With Kp=100 and ImplicitActuator, computed_torque
ranges 326-554 Nm -- always exceeding 9.5 Nm on every step, making the constraint 100% violated
and fundamentally unsatisfiable.

**Evidence:**
- `computed_torque_abs_max`: 326-554 Nm (always >> 9.5 Nm limit)
- `applied_torque_abs_max`: 12.0-12.5 Nm (post-clamp by effort_limit_sim=13 Nm)
- `effort_saturation_frac`: 78-95% (PD almost always requests more than actuator can deliver)
- Torque violation rate: ~100% (every step), budget: 20% -> unsatisfiable by 5x

**Impact on training:** The unsatisfiable constraint created constant barrier gradient at the
alpha*d_k floor margin. This gradient:
1. Provided no directional information (100% vs 99% violation = same barrier pressure)
2. Dominated reward signal (barrier:reward ratio still ~4:1 even after alpha fix)
3. Pushed exploration down (noise_std 0.61 -> 0.41, entropy 0.73)
4. Caused encoder grad_norm spikes (19680 at iter 156) when cost_surrs pushed margin to clamp floor

**Asset spec:** Hero Agent ALBC arm uses effort_limit_sim=13.0 Nm (PhysX hard cap, above motor
stall torque 9.5 Nm). The constraint should measure actual motor output (applied_torque), not
the PD controller's unbounded internal computation. The reward `joint_torque` already correctly
uses `applied_torque`.

### Fixed
- `mdp/constraints.py`: `torque_limit_cost()` now uses `_robot.data.applied_torque` instead of
  `_robot.data.computed_torque`. With applied_torque, violation is achievable (~70-80% initially)
  and decreases as the policy learns smoother control, providing actionable barrier gradient.

### Notes
- Velocity constraint (limit=4.189 rad/s) is correct: checks actual joint_vel against real motor
  max speed. 91% violation rate is high but physically achievable, not a metric error.
- With torque constraint fixed, barrier gradient should focus on velocity + yaw_vel, allowing
  reward (especially torque/smoothness components) to improve.
- Encoder grad_norm spikes should reduce: the constant noise from the unsatisfiable torque
  constraint was a major source of barrier gradient instability.

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
