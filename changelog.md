# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [2026-03-26] Add gradient decomposition diagnostics to ConstraintTRPO

### Context
Post-bugfix run (2026-03-26_20-12-54, 429 iters) showed grad_norm still growing from
O(3) to O(18K) with max 42M. Initial analysis assumed barrier gradient dominated, but
surrogate decomposition revealed reward_surr spikes to 0.66 at high-grad iters (should
be ~0 if ratio=1.0). This implies ratio deviates from 1.0 at the first surrogate call,
which should not happen if policy params are unchanged since rollout. Root cause unclear
-- added diagnostic logging to decompose gradient into reward vs barrier components and
track ratio statistics to identify the exact source of gradient explosion.

### Added
- `algorithms/constraint_trpo.py`: Gradient decomposition in `_trpo_step()`: computes
  reward-only gradient separately, derives barrier gradient by subtraction. Logs ratio
  stats (mean/min/max), reward_surr value, and minimum barrier margin per iteration.
- `runners/constraint_encoder_runner.py`: 7 diagnostic TB metrics (`Diag/reward_grad_norm`,
  `Diag/barrier_grad_norm`, `Diag/ratio_mean`, `Diag/ratio_max`, `Diag/ratio_min`,
  `Diag/reward_surr`, `Diag/margin_min`)

### Notes
- Diagnostics are temporary -- will be removed once root cause is identified
- Key question to answer: is gradient explosion from reward gradient, barrier gradient, or both?
- If ratio_mean deviates from 1.0, investigate EmpiricalNormalization drift (still enabled
  on actor/encoder/critic despite memory claiming "fixed analytical" replacement)

## [2026-03-26] Fix 3 gradient explosion bugs in ConstraintTRPO

### Context
Analysis of run 2026-03-26_18-12-25 (delta EE mode, 580 iters) revealed training was
completely broken by gradient explosions -- not a methodology issue, but algorithmic bugs.

Symptoms: grad_norm grew from O(1) to O(1e16) over 500 iterations, shs peaked at 5.97e11,
step_norm collapsed to 0.01 (policy effectively frozen), encoder_grad_norm hit inf in 130/458
iterations. At iter 562+ the system entered permanent failure: grad=inf every iteration,
line search always fails, encoder permanently frozen.

Root cause traced to cost advantage standardization dividing by (std + 1e-8) for binary
constraints (accum_rot, overshoot, attitude_abs) that have near-zero std because they
rarely fire. A single non-zero cost advantage sample (e.g., 0.1) gets amplified by
0.1/1e-8 = 1e7, which flows through the barrier surrogate into TRPO gradient.
Progressive deterioration: each spike slightly perturbs the policy, changing violation
patterns, creating a positive feedback loop until infinity.

Two secondary bugs identified: (1) encoder update used pre-TRPO old_log_prob as baseline,
causing ratio drift over 5 encoder epochs; (2) no guard against inf gradients after
clip_grad_norm_, which produces NaN parameters when clipping inf.

### Fixed
- `algorithms/constraint_trpo.py`: Cost advantage standardization epsilon changed from `(std + 1e-8)` to `std.clamp(min=1.0)` -- binary constraints with near-zero std keep natural scale instead of 1e8x amplification
- `algorithms/constraint_trpo.py`: Encoder update now re-snapshots log_prob AFTER TRPO step (was using pre-TRPO baseline, causing ratio explosion over encoder epochs)
- `algorithms/constraint_trpo.py`: Added gradient norm clipping on TRPO surrogate gradient before CG solver (defense-in-depth against barrier gradient spikes)
- `algorithms/constraint_trpo.py`: Added `isfinite` guard after encoder `clip_grad_norm_` -- skips optimizer step when gradients are inf/NaN to prevent parameter corruption

### Notes
- Bug 1 (cost_adv 0-division) is the root cause; Bugs 2-3 are downstream amplifiers
- `max_grad_norm=1.0` already existed for value function; now also applied to TRPO policy gradient
- The `clamp(min=1.0)` means constraints with std < 1.0 get centering only (no scaling), std > 1.0 get full standardization
- Needs smoke test to verify gradient stability before full training run

## [2026-03-26] Switch to delta EE action mode: fix arm freeze root cause

### Context
Systematic root cause analysis of arm freeze across all ee_position runs revealed the
fundamental issue is not tanh squashing or barrier tuning, but the **action parameterization
itself**. In absolute ee_position mode, max extension = max restoring torque, so the
physical optimum lies at the action boundary (action_size=1.41 = sqrt(2), both axes
at tanh saturation). At the boundary (pre-tanh mu~2.65):
- g2_std = 0.00 deg (literally zero joint diversity from sampling)
- EE position range = 0.022m (2.2cm out of 0.92m workspace)
- All sampled actions produce identical physical outcomes -> advantage = noise
- TRPO gradient has no directional signal -> policy frozen permanently

This is a **structural trap**: no gradient-based method can escape because the reward
surface is flat in the sampled action region. Sigma optimizer compounds the problem by
monotonically reducing noise_std (1.0->0.46, never increases) since boundary dynamics
make variance reduction always "locally optimal". Smoothness penalty (weight=-0.5)
accelerates trap entry by penalizing action changes, collapsing act_rate from 1.18 to
0.05 within 80 iterations.

Analysis of RSL-RL, HORA, Isaac Lab Factory, and hero_agent revealed that ALL working
systems use delta/velocity actions where optimal steady-state = action(0,0) = center of
action space. No system uses absolute EE position with TRPO.

Solution: delta EE mode where actions specify EE displacement per control step, not
absolute position. Current EE computed via FK, delta added, then IK + rate limiting.
Simultaneously removed tanh squashing (unnecessary with centered action space) and
reduced smoothness penalty 10x.

Smoke test (10 iter, 64 envs) confirmed: action_size_mean=0.74 (not boundary-saturated),
action_rate_mean=0.74 (active movement). Previous runs always showed 1.41/0.00.

### Added
- `albc_env.py`: `_compute_ee_position()` FK helper method (joint angles -> EE xy)
- `albc_env.py`: `_apply_ee_delta_action()` method: FK(current joints) + scaled delta + IK + rate limit
- `config.py`: `ee_delta_scale: float = 0.02` (0.02m/step at 50Hz = 1.0 m/s max)
- `docs/arm-freeze-analysis.md`: Complete root cause analysis document with 6 hypotheses, 5 analysis tracks, numerical evidence

### Changed
- `config.py`: `action_mode` default "ee_position" -> "ee_delta", docstring updated for 3 modes
- `config.py`: `smoothness_weight` -0.5 -> -0.05 (H2: stillness attractor trigger, 10x reduction)
- `albc_env.py`: `_pre_physics_step()` dispatch now handles "ee_delta" mode
- `albc_env.py`: `_reset_action_buffers()` refactored to use `_compute_ee_position()` for FK; ee_delta uses zero init (action=0 = hold position)
- `encoder/actor_critic_encoder.py`: `act()` removed tanh squashing, replaced with `.clamp(-1, 1)`
- `encoder/actor_critic_encoder.py`: `act_inference()` removed tanh, replaced with `.clamp(-1, 1)`
- `encoder/actor_critic_encoder.py`: `get_actions_log_prob()` docstring updated (no more raw/squashed distinction)
- `algorithms/constraint_trpo.py`: Removed `storage.raw_actions` (no longer needed without tanh)
- `algorithms/constraint_trpo.py`: `act()` simplified: uses clipped actions directly for log_prob
- `algorithms/constraint_trpo.py`: `update()` uses `storage.actions` instead of `storage.raw_actions`

### Removed
- `encoder/actor_critic_encoder.py`: `last_raw_actions` attribute (tanh removed, no raw/squashed split)
- `algorithms/constraint_trpo.py`: `storage.raw_actions` tensor and all raw action handling

### Notes
- ee_delta_scale=0.02m at 50Hz: full workspace traverse (~0.46m) in ~0.46s at max action
- entropy_coef stays at 0; consider 0.001-0.01 if sigma still collapses in delta mode
- torque_weight (-0.001) kept unchanged; review if delta mode training shows issues
- H4 (constraint implicit stillness) was REJECTED: barrier gradient only 15% of TRPO total
- Full analysis: `constrained_albc/docs/arm-freeze-analysis.md`

## [2026-03-26] Fix ee_position mode: rate limiting + reset initialization + quadratic reward

### Context
After removing fix2 (barrier weights), the isolation test run 2026-03-26_15-11-51 still showed
identical arm freeze (act_size=1.41, act_rate=0.02, roll_err=43 deg). Comparing configs between
the working run (2026-03-25_18-22-28) and all failed runs revealed the true root cause:
`action_mode: ee_position` (commit 851f946d) was added AFTER the working run, which used
`joint_velocity` mode.

The fundamental problem: ee_position mode applies IK-computed joint angles directly as position
targets, with no rate limiting. In joint_velocity mode, the integrator naturally limits joint
position delta to `max_vel * dt = 0.126 rad/step`. Without this, random exploration during early
training causes huge joint angle jumps → high torque/velocity → constraint costs spike →
barrier gradient pushes toward "don't move" → arm freeze.

Additionally, the reset initialization was wrong for ee_position: action buffers were zeroed
(action=[0,0] = "EE at center"), but joints start at random positions. This caused a false
smoothness spike on the first step of every episode. `_joint_pos_targets` was never reset,
leaving stale targets from the previous episode.

Reference: `/workspace/references/abpc_dynamixel_control/src/advanced_albc_controller.cpp`
operates at 50Hz with smooth EE target changes from IMU-derived PID/TD controller. The ABPC
servo tracks via internal PD at ~1kHz. Our implementation must rate-limit since RL actions
lack the natural smoothness of a PID controller.

### Changed
- `albc_env.py`: Added joint position rate limiting to `_apply_ee_position_action()`. IK target
  delta is clamped to `max_joint_velocity * control_dt` (= 0.126 rad/step) with atan2 wrapping
  for shortest-path angular interpolation. Matches joint_velocity mode's implicit rate limit.
- `albc_env.py`: `_reset_action_buffers()` now initializes `_joint_pos_targets` to current joint
  position for both modes. For ee_position mode, action buffers are initialized via FK of current
  joint position (normalized to [-1,1]) instead of zero, preventing first-step smoothness spike.
- `config.py`: Re-applied quadratic reward (command_weight=-1.0, command_type="quadratic") for
  testing with rate limiting. Quadratic gradient = 2*c*e never vanishes, addressing one-axis collapse.
- `rewards.py`: Updated `ALBCRewardCfg` defaults to quadratic (command_weight=-1.0, command_type="quadratic")

### Notes
- Fix2 (barrier weights) removed in previous commit (a1bcb86f), kept removed
- First test of quadratic reward WITHOUT fix2 and WITH rate limiting
- Elbow-up IK convention (g2 >= 0) restricts joint space to half; may need revisiting
- Proprio history includes prev_actions whose distribution changes between modes

## [2026-03-26] Fix raw action storage for tanh squashing + barrier_t=100

### Context
Run 2026-03-26_16-55-58 (first tanh squashing run) showed arm freeze resolved -- arm was
actively moving (act_size=0.91, act_rate=0.80, jnt_vel=3.49 vs previous 0.09). However,
`TRPO/encoder_grad_norm` showed 34/248 iterations at `inf` (14%), with raw values reaching
10^12 before grad clipping. `Constraint/barrier_penalty` also showed `-inf` values. TRPO
grad norm reached millions. Surrogate loss spiked to -40+. All fundamentally broken.

Root cause: `get_actions_log_prob()` was calling `atanh(actions.clamp(-0.999, 0.999))` to
invert the tanh squashing, but this is numerically lossy. Example: raw=4.0 -> tanh=0.99933
-> atanh=3.654 (not 4.0). When encoder shifts the distribution, the log_prob at the wrong
raw value produces extreme importance sampling ratios: exp(log_prob_new - log_prob_old)
reaching 10^6+, causing gradient explosion through the entire TRPO pipeline.

The correct approach: store raw (pre-tanh) actions in the rollout buffer during collection,
and use them directly for all log_prob/ratio computations. This eliminates the lossy atanh
round-trip entirely. The tanh-squashed actions are still sent to the environment (bounded
to (-1,1) for workspace mapping), but all probability computations use the exact raw values.

Additionally, barrier_t increased from 50 to 100 (paper nominal). Three constraints were
over budget (joint_torque 2.2x, joint_vel_limit 2.6x, yaw_vel 1.6x), causing barrier
gradient spikes up to +1.29. At t=100, barrier gradient is halved: 1/(margin*100) vs
1/(margin*50), reducing spike severity.

### Changed
- `encoder/actor_critic_encoder.py`: `act()` stores `last_raw_actions` (pre-tanh sample).
  `get_actions_log_prob()` now expects raw actions directly -- atanh inversion removed entirely.
- `algorithms/constraint_trpo.py`: Added `storage.raw_actions` tensor in `init_storage()`.
  Collection stores raw actions alongside squashed actions. `update()` uses `raw_actions` for
  all surrogate/ratio computations instead of squashed actions.
- `algorithms/constraint_trpo.py`: `barrier_t` default 50.0 -> 100.0 (paper nominal)
- `agents/rsl_rl_ppo_cfg.py`: `barrier_t` 50.0 -> 100.0, updated docstring

### Fixed
- `encoder/actor_critic_encoder.py`: Eliminated lossy atanh round-trip that caused importance
  sampling ratio explosion (10^12) and gradient blow-up in TRPO surrogate and encoder update

### Notes
- Raw actions stored separately from squashed actions: env receives tanh(raw), storage keeps raw
- `actions_flat` in update() is now raw (pre-tanh), not squashed -- all log_prob calls are exact
- barrier_t=100 halves barrier gradient magnitude, reducing spike-induced instability
- The `action_mean`/`action_std` properties still return raw Gaussian params (correct for KL)

## [2026-03-26] Add tanh squashing to actor output + expand workspace radius

### Context
Run 2026-03-26_15-43-39 (with rate limiting + quadratic reward) still showed complete failure:
command reward declining to -73, attitude error 32 deg, act_size=1.41 (saturated at sqrt(2)),
act_rate=0.00 (constant actions). TRPO surrogate_loss exactly matched barrier_penalty (-0.249)
throughout training, meaning reward surrogate contribution was ~zero.

Root cause analysis: the actor MLP outputs unbounded mu, and Gaussian sampling produces actions
well beyond [-1, 1]. These are scaled by workspace_radius=0.40 and then clamped at r_max=0.461m.
Multiple different actions (e.g., 1.2, 1.5, 2.0) all map to the same physical EE position after
clamping, creating a FLAT reward landscape. Policy gradient = 0 in these regions because different
actions produce identical outcomes. The policy converges to constant extreme actions and cannot
escape this local optimum.

Solution: apply tanh to raw Gaussian samples, bounding actions to (-1, 1). Key mathematical
insight: KL divergence is invariant under bijective transforms (tanh is a diffeomorphism from
R to (-1,1)). The Jacobian of tanh cancels in the importance sampling ratio. Therefore: no
changes needed to KL computation, Fisher vector product, surrogate loss, or line search.

tanh chosen over softsign because: workspace 95% requires raw=1.83 (tanh) vs raw=19 (softsign),
matching typical MLP output range. tanh provides 19x better exploration at y=0.9. Standard
practice (SAC). softsign's polynomial gradient decay advantage is not needed with proper init.

workspace_radius increased from 0.40 to 0.461 (= L1+L2-0.005) so tanh output (-1,1) maps to
the full reachable workspace. The diagonal workspace clamp remains as a safety net for cases
where both axes are near-maximal (radius = R*sqrt(2) > r_max).

### Changed
- `encoder/actor_critic_encoder.py`: `act()` now returns `tanh(distribution.sample())` instead
  of raw `distribution.sample()`. All actions are bounded to (-1, 1) by construction.
- `encoder/actor_critic_encoder.py`: `act_inference()` returns `tanh(actor(obs))` for consistency.
- `encoder/actor_critic_encoder.py`: `get_actions_log_prob()` inverts squashed actions via
  `atanh(actions.clamp(-0.999, 0.999))` then computes Gaussian log_prob. Jacobian correction
  not needed (cancels in importance sampling ratio).
- `config.py`: `workspace_radius` 0.40 -> 0.461 (full workspace coverage, matches r_max)

### Notes
- KL, FVP, surrogate, line search, value function, constraints: NO changes (KL invariance)
- `action_mean`/`action_std` properties still return raw Gaussian parameters (correct for KL)
- `_update_action_buffers` clamp(-1,1) kept for joint_velocity mode compatibility
- entropy_coef=0.0 so Gaussian vs squashed entropy difference is moot
- Existing checkpoints incompatible (action distribution semantics changed), fresh training needed
- atanh numerical stability: clamp to (-0.999, 0.999) prevents inf at boundaries

## [2026-03-26] Remove violation-proportional barrier weights (fix2)

### Context
Isolation test (exponential reward + fix2 barrier weights, run 2026-03-26_14-59-24) confirmed
fix2 is the root cause of arm freeze, not fix1 (quadratic reward). Run showed identical failure:
act_size=1.41 (saturated), act_rate=0.02, roll_err=42 deg, pitch_err=33 deg.

Analysis of early-iteration barrier weights revealed the mechanism:
- At iter 10: joint_vel_limit w_k=5.22, joint_torque w_k=3.42 (barrier weights from fix2)
- Total barrier gradient: 0.75 (fix2) vs 0.20 (standard) -- 3.75x amplification
- Positive feedback loop: exploration -> high cost -> amplified barrier -> "don't move" -> arm freeze
- Working run (2026-03-25_18-22-28, no fix2) had same initial cost levels but standard barrier
  allowed gradual cost reduction without arm freeze

Fix2 was introduced to compensate for "constant gradient" from adaptive thresholds (barrier
structural analysis problem 1). But the working run proved constant gradient IS sufficient --
it's a feature, not a bug. The adaptive threshold margin (alpha*d_k) provides stable, consistent
constraint pressure that lets the policy explore while gradually satisfying constraints.

### Removed
- `constraint_trpo.py`: Removed `barrier_weights = (mean_cost_returns / d_k).clamp(min=1.0)`
  computation and its usage in both TRPO surrogate and sigma update barriers. Restored standard
  log barrier: `-sum_k log(margin_k) / t`

### Notes
- This fully reverts the fix2 portion of commit 7239ff6f
- Combined with the fix1 revert (previous entry), config now matches working run 2026-03-25_18-22-28
- The one-axis collapse problem (which motivated fix1/fix2) still needs a different solution

## [2026-03-26] Revert to exponential reward for isolation test

### Context
Quadratic reward (fix1 from earlier session) produced consistent arm-freeze failure across
multiple training runs: arm saturates at action boundary (act_size=1.41), action rate drops
to ~0.03, and command reward monotonically worsens while smoothness/torque improve. The
arm-freeze pattern was identical with command_weight=-1.0, -7.0, and even with smoothness
completely removed (weight=0.0).

Deep investigation of the optimization pipeline revealed:
1. Advantage normalization (double: storage + update) erases absolute reward magnitude,
   making weight/coefficient tuning irrelevant.
2. In an all-negative reward landscape (quadratic penalty), the "least negative" timesteps
   (which get positive advantage after normalization) are those with least movement, not
   those with lowest attitude error. This systematically directs the policy gradient toward
   "freeze arm" rather than "find good position."
3. Exponential reward (positive per-step values) avoids this by giving positive advantages
   to timesteps with genuinely low error, regardless of movement.

To isolate whether fix1 (quadratic reward) or fix2 (barrier weights) caused the failure,
reverted reward to exponential (matching working run 2026-03-25_18-22-28) while keeping
fix2 (violation-proportional barrier weights) active.

### Changed
- `mdp/rewards.py`: Restored exponential mode in `command_reward()` with `command_type`
  parameter ('exponential' or 'quadratic'). Default: exponential with coeff 5.0/7.5.
  Re-added `command_type` field to `ALBCRewardCfg`. torque_weight default: -0.001.
- `config.py`: Reverted reward config to exponential (command_weight=+5.0, coeff_roll=5.0,
  coeff_pitch=7.5, smoothness_weight=-0.5, torque_weight=-0.001)
- `albc_env.py`: Pass `command_type` from config to `command_reward` params

### Notes
- Fix2 (violation-proportional barrier weights in constraint_trpo.py) is PRESERVED
- If this run works: fix1 (quadratic) was the problem, exponential reward is necessary
- If this run fails: fix2 (barrier weights) is the problem or the combination matters
- The original one-axis collapse problem (exponential gradient vanishing) may need a
  different solution than quadratic penalty

## [2026-03-26] Quadratic reward + violation-proportional barrier weights

### Context
Diagnosed the "one-axis collapse" problem: EE position mode run `2026-03-26_10-47-16`
showed pitch converging to 6.5 deg while roll diverged to 49 deg -- the EXACT INVERSE
of joint velocity mode runs (roll 8 deg, pitch 20 deg). The pattern was consistent:
whichever axis learned first monopolized the optimizer, and the other axis never recovered.

Deep analysis of the TRPO gradient dynamics revealed the root cause was NOT the optimizer
but the reward function: `exp(-c*e^2)` has vanishing gradient at large errors (>40 deg,
gradient falls to 1.8% of peak). Combined with TRPO's shared KL budget, the axis with
stronger gradient consumed 99% of the update budget. This created a race condition: the
weak axis's update rate was slower than its physical drift rate, causing permanent divergence.

Key insight from cross-system comparison: quadruped locomotion systems NEVER have this
problem because they use quadratic (`-e^2`) or linear (`-|e|`) penalties whose gradients
NEVER vanish. The gradient ratio between axes is constant (cp/cr) at all error levels,
preventing KL budget starvation. Per-axis advantage decomposition was considered but
rejected as over-engineering -- the real fix is using the right reward function.

Also identified why 3 constraints (joint_torque, joint_vel_limit, yaw_vel) were diverging:
the adaptive barrier threshold `d_k^i = max(d_k, J_C_k + alpha*d_k)` produces a CONSTANT
margin `alpha*d_k` when over budget, giving CONSTANT barrier gradient regardless of violation
severity. With reward gradient at 2.12 vs barrier gradient at 0.05 (ratio 42:1), the
reward always won.

### Changed
- `mdp/rewards.py`: Replaced exponential reward `exp(-c*e^2)` with quadratic penalty
  `c*e^2` (positive output, used with negative weight). Gradient = 2*c*e (linear in error,
  never vanishes). Pitch/roll gradient ratio = cp/cr = 1.5 (constant at all error levels).
  command_weight changed from +5.0 to -1.0, coeff_roll 5.0 -> 1.0, coeff_pitch 7.5 -> 1.5.
  Default torque_weight synced to -0.0001 (was -0.001 in default, already -0.0001 in config).
- `config.py`: Updated ALBCRewardCfg instantiation to match new quadratic defaults
- `albc_env.py`: Removed `command_type` parameter from reward term params, updated docstring
- `algorithms/constraint_trpo.py`: Added violation-proportional barrier weights
  `w_k = max(1, J_C_k/d_k)`. Violated constraints get amplified barrier gradient proportional
  to how far they exceed budget. Applied in both TRPO surrogate and sigma Adam step.
  Satisfied constraints (w_k=1) are completely unchanged (backward compatible).

### Removed
- `mdp/rewards.py`: Deleted all non-exponential command_type variants (laplacian,
  min_laplacian, smooth_min_laplacian, quadratic fallback). The exponential kernel was
  also removed as part of the switch to quadratic. `command_type` config field deleted.

### Notes
- Quadratic reward is unbounded negative (larger error = more negative). Episode total at
  20 deg both axes: ~-23 (vs old exponential ~+71). Termination penalty -10.0 is comparable.
- Barrier weight growth is uncapped. If needed, `clamp(min=1.0, max=N)` can be added later.
- The 1.5x pitch coefficient ratio preserved from old config (was 7.5/5.0 = 1.5).

## [2026-03-26] Add EE position action mode for constrained ALBC

### Context
Analysis of run `2026-03-25_18-22-28` (2500 iter, headroom fix) showed persistent pitch
error asymmetry: roll converged to 7.8 deg while pitch plateaued at 20 deg. Training
trajectory revealed the policy locked into a roll-priority strategy at iter 25 (roll 16
deg, pitch 39 deg) and never recovered. Exploration was NOT the issue -- noise_std
remained high until iter 800 but pitch showed no improvement.

Root cause: the additive reward structure `exp(-c_r*e_r^2) + exp(-c_p*e_p^2)` combined
with TRPO's conservative trust region created a local optimum. The policy learned to
extend the arm along Y (good for roll), creating x_EE near 0 (bad for pitch). Escaping
this requires temporarily worsening roll to reposition the arm, which TRPO's small
KL steps cannot achieve.

Additionally, the joint velocity action space required the policy to implicitly learn the
FK/IK/Jacobian mapping -- a nonlinear function of joint configuration that TDC computes
analytically. TDC achieves ~5 deg on both axes, proving the physics is not the limitation.

Solution: change the action space from joint velocity commands to desired EE position
(x, y) in body frame, with analytical 2-link IK converting to joint angle targets. This
makes the action semantics match the physics: action[0] controls x_EE (pitch torque),
action[1] controls y_EE (roll torque). No integration, no drift, no implicit Jacobian
learning required.

Initial test showed torque penalty scale increased ~10x due to abrupt target position
changes (no velocity integration smoothing). Reduced torque_weight accordingly.

### Added
- `albc_env.py`: `_apply_ee_position_action()` method -- analytical 2-link IK converting
  normalized EE position actions to joint position targets with workspace radius clamping
- `config.py`: `action_mode` parameter ("ee_position" default, "joint_velocity" legacy)
- `config.py`: `workspace_radius` parameter (0.40m, below kinematic max 0.466m for margin)

### Changed
- `albc_env.py`: `_pre_physics_step()` now branches on `action_mode`
- `config.py`: `torque_weight` -0.001 -> -0.0001 (10x reduction to match EE position mode
  torque scale; PD actuator generates larger torques with direct position targets)

### Notes
- Legacy `joint_velocity` mode preserved for backward compatibility
- Smoothness penalty now operates on EE position changes (da = EE_t - EE_{t-1}) which is
  more physically meaningful than joint velocity changes
- Encoder z_sweep fix also committed this session (separate commit dbee3bcc)
- All previous encoder z_sweep results for constrained ALBC were invalid (wrong indices)

## [2026-03-26] Fix encoder z_sweep dimension indexing for 280D concatenated input

### Context
Ran encoder z_sweep on constrained ALBC run `2026-03-25_18-22-28` (post-headroom-fix,
2500 iter) and noticed only Payload Mass showed sensitivity (13/13 dims, range 0.69)
while all other DR parameters had near-zero response. This led to a false "encoder
collapse" diagnosis -- z_min/z_max at softsign boundaries was interpreted as saturation.

Root cause: the constrained ALBC encoder takes a **280D concatenated input**:
`cat([policy_obs(13), history(240), privileged(27)])`. The z_sweep script's
`build_sweep_params_from_checkpoint()` only handled `input_dim == 28` (privileged-only)
and `input_dim == 19` (hero_agent). For 280D input, it fell through to the 19D hero_agent
path, sweeping indices 0-18 which target **policy_obs and early history** instead of the
privileged obs at indices 253-279.

Result: "Payload Mass" was actually sweeping `policy_obs[14]` (history buffer element),
not the actual payload mass at `privileged[10]` (index 263). The strong response was due
to injecting out-of-distribution values into history, not real DR sensitivity.

After fix: encoder shows **excellent** DR sensitivity across all 27 privileged parameters.
Most parameters activate 10-13/13 z dimensions with max ranges up to 1.71. The encoder
is NOT collapsed -- it has learned a rich 13D representation of the full 27D privileged
observation space. Previous sessions' z_sweep analyses for constrained ALBC (including
run `2026-03-25_15-01-22`) were also invalid.

### Fixed
- `scripts/analysis/common.py`: Added 280D concatenated input handling to
  `build_sweep_params_from_checkpoint()`. Detects full encoder input (>=100D), extracts
  the privileged portion at the end, and applies correct index offset (253 for 280D).
  Added `_build_constrained_albc_27d_sweep()` helper with proper DR ranges for all 27
  privileged dimensions. Refactored 28D case to reuse helper with yaw quad damping.

### Notes
- All previous constrained ALBC encoder z_sweep results are invalid and should be re-run.
- The encoder is working well; the real bottleneck is policy optimization (plateau at 5%,
  pitch error 20 deg, step_norm=0.01) and constraint costs still above budget.
- Binary cost gradient IS present via cost advantage (GAE), contrary to earlier analysis.
  The original paper uses binary costs successfully.
- Training run analysis (2026-03-25_18-22-28): reward 103, roll 7.8 deg, pitch 20 deg,
  ls_success=1.00, entropy collapsed (-0.38), noise at floor (0.20).

## [2026-03-25] TRPO max_kl increase + encoder KL gating fix

### Context
Analysis of run `2026-03-24_20-18-09` (max_kl=0.002, 2500 iter) showed TRPO step_norm
decaying from 0.18 to 0.009 while grad_norm grew to 0.5 -- policy wanted larger changes
but trust region blocked them. max_kl raised from 0.002 to 0.005.

Run `2026-03-25_10-46-21` (max_kl=0.005, 2190 iter) confirmed step_norm increased 40-55%
as intended, but exposed a critical encoder freeze: z_std stuck at 0.12 for the entire
run despite enc_grad growing from 0.002 to 0.17. Raw data showed `enc_added` (total_kl -
pre_encoder_kl) = 0.000 from iter 200 onward -- every encoder epoch was reverted by
KL gating.

Root cause: `max_encoder_kl=0.003` was derived from `max_kl(0.002) * kl_margin(1.5)`.
When max_kl was raised to 0.005, the TRPO step consumed more KL (pre_encoder_kl ~0.005
vs ~0.002), making the encoder gradient landscape steeper. Combined with encoder_lr=0.001
(3.3x higher than the working prior run), a single encoder epoch overshot the 0.003 KL
budget immediately. Additionally, Adam optimizer state was poisoned: `optimizer.step()`
executes before reversion, accumulating momentum/second-moment for gradients that were
never applied (2000+ iterations of phantom updates).

Comparison with working run `2026-03-24_20-18-09` (max_kl=0.002, enc_lr=0.0003):
encoder expanded z_std 0.12->0.71 in first 200 iters, then enc_added=0 afterward (KL
gating kicked in, but z was already expanded). Current run never got that initial window.

Additional analysis uncovered three structural issues for future investigation:
1. Barrier adaptive threshold locks margin at `alpha*d_k` (0.4/0.2) when violating --
   gradient never steepens regardless of violation severity.
2. Barrier gradient ~300x weaker than reward gradient in TRPO surrogate.
3. Smoothness penalty penalizes action changes (da), not action magnitude (a) --
   sustained saturation at a=[0.9,0.9] incurs zero smoothness penalty.

### Changed
- `agents/rsl_rl_ppo_cfg.py`: max_kl 0.002 -> 0.005 (prior commit)
- `agents/rsl_rl_ppo_cfg.py`: max_encoder_kl 0.003 -> 0.0075 (proportional to max_kl)
- `agents/rsl_rl_ppo_cfg.py`: encoder_lr 1e-3 -> 3e-4 (reduce per-epoch KL shift)

### Notes
- Encoder freeze diagnosis: enc_grad growing != encoder learning. Must check enc_added
  (total_kl - pre_encoder_kl) to verify updates are actually applied.
- Adam state poisoning: if encoder was frozen for many iterations, consider resetting
  encoder_optimizer.state on restart.
- Constraint/action issues (barrier weakness, action magnitude) are separate problems
  to address after encoder fix is validated.

## [2026-03-25] Encoder fix validation + torque_weight increase

### Context
Run `2026-03-25_15-01-22` (encoder fix applied: max_encoder_kl=0.0075, encoder_lr=3e-4)
completed 2500 iterations. Encoder fix confirmed working: enc_added 0.003-0.007 throughout
(vs 0.000 in previous frozen run), z_std expanded 0.13 -> 0.87.

encoder_z_sweep comparison proved encoder IS learning meaningful representations:
- NEW run (model_2499): Payload Mass 11/13 dims active (max range 0.74), Main CoG Z
  5/13, Main Ixx 2/13. Sensitivity growing over training.
- OLD frozen run (model_950): 1/13 dim barely active (max range 0.052). Essentially dead.

However, attitude error (~10 deg) and reward (~110) are identical between runs.
Encoder learning did not translate to performance improvement. Root cause analysis
of the barrier/TRPO system identified 5 structural issues:

1. Adaptive threshold pins barrier margin at alpha*d_k (0.4/0.2) when violating --
   gradient coefficient 1/(t*alpha*d_k) is constant regardless of violation depth.
2. Cost advantage standardization (mean subtraction) removes absolute-level signal.
3. TRPO trust region normalization cancels barrier_t scaling when reward gradient is
   ~0 (plateau). surrogate decomposition confirmed barrier is 100% of gradient.
4. No action magnitude penalty: smoothness penalizes da (rate), torque_weight=-0.001
   is 900x weaker than command reward. Policy sustains action saturation (1.25) freely.
5. Cost value loss d_k^2 normalization weakens critic gradient for large-budget constraints.

First intervention: increase torque_weight to create reward gradient aligned with
constraint reduction, breaking the TRPO scale-invariance deadlock (problem 3).

### Changed
- `config.py`: torque_weight -0.001 -> -0.01 (10x, ~6% of command reward magnitude)
- `mdp/rewards.py`: Updated torque_weight default and docstring with rationale

### Notes
- barrier_t alone is ineffective when reward gradient is zero (TRPO normalizes away scaling)
- torque_weight creates g_reward component aligned with barrier direction, enabling barrier_t
  to affect gradient direction mixing ratio
- Future fixes planned: violation penalty (B), standardization fix (C), d_k^2 removal (D)
- Encoder z sensitivity is heavily Payload Mass biased (11/13 dims). Other DR params weak.

## [2026-03-25] torque_weight -0.01 insufficient, escalate to -0.05

### Context
Run `2026-03-25_17-35-34` (torque_weight=-0.01) reached iter 387. Mid-run comparison
with previous run (torque_weight=-0.001) at identical iterations showed:
- Torque penalty now visible: -0.53 (was -0.06), ~8% of command reward
- Constraint costs reduced ~15% early (jt: 25.7 vs 39.1 at iter 50)
- But action_size plateaued at 1.37 (was 1.36) -- no meaningful reduction
- Constraint costs re-converged to near-violation levels (jt=39.6, jv=22.4 at plateau)
- -0.01 insufficient to break the "max action = max reward" equilibrium

Escalating to -0.05 (~30% of command reward) to create stronger gradient pressure
against action saturation. Risk: tracking performance may degrade if torque penalty
dominates. Monitoring needed.

### Changed
- `config.py`: torque_weight -0.01 -> -0.05
- `mdp/rewards.py`: Updated default and docstring with escalation rationale

## [2026-03-25] Constraint headroom fix: decouple PhysX limits from constraint thresholds

### Context
Root cause analysis revealed why joint_torque and joint_vel_limit constraints were
permanently violated (cr=40 vs d_k=20, cr=24 vs d_k=10): zero headroom between the
action space boundary and constraint threshold.

Velocity: `max_joint_velocity = 4*pi/3 = 4.189 rad/s` (action scaling) was identical
to `limit_rad_per_s = 4.189 rad/s` (constraint threshold). Action=1.0 immediately hits
the constraint. Meanwhile PhysX allowed up to `velocity_limit_sim = 6.28 rad/s`.

Effort: `effort_limit_sim = 9.5 Nm` (PhysX cap) was the same value the constraint read
via `_robot.data.joint_effort_limits`. DR further reduced this to 6.65-9.5 Nm, making
the constraint structurally unavoidable during transients with Kp up to 120 Nm/rad.

torque_weight escalation (-0.001 -> -0.01 -> -0.05) was a reward workaround for the
constraint system's structural failure. Run `2026-03-25_17-35-34` (torque_weight=-0.01)
showed ~15% constraint reduction but action_size stayed at 1.37. The real fix is giving
the policy headroom to operate below constraint thresholds within the action space.

Design pattern follows effort_limit: PhysX hard cap > constraint threshold (motor spec).
Constraint thresholds are fixed at motor specs (no DR) -- DR affects physics only.

### Changed
- `hero_agent.py`: effort_limit_sim 9.5 -> 13.0 Nm (PhysX hard cap, 27% above motor spec)
- `config.py`: max_joint_velocity 4*pi/3 -> 2*pi rad/s (matches PhysX velocity_limit_sim=6.28, 33% headroom)
- `config.py`: torque_weight -0.05 -> -0.001 (reverted, constraint system handles it now)
- `config.py`: effort_limit_cost params added (limit_nm=9.5, fixed motor spec threshold)
- `mdp/constraints.py`: effort_limit_cost uses fixed threshold instead of DR'd joint_effort_limits
- `mdp/rewards.py`: torque_weight reverted to -0.001 with updated docstring

### Notes
- All 6 constraints checked for headroom issues: only joint_torque and joint_vel_limit affected
- Constraint thresholds have no DR (fixed at motor specs). DR affects PhysX physics only.
- accum_rot, attitude_abs, overshoot, yaw_vel: already have adequate headroom, unchanged
- The 5 barrier structural issues (adaptive threshold, standardization, TRPO invariance,
  d_k^2 normalization) remain. This fix addresses the most basic prerequisite: the policy
  must be ABLE to satisfy constraints within the action space.

## [2026-03-24] Encoder dynamic input integration (NORBC alignment)

### Context
Two consecutive training runs with constrained TRPO failed with different encoder
failure modes:
- Run `2026-03-24_19-05-41` (546 iter): encoder gradient death (enc_grad -> 0.0005).
  Roll error oscillated 10-20 deg, constraints (joint_torque, joint_vel_limit) diverging.
- Run `2026-03-24_19-45-59` (181 iter, with reconstruction auxiliary loss): encoder z
  collapse (z_std 0.34 -> 0.088). Pitch error exploded to 39 deg. Decoder learned
  degenerate solution (mapped collapsed z to mean privileged_obs, ignoring z content).

Root cause analysis revealed the fundamental issue: **encoder input was exclusively
static DR parameters (27D), making z constant throughout each 3000-step episode.**
In NORBC/ANYmal/RMA papers, encoder input includes dynamic quantities (body velocity,
contact forces, terrain scans), making z time-varying and naturally creating policy-
encoder coupling. With static-only input, unique z samples per iteration = 4096
(num_envs) vs 262K (num_envs x steps) in the reference papers -- a 64x difference
in encoder gradient diversity.

The hero_agent PPO encoder already solved this by using
`cat([policy_obs, hist_flat, privileged])` as encoder input. The constrained_albc
encoder used only `privileged` -- a divergence from the reference architecture.

Reconstruction auxiliary loss was reverted (commit 4e8d218b) as it failed to prevent
z collapse and was treating symptoms rather than the root cause.

### Changed
- `encoder/actor_critic_encoder.py`: Encoder input changed from `privileged(27D)` to
  `cat([policy_obs(13), hist_flat(240), privileged(27)]) = 280D`, matching hero_agent
  and NORBC encoder design. Switched from `_FixedNormalization` (analytical DR stats)
  to `EmpiricalNormalization` (online Welford) since dynamic inputs have no fixed
  distribution. Added `_encode_from_parts()` helper to avoid double hist_flat computation
  in `_get_combined_obs()`. Fixed `_handle_dim_mismatch()` to use `encoder_input_dim`
  instead of `privileged_dim`. Added normalizer buffer dimension mismatch detection in
  `load_state_dict()`.
- `agents/rsl_rl_ppo_cfg.py`: `encoder_hidden_dims` [128, 64] -> [256, 128, 64] to
  handle 280D input (matching hero_agent). 280D -> 128 was too aggressive a compression.
- `encoder/actor_critic_encoder_constrained.py`: Updated docstring (encoder 280D input).
- `config.py`: Updated network dimension docstring to reflect 280D encoder input.

### Removed
- `encoder/actor_critic_encoder.py`: Removed `_FixedNormalization` class and
  `_build_fixed_encoder_normalizer()` method (100+ lines of analytical DR stats).
  No longer needed with EmpiricalNormalization.

### Notes
- Reconstruction auxiliary loss was tried and failed: decoder learned degenerate solution
  (predicting mean privileged_obs with collapsed z). Root cause was static encoder input,
  not lack of auxiliary gradient. Do not re-attempt auxiliary losses on encoder.
- Old checkpoints (27D encoder input) will auto-reinitialize encoder via dim mismatch
  detection. This is intended since all recent runs failed.
- constraint_trpo.py required NO changes: encoder_prefixes, _update_encoder(), and KL
  rollback all work transparently with the new architecture.

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

## [2026-03-23 Summary] Modified IPO barrier, TRPO tuning, DORAEMON IS improvements (14 sessions)

### Context
Intensive tuning day focused on making constrained TRPO work end-to-end. Started from
the Lagrangian->log-barrier (Modified IPO) migration and went through noise_std runaway
fix, max_kl scaling for 2D actions, command reward tuning, DORAEMON IS improvements, and
encoder activation change. Key discoveries: entropy_coef must be 0 (constant upward
pressure on sigma with no counterforce during reward plateau), barrier_alpha was 15x too
large (0.3 vs paper's 0.02), and max_kl must scale with action dimensionality (0.01*2/12
= 0.002 for 2D actions). DORAEMON IS estimator upgraded with soft traversability and ESS
monitoring. Encoder activation changed from tanh to softsign (gradient 1/(1+|x|)^2 vs
sech^2(x); softsign 7x better at z=2). Full code simplification and weight_decay fix for
encoder saturation (P0 bug: both param groups had WD=0).

### Changed
- `algorithms/constraint_trpo.py`: Lagrangian replaced with Modified IPO log-barrier.
  entropy_coef default 0.01->0. barrier_alpha 0.3->0.02. max_kl default 0.01->0.002.
  max_encoder_kl 0.016->0.003. Added adaptive thresholds, per-constraint cost advantage
  standardization, TRPO step diagnostic logging (shs, step_norm, grad_norm, backtracks).
  Encoder weight_decay 0->1e-4 (P0 fix). EmpiricalNormalization replaced with fixed
  analytical normalization for encoder.
- `agents/rsl_rl_ppo_cfg.py`: max_kl 0.01->0.002, max_encoder_kl 0.016->0.003,
  entropy_coef 0.01->0, barrier_alpha 0.3->0.02. Added DORAEMON soft traversability
  and ESS monitoring config. Privileged obs 23D->28D (added joint gains, damping params).
- `config.py`: Reward architecture: exponential command (k_c=5/7.5) + torque penalty
  (-0.001). command_weight=5.0. Removed Laplacian reward variants. DR curriculum
  expanded with DORAEMON sensitivity logging.
- `mdp/rewards.py`: command_reward switched to coefficient form exp(-c*e^2) with
  c=5 (roll), c=7.5 (pitch). Removed sigma-based parameterization.
- `mdp/events.py`: DORAEMON IS estimator: soft traversability (exp(-err/threshold)
  instead of binary), ESS ratio monitoring (ess/N), per-param sensitivity logging.
- `encoder/actor_critic_encoder.py`: Activation tanh->softsign. EmpiricalNorm->fixed
  analytical. weight_decay fix: encoder 1e-4, actor/critic 0. Architecture [256,128,64].
- `runners/constraint_encoder_runner.py`: WandB logging dedup, TRPO step quality metrics,
  DORAEMON sensitivity per-param logging.

### Notes
- Constraint compliance improved after barrier_alpha fix: joint_torque 15.3 (dk=20),
  joint_vel 9.0 (dk=10) in the max_kl=0.002 run. Later degraded when encoder changes
  were applied (see 03-24 entries).
- P0 encoder saturation: sigmoid caused 13/13 dims binary by iter 50. Softsign + WD fix
  resolved it (z_std=0.94, 7x gradient improvement at z=2).
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

## [2026-03-22 Summary] Recovery mode removal + Lagrangian migration (2 sessions)

### Context
Recovery mode created deterministic safe/recovery oscillation (mean_cycle=159 iter) --
the policy alternated between "optimize reward" and "minimize cost" phases indefinitely.
Removed recovery mode; barrier-only approach then failed (all constraints OVER, barrier
penalty ~0). Migrated to adaptive Lagrangian multipliers (linear penalty + dual ascent)
as interim solution before returning to Modified IPO on 03-23.

### Changed
- `algorithms/constraint_trpo.py`: Removed recovery mode (73 lines). Replaced quadratic
  barrier with Lagrangian penalty (lambda_k * E[ratio * A_cost], dual ascent, lambda_max=0.5).
- `agents/rsl_rl_ppo_cfg.py`: Added lambda_lr=0.035, lambda_max=0.5.
- `runners/constraint_encoder_runner.py`: Lambda checkpoint, per-constraint lambda logging.

## [2026-03-21 Summary] Barrier bug fix, encoder input, privileged obs expansion (5 sessions)

### Context
Major debugging day. Found barrier penalty was structurally zero: per-constraint cost
advantage standardization set E[A_cost]=0, making cost_surr^2 gradient identically zero.
Fixed by using raw (unstandardized) cost advantages for barrier, standardized for recovery.
Also fixed recovery cycling (64-iter oscillation from gradient discontinuity at mode
transitions). Expanded privileged obs from 19D to 23D (added joint stiffness, damping,
effort limit, body damping, mass; removed negligible CoG x/y). Changed encoder input from
276D (policy+hist+privileged) to privileged-only 23D (HORA Phase 1 style). Disabled
z_bounds_loss (false saturation diagnostic: batch min/max naturally reaches +-0.95 with
4096 envs). Increased encoder epochs 1->3 with KL gating safety.

### Fixed
- `algorithms/constraint_trpo.py`: Barrier gradient was identically zero due to
  standardized cost advantages (E[A_cost]=0 by construction). Fixed with raw buffer.
- `algorithms/constraint_trpo.py`: Recovery mode barrier exclusion caused 500:1 gradient
  discontinuity. Barrier now stays active through mode transitions, margin_min 0.01->0.1.

### Changed
- `encoder/actor_critic_encoder.py`: Encoder input 276D->23D (privileged-only). 12x
  parameter reduction (70656->5888 first layer).
- `mdp/observations.py`: Privileged obs 19D->23D (+joint stiffness/damping/effort_limit,
  +body damping/mass, -CoG x/y).
- `agents/rsl_rl_ppo_cfg.py`: num_encoder_epochs 1->3, z_bounds_coef 0.3->0.0,
  recovery_threshold_frac 0.6->0.4, beta 0.05->0.02, joint_vel_limit budget 0.05->0.10.
- `config.py`: privileged dim 19->23, joint_vel_limit budget synced.

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
