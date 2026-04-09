# Constrained ALBC Development History

## Overview

Development of a constrained RL system (NORBC-style) for 2-DOF underwater vehicle arm control,
spanning 2026-03-05 to 2026-03-26 across 8 phases and 85+ commits. The system evolved through
three constraint enforcement paradigms (IPO log-barrier -> Lagrangian primal-dual -> Modified
IPO / C-TRPO), multiple encoder architectures (privileged-only -> history-augmented -> full
concatenated input), a critical action parameterization shift (joint velocity -> absolute EE
position -> delta EE -> Joint PD), and a final paper-aligned architecture overhaul. The most
valuable lessons came from root cause analyses of gradient explosions, arm freeze, and encoder
starvation -- problems that were repeatedly misdiagnosed before their true structural causes
were identified.

## Phase 1: Initial Build-Out and IPO Barrier (2026-03-05 -- 2026-03-10)

### Summary
Built the constrained TRPO + IPO log-barrier system from scratch. Expanded constraints from
3 to 8 terms. Discovered that barrier_t is critical for arm movement (too low = arm freeze)
and that effort_limit DR creates structurally unavoidable constraint violations.

### Key Changes
- Implemented full TRPO + IPO barrier (~600 lines) in `constraint_trpo.py`
- Multi-head cost critic with `F.softplus()` output in `actor_critic_encoder_constrained.py`
- Asymmetric critic (raw privileged input) aligned with NORBC
- Constraints expanded: 3 -> 8 cost functions (joint torque, velocity, effort, attitude,
  accumulated rotation, overshoot, yaw velocity)
- barrier_t tuned through range 1 -> 10 -> 50 -> 100 (final: 50 initial, 100 final).
  t=10 caused cost gradient dominance -> 80% line search failure -> arm freeze
- Unified actuator DR (Kp 40-120, Kd 0.5-5.0, effort 0.7-1.0x)
- Noise floor unified to 0.25 across all modes
- `velocity_limit_sim` corrected to 4.19 rad/s (datasheet 40 rpm)

### Key Lessons
- barrier_t too low makes barrier gradient dominate reward -> arm freeze
- effort_limit using per-env DR'd limits creates unavoidable violations during transients
- Cost critic must use softplus (non-negative output for cost values)

## Phase 2: Lagrangian Migration and Entropy Crisis (2026-03-16 -- 2026-03-17)

### Summary
IPO barrier assumed feasible start, but random policy starts infeasible. Barrier's easiest
path to reduce penalty was killing action variance (entropy collapse). Migrated to Lagrangian
primal-dual, which also failed (lambda hysteresis, entropy dilemma). Then migrated to C-TRPO
(barrier-based, Muller et al. ICML 2025). Fixed critical encoder gradient death (50x drop)
by replacing cached gradients with multi-step fresh forward passes. Replaced HistoryTCN with
raw flatten concat (960MB -> 260MB VRAM, TRPO full-batch compatible).

### Key Changes
- IPO -> Lagrangian: lambda starts at 0, grows with violations. Detached std from cost
  gradient. Added reward advantage normalization and line-search-gated updates.
- Lagrangian -> C-TRPO: Barrier-based with safe/recovery modes. Removed lambda_k,
  log_alpha, alpha_optimizer.
- Encoder gradient death fix: cached gradients replaced with multi-step fresh forward passes
  (enc_grad restored from 8.3e-4 to 0.04)
- HistoryTCN replaced with raw flatten concat (hist(N,30,8) -> flatten(N,240))
- eval_dr.py unified DR conditions for fair TDC vs Encoder comparison
- TDC output latency buffer added for evaluation fairness

### Key Lessons
- IPO log-barrier is unsuitable when policy starts infeasible (barrier kills exploration)
- Lagrangian primal-dual has fundamental entropy dilemma in TRPO: alpha>0 -> unbounded
  entropy growth, alpha=0 -> collapse. Seven tuning sessions confirmed this is unsolvable.
- TRPO full-batch requirement makes TCN (960MB) impractical; raw flatten (260MB) is sufficient
- Encoder z sweep after 2500 iter C-TRPO: 10/13 dims near-constant, cosine sim 0.9482.
  Encoder not learning from privileged info alone.

## Phase 3: C-TRPO Stabilization and Encoder Architecture (2026-03-18 -- 2026-03-21)

### Summary
Multi-epoch encoder update caused uncontrolled KL drift (0.08-5.6, up to 560x above TRPO
budget). Reverted to 1 epoch. Comprehensive code review found 3 critical bugs. Barrier
penalty was structurally zero due to cost advantage standardization. Recovery mode caused
deterministic safe/recovery oscillation (159-iter cycles). Encoder input changed from
privileged-only to full concatenated (policy+history+privileged = 280D, later reverted to
privileged-only 23D, then back to 280D).

### Key Changes
- Encoder epochs reverted 5 -> 1 (multi-step incompatible with TRPO trust region)
- encoder_lr reverted 1e-3 -> 3e-4
- Critical bug fix: barrier penalty was identically zero because per-constraint cost
  advantage standardization set E[A_cost]=0. Fixed by using raw (unstandardized) cost
  advantages for barrier, standardized for recovery.
- Recovery mode removed: caused deterministic 159-iter safe/recovery oscillation. The
  barrier-only approach is structurally cleaner.
- Lagrangian mechanism tried as interim (lambda_k, dual ascent, lambda_max=0.5), then
  replaced by Modified IPO on 03-23
- prev_actions_obs causal violation fixed (was using current-step actions as "previous")
- effort_limit_cost fixed to use per-joint comparison (was aggregating across joints)
- Joint DR fixed: was running in debug mode (always sampling, ignoring enable flag)
- Privileged obs expanded 19D -> 23D (added joint stiffness/damping/effort_limit,
  body damping/mass; removed negligible CoG x/y)

### Key Lessons
- Multi-epoch encoder in TRPO causes indirect distribution shift that TRPO cannot bound.
  TRPO constrains actor KL but encoder changes actor input, causing unbounded shift.
- Cost advantage standardization (mean subtraction) removes absolute-level signal. This
  makes barrier penalty structurally zero. Raw advantages preserve violation magnitude.
- Recovery mode creates a bistable system, not a convergent one. The policy alternates
  between "optimize reward" and "minimize cost" indefinitely.

## Phase 4: Entropy Solutions and Exploration (2026-03-20 -- 2026-03-23)

### Summary
Systematic attack on entropy collapse in C-TRPO. min_std=0.2 floor failed (std monotonically
collapsed to floor, stayed there permanently). EAPO (Entropy Advantage Policy Optimization)
implemented but was superseded by sigma decoupling from TRPO. The root cause: in 2D action
space, sigma consumes ~33% of KL budget (vs ~4-8% in 12D locomotion), so TRPO preferentially
reduces sigma. Solution: decouple log_std from TRPO natural gradient, give it a separate
Adam optimizer with score-function gradient.

### Key Changes
- min_std=0.2 floor: insufficient. Prevents going below 0.2 but no mechanism to resist
  reduction. noise_std monotonically decreased to floor by iter 80 in every run.
- EAPO implemented: per-sample entropy advantage A_H = normalize(-log_prob), adaptive tau
  via SAC v2 dual gradient. Theoretically sound but superseded by simpler sigma decoupling.
- Laplacian reward: exp(-|e|/sigma) replaced quadratic for better near-zero gradient.
  min_laplacian variant: worst-axis determines reward, preventing better axis from
  dominating gradient (93:7 ratio -> 100% on worst axis).
- Sigma decoupled from TRPO: log_std moved to separate Adam optimizer (std_lr). Score-function
  equilibrium ((a-mu)^2 - sigma^2)/sigma^3 naturally finds balance. Post-TRPO baseline
  re-snapshot ensures IS ratio starts at 1.0 for sigma update.
- yaw_quad_damp removed from privileged obs (27D -> 26D... then later 28D -> 27D). Encoder
  z sweep showed 13/13 dims dominated by yaw_quad_damp (range 1.37-1.85), a parameter ALBC
  cannot act on. Removing it freed encoder capacity for actionable information.
- std_lr tuned: 1e-4 (too slow, sigma stuck at 0.98 for 310 iter) -> 3e-3 (equilibrium
  in ~100 iter). Score-function gradient is self-correcting, making higher LR safe.
- entropy_coef: 0.01 -> 0.005 -> 0.001 -> 0.0 (final). Each reduction was needed because
  TRPO's single natural gradient step amplifies entropy bonus vs PPO's ~20 mini-batch steps.
- barrier_alpha: 0.3 -> 0.02 (paper value). Was 15x too weak. Gradient was 0.0067
  (0.7% of reward) vs paper's 0.10 (10% of reward).

### Key Lessons
- In low-dimensional action spaces, sigma dominates KL budget. Decoupling is essential.
- entropy_coef is structurally unstable in TRPO (competes with reward for single KL budget).
  Any positive value either dominates or is negligible, with no stable equilibrium.
- barrier_alpha=0.02 (paper value) gives 15x stronger constraint correction than 0.3.
  The original 0.3 was from a misreading of the paper.
- Score-function gradient for sigma is self-correcting: overshooting triggers corrective
  gradient, making higher LR safe unlike actor/encoder parameters.

## Phase 5: Encoder Learning and Constraint Headroom (2026-03-24 -- 2026-03-25)

### Summary
Encoder learning was blocked by two interacting problems: (1) KL gating too tight after
max_kl increase, and (2) encoder input was exclusively static DR parameters. Constraint
violations were structurally unavoidable due to zero headroom between action space boundaries
and constraint thresholds. After fixing these, encoder learned rich representations but
attitude error did not improve, revealing 5 structural barrier problems.

### Key Changes
- max_kl increased 0.002 -> 0.005 (step_norm was decaying while grad_norm grew, indicating
  trust region was too tight). step_norm increased 40-55% as intended.
- Encoder KL gating fix: max_encoder_kl=0.003 was derived from max_kl=0.002. After max_kl
  increase, TRPO consumed more KL, making encoder overshoot 0.003 budget immediately.
  Fix: max_encoder_kl 0.003 -> 0.0075, encoder_lr 1e-3 -> 3e-4. Adam state poisoning
  identified (optimizer.step() executes before reversion, accumulating phantom momentum).
- Encoder input changed from privileged-only (27D) to full concatenated
  (policy_obs(13) + hist_flat(240) + privileged(27) = 280D). Static-only input produced
  only 4096 unique z samples per iteration (vs 262K with dynamic input). This matched
  NORBC/ANYmal/RMA reference architectures.
- Encoder hidden dims [128, 64] -> [256, 128, 64] for 280D input.
- EmpiricalNormalization replaced _FixedNormalization (dynamic inputs have no fixed distribution).
- Reconstruction auxiliary loss tried and failed: decoder learned degenerate solution
  (predicting mean privileged_obs with collapsed z, ignoring z content).
- Constraint headroom fix: velocity limit = action scaling (4.189 rad/s) was identical to
  constraint threshold. Action=1.0 immediately hit constraint. Fix: max_joint_velocity
  increased to 2*pi (33% headroom above constraint), effort_limit_sim 9.5 -> 13.0 Nm
  (27% above motor spec). Constraint thresholds fixed at motor specs (no DR).
- torque_weight escalation: -0.001 -> -0.01 -> -0.05 -> -0.001 (reverted after headroom fix
  made reward workaround unnecessary). -0.01 was ~8% of command reward (insufficient),
  -0.05 was ~30% (risk of tracking degradation).
- DORAEMON threshold: success_threshold_deg 10 -> 15 (pitch_err=12.7 deg caused most
  episodes to fail, starving encoder of diverse DR signal).

### Key Lessons
- enc_grad growing != encoder learning. Must check enc_added (total_kl - pre_encoder_kl)
  to verify updates are actually applied. Gradient can grow while all updates are reverted.
- Adam optimizer state poisoning: step() executes before KL check reverts params. Momentum
  accumulates for 2000+ phantom updates, corrupting future steps.
- Static-only encoder input produces 64x fewer unique z samples than dynamic input. This is
  the root cause of encoder gradient death in ALBC, not activation function choice.
- Reconstruction auxiliary loss fails when encoder input is static: decoder maps collapsed z
  to mean prediction. Do not re-attempt auxiliary losses on encoder.
- Zero headroom between action space and constraint threshold makes constraints structurally
  unavoidable. PhysX hard cap must exceed constraint threshold (motor spec).
- 5 structural barrier problems identified (all confirmed by numerical analysis):
  1. Adaptive threshold pins barrier margin at alpha*d_k when violating -- constant gradient
  2. Cost advantage standardization removes absolute-level signal
  3. TRPO trust region normalization cancels barrier_t scaling at reward plateau
  4. No action magnitude penalty (smoothness penalizes rate, not magnitude)
  5. Cost value loss d_k^2 normalization weakens critic for large-budget constraints

## Phase 6: Action Parameterization and Gradient Stability (2026-03-26)

### Summary
The most productive and turbulent day. Discovered that absolute EE position action mode
creates a structural trap (optimal position at action boundary -> tanh saturation -> zero
gradient -> permanent arm freeze). Switched to delta EE mode. Fixed 3 gradient explosion
bugs in C-TRPO (cost advantage 0-division, encoder baseline drift, inf gradient propagation).
Added tanh squashing with proper raw action storage. Multiple reward function experiments
(exponential vs quadratic) revealed that quadratic penalty in all-negative landscape
systematically directs gradient toward "freeze arm".

### Key Changes

**Action parameterization evolution:**
- joint_velocity (working but pitch-biased) -> ee_position (arm freeze) -> ee_delta (final)
- ee_position arm freeze root cause: max extension = max restoring torque, so physical
  optimum is at action boundary (action_size=1.41 = sqrt(2)). At boundary, pre-tanh mu~2.65,
  g2_std=0.00 deg (zero joint diversity), EE range=0.022m (2.2cm of 0.92m workspace).
  No gradient-based method can escape because reward surface is flat in sampled region.
- Rate limiting added for ee_position mode (max_joint_velocity * control_dt = 0.126 rad/step)
  but did not solve the boundary saturation problem.
- Delta EE mode: actions specify displacement per control step. Current EE via FK, delta
  added, then IK + rate limit. Optimal steady-state = action(0,0) = center of action space.
  ee_delta_scale=0.02m at 50Hz = 1.0 m/s max. Smoke test confirmed: action_size=0.74
  (not boundary-saturated), action_rate=0.74 (active movement).

**Reward function experiments (all ultimately kept exponential):**
- Quadratic penalty (-c*e^2): gradient never vanishes but all-negative landscape means
  "least negative" timesteps (positive advantage after normalization) = least movement.
  Systematically directs policy gradient toward "freeze arm".
- Violation-proportional barrier weights (w_k = max(1, J_C_k/d_k)): amplified barrier 3.75x,
  creating positive feedback loop (exploration -> high cost -> amplified barrier -> freeze).
  Isolation test confirmed this was root cause of arm freeze, not quadratic reward.
- Final: exponential reward exp(-c*e^2) with c=5.0/7.5, command_weight=+5.0

**Gradient explosion fixes (3 bugs, all in constraint_trpo.py):**
1. Root cause: cost advantage standardization divided by (std + 1e-8) for binary constraints
   with near-zero std. Single non-zero sample amplified 0.1/1e-8 = 1e7. Progressive
   deterioration via positive feedback loop until inf at iter 562+.
   Fix: std.clamp(min=1.0) -- binary constraints get centering only, no scaling.
2. Encoder update used pre-TRPO old_log_prob as baseline, causing ratio explosion over
   5 encoder epochs. Fix: re-snapshot log_prob after TRPO step.
3. No guard against inf gradients after clip_grad_norm_ (inf -> NaN parameters).
   Fix: isfinite guard, skip optimizer step on inf/NaN.

**Tanh squashing:**
- Added to bound actions to (-1, 1). KL divergence invariant under bijective transforms.
- Critical bug: atanh inversion is numerically lossy. raw=4.0 -> tanh=0.99933 ->
  atanh=3.654 (not 4.0). Caused importance sampling ratio explosion (10^12).
  Fix: store raw (pre-tanh) actions in rollout buffer, use directly for all log_prob
  computations. Tanh removed entirely when switching to delta EE mode (unnecessary with
  centered action space).

**Other changes:**
- Torque reward fixed: was using computed_torque (pre-clamp, up to 100s Nm) instead of
  applied_torque (post-clamp, max 9.5 Nm). Torque penalty dominated command tracking by 58x.
- Command tracking weight k_c increased -1.0 -> -8.0
- 9 Dynamics/* metrics ported from hero_agent for actuator monitoring
- Gradient decomposition diagnostics added (reward vs barrier gradient, ratio stats,
  margin tracking) -- revealed reward_surr spikes to 0.66 at high-grad iters
- workspace_radius 0.40 -> 0.461 (full reachable workspace for tanh output)
- smoothness_weight -0.5 -> -0.05 (10x reduction, was accelerating boundary trap)
- Encoder z_sweep dimension indexing fixed for 280D concatenated input. Previous results
  were sweeping policy_obs indices instead of privileged obs at indices 253-279. All
  prior constrained ALBC z_sweep analyses were invalid. After fix: encoder shows excellent
  sensitivity across all 27 privileged parameters (10-13/13 dims active per param).
- barrier_t 50 -> 100 (paper nominal). Halves barrier gradient: 1/(margin*100) vs 1/(margin*50).

### Key Lessons
- Absolute EE position is structurally incompatible with policy gradient methods when the
  physical optimum lies at the action boundary. All working systems (RSL-RL, HORA, Isaac Lab
  Factory, hero_agent) use delta/velocity actions where optimal = center of action space.
- atanh is not the inverse of tanh in floating point. Store raw actions, never reconstruct.
- Cost advantage standardization with epsilon=1e-8 is catastrophic for binary constraints.
  A single non-zero sample gets amplified by 1e7-1e8, creating gradient explosion.
- Quadratic reward in all-negative landscape creates perverse incentives under advantage
  normalization. Exponential reward (positive per-step values) avoids this.
- Violation-proportional barrier weights create positive feedback: exploration -> cost ->
  amplified barrier -> "don't move" -> arm freeze. Standard log barrier (constant weights)
  allows gradual cost reduction.
- Constraint system analysis: torque_limit_cost correctly uses computed_torque (pre-clamp)
  to detect when PD demands exceed physical limits. Reward should use applied_torque
  (post-clamp) to measure actual energy consumption.
- grad_norm growing from O(1) to O(1e16) over 500 iterations is always a bug, not a
  methodology issue. Look for division by near-zero first.
- encoder_z_sweep results depend critically on correct index mapping. For concatenated
  encoder input (280D), privileged obs start at index 253, not index 0.

## Phase 7: Logging and Monitoring Fixes (2026-03-26, late)

### Summary
Fixed logging artifacts and added diagnostic tooling for continued development.

### Key Changes
- Line search failure logging artifact: barrier_penalty and entropy spiked on failure because
  surrogate() closure overwrites monitoring vars on each backtracking attempt. Fix: recalculate
  surrogate() with reverted params after failure.
- Gradient decomposition diagnostics: reward-only gradient computed separately, barrier
  gradient derived by subtraction. 7 diagnostic TB metrics added (reward/barrier grad_norm,
  ratio stats, reward_surr, margin_min).

### Key Lessons
- In interior point methods, monitoring variables must reflect the accepted state, not the
  last rejected candidate. log(margin) diverges as margin approaches zero (Boyd & Vandenberghe).
- Diagnostic logging should decompose gradients by source (reward vs barrier) to identify
  which component drives instability.

## Phase 8: Paper-Aligned Architecture Overhaul (2026-03-26)

### Summary
Major restructuring to align with NORBC paper's Teacher-Student framework.
670 insertions, 2,902 deletions across 16 files. This overhaul set the stage for the
03-27 ablation study that isolated encoder integration as the sole remaining problem.

### Key Changes

**Network Architecture (3-Layer MLP)**
- Encoder hidden: [128, 64] -> [256, 128, 64]
- Actor hidden: [128, 64] -> [256, 128, 64]
- Critic hidden: [256, 128] -> [512, 256, 128]
- Value backbone: [256] -> [512, 256, 128] -> 64D features (shared multi-head)

**Observation Redesign (14D + 23D)**
- Policy obs o_t (14D): euler(3) + ang_vel(3) + att_err_rp(2) + joint_pos(2) + joint_vel(2) + prev_actions(2)
- Privileged p_t (23D): hydro(6) + inertia(4) + damping(4) + mass(2) + payload(4) + joint(2) + density(1)
- Encoder input: p_t only (was 280D with history + policy_obs + privileged)

**Action Space: EE Delta -> Joint PD Targets**
- New: `q_des = q_nominal + action_scale * a_t` (5 lines, 1 method)
- Removed: FK + analytical IK + workspace clamp + rate limiting (65 lines, 3 methods)
- `nominal_joint_pos = (0.0, pi)`, `action_scale = pi`

**Control Frequency: 1:40 Ratio**
- Physics PD: 2000Hz (dt=0.0005s), Policy: 50Hz (decimation=40, control_decimation=1)
- Was: 200Hz PD (dt=0.005, decimation=1, control_decimation=4)

**Shared Backbone Multi-Head Value Function**
- Shared backbone: cat([o_t, p_t]) -> MLP[512,256,128] -> 64D features
- Reward head: Linear(64 -> 1), Cost head: Linear(64 -> K=4)

**Code Simplification**
- Deleted `utils/debug_vis.py` (333 lines)
- Removed proprio history (`_get_proprio_features`, `_update_proprio_hist`, EMA buffers)
- Removed equilibrium joint init mode
- Naming: C-TRPO -> TRPO + IPO (all files)

### Files
- `encoder/actor_critic_encoder.py`, `encoder/actor_critic_encoder_constrained.py`
- `agents/rsl_rl_ppo_cfg.py`, `albc_env.py`, `config.py`
- `mdp/events.py`, `mdp/observations.py`

---

## Appendix: Commit History (2026-03-20 -- 2026-03-26, 85 commits)

Granular commit-level history for the constrained ALBC extraction and development period.

### A. Extraction and Refactoring (2026-03-20)

| Commit | Description |
|--------|-------------|
| f24f409 | Extract constrained_albc as standalone package from hero_agent |
| 86f8274 | Flatten agents/ config hierarchy from 3-level to 2-level |
| 912f25d | Clean up utils/ dead code, docstrings, magic numbers |
| 9dc2e04 | Simplify runners/ with property extraction and pattern dedup |
| cd63b8a | Remove 5 unused constraint cost functions from mdp/ |
| bf3ff43 | Remove hardcoded num_constraints/budgets duplicates |
| 1834347 | Simplify constraint_trpo.py internal duplication |
| 40592e3 | Structural cleanup of encoder/ activation, helpers, OCP |
| 4b990ed | Extract helpers from monolithic methods, remove dead code |
| cea93c5 | Flatten 3-level runner hierarchy to single class |
| 9978524 | Simplify constraint_trpo.py surrogates, logging, cost GAE |
| 1dfb2c1 | Remove dead branches from encoder (465->348, 131->110 LOC) |
| 7b65d6e | Simplify constrained ALBC MDP module (rewards, events, constraints, observations) |
| 8615e27 | Merge 4-class config hierarchy into single ConstrainedALBCEnvCfg |
| e2afd27 | Fix config naming, delete doraemon.py (old), remove DR infeasibility logging |
| 72be9ac | Remove enable_payload conditional and state_space guards |
| 762d7c9 | Remove backward compat from encoder load_state_dict |
| cbd2dd2 | Remove kwargs, rnd, and hasattr guards from ConstraintTRPO |
| efb4e3e | Final DORAEMON cleanup + DRSampler.get() signature simplification |

### B. Bug Fixes and Code Review (2026-03-21 -- 2026-03-22)

| Commit | Description |
|--------|-------------|
| d36dbc4 | Remove unused policy config fields |
| 2b610fa | Format actor_critic_encoder_constrained.py |
| 53b4f70 | Initialize _last_* monitoring attrs + fix docstring |
| f6089fb | Code review fixes + runtime integration bugs |
| 3d70b5d | _prev_joint_pos reset timing + control_dt latent bug |
| 3f48317 | Encoder code review: DRY, no_grad perf, backward compat |
| 7c206f7 | Save encoder optimizer on checkpoint + NaN guard |
| 59e37eb | MDP code review: 3 critical bugs + 5 theoretical fixes |
| 8026720 | Remove redundant PBRS progress reward |
| 8350642 | C-TRPO mode oscillation fix: EMA smoothing + critic LR gating |
| 56b38c5 | 5-7 deg plateau fix: Laplacian reward + noise floor + overshoot relaxation |

### C. Reward Tuning and Constraint Improvements (2026-03-22 -- 2026-03-24)

| Commit | Description |
|--------|-------------|
| 0ee185d | Add TRPO step quality diagnostic logging |
| ec47ede | Deduplicate WandB metrics + update train-analyze skill |
| 0783337 | Add exponential command kernel + joint torque penalty |
| 706accd | Per-axis sigma for roll/pitch asymmetry in command reward |
| 7028fe4 | Sigma -> direct coefficient form + k_c=100 scaling |
| 4a00c06 | Tune exponential coefficients c=5/7.5, revert k_c=5 |
| 6abe899 | entropy_coef=0, barrier_alpha=0.02 (NORBC paper) |
| d0dd0e1 | Scale max_kl=0.002 for 2D action space (per-dim KL normalization) |
| 4de20a2 | Decouple sigma from TRPO + remove yaw_quad_damp from privileged obs |
| 6935ac7 | Increase std_lr 1e-4 -> 3e-3 for faster sigma equilibrium |
| b260ba5 | encoder_lr 3e-4->1e-3, encoder_epochs 3->5, DORAEMON threshold 10->15 |
| ad5c945 | Add reconstruction auxiliary loss for encoder gradient survival |
| 4e8d218 | Revert reconstruction auxiliary loss (failed experiment) |
| 8db7660 | Encoder dynamic input (280D) matching NORBC architecture |
| f80f293 | Increase TRPO max_kl from 0.002 to 0.005 |
| a328c62 | Encoder KL gating freeze + max_encoder_kl proportional scaling |

### D. Action Mode Evolution (2026-03-24 -- 2026-03-26)

| Commit | Description |
|--------|-------------|
| 961241c | Add EE position action mode with analytical IK |
| 344ed08 | Tune torque_weight from -0.001 to -0.01 |
| 58bf466 | Escalate torque_weight from -0.01 to -0.05 |
| 7239ff6 | Quadratic reward + violation-proportional barrier weights |
| 96540850 | Reduce command_weight from -5.0 to -1.0 |
| 5c2d203 | Revert: restore exponential reward for fix1/fix2 isolation test |
| a1bcb86 | Revert: remove violation-proportional barrier weights |
| c909d21 | EE position rate limiting + reset init + quadratic reward |
| d834779 | Add tanh squashing to actor output + expand workspace radius |
| 9abb758 | Store raw actions to eliminate lossy atanh + barrier_t=100 |
| 7da1146 | Switch to delta EE action mode to fix arm freeze root cause |
| 641db83 | Fix 3 gradient explosion bugs in ConstraintTRPO |
| bcb6e2f | Add gradient decomposition diagnostics to ConstraintTRPO |

### Commit History Key Lessons

1. **Encoder auxiliary loss failed**: Reconstruction loss (ad5c945) did not help; reverted (4e8d218)
2. **Barrier parameter sensitivity**: barrier_t=10 caused 80% LS failure; fixed to 100 (9abb758)
3. **Action space evolution**: absolute EE -> EE position -> delta EE -> Joint PD targets
4. **Reward engineering**: simple -> exponential -> Laplacian. Final: tracking + linear_error + settling + penalties
5. **TRPO scaling**: max_kl needs per-dimension normalization for low-DOF systems (d0dd0e1)
