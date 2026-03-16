# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [2026-03-16] Lambda LR warmup + d_k^2 cost value norm + budget recalibration

### Context
Run `2026-03-16_11-20-25` (179 iters) confirmed entropy fix works: entropy stable at
2.33 (was 0.07), noise_std 0.78 (was 0.25), arm actually moving (act_size=0.92 vs 0.36),
line search 100% success, reward growing (23.8 at iter 179).

**Problem 1 -- lambda too fast**: lambda grows too fast, constraint pressure dominates
before policy learns. effort_limit lambda=15.6, yaw_vel=14.7 at iter 179 (near
lambda_max=20). Total constraint gradient ~42x reward gradient. Attitude error stuck
at 27-29 deg. Solution: lambda LR warmup (linear ramp from 0).

**Problem 2 -- cost value scale**: cost_value_loss=19.0 (vs value_loss=0.01) because
joint_vel MSE~27000 dominates accum_rot MSE~1e-7 within cost_critic's shared hidden
layers. Solution: d_k^2-normalized cost value loss.

**Problem 3 -- budgets had no empirical basis**: Run `2026-03-16_11-40-48` (212 iters,
with warmup) showed warmup working (roll=8.3 deg at step 50) but 5/8 constraints still
OVER budget after warmup ends. Measured natural cost levels from unconstrained
Encoder-Base (`2026-03-05_11-41-18`, 2000 iters, roll=5.7, pitch=6.3 deg):
- oscillation: budget was 1.14x natural (almost no room for exploration)
- yaw_vel: budget was 0.58x natural (IMPOSSIBLE to satisfy while controlling attitude)
- effort_limit: budget was 16.7x natural (generous, no change needed)
- joint_vel: budget was 2.0x natural (fine, no change needed)
Recalibrated to ~1.5x natural. Also increased warmup from 15% to 30%.

### Changed
- `algorithms/constraint_trpo.py`: Added `lambda_warmup_frac` parameter (default 0.30) -- linearly ramps lr_lambda from 0 to target over warmup period (30% of max_iterations = 750 iters for max_iter=2500)
- `algorithms/constraint_trpo.py`: `set_max_iterations()` activated from no-op -- now computes `_lambda_warmup_end` from `lambda_warmup_frac * max_iterations`
- `algorithms/constraint_trpo.py`: Dual update uses `effective_lr = lr_lambda * min(1.0, iteration / warmup_end)` instead of fixed `lr_lambda`
- `algorithms/constraint_trpo.py`: Cost value loss changed from `MSE.mean()` to `(per_k_mse / d_k^2).mean()` -- equalizes gradient across constraints with different cost return scales (joint_vel MSE 27000 -> 0.68, oscillation 4290 -> 4.77; 80:1 range compressed to ~20:1)
- `algorithms/constraint_trpo.py`: Added `lambda_lr_eff` to loss dict for WandB monitoring
- `algorithms/constraint_trpo.py`: Updated module docstring with lambda warmup and cost value normalization design decisions
- `agents/rsl_rl_ppo_cfg.py`: Added `lambda_warmup_frac: float = 0.30`, updated `constraint_budgets` oscillation 0.3->0.4, yaw_vel 0.15->0.4
- `config.py`: Constraint budgets recalibrated based on unconstrained Encoder-Base natural cost levels: oscillation D_k 0.3->0.4 (1.52x natural), yaw_vel D_k 0.15->0.4 (1.55x natural)

### Notes
- Warmup timeline (2500 iters): iter 0-750 lambda LR ramps 0 -> 0.035 (reward learning dominant), iter 750+ full lambda LR
- Budget calibration method: run unconstrained Encoder-Base to convergence, measure mean per-step cost for each constraint, set budget to 1.5x that level
- Natural per-step costs (Encoder-Base converged): effort_limit=0.003, joint_vel=0.978, oscillation=0.263, yaw_vel=0.258
- yaw_vel was the most egregious: budget 0.15 was BELOW the natural 0.258 -- policy could never satisfy it while maintaining attitude control
- cost_value_loss dropped from 19.0 to 0.06 with d_k^2 normalization

## [2026-03-16] Remove entropy_coef + noise ceiling (post detach-std analysis)

### Context
Training run `2026-03-16_11-05-17` (315 iters) with detached-std cost gradient showed
entropy and noise_std LOCKED at ceiling (entropy=2.838, std=1.0) for ALL 315 iterations.
The detach fix successfully blocked cost gradient from collapsing std, but with entropy_coef
still active (0.02), the only remaining force on std was upward (entropy bonus). Reward
gradient at std=1.0 is too weak to push down (actions are effectively random with ~32%
probability mass outside [-1,1]).

Result: policy learned ~20 reward (vs potential ~80), attitude error 25-30 deg, 5/8
constraints over budget, effort_limit and yaw_vel lambdas saturated to 20.0 (max).

Decision: entropy_coef was originally needed to counteract cost-gradient-driven collapse.
With cost gradient detached from std, there is no collapse pressure to counteract.
Setting entropy_coef=0 lets reward gradient alone control variance (natural equilibrium).
Noise ceiling also removed (no upward pressure without entropy bonus). Floor lowered
to 0.1 (numerical safety only, not behavior-modifying).

### Changed
- `agents/rsl_rl_ppo_cfg.py`: `entropy_coef` 0.02 -> 0.0 in `RslRlConstraintTRPOAlgorithmCfg` (cost gradient detached from std, no collapse to counteract)
- `algorithms/constraint_trpo.py`: Removed noise ceiling (`max_log_std`), lowered floor from `log(0.25)` to `log(0.1)` (numerical safety only)
- `runners/base_runner.py`: `_apply_noise_floor()` removed ceiling (`max_std`), lowered floor from 0.25 to 0.1

### Notes
- With entropy_coef=0 and detached cost: only reward gradient controls std
- Initial std=1.0 (from init_noise_std=1.0 in policy config); reward gradient should push it down naturally
- TRPO KL constraint limits rate of std change, providing natural annealing
- Floor 0.1 is purely numerical: prevents log_prob divergence when std -> 0
- Unconstrained Encoder-Base also has entropy_coef=0.005 but noise naturally converges -- reward alone is sufficient

## [2026-03-16] Fix entropy collapse: detach std from cost gradient + d_k normalization

### Context
Lagrangian constraint enforcement (previous session) eliminated log-barrier structural
issues but entropy still collapsed to floor (0.065) by iter 50. Training log analysis
(run 2026-03-16_10-32-34, 278 iters) showed yaw_vel lambda saturating to 20 at iter 20,
total constraint pressure ~48x vs reward ~1x. Policy converged to "don't move arm"
(satisfies 5/8 constraints trivially). Encoder-Base (no constraints) achieves <10 deg
error, confirming the reward structure is sufficient.

Root cause: For Gaussian policies, the cost surrogate gradient flows through `log_std`.
Reducing variance is the most efficient way to reduce ALL constraint costs simultaneously
-- this is structural and orthogonal to the enforcement mechanism (barrier OR Lagrangian).
Previous fixes (noise floor, ceiling, barrier_t, warmup, lambda_max) were all band-aids
that delayed but couldn't prevent the collapse.

Secondary issue: Dual update `lambda += lr * (J_C - d_k)` is scale-dependent. d_k ranges
from 1.0 to 200.0 (100x), so continuous constraints (large d_k) dominate lambda growth.

### Changed
- `algorithms/constraint_trpo.py`: Added `_log_prob_mean_only()` helper -- computes Gaussian log_prob with `std.detach()`, so `d(log_prob)/d(log_std) = 0`
- `algorithms/constraint_trpo.py`: `update()` cost surrogate now uses `ratio_cost` (from detached-std log_prob) instead of `ratio` -- constraint gradient flows through action mean only, not variance
- `algorithms/constraint_trpo.py`: `_linearized_surrogate()` uses same `ratio_cost` for code consistency (always called under `torch.no_grad()` so detachment is a no-op for values, but keeps gradient path consistent)
- `algorithms/constraint_trpo.py`: Dual update normalized by d_k: `lambda += lr * (J_C - d_k) / d_k` -- violation expressed as fraction of budget, equalizing scale across constraints

### Notes
- Reward surrogate: full ratio (gradient to both mean and std) -- unchanged
- Entropy term: gradient to std only -- unchanged (maintains exploration incentive)
- TRPO core (CG, FVP, line search, KL): all use full distribution geometry -- unchanged
- This removes the MECHANISM of entropy collapse (cost gradient to log_std), not just the symptoms
- Noise floor/ceiling kept as safety net but should rarely trigger now
- Expected: entropy > 0.5 at iter 50 (was 0.065), all 8 lambdas responsive (was 5 stuck at 0)

## [2026-03-16] Replace IPO log-barrier with Lagrangian (primal-dual) constraint enforcement

### Context
Previous session identified the root cause of entropy collapse: IPO log-barrier assumes
a feasible start (all constraints satisfied), but our random policy starts infeasible.
The barrier gradient `1/(t * margin)` at maximum from iter 0 drives the easiest
optimization path -- reducing action variance (which reduces ALL constraint costs
simultaneously). noise_std collapsed to 0.25 floor by iter 50, staying there for 550
iterations. Reward plateaued at ~30, attitude error 17-18 deg.

Previous fixes (barrier_t tuning, noise floors, n_active normalization, adaptive threshold,
entropy_coef) were all band-aids that delayed but couldn't prevent this structural problem.

Solution: replace log-barrier with Lagrangian primal-dual constraint enforcement.
`lambda_k` starts at 0 (no initial constraint pressure), grows linearly with violation
via dual ascent `lambda_k = clamp(lambda_k + lr*(J_C_k - d_k), 0, max)`. This pushes
action *mean* toward constraint-satisfying directions rather than collapsing variance.

### Changed
- `algorithms/constraint_trpo.py`: Replaced barrier params (`barrier_t`, `barrier_t_final`, `barrier_t_schedule_frac`) with Lagrangian params (`lr_lambda=0.035`, `lambda_max=20.0`, `lambda_init=0.0`). Added `self.lambda_k` tensor (one per constraint).
- `algorithms/constraint_trpo.py`: Cost surrogate changed from `cost_adv_k / ((1-gamma) * barrier_t * margin)` to `lambda_k[k] * cost_adv_k` in both `_linearized_surrogate()` and `update()`
- `algorithms/constraint_trpo.py`: Added dual variable update (step 4) after value update: `lambda_k = clamp(lambda_k + lr_lambda * (J_C_k - d_k), 0, lambda_max)`
- `algorithms/constraint_trpo.py`: `_linearized_surrogate()` and `_line_search()` signatures simplified (removed `mean_cost_returns` parameter)
- `algorithms/constraint_trpo.py`: Monitoring metrics changed from `_last_margins` to `_last_violations` + `_last_lambdas`. Added `lambda_mean` to loss_dict.
- `agents/rsl_rl_ppo_cfg.py`: `RslRlConstraintTRPOAlgorithmCfg` replaced `barrier_t`/`barrier_t_final`/`barrier_t_schedule_frac` with `lr_lambda`/`lambda_max`/`lambda_init`
- `runners/constraint_encoder_runner.py`: Logging replaced `barrier_t`/`margin_*` metrics with `lambda_*`/`violation_*` metrics. Added `lambda_mean`/`lambda_max` aggregate stats.
- `runners/constraint_encoder_runner.py`: Added `save()`/`load()` overrides for `lambda_k` checkpoint persistence (`lambda_state.pt` alongside model checkpoint)

### Removed
- `algorithms/constraint_trpo.py`: `update_barrier_schedule()` method, barrier schedule instance vars. `set_max_iterations()` kept as no-op stub (called by BaseRunner.learn).
- `mdp/constraints.py`: Dead fields from `ALBCConstraintCfg`: `barrier_t`, `barrier_t_final`, `barrier_t_schedule_frac`, `adaptive_threshold_alpha`

### Notes
- TRPO core unchanged: CG solver, Fisher-vector product, line search, KL divergence
- Cost GAE unchanged: per-constraint standardization preserved
- Noise floor/ceiling kept as safety net (should rarely be hit with Lagrangian)
- Hyperparameter rationale: lr_lambda=0.035 gives ~0.28/iter growth for fully violated binary constraint (d_k=2), reaching lambda=5 in ~18 iters. lambda_max=20.0 is a conservative cap.
- Dual update ordering: policy -> value -> dual. lambda_k updated on current rollout, takes effect next iteration (standard primal-dual one-step lag).

## [2026-03-16] Replace adaptive threshold with pure barrier + root cause analysis

### Context
Analyzed why constraint cost_returns diverge and never decrease. Identified the adaptive
threshold mechanism `d_k_adaptive = max(d_k, J_C + alpha*d_k)` as a "moving goalpost":
when costs exceed budget, the threshold follows costs upward, keeping barrier margin
fixed at `0.1*d_k` regardless of how far over budget. This prevents increasing pressure
to push costs back toward budget. Also removed n_active normalization (was diluting
enforcement, introduced as a workaround for the adaptive threshold problem).

Replaced with pure barrier: `margin = max(d_k - J_C, 0.01*d_k)`. Creates increasing
pressure as cost approaches budget. Floor at 0.01*d_k (10x tighter than previous 0.1*d_k)
for over-budget constraints.

Training run `2026-03-16_09-21-04` (600 iter): constraints came closer to budget than
any previous run (yaw_vel from 29% over to 0.1% over, singularity 0.5% over, effort_limit
well under). However, **entropy collapsed at iter 50** (noise_std hit 0.25 floor and stayed
for 550 iterations). Reward plateaued at ~30 from iter 200. Attitude error 17-18 degrees.

Root cause analysis: IPO log-barrier is structurally incompatible with infeasible start.
The barrier gradient's easiest optimization path is to reduce action variance (reducing
noise reduces all constraint costs simultaneously). This is mathematically correct but
kills exploration. Budget warmup, noise floors, and barrier_t tuning are all band-aids
that delay but don't prevent this outcome -- once constraints approach budget, the same
dynamic recurs. Fundamental fix requires switching constraint enforcement from log-barrier
to Lagrangian (primal-dual) method, where lambda starts at 0 and grows gradually.

### Changed
- `algorithms/constraint_trpo.py`: Replaced adaptive threshold with pure barrier margin `max(d_k - J_C, 0.01*d_k)` in both `_linearized_surrogate()` and `update()`. Removed `adaptive_threshold_scale`, `d_k_adaptive`. Added `_last_margins` storage for logging.
- `algorithms/constraint_trpo.py`: Removed n_active normalization from both cost surrogate loops (was dividing by count of active constraints)
- `algorithms/constraint_trpo.py`: Noise ceiling `max_log_std` reduced from `log(2.0)` to `log(1.0)`
- `runners/constraint_encoder_runner.py`: Logging changed from `d_k_adaptive` to `margin` and `d_k` per constraint
- `agents/rsl_rl_ppo_cfg.py`: Removed `adaptive_threshold_alpha` field from `RslRlConstraintTRPOAlgorithmCfg`
- `runners/base_runner.py`: Noise ceiling `max_std` reduced from 2.0 to 1.0

### Removed
- `algorithms/constraint_trpo.py`: `adaptive_threshold_alpha` parameter, `self.adaptive_threshold_scale`, `self.d_k_adaptive` tensor, n_active counting/normalization logic

### Notes
- Pure barrier improved constraint enforcement vs adaptive threshold, but did NOT fix the fundamental entropy collapse problem
- IPO log-barrier assumes feasible start (all constraints satisfied); we start infeasible (random policy violates multiple constraints)
- All previous fixes (barrier_t tuning, noise floor/ceiling, entropy_coef, n_active normalization) were treating symptoms, not root cause
- Next step: replace IPO log-barrier with Lagrangian (primal-dual) constraint enforcement -- lambda starts at 0, grows linearly with violations, naturally handles infeasible start without crushing exploration
- The analyze_training.py script was also updated (replaced `_dk_expanding` with `_margin_at_floor`, FLOOR/OVER alerts) but is not git-tracked

## [2026-03-16] Entropy overcorrection fix: revert entropy_coef + add noise ceiling

### Context
Run `2026-03-16_04-17-31` (130 iter) with previous session's fixes showed entropy
EXPLOSION instead of collapse: noise_std=1678, entropy=17.55. The combined effect of
barrier_t 5x reduction + n_active normalization ~3x + entropy_coef 2x created ~30x
swing, overshooting from barrier-dominated to entropy-dominated.

Root cause: Gaussian entropy gradient is always positive (d(entropy)/d(log_std) = 1),
so without sufficient counterbalance from barrier or reward, noise_std grows without
bound. The existing noise floor (min_std=0.25) had no symmetric ceiling.

Fix: reverted entropy_coef 0.04->0.02 (barrier_t + n_active alone provide sufficient
collapse prevention), added max_std=2.0 ceiling in both constraint_trpo.py and
base_runner.py (symmetric counterpart to existing floor).

Run `2026-03-16_04-27-48` (276 iter) with this fix shows best results to date:
reward 42.07, roll_err 11.59 deg, pitch_err 13.40 deg, noise_std=0.48 (natural
equilibrium, neither floor nor ceiling), ls_success 100%, z_range [-0.94, 0.91].

### Changed
- `agents/rsl_rl_ppo_cfg.py`: `entropy_coef` 0.04 -> 0.02 (reverted -- barrier_t + n_active fixes are sufficient)
- `algorithms/constraint_trpo.py`: Added `max_log_std = math.log(2.0)` ceiling alongside existing `min_log_std = math.log(0.25)` floor
- `runners/base_runner.py`: Added `max_std = 2.0` ceiling alongside existing `min_std = 0.25` floor in `_apply_noise_floor()`

### Notes
- Three training runs today: (1) collapse (barrier_t=10, entropy=0.02), (2) explosion (barrier_t=50, entropy=0.04), (3) balanced (barrier_t=50, entropy=0.02, noise ceiling)
- noise_std=0.48 at iter 276 confirms natural equilibrium exists between entropy bonus and barrier pressure
- Noise ceiling (2.0) was NOT hit -- it's a safety net, not the balancing mechanism
- The effective fix chain: barrier_t 10->50 (5x weaker barriers) + n_active normalization (~3x when 3 active) = ~15x reduction in barrier dominance, sufficient to prevent collapse without inflating entropy_coef

## [2026-03-16] Entropy collapse: parameter + structural fix (active constraint normalization)

### Context
Post-fix training run `2026-03-16_03-51-07` (261 iter) confirmed that 6 bug fixes
from earlier session improved reward (+5, 30.96 vs 26.04) and encoder gradients (3.6x),
but entropy STILL collapsed (noise_std=0.25 floor, entropy=0.07).

Root cause analysis of the IPO barrier gradient math:
- 3 constraints (oscillation, singularity, yaw_vel) at ~91% budget simultaneously
- Each has margin ~0.09*d_k, producing barrier gradient scale ~200 per constraint
- Combined scale ~500+, vs entropy_coef=0.02: a ~25,000x gradient imbalance
- Also discovered barrier_t fix from 2026-03-09 was never applied (still 10/50, not 50/100)

Decision: parameter fixes (barrier_t, entropy_coef) + lightweight structural fix
(normalize cost_surrogate by number of simultaneously active constraints).

### Changed
- `agents/rsl_rl_ppo_cfg.py`: `barrier_t` 10.0 -> 50.0, `barrier_t_final` 50.0 -> 100.0 (planned fix from 2026-03-09 finally applied)
- `agents/rsl_rl_ppo_cfg.py`: `entropy_coef` 0.02 -> 0.04 (2x increase to counterbalance residual constraint pressure)
- `mdp/constraints.py`: `barrier_t` 1.0 -> 50.0, `barrier_t_final` 50.0 -> 100.0 (dead field synced for consistency)
- `algorithms/constraint_trpo.py`: Added active constraint normalization in both `_linearized_surrogate()` and `update()` cost_surr loops -- when multiple constraints have margin within 2x of floor (0.1*d_k), divide cost_surrogate by n_active to prevent combined barrier gradient from overwhelming entropy

### Notes
- Expected effect: barrier gradient pressure reduced ~6x (barrier_t 2x + n_active ~3x) + entropy 2x = ~12x relative improvement
- "Active" threshold: raw_margin <= 2.0 * margin_floor. With 0-1 active constraints, behavior is identical to before
- Noise floor (0.25) intentionally NOT changed -- structural fixes should prevent hitting it
- Previous session's 6 fixes remain: energy/smoothness=0, per-env effort limits, cost value clamp, NaN guard, line_search_cost_margin removal

## [2026-03-16] Code review: fix 6 bugs in ConstraintTRPO pipeline

### Context
Thorough analysis of ConstraintTRPO algorithm (constraint_trpo.py, config.py,
constraints.py, rsl_rl_ppo_cfg.py) while a pre-fix training run was executing.
Identified 6 issues ranging from correctness bugs to theoretical inconsistencies.

Training log analysis of run `2026-03-16_03-27-02` (548/600 iters, pre-fix) confirmed:
- Entropy collapsed to 0.07, noise_std hit 0.25 floor by mid-training
- Singularity cost increasing (8.5 -> 12.6), arm drifting toward singularity
- 3 constraints at 88-92% of budget (oscillation, singularity, yaw_vel)
- Roll/pitch error plateaued at 16-19 deg (same pattern as previous runs)
- Energy/smoothness double-counting identified as primary entropy collapse driver

### Fixed
- `algorithms/constraint_trpo.py`: Cost value loss now clamps targets to >=0 -- softplus-bounded V_cost (>=0) could never match negative GAE targets, causing systematic estimation bias
- `algorithms/constraint_trpo.py`: Added NaN guard to per-constraint cost advantage normalization -- non-finite advantages now zeroed with warning instead of propagating
- `algorithms/constraint_trpo.py`: Clarified log_std clamp comment -- clamp happens BEFORE KL measurement so logged metric reflects actual policy state
- `mdp/constraints.py`: `effort_limit_cost` now uses per-env DR'd limits (`_robot.data.joint_effort_limits`) instead of cached scalar `env._default_effort_limit` -- envs with weaker DR'd motors (0.7x) were not properly constrained
- `config.py`: `HeroAgentConstrainedEncoderEnvCfg.reward` set `energy_weight=0.0, smoothness_weight=0.0` -- energy penalty overlapped with joint_vel constraint, smoothness penalty overlapped with oscillation constraint, creating double gradient pressure that killed exploration

### Removed
- `algorithms/constraint_trpo.py`: Removed unused `line_search_cost_margin` parameter from `__init__` and `self` storage (dead code from early prototype)
- `agents/rsl_rl_ppo_cfg.py`: Removed matching `line_search_cost_margin` field from `RslRlConstraintTRPOAlgorithmCfg`

### Notes
- Highest-impact fix: energy/smoothness removal (#5) -- directly addresses the entropy collapse that plagued all previous runs
- Next step: retrain with these fixes and monitor whether noise_std stays above floor
- If entropy still collapses: consider raising entropy_coef (0.02 -> 0.03) or noise floor (0.25 -> 0.3)

## [2026-03-10] Entropy collapse fix: raise noise floor + remove effort_limit DR conflict

### Context
Analysis of constrained_encoder_base run `klmm0hqj` (372 steps, barrier_t=50/100) showed
policy learning reward (0 -> 1.5, attitude error 30 -> 15 deg) but arm freezing at singularity.

Previous diagnosis was wrong: line_search_success was 99.4% (333/335), NOT 80% failure.
Barrier_t=50/100 is fine. The actual root cause chain:
1. Entropy collapsed by step 50 (2.84 -> -0.69), noise_std hit floor (0.98 -> 0.15)
2. Two inconsistent noise floors existed: constraint_trpo.py (0.1) vs base_runner.py (0.15)
3. std=0.15 -> 95% of actions within +-0.3 of mean -> exploration dies
4. "Don't move arm" is rational under low exploration (avoids 5 arm-related constraint costs)
5. Arm drifts to singularity (joint_pos_mean: 1.6 -> 5.4), costs explode
6. Narrow signal -> encoder z saturates (+-0.98 by step 50)

Additionally, `HeroAgentConstrainedEncoderEnvCfg` overrode `joint_effort_limit_range=(1.3, 1.5)`,
allowing PhysX 130-150% stall torque while `effort_limit_cost` checks against 100% -- a permanent
training conflict where normal actuator behavior always violates the constraint.

Barrier_t reverted from 50/100 back to 10/50 (both values produce >99% ls_success; lower
barrier gives tighter constraint enforcement from the start).

### Changed
- `algorithms/constraint_trpo.py`: `min_log_std` from `log(0.1)` to `log(0.25)` -- unified with base_runner floor
- `runners/base_runner.py`: `min_std` from 0.15 to 0.25 -- at std=0.25, entropy ~0.07 (vs -0.95 at 0.15)
- `config.py`: Removed `joint_effort_limit_range=(1.3, 1.5)` override from `HeroAgentConstrainedEncoderEnvCfg`, restoring unified default (0.7, 1.0)
- `agents/rsl_rl_ppo_cfg.py`: `barrier_t` 50.0 -> 10.0, `barrier_t_final` 100.0 -> 50.0 (revert to original)
- `mdp/constraints.py`: `ALBCConstraintCfg.barrier_t` 50.0 -> 1.0, `barrier_t_final` 100.0 -> 50.0 (dead field, synced with revert)

### Notes
- std=0.25 gives 95% of samples within +-0.5 of mean for [-1,1] actions -- wider exploration without excessive noise
- Two noise floors now unified at 0.25 (constraint_trpo.py post-step + base_runner.py per-iteration)
- If arm still freezes at step 200: next step is relax singularity budget 0.15 -> 0.3
- Follow-up: make noise floor a config field instead of hardcoded

## [2026-03-09] barrier_t fix: 10->50 initial, 50->100 final

### Context
Constrained-Encoder-Base training (320 iterations) converged to a local optimum where
the arm stopped moving entirely. Full root cause chain: low barrier_t (10-20) caused
cost gradient to dominate policy_loss -> line search failed 80% of the time (success=0
from step ~80) -> actor params reverted every iteration (no learning) -> encoder gated
out (no policy-loss gradient) -> z saturated to [-1, +1] (z_bounds_loss too weak alone)
-> encoder grad_norm -> 0 (dead) -> entropy collapsed (2.0 -> 0 by step 75) ->
noise_std hit floor (0.15) -> arm froze.

NORBC paper nominal barrier_t=100. Paper ablation (Table II): t=10 caused constraint
violations. Our schedule was 10->50. Increasing to 50->100 reduces cost gradient
amplification 5x at initialization, allowing reward gradient to compete and line search
to succeed.

Also synced the dead field `ALBCConstraintCfg.barrier_t` (was 1.0, never read by runner)
to 50.0 for code consistency.

### Changed
- `agents/rsl_rl_ppo_cfg.py`: `barrier_t` 10.0 -> 50.0, `barrier_t_final` 50.0 -> 100.0 in `RslRlConstraintTRPOAlgorithmCfg`
- `mdp/constraints.py`: `ALBCConstraintCfg.barrier_t` 1.0 -> 50.0, `barrier_t_final` 50.0 -> 100.0 (dead field, synced for consistency)

### Notes
- `barrier_t_schedule_frac=0.4` unchanged -- annealing completes at 40% of max_iterations
- Next run key metrics: line_search_success > 50%, entropy > 0.5 past step 100, encoder/grad_norm > 0, joint_vel_abs_max > 1.0
- Follow-up issues identified: (1) line_search_cost_margin configured but unused, (2) ALBCConstraintCfg dead fields cleanup, (3) encoder KL guard if spike persists

---

## [2026-03-09 Summary] NORBC conformance + actuator DR unification

### Context
Systematic NORBC paper comparison and actuator parameter audit. Asymmetric critic (raw
privileged 19D instead of encoder z 13D) was the highest-impact change. Also unified all
environments to use physically-grounded actuator DR ranges from Dynamixel XW540-T260-R
datasheet. Per-constraint cost advantage normalization added (NORBC Sec IV-B).

### Changed
- `encoder/actor_critic_encoder.py`: Asymmetric critic (`_get_critic_obs()` bypassing encoder, 32D input)
- `encoder/actor_critic_encoder_constrained.py`: Cost critic asymmetric, K mismatch detection
- `algorithms/constraint_trpo.py`: Update order threshold->policy->value, instantaneous threshold, pure MSE value, per-constraint cost advantage standardization
- `config.py`: Unified actuator DR (Kp 40-120, Kd 0.5-5.0, effort 0.7-1.0), removed TDC overrides
- `hero_agent.py`: `velocity_limit_sim` 6.28 -> 4.19 rad/s (datasheet 40 rpm)

### Removed
- `algorithms/constraint_trpo.py`: `encoder_value_grad_scale`, `adaptive_ema_alpha`, `_compute_barrier_loss()`

## [2026-03-08 Summary] Constraint system build-out and stabilization

### Context
Built constraint system from 3 to 8 terms, converted 3 binary costs to continuous (NORBC
average type), added cost critic softplus fix, and tuned barrier schedule. Multiple rounds
of budget tuning driven by WandB analysis. Key fix: cost critic negative V_cost (softplus
output activation) caused catastrophic instability at step ~650.

### Changed
- `mdp/constraints.py`: ConstraintTermCfg registry, 8 cost functions (final: accum_rot, attitude_abs, singularity, effort_limit binary; joint_vel, oscillation, yaw_vel, cob_cog continuous)
- `algorithms/constraint_trpo.py`: barrier_t 1.0->10.0, value_lr 3e-4->1e-3, barrier_t_schedule_frac, mean_cost_returns clamp
- `encoder/actor_critic_encoder_constrained.py`: `F.softplus()` on cost critic output, K mismatch detection
- `config.py`: 8-term constraint list, budgets tuned (joint_vel=1.0, singularity=0.15, attitude_err=0.087, attitude_abs limit 60 deg)
- `runners/constraint_encoder_runner.py`: Named constraint logging, auto-sync K, lambda checkpoint

## [2026-03-05/06 Summary] ConstraintTRPO initial implementation

### Context
Full NORBC-style constrained RL (IPO + TRPO) implementation. 6 critical bugs fixed across
5 debugging rounds. Encoder value gradient experiments (shelved: too aggressive at any scale).
Entropy_coef finalized at 0.005. Equilibrium joint init implemented then reverted.

### Added
- `algorithms/constraint_trpo.py`: Full TRPO + IPO (~600 lines)
- `encoder/actor_critic_encoder_constrained.py`: Multi-head cost critic
- `runners/constraint_encoder_runner.py`: Constraint metrics logging
- `mdp/constraints.py`: Initial 3 binary cost functions, `compute_all_costs()`
- `algorithms/ppo_patch.py`, `docs/THEORETICAL_ANALYSIS.md`, `play.py`, `eval_dr_comparison.py`

### Changed
- `base_env.py`: Accumulated rotation tracking, cost computation, constraint buffers
- `config.py`: Registered `Isaac-HeroAgent-Constrained-Encoder-Base-v0`

### Notes
- Baseline run (15-30-39) best overall: mean_reward 80, attitude error 8-10 deg
