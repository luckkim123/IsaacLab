# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

For entries before 2026-03-27, see [changelog_legacy.md](changelog_legacy.md).
For the encoder ablation study (Steps 0-19), see
[encoder_ablation.md](source/isaaclab_tasks/isaaclab_tasks/direct/constrained_albc/docs/encoder_ablation.md).

---

## [2026-03-30] Frozen Encoder: Normalization Mismatch Fix + z_init_scale Experiment

### Context

Encoder z sweep analysis revealed the frozen encoder was producing constant (saturated)
z output -- 13/13 dimensions pinned at |z| > 0.999 regardless of DR parameter variation.
Root cause: the offline encoder was trained WITH static min-max normalization
(`(2x - upper - lower) / (upper - lower)` -> [-1, 1]), but the frozen encoder deployment
did not load these bounds from the checkpoint. Raw privileged obs (body_mass~9,
stiffness~80, water_density~1010) caused extreme pre-activation values, saturating softsign.

The fix auto-loads normalization bounds from the offline encoder checkpoint via
`register_buffer()` in `_load_pretrained_encoder()`. After fix: 0/13 saturated dims,
116/299 active param-z pairs (vs 0/299 before).

Additionally, `z_init_scale` was changed from 0.01 to 1.0. The 0.01 scaling was designed
for hist-only warm-start (prevent z from disrupting pre-trained actor) but is
counterproductive when training from scratch -- it forces the actor to spend iterations
re-learning to upweight z.

### Fixed
- `encoder/actor_critic_frozen_encoder.py`: `_load_pretrained_encoder()` now auto-loads
  `enc_obs_lower`/`enc_obs_upper` from checkpoint and registers buffers + sets
  `_has_static_enc_norm=True`, even when base class was initialized without bounds.

### Changed
- `agents/rsl_rl_ppo_cfg.py`: `_FrozenEncoderPolicyCfg.z_init_scale` 0.01 -> 1.0

### Experimental Results

Encoder z sweep (offline encoder checkpoint):

| Condition | Active z-param pairs | Saturated dims |
|-----------|---------------------|----------------|
| With static normalization | 116/299 | 0/13 |
| Without normalization (bug) | 0/299 | 13/13 |

Training comparison (500 iterations, last 50 avg):

| Run | Roll (deg) | Pitch (deg) | Reward | Terminations |
|-----|-----------|------------|--------|-------------|
| Hist-Only baseline | **9.78** | **7.41** | **-12.18** | 4.0% |
| Frozen(norm bug) | 9.77 | 7.51 | -13.29 | 2.9% |
| Frozen(z=0.01, norm fix) | 10.27 | 7.97 | -13.76 | 3.3% |
| Frozen(z=1.0, norm fix) | 9.85 | 7.89 | -14.64 | 3.3% |

z_init_scale=1.0 improved roll by 0.41 deg over z=0.01 and stabilized convergence slope
(roll: +0.003/iter vs +0.007/iter). However, frozen encoder still does not beat hist-only
on attitude accuracy. Encoder z_std=0.74 (healthy), z_mean=-0.16 (centered) -- encoder
itself is functioning correctly.

### Notes
- The normalization bug affected ALL previous frozen encoder experiments (2026-03-29 ~
  2026-03-30). The encoder was always producing constant output.
- Despite the fix, frozen encoder does not yet outperform hist-only. Possible causes:
  (1) offline encoder trained to predict V_critic, not attitude error directly;
  (2) 240D history already encodes sufficient dynamics info, making z redundant;
  (3) actor needs warm-start from hist-only to leverage z effectively.
- Analysis plots saved to `logs/offline_encoder/encoder_analysis/`.

---

## [2026-03-30] Constrained ALBC Codebase Refactoring

### Context

Ablation study (Steps 0-19, 20+ experiments) accumulated 30 debug tasks, 33 RunnerCfgs,
12 EnvCfgs, and 4 ablation-only parameters. With ablation conclusions preserved in
`encoder_ablation.md`, removed all debug/ablation code, keeping only 3 production tasks.

### Changed
- `agents/rsl_rl_ppo_cfg.py`: 1625 -> 336 lines. Removed 30+ ablation RunnerCfgs.
  Renamed `_DebugPolicyCfg` -> `_HistOnlyPolicyCfg`. Removed `noise_std_type` from
  `_EncoderPolicyCfg`. Fixed pre-existing double `@configclass` on
  `_FrozenEncoderAlgorithmCfg`.
- `config.py`: 529 -> 410 lines. Removed 8 debug/ablation EnvCfg classes.
- `encoder/actor_critic_encoder.py`: 399 -> 357 lines. Removed ablation parameters
  (`noise_std_type`, `clamp_actions`, `symmetric_critic`, `z_bounds_coef`,
  `z_bounds_soft_bound`). Deleted `z_bounds_loss()` method. Hardcoded log_std,
  no-clamp, asymmetric critic.
- `__init__.py`: 379 -> 60 lines. Removed 30 `gym.register()` blocks.
- `encoder/__init__.py`: Removed `ActorCriticConstrained` export.

### Removed
- `encoder/actor_critic_constrained.py`: 43-line Step 3 ablation-only class deleted.
- 30 debug task registrations (e.g., `Isaac-Constrained-ALBC-Debug-*`)

### Notes
- Total ~1812 lines removed across 5 files (1 deleted, 4 rewritten).
- 3 production tasks retained: Encoder-v0, HardDR-HistOnly-v0, HardDR-FrozenEncoder-v0.
- Checkpoint backward compatibility maintained via `load_state_dict(..., strict=False)`
  and `**kwargs` catch in `ActorCriticEncoder.__init__`.

---

## [2026-03-30] Frozen Encoder: Three Critical Fixes

### Context

Frozen encoder fine-tuning (offline pipeline Step 3) had noise_std explosion preventing
any learning. Systematic investigation found three independent bugs.

**Bug 1: `_normalize_storage_values()` overwrote normalized advantages.**
`storage.compute_returns()` normalizes advantages, then `_normalize_storage_values()`
recomputed advantages as `returns_norm - values_norm`, introducing bias (mean ~-0.66).
Surrogate loss ~1.0 at iter 0 (normal: ~0.002) drove immediate noise_std explosion.

**Bug 2: Critic received less information than actor.**
`_get_critic_obs()` returned `cat([o_t, p_t])` = 37D while actor received
`cat([o_t, hist, z])` = 267D. Critic was blind to 240D of proprioceptive history.
All previous encoder experiments (Steps 4-19) were affected.

**Bug 3: Missing denormalization during rollout (HORA mismatch).**
HORA denormalizes critic output during rollout so GAE operates on raw-scale values.
Our implementation stored normalized values directly.

### Fixed
- `runners/constraint_encoder_runner.py`: Removed advantages recomputation in
  `_normalize_storage_values()`
- `encoder/actor_critic_encoder.py`: `_get_critic_obs()` now returns
  `cat([o_t, hist, p_t])` = 277D. `num_critic_obs` includes `proprio_hist_dim`.
- `runners/constraint_encoder_runner.py`: HORA-style value normalization --
  denormalize stored values before GAE, denormalize last_values for bootstrap,
  then normalize values/returns after GAE for critic targets.

### Changed
- `agents/rsl_rl_ppo_cfg.py`: `ALBCHardDRFrozenEncoderRunnerCfg` obs_groups
  critic now includes `proprio_hist`. Added `hist_only_checkpoint` field.
- `encoder/actor_critic_frozen_encoder.py`: `load_history_only_weights()` now
  copies `log_std`/`std` parameter from hist_only checkpoint.

### Experimental Results

| Metric | Frozen Encoder (499 iters) | Hist Only | Delta |
|--------|---------------------------|-----------|-------|
| Best roll | 6.9 deg | 7.0 deg | -0.1 |
| Best pitch | 5.7 deg | 5.6 deg | +0.1 |
| Final roll | 11.8 deg | 8.7 deg | +3.1 |
| Final pitch | 6.9 deg | 6.5 deg | +0.4 |
| noise_std | 0.065 | 0.153 | -0.088 |

Training stable. Best performance comparable but encoder z not yet providing measurable
advantage over history-only baseline. Final roll has more variance (7-12 deg oscillation).

### Notes
- All previous encoder experiments (Steps 4-19) had the critic bug (37D instead of 277D).
  The "encoder destabilizes training" conclusion may need revision.
- Offline encoder quality verified: z explains 70.3% additional V_critic variance
  (R^2: 0.088 -> 0.791).
- Untested: actor warm-start, encoder unfreezing after convergence, online encoder
  with fixed critic.

---

## [2026-03-29] Offline Encoder Pipeline

### Context

After 15+ online encoder experiments failed (see
[encoder_ablation.md](source/isaaclab_tasks/isaaclab_tasks/direct/constrained_albc/docs/encoder_ablation.md)),
pivoted to offline training: (1) collect rollouts from trained history-only policy,
(2) train encoder supervised with value prediction bottleneck, (3) fine-tune actor
with frozen encoder.

Root cause of online failure: `sample().clamp(-1,1)` in `ActorCriticEncoder.act()`
concentrates actions at boundaries, amplifying KL 100x. Secondary: env-level clamp
positive feedback on noise_std when encoder makes advantages noisy.

### Added
- `scripts/analysis/collect_rollouts.py`: Rollout data collection from trained policy.
  Collects (o_t, privileged, V_critic) per step.
- `scripts/analysis/train_offline_encoder.py`: Supervised encoder training.
  Architecture: p_t(23D)->MLP[256,128,64]->softsign->z(13D),
  value head: cat([o_t, z])->Linear->V_hat, loss=MSE(V_hat, V_critic).
- `encoder/actor_critic_frozen_encoder.py`: `ActorCriticFrozenEncoder` -- encoder
  frozen (requires_grad=False), pre-trained weights loaded, z-related actor weights
  init to near-zero (scale=0.01). `load_history_only_weights()` for warm-start.
- `config.py`: `ALBCHardDRFrozenEncoderEnvCfg` -- Hard DR + state_space=23 +
  history(30, stride=1).
- `agents/rsl_rl_ppo_cfg.py`: `_FrozenEncoderAlgorithmCfg` (use_encoder_update=False),
  `_FrozenEncoderPolicyCfg`, `ALBCHardDRFrozenEncoderRunnerCfg`.
- `__init__.py`: Registered `Isaac-Constrained-ALBC-HardDR-FrozenEncoder-v0`.

### Experimental Results
- Rollout collection: 207,360 transitions (50 episodes, 512 envs).
- Offline encoder: val_loss 13.67->2.43, z_std=0.315 (non-trivial output).

---

## [2026-03-29] Hard DR Environment

### Context

History-only baseline achieves 3.0 deg with standard DR -- encoder has no gap to close.
Created "hard DR" environment where history-only degrades to ~10 deg, providing headroom
for encoder benefit.

### Added
- `config.py`: `HardDomainRandomizationCfg` -- aggressive DR:
  added_mass +-40% (was +-15%), body_mass +-25% (was +-10%), volume +-25% (was +-10%),
  CoG/CoB offsets doubled, inertia (0.5, 1.8), payload 0-2kg
- `config.py`: `ALBCHardDRHistOnlyEnvCfg` -- hard DR + history(30) + ocean current
- `agents/rsl_rl_ppo_cfg.py`: `ALBCHardDRHistOnlyRunnerCfg`
- `__init__.py`: Registered `Isaac-Constrained-ALBC-HardDR-HistOnly-v0`

---

## [2026-03-29] Encoder Ablation: Root Cause Found

### Summary

20+ experiments (Steps 0-19) systematically isolated why encoder destabilizes training.
Full details: [encoder_ablation.md](source/isaaclab_tasks/isaaclab_tasks/direct/constrained_albc/docs/encoder_ablation.md).

**Root cause:** `sample().clamp(-1,1)` in `ActorCriticEncoder.act()`.
Actions pile at boundaries -> sharp log_prob gradients -> KL 100x amplification ->
adaptive LR death. Removing clamp (Step 17) reduced encoder KL from 0.88 to 0.003 --
but noise_std exploded due to env-level clamp positive feedback loop.

10 hypotheses tested and disproved (EmpiricalNorm, encoder gradient, encoder freeze,
init LR, update path, history, critic asymmetry, normalization, std type, action clamp
alone). Online co-training structurally unstable in 2D action space.

### Added
- `encoder/actor_critic_encoder.py`: `noise_std_type`, `clamp_actions`,
  `symmetric_critic` params. Static min-max normalization support.
  HORA-style `actor_obs_normalizer` (excludes z).
- `config.py`: `proprio_history_stride` field. Debug/ablation env configs (Steps 0-19).
- `agents/rsl_rl_ppo_cfg.py`: 15+ runner configs for ablation steps.
- `albc_env.py`: Strided proprioceptive history recording.
- `runners/constraint_encoder_runner.py`: `normalize_value` flag.
- `__init__.py`: 15+ debug task registrations.

### Changed (rsl_rl/algorithms/ppo.py -- external, not git-tracked)
- Added `use_encoder_update`, `reward_scale`, `min_lr`, `max_lr`, `encoder_grad_scale`.
- **Needs reapply on container rebuild.**

---

## [2026-03-27] Action Parameterization and Reward Tuning

### Summary

Three fixes: (1) torque constraint measured unbounded PD internal computation instead of
actual motor output (100% violated, unsatisfiable), (2) Gaussian noise in absolute joint
targets created 115 deg/step jitter (91% effort saturation), switched to delta action,
(3) tuned delta_scale and reward weights.

### Fixed
- `mdp/constraints.py`: `torque_limit_cost()` uses `applied_torque` (post-clamp, max
  9.5 Nm) instead of `computed_torque` (PD internal, 326-554 Nm)

### Changed
- `config.py`: `action_scale: float = pi` -> `delta_scale: float = 0.08`
- `albc_env.py`: `_apply_joint_pd_action()` from absolute to delta accumulation
  (`q_des += delta_scale * a_t`, clamped to joint limits)
- `config.py`: `k_tau` -0.001 -> -0.01, `k_s` -0.05 -> -0.2

### Experimental Results (delta action first run, 139 iters)

| Category | Metric | Absolute | Delta |
|----------|--------|----------|-------|
| Dynamics | effort_saturation | 91% | 2.2% |
| | torque cost_return | 92 | 4.5 (within budget) |
| Attitude | Roll / Pitch | 17/13 deg | 21.6/18.8 deg |

Delta action solved dynamics (effort/torque within limits) at the cost of slower attitude
convergence (delta_scale bandwidth). Tuned from 0.05 to 0.08 (0.39s to 90 deg).

---

## [2026-03-27] TRPO+IPO Algorithm Fixes (NORBC Paper Alignment)

### Summary

Six structural fixes aligning ConstraintTRPO with NORBC paper (Muller et al., ICML 2025).
Combined effect: reward -78.80 -> -37.36 (2x), roll 29.2 -> 18.0 deg, pitch 26.5 ->
11.9 deg, z saturation eliminated ([-0.99,0.99] -> [-0.53,0.40]).

### Fixes

1. **Line search logging artifact**: `surrogate()` closure overwrites monitoring vars on
   each backtracking attempt. Fixed: recalculate with reverted params after failure.

2. **Cost critic d_k^2 normalization**: Non-standard, ineffective (yaw_vel contributed
   98.6% of loss). Changed to plain MSE (OmniSafe/CPO convention).

3. **Encoder LS gating removed**: Encoder received zero gradient on line search failure,
   creating starvation loop. No precedent in HORA/RMA/RSL-RL.

4. **Encoder integrated into TRPO trust region**: Separate Adam encoder update destroyed
   trust region (post-encoder KL: 27.6x budget avg, max 1153.4x). Moved encoder params
   into TRPO CG + line search (matching NORBC joint training).

5. **Missing 1/(1-gamma) in IPO barrier**: With cost_gamma=0.99, factor=100. Barrier
   estimated margin change 100x too small without this factor.

6. **Per-constraint cost advantage standardization**: Restored NORBC Sec IV-B
   `(A_Ck - mean) / (std + 1e-8)`. Raw advantages near-zero when deeply infeasible.

### Changed
- `algorithms/constraint_trpo.py`: Cost value loss plain MSE, LS gating removed,
  encoder params in `_policy_params`, 1/(1-gamma) factor added, per-constraint
  standardization restored
- `agents/rsl_rl_ppo_cfg.py`: `barrier_alpha` 0.02 -> 0.05, removed
  `num_encoder_epochs`/`encoder_lr`

### Removed
- `algorithms/constraint_trpo.py`: `_update_encoder()`, `encoder_optimizer`,
  `_encoder_params`, `_has_encoder_params`, `_last_pre_encoder_kl`
