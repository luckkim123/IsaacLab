# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

For entries before 2026-03-27, see [changelog_legacy.md](changelog_legacy.md).
For the encoder ablation study (Steps 0-19), see
[encoder_ablation.md](source/isaaclab_tasks/isaaclab_tasks/direct/constrained_albc/docs/encoder_ablation.md).

---

## [2026-03-30] Asymmetric Critic Test + Pre-Softsign LayerNorm

### Context

Two experiments to improve online encoder training:

**Experiment 1: Asymmetric critic (critic sees z + p_t).**
Tested whether critic receiving both z and raw privileged obs p_t would provide
encoder gradient from value loss while using separate actor/critic MLPs.
Result: **shortcut problem confirmed.** Critic immediately ignores z in favor of
p_t (easier path to value prediction). z_std goes to ~1 instantly and stays
constant -- encoder receives no meaningful gradient from either path. Actor also
can't leverage z (chicken-and-egg: z is noise -> actor ignores z -> no gradient
to shape z).

**Experiment 2: Pre-softsign LayerNorm (shared backbone).**
Root cause analysis of z saturation in original shared backbone run: encoder
weight growth causes pre-softsign MLP output to explode (|x| mean: 0.44 at init
-> 8.50 at iter 499). Softsign gradient = 1/(1+|x|)^2 vanishes (75% of outputs
have gradient < 0.05 by iter 350), trapping z near boundaries.

Added LayerNorm between encoder MLP output and softsign activation. LayerNorm
normalizes output to ~N(0,1), keeping softsign in its responsive range regardless
of weight magnitude.

Result: z saturation eliminated. Encoder learns meaningful representations --
Body Mass 12/13 active dims, Main Volume 10/13, CoG/CoB 10/13. However, noise_std
drops too fast (entropy_coef=0.0, no min_std floor), causing premature exploration
collapse. Final performance still worse than hist-only baseline.

### Added
- `config.py`: `ALBCHardDRAsymmetricEncoderEnvCfg` -- inherits SharedBackbone env
- `agents/rsl_rl_ppo_cfg.py`: `_AsymmetricEncoderPolicyCfg` (critic_uses_z=True,
  shared_backbone=False, critic_obs_normalization=False),
  `ALBCHardDRAsymmetricEncoderRunnerCfg`
- `__init__.py`: Registered `Isaac-Constrained-ALBC-HardDR-AsymmetricEncoder-v0`

### Changed
- `encoder/actor_critic_encoder.py`: Added `critic_uses_z` param -- when True,
  `_get_critic_obs()` includes z via `_encode(obs)`, making critic input
  cat([o_t, hist, z, p_t]). Added `encoder_output_norm` param -- when True,
  inserts `nn.LayerNorm(latent_dim)` between encoder MLP and softsign activation.
  Updated `_encode()` flow: MLP -> LayerNorm -> softsign.
- `agents/rsl_rl_ppo_cfg.py`: `_SharedBackbonePolicyCfg` now sets
  `encoder_output_norm: bool = True`. Both shared backbone and asymmetric configs
  use LayerNorm.
- `scripts/analysis/encoder_z_sweep.py`: Detects `_encoder_output_norm` in
  checkpoint and includes LayerNorm in reconstructed encoder Sequential.
  `build_encoder_mlp()` gains `output_norm` parameter.

### Experimental Results

**Shared Backbone + LayerNorm (500 iters) vs Hist-Only (500 iters):**

| Metric | Shared BB + LN | Hist-Only | Delta |
|--------|:--------------:|:---------:|:-----:|
| Roll | 9.96 deg | 8.74 deg | +1.22 |
| Pitch | 9.74 deg | 6.54 deg | +3.20 |
| Reward | -18.22 | -10.75 | -7.47 |
| noise_std | 0.15 | 0.15 | 0 |
| z_std | 0.56 | -- | -- |

**Encoder z sweep comparison (shared backbone, iter 499):**

| Condition | |z|>0.9 | Softsign grad mean | Body Mass active | Main Vol active |
|-----------|:------:|:-----------------:|:----------------:|:--------------:|
| No LayerNorm | 44% | 0.063 | 12/13 (saturated) | 10/13 (saturated) |
| With LayerNorm | ~0% | ~0.25 (healthy) | 12/13 (real variation) | 10/13 (real variation) |

**Root cause data (no LayerNorm, model_0 vs model_499):**

| Metric | Init (iter 0) | Trained (iter 499) |
|--------|:------------:|:-----------------:|
| Pre-softsign |x| mean | 0.44 | 8.50 |
| |x| > 3 fraction | 0% | 80% |
| Softsign gradient mean | 0.548 | 0.063 |
| Encoder weight std (hidden) | 0.04-0.07 | 0.12-0.13 |

**Asymmetric Critic + LayerNorm (500 iters):**

Previous asymmetric run (no LayerNorm) showed z_std -> 1 immediately (shortcut +
saturation confounded). Re-ran with LayerNorm to isolate the shortcut effect.
Result: encoder DOES learn -- z_std=0.70 (stable), z responds to DR parameters.
Shortcut is not total: actor gradient alone (with LayerNorm) is sufficient to
train the encoder. Encoder even shows broader DR sensitivity than shared backbone
(quad_damp_roll 0.06->0.47, water_density 0.04->0.26, buoy_cog_z 0.02->0.14).

| Metric | Hist-Only | Shared BB + LN | Asymmetric + LN |
|--------|:---------:|:--------------:|:---------------:|
| Roll | **8.74** | 9.96 | 9.77 |
| Pitch | **6.54** | 9.74 | 10.16 |
| Reward | **-10.75** | -18.22 | -19.74 |
| noise_std | 0.15 | 0.15 | 0.14 |
| z_std | -- | 0.56 | 0.70 |

Encoder z sweep (asymmetric + LN, iter 499): Body Mass 10/13 active (max 1.06),
Main CoG Z 8/13 (max 0.90), Quad Damp Roll 3/13 (max 0.47), Water Density 2/13
(max 0.26). More diverse than shared backbone but still no performance gain.

### Notes
- Asymmetric critic WITHOUT LayerNorm: shortcut + saturation -> encoder fails.
  WITH LayerNorm: encoder learns via actor gradient alone. The earlier "shortcut
  conclusively disproved" conclusion was wrong -- the issue was saturation, not
  shortcut exclusively.
- Both encoder architectures (shared BB, asymmetric) learn encoder representations
  but neither beats hist-only. Common bottleneck: noise_std collapse (entropy_coef=0,
  no min_std floor).
- encoder_z_sweep.py verified to produce identical output as training forward pass
  (max diff = 0.00e+00).

---

## [2026-03-30] Encoder Input Reduction + Hyperparameter Ablations

### Context

Continued encoder experiments from previous session. Three ablations tested on
the asymmetric critic + LayerNorm architecture to improve encoder-based policy:

**Experiment 3: entropy_coef=0.001 (asymmetric + LN, 23D->13D, 500 iters).**
Hypothesis: noise_std collapse (0.14, LOW) is the bottleneck. Small entropy bonus
should maintain exploration. Result: noise_std improved marginally (0.14->0.17) but
still LOW. Roll WORSENED (9.77->13.59 deg), pitch slightly better (10.16->9.69 deg).
Entropy bonus interfered with exploitation without sufficiently maintaining exploration.

**Experiment 4: encoder [128, 64] 2-hidden layer (asymmetric + LN, 23D->13D, 500 iters).**
Hypothesis: 3-layer [256,128,64] encoder (~49K params) is over-parameterized for
23D->13D compression. Smaller encoder should learn faster. Result: performance degraded
(not fully analyzed, user observed instability and moved on).

**Experiment 5: Reduced encoder input 15D->6D (asymmetric + LN, no ocean current).**
Based on z-sweep sensitivity analysis, dropped 8 input dims with near-zero encoder
response (buoy CoG/CoB Z, main/buoy Ixx/Iyy, payload CoG Z, water density). Kept
10 clearly important + 3 borderline/suspicious + 2 physically important (payload CoG XY).
Also removed ocean current from DR. Result: severe instability at 145 iters (roll 23 deg,
pitch 39 deg), compression ratio 2.5:1 likely too aggressive.

**Experiment 6: Increased output to 9D (15D->9D, asymmetric + LN, no ocean current, 500 iters).**
Raised latent dim from 6 to 9 (compression ratio 1.67:1, close to original 1.77:1).
Result: much better than 6D -- roll 12.96 deg, pitch 11.06 deg. Encoder z sweep shows
improved sensitivity to CoG Z (3.5x), CoB Z (2.9x), and Lin Damp Roll (0->0.46) vs
23D->13D. But performance still worse than hist-only and 23D->13D asymmetric.
noise_std=0.13 (LOW) remains the common bottleneck across ALL encoder experiments.

### Added
- `agents/rsl_rl_ppo_cfg.py`: 15D encoder bounds (`_ENC_OBS_INDICES_15D`,
  `_ENC_OBS_15D_LOWER`, `_ENC_OBS_15D_UPPER`) selected by z-sweep sensitivity analysis
- `scripts/analysis/common.py`: `_build_reduced_encoder_sweep()` for reduced encoder
  z sweep parameter mapping

### Changed
- `encoder/actor_critic_encoder.py`: Added `encoder_obs_indices` parameter.
  When provided, `_encode()` selects subset of privileged dims before normalization.
  Encoder input_dim matches len(indices), bounds validated against selected dims.
- `agents/rsl_rl_ppo_cfg.py`: `_AsymmetricEncoderPolicyCfg` updated to use 15D input
  (encoder_obs_indices), 9D output (encoder_latent_dim=9), entropy_coef=0.0 (restored).
  Encoder hidden dims restored to [256,128,64].
- `config.py`: `ALBCHardDRAsymmetricEncoderEnvCfg` now disables ocean current
  (max_velocity=0, noise_scale=0)
- `scripts/analysis/common.py`: `get_encoder_architecture_from_checkpoint()` detects
  softsign for 15D+ encoders with static bounds. `build_sweep_params_from_checkpoint()`
  routes non-23D static-bound encoders to reduced sweep builder.

### Experimental Results

**All experiments: asymmetric critic + LayerNorm, 500 iters unless noted:**

| Metric | Hist-Only | 23D->13D (ent=0) | 23D->13D (ent=0.001) | **15D->9D (ent=0)** |
|--------|:---------:|:----------------:|:--------------------:|:-------------------:|
| Roll | **8.74** | **9.77** | 13.59 | 12.96 |
| Pitch | **6.54** | **10.16** | 9.69 | 11.06 |
| Reward | **-10.75** | **-19.74** | -17.49 | -19.75 |
| noise_std | 0.15 | 0.14 | 0.17 | 0.13 |
| z_std | -- | 0.70 | 0.75 | 0.70 |

**15D->9D encoder z sweep improvements vs 23D->13D:**

| Parameter | 23D->13D max range | 15D->9D max range | Change |
|-----------|:------------------:|:-----------------:|:------:|
| Main CoG Z | 0.32 | **1.13** | 3.5x |
| Main CoB Z | 0.35 | **1.00** | 2.9x |
| Lin Damp Roll | 0.04 | **0.46** | 11x |
| Body Mass | 1.75 | 1.59 | -9% |
| Main Volume | 1.70 | 1.40 | -18% |

### Notes
- Input reduction improved encoder's sensitivity to secondary parameters (CoG, CoB,
  damping) by removing noise from uninformative dims. But performance did not improve.
- All encoder experiments share noise_std collapse (entropy_coef=0, no min_std floor).
  This is likely the fundamental bottleneck -- encoder produces good z but policy
  can't explore to exploit it.
- 15D->6D was too aggressive (pitch 39 deg at 145 iters). 15D->9D is viable.
- Ocean current removed from asymmetric env to focus on pure DR adaptation.

---

## [2026-03-30] Shared Backbone Encoder: Online End-to-End PPO

### Context

Offline encoder experiments showed that value prediction provides a strong learning
signal for the encoder (R^2 0.088 -> 0.791). However, the existing online encoder
architecture (separate mode) only gave the encoder gradient from the actor/policy
loss -- the critic used privileged obs directly, bypassing the encoder entirely.

Designed a shared backbone architecture where the encoder receives gradient from
BOTH actor and critic losses. The critic uses z (not raw privileged obs) as its
only path to privileged information, replicating the offline encoder's success
condition (value-prediction trains encoder) in online end-to-end training.

Previous online encoder experiments (Steps 5a-5b) failed with shared backbone due
to `sample().clamp(-1,1)` causing KL death (root cause identified in Step 17, clamp
since removed). With clamp removed + PPO single optimizer, shared backbone is stable.

### Added
- `config.py`: `ALBCHardDRSharedBackboneEnvCfg` -- Hard DR + 15-step strided
  history (stride=5, 120D) for shared backbone experiments
- `agents/rsl_rl_ppo_cfg.py`: `_SharedBackboneAlgorithmCfg` (PPO, use_encoder_update=False),
  `_SharedBackbonePolicyCfg` (shared_backbone=True, static min-max norm, proprio_hist_dim=120),
  `ALBCHardDRSharedBackboneRunnerCfg`
- `__init__.py`: Registered `Isaac-Constrained-ALBC-HardDR-SharedBackbone-v0`

### Changed
- `scripts/analysis/common.py`: Added `_build_constrained_albc_23d_sweep()` for 23D
  privileged obs z sweep. Fixed activation detection: `input_dim >= 23` -> softsign
  (was `>= 28`, missed 23D/27D). Added `enc_obs_lower`/`enc_obs_upper` params to
  `build_sweep_params_from_checkpoint()`.
- `scripts/analysis/encoder_z_sweep.py`: Rewritten to support static min-max
  normalization. Added `NormMode` dataclass, `load_encoder()` now detects and uses
  static bounds from checkpoint (`_enc_obs_lower`/`_enc_obs_upper` or top-level keys).

### Experimental Results

Training (shared backbone, 342 iters, PPO, HardDR):

| Metric | Shared Backbone | Hist-Only (500 iters) | Delta |
|--------|----------------|----------------------|-------|
| Roll | 8.56 deg | 8.74 deg | -0.18 |
| Pitch | 9.39 deg | 6.54 deg | +2.85 |
| Reward | -13.60 | -10.75 | -2.85 |
| noise_std | 0.21 | 0.15 | +0.06 |

Encoder z sweep (model_350):
- 75/299 active param-z pairs (range > 0.05)
- 0/13 saturated dims
- Top responsive: body_mass (10/13 active), main_vol (8/13), main_CoG_z (8/13)
- z at nominal: 12/13 dims near |z| > 0.7 (boundary bias), only z_11 (-0.17)
  has full dynamic range. Effective encoder capacity ~2-3 dimensions out of 13.

### Notes
- Training is stable (no KL death, no noise_std explosion) -- first successful
  online encoder training since ablation Steps 5a-5b
- Encoder IS learning domain info (z responds to DR parameters), but most z
  dimensions are near softsign boundary, reducing effective capacity
- Pitch 2.85 deg worse than hist-only, possibly due to history dimension gap
  (120D vs 240D) and/or symmetric critic limitation
- Offline encoder z sweep comparison was not properly validated -- inline analysis
  had softsign not applied, producing incorrect z ranges. Needs re-run with
  corrected `encoder_z_sweep.py` for fair comparison.

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
