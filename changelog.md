# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

For entries before 2026-03-27, see [changelog_legacy.md](changelog_legacy.md).

---

## [2026-03-30] Frozen Encoder: Three Critical Fixes

### Context

Frozen encoder fine-tuning (offline pipeline Step 3) had noise_std explosion
preventing any learning. Systematic investigation found three independent bugs,
each requiring a separate fix. All three fixed in this session.

**Bug 1: `_normalize_storage_values()` overwrote normalized advantages.**
`storage.compute_returns()` normalizes advantages (mean=0, std=1), then
`_normalize_storage_values()` recomputed advantages as `returns_norm - values_norm`,
introducing systematic bias (mean ~-0.66). This caused surrogate loss of ~1.0 at
iter 0 (normal: ~0.002), driving immediate noise_std explosion.
Fix: removed advantages recomputation, preserving already-normalized values.

**Bug 2: Critic received less information than actor.**
`_get_critic_obs()` returned `cat([o_t, p_t])` = 37D while actor received
`cat([o_t, hist, z])` = 267D. Critic was blind to 240D of proprioceptive history.
This made advantages poorly correlated with actor behavior, destabilizing sigma.
All previous encoder experiments (Steps 4-19 ablation study) were affected.
Fix: critic now receives `cat([o_t, hist, p_t])` = 277D.

**Bug 3: Missing denormalization during rollout (HORA mismatch).**
HORA denormalizes critic output during rollout (`value_mean_std(v, unnorm=True)`)
so GAE operates on raw-scale values. Our implementation stored normalized values
directly, causing GAE to mix raw rewards with normalized values as critic converged.
Fix: denormalize stored values before GAE, compute raw-scale last_values for
bootstrap, then re-normalize after GAE for critic targets.

**After all three fixes:** noise_std monotonically decreases (0.999 -> 0.065),
training stable for 499 iterations. Performance: roll 6.9 deg best (vs hist_only
7.0), pitch 5.7 deg best (vs hist_only 5.6). Final performance similar to
hist_only baseline -- encoder z not yet providing additional benefit, but training
is now stable.

Also added actor warm-start mechanism (`load_history_only_weights()` in runner)
and log_std transfer from hist_only checkpoint. These are available via
`hist_only_checkpoint` config but currently disabled (empty string).

### Fixed
- `runners/constraint_encoder_runner.py`: Removed advantages recomputation in
  `_normalize_storage_values()` (was overwriting normalized advantages with biased
  values, causing surrogate ~1.0 and noise_std explosion)
- `encoder/actor_critic_encoder.py`: `_get_critic_obs()` now returns
  `cat([o_t, hist, p_t])` = 277D instead of `cat([o_t, p_t])` = 37D. Critic
  `num_critic_obs` calculation updated to include `proprio_hist_dim`.
- `runners/constraint_encoder_runner.py`: HORA-style value normalization --
  denormalize stored values before GAE, denormalize last_values for bootstrap,
  then normalize values/returns after GAE for critic targets.

### Changed
- `agents/rsl_rl_ppo_cfg.py`: `ALBCHardDRFrozenEncoderRunnerCfg` obs_groups
  critic now includes `proprio_hist`. Added `hist_only_checkpoint` field (default
  empty, for optional actor warm-start).
- `encoder/actor_critic_frozen_encoder.py`: `load_history_only_weights()` now
  copies `log_std`/`std` parameter from hist_only checkpoint.

### Experimental Results

**Frozen encoder (499 iters, Hard DR, no warm-start):**

| Metric | Frozen Encoder | Hist Only | Delta |
|--------|---------------|-----------|-------|
| Best roll | 6.9 deg | 7.0 deg | -0.1 |
| Best pitch | 5.7 deg | 5.6 deg | +0.1 |
| Final roll | 11.8 deg | 8.7 deg | +3.1 |
| Final pitch | 6.9 deg | 6.5 deg | +0.4 |
| noise_std | 0.065 | 0.153 | -0.088 |

Best performance is comparable but frozen encoder has more variance in later
iterations (roll oscillates 7-12 deg vs hist_only stable 8-9 deg). Encoder z
is not yet providing measurable advantage over history-only baseline.

### Notes
- All previous encoder experiments (Steps 4-19) had the critic bug (37D instead
  of 277D). The "encoder destabilizes training" conclusion may need revision.
- Actor warm-start from hist_only available but not tested in this session
  (`hist_only_checkpoint = ""`).
- Offline encoder quality verified: z explains 70.3% additional V_critic variance
  beyond o_t alone (R^2: 0.088 -> 0.791). Encoder captures meaningful information.
- Next steps: (1) test actor warm-start, (2) test encoder unfreezing after
  actor convergence, (3) revisit online encoder training with fixed critic.

---

## [2026-03-29] Offline Encoder Pipeline Implementation

### Context

After 15+ online encoder experiments failed due to 2D action space instability,
pivoted to offline encoder training: (1) collect rollouts from trained history-only
policy, (2) train encoder supervised with value prediction bottleneck, (3) fine-tune
actor with frozen encoder.

Hard DR baseline (history-only, no encoder) achieves 8.7/6.5 deg attitude error --
the gap that encoder should close. Offline encoder trained successfully:
loss 13.67->2.43 (converged), z_std=0.315 (non-trivial, not collapsed).

Frozen encoder fine-tuning encountered noise_std explosion (>3000). Root cause:
PPO detected encoder parameters and used `_update_encoder_ppo()` path (per-epoch
LR adaptation, weight_decay=0) even though encoder is frozen. Fixed by adding
`use_encoder_update=False` via `_FrozenEncoderAlgorithmCfg`. Second run still
shows noise_std increase -- investigation ongoing (may be the same env-level clamp
positive feedback issue; encoder quality also needs verification).

### Added
- `scripts/analysis/collect_rollouts.py`: Rollout data collection from trained policy.
  Collects (o_t, privileged, V_critic) per step. Uses `--resume_path` for direct
  checkpoint path. Overrides `state_space=23` to access privileged obs from
  history-only env (which has state_space=0).
- `scripts/analysis/train_offline_encoder.py`: Supervised encoder training with
  value prediction bottleneck. Architecture: p_t(23D)->MLP[256,128,64]->softsign->z(13D),
  value head: cat([o_t, z])->Linear->V_hat, loss=MSE(V_hat, V_critic). Pure PyTorch
  (no Isaac Sim). Empirical static norm bounds from data.
- `encoder/actor_critic_frozen_encoder.py`: `ActorCriticFrozenEncoder` subclass of
  `ActorCriticEncoder`. Encoder frozen (requires_grad=False), pre-trained weights
  loaded from offline checkpoint. z-related actor weights init to near-zero (scale=0.01).
  `load_history_only_weights()` for warm-start from history-only checkpoint.
- `config.py`: `ALBCHardDRFrozenEncoderEnvCfg` -- Hard DR + state_space=23 +
  history(30, stride=1) matching history-only baseline for actor warm-start.
- `agents/rsl_rl_ppo_cfg.py`: `_FrozenEncoderAlgorithmCfg` (use_encoder_update=False),
  `_FrozenEncoderPolicyCfg` (pretrained_encoder_path, z_init_scale=0.01,
  clamp_actions=False), `ALBCHardDRFrozenEncoderRunnerCfg`.
- `__init__.py`: Registered `Isaac-Constrained-ALBC-HardDR-FrozenEncoder-v0`.

### Changed
- `encoder/__init__.py`: Added `ActorCriticFrozenEncoder` export.
- `runners/constraint_encoder_runner.py`: Added policy class registration in
  `_runner_module` namespace (ALBCActorCriticEncoder, ALBCActorCriticEncoderConstrained,
  ALBCActorCriticFrozenEncoder) for `eval()` resolution in OnPolicyRunner.

### Experimental Results
- Rollout collection: 207,360 transitions from 50 episodes (512 envs).
- Offline encoder training: val_loss 2.43, z_std=0.315 (encoder producing varied output).
- Frozen encoder fine-tuning: noise_std explosion persists even with use_encoder_update=False.
  Root cause under investigation.

### Notes
- noise_std explosion in frozen encoder run needs root cause analysis. Possible causes:
  (1) ActorCriticEncoder class has other differences from ActorCritic beyond update path,
  (2) env-level clamp positive feedback still applies even with frozen encoder,
  (3) offline encoder quality may be insufficient (z_std=0.315, need z_sweep analysis).
- `pretrained_encoder_path` hardcoded to `logs/offline_encoder/encoder.pt` in config.
  Should be parameterized for production use.

---

## [2026-03-29] Hard DR Environment for Offline Encoder Experiments

### Context

After 15+ online encoder experiments (Steps 4-19) all failed to match history-only
baseline (Step 4d, 3.0 deg), pivoted to offline encoder approach. The fundamental
issue: 2D action space makes online encoder co-training structurally unstable
(KL sensitivity, env-level clamp positive feedback on noise_std).

Key insight: Step 4d's actor already operates without privileged info (uses only
o_t + history). The encoder/adapt pipeline's purpose (deploy without privileged
info) is already achieved. However, encoder still has value for online adaptation
to dynamics outside DR range -- real robot parameters (added mass, CoG, CoB) are
estimated, not measured, so DR range uncertainty is the primary motivation.

Strategy: Create a "hard DR" environment where history-only degrades to ~10 deg
(vs 3 deg with standard DR), then show offline encoder closes the gap.

### Added
- `config.py`: `HardDomainRandomizationCfg` -- aggressive DR with wider ranges:
  added_mass +-40% (was +-15%), body_mass +-25% (was +-10%), volume +-25% (was +-10%),
  CoG/CoB offsets doubled, inertia (0.5, 1.8) (was 0.75, 1.3), payload 0-2kg (was 0-1kg)
- `config.py`: `ALBCHardDRHistOnlyEnvCfg` -- hard DR + history(30) + ocean current
  (0.5/0.5/0.25 m/s) + no encoder
- `config.py`: `ALBCHardDREncoderEnvCfg` -- hard DR + encoder + strided history(15x5)
  for future offline encoder fine-tuning
- `agents/rsl_rl_ppo_cfg.py`: `ALBCHardDRHistOnlyRunnerCfg` -- PPO runner for hard DR baseline
- `__init__.py`: Registered `Isaac-Constrained-ALBC-HardDR-HistOnly-v0`

### Notes
- DR ranges may need tuning after first run. Target: ~10 deg error (5 deg = too easy,
  20 deg = too hard). Adjust added_mass_scale and volume_scale first.
- Ocean current enabled for the first time in constrained_albc (was always 0,0,0).
- Offline encoder pipeline (data collection, supervised training, fine-tuning) is next
  step after hard DR baseline is established.

---

## [2026-03-29] Step 19: HORA-Aligned Encoder Training

### Context

Step 18 (scalar_std + no_clamp) confirmed KL fix works but revealed std explosion
(1.0 -> 18.5, same direction as Step 17's 148). Investigation into the mechanism:

**Root cause of std explosion: env-level clamp positive feedback.**
- `albc_env.py:320` clamps actions to [-1,1] before physics
- `ppo.py:182` stores UNCLAMPED actions in rollout buffer
- `ppo.py:297` computes log_prob on UNCLAMPED actions during update
- Reward comes from CLAMPED action outcome -> log_prob/advantage mismatch
- When std > ~1.5: 62%+ actions clamped -> different unclamped actions produce
  identical physical outcomes -> score function gradient loses corrective signal
  -> std drifts upward -> more clamping -> positive feedback loop
- Step 4d (no encoder): std decreases from 1.0 to 0.2 before reaching threshold
  because advantage structure is stable (no encoder z changes)
- With encoder: z changes make advantages noisy -> std fails to decrease early
  -> crosses ~1.5 threshold -> positive feedback -> explosion

**HORA code analysis confirms identical structure:**
- `hora/algo/models/models.py:97`: act() returns unclamped sample (same as us)
- `hora/algo/ppo/ppo.py:324-325`: stores unclamped actions (same mismatch)
- `hora/algo/ppo/ppo.py:327`: clamps before env.step() (same as albc_env.py:320)
- `hora/algo/ppo/ppo.py:332`: `shaped_rewards = 0.01 * rewards` (KEY DIFFERENCE)
- HORA has the same mismatch but avoids positive feedback via reward_scale=0.01:
  reduced gradient -> smaller encoder z changes -> stable advantages ->
  std decreases before reaching ~1.5 threshold

**scalar_std hypothesis disproved:** Step 18 showed scalar std explodes identically
to log_std (18.5 vs 148, same mechanism). Reverted to log_std (RL standard,
HORA default, guarantees std > 0).

### Changed
- `encoder/actor_critic_encoder.py`: Removed `log_std.clamp(-10.0, 5.0)` from
  `_update_distribution()`. Now matches base ActorCritic: `std = exp(log_std)`.
  The clamp was NOT present in HORA or rsl_rl's ActorCritic. exp(5)=148 was
  the ceiling that Step 17 hit.

### Added
- `agents/rsl_rl_ppo_cfg.py`: `_HoraAlignedPolicyCfg` (clamp_actions=False,
  log_std default), `_HoraAlignedAlgorithmCfg` (reward_scale=0.01, lr=5e-3,
  min_lr=1e-6, desired_kl=0.02), `ALBCDebugPPOHoraAlignedRunnerCfg` (Step 19)
- `__init__.py`: Registered `Isaac-Constrained-ALBC-Debug-PPO-HoraAligned-v0`

### Experimental Results

**Step 18 (scalar_std + no_clamp, 151 iters):**
KL=0.01 (fix confirmed), LR=0.01 (max, stable). But noise_std exploded to 18.5
(monotonic ^^^^). Scalar std hypothesis DISPROVED -- both parameterizations explode
with encoder. Roll/Pitch 16/14.5 deg, z_range [-1.0, 1.0] SAT. action_size=1.41
(boundary). Env-level clamp positive feedback mechanism identified (see Context).

**Step 19 (HORA-aligned, 500 iters):**

| Metric | Value | vs Step 18 | vs Step 4d (target) |
|--------|-------|-----------|---------------------|
| KL | 9.9e-4 | = (both fixed) | = |
| LR | 0.01 (max) | = | = |
| noise_std | 7.66 (oscillating) | better (not monotonic) | 0.20 (target) |
| Roll/Pitch | 16.8/14.5 | = | 3.0/3.8 |
| value_loss | 0.16 | better (0.46) | - |
| z_range | [-1.0, 1.0] SAT | = | N/A |

Key finding: noise_std OSCILLATES instead of monotonically exploding.
Periodic drops to ~0.3 (reward_scale corrective signal works!) then spikes
back to 6-11 (env-clamp positive feedback wins). Amplitude growing (6->11->8).
reward_scale=0.01 provides corrective signal but is insufficient to prevent
re-entry into the positive feedback regime (std > ~1.5).

Remaining HORA differences suspected to contribute to oscillation:
- horizon: HORA=8 vs ours=64 (fresher advantages in HORA)
- num_envs: HORA=16384 vs ours=4096 (lower variance in HORA)
- bounds_loss: HORA=0.0001 on mu vs ours=none

### Notes
- reward_scale=0.01 is HORA's exact value. May need tuning for our 2D system
  (HORA: 16D actions, horizon=8, 16384 envs vs our 2D, horizon=64, 4096 envs).
- value_loss=0.16 is healthy (Step 12 had 7.2e-4 with clamp, critic was starved).
- noise_std oscillation pattern suggests system is near stability boundary.
  Reducing horizon or increasing reward_scale correction may push it to stable.
- scalar_std hypothesis fully disproved: reverted to log_std (RL standard, HORA default).

---

## [2026-03-29] Step 18: Scalar Std + No Clamp (Combined Root Cause Fix)

### Context

Steps 16-17 individually isolated the two components of the KL spike root cause:
- Step 16 (scalar_std alone): DISPROVED -- clamp still present, 32% boundary concentration
- Step 17 (no_clamp alone): BREAKTHROUGH -- KL 100x reduced (0.88 -> 0.003), but
  noise_std exploded to 148 (log_std exp() has no upper bound without clamp regularization)

The combined fix matches rsl_rl's base ActorCritic exactly (Step 4d, 3.0 deg success):
- `noise_std_type = "scalar"`: `self.std = Parameter(1.0)`, additive gradient, naturally bounded
- `clamp_actions = False`: `act()` returns `distribution.sample()` without `.clamp(-1, 1)`

Code-level verification:
- rsl_rl ActorCritic (Step 4d): `noise_std_type="scalar"` (default), no clamp in `act()` (line 152)
- ActorCriticEncoder (Steps 4-17): `noise_std_type="log"` (default), `sample.clamp(-1,1)` (line 311)
- Environment safety: `albc_env.py:320` always applies `actions.clone().clamp(-1, 1)` regardless

### Added
- `agents/rsl_rl_ppo_cfg.py`: `_ScalarStdNoClampPolicyCfg` (noise_std_type="scalar", clamp_actions=False),
  `ALBCDebugPPOScalarStdNoClampRunnerCfg` (Step 18 runner)
- `__init__.py`: Registered `Isaac-Constrained-ALBC-Debug-PPO-ScalarStdNoClamp-v0`

### Notes
- reward_scale=0.01 intentionally NOT combined -- root cause (clamp) is removed, gradient
  reduction unnecessary. Available as contingency if Step 18 fails.
- Total hypotheses tested: 12 (10 disproved, 1 breakthrough, 1 combined fix pending)
- If successful, next step is encoder validation (z_sweep) then constraint reintegration

---

## [2026-03-29] Steps 15-17: Isolating ActorCriticEncoder Structural Root Cause

### Context

After Steps 13-14 disproved all HORA-aligned learning dynamics hypotheses, focus shifted
to code-level structural differences between ActorCritic (Step 4d, 3.0 deg success) and
ActorCriticEncoder (all encoder experiments, ~22 deg failure). Three isolation experiments:

**Step 15: Symmetric Critic (DISPROVED)**
Hypothesis: privileged critic's accurate advantages cause large gradients -> KL spike.
Changed critic from asymmetric `cat([o_t, p_t])` to symmetric `cat([o_t, hist])` (134D,
matching Step 4d's critic input).

| Step | Config | Roll/Pitch | noise_std | LR | Encoder KL iter-0 |
|------|--------|-----------|-----------|-----|-------------------|
| 8a | Q1Q3 baseline | 23.3/20.0 | 0.96 | 1.5e-5 | 0.88 |
| 15 | Symmetric critic | 22.9/21.0 | 0.97 | 1.8e-5 | 0.65 |

Result: DISPROVED. Identical failure pattern. Critic asymmetry is not the cause.

**Step 16: Scalar noise_std (DISPROVED)**
Hypothesis: log_std parameterization (exp-based, multiplicative gradient) prevents
noise_std from decreasing. Changed to scalar std (additive gradient, matching ActorCritic).

| Step | Config | Roll/Pitch | noise_std | LR | Encoder KL iter-0 |
|------|--------|-----------|-----------|-----|-------------------|
| 8a | Q1Q3 baseline (log_std) | 23.3/20.0 | 0.96 | 1.5e-5 | 0.88 |
| 16 | Scalar std | 22.9/18.6 | 0.98 | 2.6e-5 | 0.88 |

Result: DISPROVED. Identical failure. noise_std parameterization is not the cause alone.

**Step 17: No Action Clamp -> BREAKTHROUGH**
Hypothesis: `sample().clamp(-1,1)` concentrates ~32% of actions at boundaries when
noise_std=1.0, creating sharp log_prob gradients that amplify KL.

| Step | Config | Roll/Pitch | noise_std | LR | Encoder KL iter-0 | KL |
|------|--------|-----------|-----------|-----|-------------------|-----|
| 8a | Q1Q3 (with clamp) | 23.3/20.0 | 0.96 CEIL | 1.5e-5 | 0.88 | 0.03 |
| **17** | **No clamp** | **10.7/9.6** | **148 EXPLODED** | **0.01 MAX** | **0.003** | **0.01** |

Result: **ACTION CLAMP CONFIRMED AS ROOT CAUSE OF KL SPIKE.**
- First time EVER that encoder KL stayed below 0.015 (100x reduction from 0.88)
- LR stable at max (0.01) throughout training -- no crash
- Roll/Pitch actively decreasing (30->10 deg trend `vvv\`)
- BUT: noise_std exploded to 148 (exp(5.0) = log_std upper clamp) because unconstrained
  log_std grows without bound when actions are not clamped

**Root cause mechanism confirmed:**
With clamp: actions pile up at [-1,1] boundaries -> small mu shift causes large log_prob
change at boundaries -> amplified surrogate gradient -> KL spike -> LR crash -> policy frozen.
Without clamp: actions spread naturally per Normal distribution -> smooth log_prob surface ->
KL stays in desired range -> LR healthy -> policy learns.

**Next step:** Combine scalar std (Step 16, prevents std explosion) + no clamp (Step 17,
prevents KL spike). This is exactly what ActorCritic (Step 4d) uses successfully.

### Added
- `encoder/actor_critic_encoder.py`: `noise_std_type` parameter ("log" default, "scalar" option). Scalar mode uses `self.std = Parameter(init_noise_std)` with additive gradient (matching ActorCritic). Log mode keeps existing `self.log_std` behavior.
- `encoder/actor_critic_encoder.py`: `clamp_actions` parameter (default True). When False, `act()` returns raw `sample()` without `.clamp(-1, 1)`, matching ActorCritic behavior.
- `encoder/actor_critic_encoder.py`: `symmetric_critic` parameter (default False). When True, critic uses `cat([o_t, hist])` instead of `cat([o_t, p_t])`.
- `agents/rsl_rl_ppo_cfg.py`: `_SymCriticPolicyCfg` (symmetric_critic=True), `ALBCDebugPPOSymCriticRunnerCfg` (Step 15)
- `agents/rsl_rl_ppo_cfg.py`: `_ScalarStdPolicyCfg` (noise_std_type="scalar"), `ALBCDebugPPOScalarStdRunnerCfg` (Step 16)
- `agents/rsl_rl_ppo_cfg.py`: `_NoClampPolicyCfg` (clamp_actions=False), `ALBCDebugPPONoClampRunnerCfg` (Step 17)
- `__init__.py`: Registered `Isaac-Constrained-ALBC-Debug-PPO-SymCritic-v0` (Step 15), `Isaac-Constrained-ALBC-Debug-PPO-ScalarStd-v0` (Step 16), `Isaac-Constrained-ALBC-Debug-PPO-NoClamp-v0` (Step 17)

### Fixed
- `encoder/actor_critic_encoder.py`: `z_bounds_loss()` device reference fixed for scalar std mode (was hardcoded to `self.log_std.device`, now conditionally uses `self.std.device`).

### Notes
- Steps 15 and 16 individually are DISPROVED as root causes
- Step 17 proves action clamp is the KL spike root cause but creates std explosion
- The combination scalar_std + no_clamp (= ActorCritic's exact config) is the next experiment
- Total hypotheses tested and disproved: 10 (EmpNorm, enc gradient, enc freeze, init LR, update path, history, critic asymmetry, normalization method, noise_std type, action clamp isolated)
- Action clamp is confirmed as root cause but needs scalar std to prevent explosion

---

## [2026-03-29] Steps 13-14: Static MinMax Norm (DISPROVED) + Encoder Freeze (DISPROVED)

### Context

Three experiments to test remaining HORA-aligned hypotheses for encoder KL spike:

**Step 13: Static min-max normalization (HORA-style)**
Hypothesis: EmpiricalNormalization's running stats update every env step causes z drift,
which is the hidden KL source (independent of encoder gradient). Replaced with HORA's
deterministic formula: `(2*x - upper - lower) / (upper - lower)` -> [-1, 1].

| Step | Config | Roll/Pitch | noise_std | LR | z_range |
|------|--------|-----------|-----------|-----|---------|
| 8a (baseline) | Q1Q3 + EmpiricalNorm | 23.3/20.0 | 0.96 | 1.5e-5 | [-0.86,0.85] |
| 13b | Q1Q3 + StaticNorm | 22.5/20.9 | 0.98 | 2.6e-5 | [-0.93,0.92] |
| 13a | Q1Q3 + StaticNorm + RS=0.01 | 22.0/30.6 | 0.92 | 4.0e-5 | [-0.93,0.91] |
| 12a (prev session) | Q1Q3 + EmpNorm + RS=0.01 | 17.9/43.8 | 0.92 | 4.0e-5 | [-0.96,0.95] |

Result: **DISPROVED**. Static norm vs EmpiricalNorm -> essentially identical results.
Improvement in 13a came entirely from reward_scale, not normalization method.

**Step 14: Encoder freeze (encoder_grad_scale=0.0)**
Critical validation: if encoder weight changes cause KL spike, then freezing encoder
(zero gradient) should restore normal PPO learning. Actor should learn using o_t + history,
treating z as fixed random noise.

| Step | Config | Roll/Pitch | noise_std | LR | z_range |
|------|--------|-----------|-----------|-----|---------|
| 4d (no encoder) | PPO + History 30x1 | 3.0/3.8 | 0.20 | 1e-2 | N/A |
| **14 (freeze)** | **PPO + Frozen encoder** | **21.5/19.5** | **0.97** | **1.8e-5** | [-0.20,0.17] |

Result: **DISPROVED**. Even with encoder completely frozen (z fixed, z_std=0.08),
PPO still fails identically: LR crashes to 1.8e-5, noise_std at 0.97 CEILING.
Encoder weight changes are NOT the root cause of KL spike.

**Implications:**
All HORA-aligned hypotheses have now been tested and either disproved or shown insufficient:
- EmpiricalNorm z drift: DISPROVED (Step 13)
- Encoder gradient magnitude: DISPROVED (Step 10, encoder_grad_scale=0.1)
- Encoder weight changes: DISPROVED (Step 14, encoder_grad_scale=0.0)
- reward_scale=0.01: partial improvement only (noise_std 0.92, still ~20 deg)
- Update path: DISPROVED (Step 11)
- History/normalization configs: insufficient (Steps 8-9)

The failure occurs whenever ActorCriticEncoder class is used, regardless of whether
the encoder is learning, frozen, or how observations are normalized. Root cause is
structural/architectural, not learning-dynamics. Remaining suspects:
1. Policy class (ActorCriticEncoder vs ActorCritic)
2. `_update_encoder_ppo()` path with frozen encoder (untested combination)
3. Interaction between obs_groups/TensorDict processing and PPO update

### Added
- `encoder/actor_critic_encoder.py`: Static min-max normalization via `encoder_obs_lower`/`encoder_obs_upper` params. When provided, uses deterministic `(2*x - upper - lower) / (upper - lower)` instead of EmpiricalNormalization. Registered as buffers for device placement.
- `agents/rsl_rl_ppo_cfg.py`: 23D privileged obs bounds (`_PRIV_OBS_LOWER`, `_PRIV_OBS_UPPER`) derived from DomainRandomizationCfg + HydrodynamicsCfg with 10% margin
- `agents/rsl_rl_ppo_cfg.py`: `_StaticNormPolicyCfg` -- Q1Q3 encoder policy with static min-max normalization
- `agents/rsl_rl_ppo_cfg.py`: `ALBCDebugPPOStaticNormRunnerCfg` (Step 13b), `ALBCDebugPPOStaticNormRSRunnerCfg` (Step 13a)
- `agents/rsl_rl_ppo_cfg.py`: `_EncFreezeAlgorithmCfg` with `encoder_grad_scale=0.0`, `ALBCDebugPPOEncFreezeRunnerCfg` (Step 14)
- `__init__.py`: Registered `Isaac-Constrained-ALBC-Debug-PPO-StaticNorm-v0`, `Isaac-Constrained-ALBC-Debug-PPO-StaticNorm-RS-v0`, `Isaac-Constrained-ALBC-Debug-PPO-EncFreeze-v0`

### Notes
- All HORA elements that can be replicated in 2D action space have been tested
- Critic asymmetry is confirmed as correct design (not a suspect)
- History size variation ruled out (all encoder experiments fail regardless of history config)
- Next investigation: isolate whether the `_update_encoder_ppo()` code path itself (vs standard `update()`) is the structural cause, tested with frozen encoder

---

## [2026-03-29] Step 11: Standard Update Path (DISPROVED) + Step 12: HORA Reward Scaling

### Context

Two experiments to diagnose encoder training failure root cause:

**Step 11 (standard update path):** Hypothesis that `_update_encoder_ppo()` with per-epoch
LR adaptation was causing failure. Bypassed custom encoder update path, routing encoder
through standard `update()` (per-minibatch LR, no mu/sigma refresh). Result: **DISPROVED**.
Standard path performed WORSE:

| Step | Update Path | Roll/Pitch | noise_std | LR | z_range | KL |
|------|------------|-----------|-----------|-----|---------|-----|
| 8a (baseline) | `_update_encoder_ppo()` | 23/20 | 0.96 CEIL | 1.5e-5 | [-0.86,0.85] | 0.03 |
| 11a (enc norm) | `update()` standard | 23/18 | 0.98 CEIL | **1.0e-5** | [-0.95,0.95] | 0.08 |
| 11a (no enc) | `update()` standard | 15/**49** | 0.97 CEIL | **1.0e-5** | [-1.00,1.00] SAT | 0.13 |

Key finding: per-minibatch mu/sigma refresh in `_update_encoder_ppo()` was actually HELPING
by keeping reported KL lower (0.03 vs 0.08-0.13). Without refresh, cumulative KL from encoder
z-drift is fully reflected, causing harder LR crash (1.0e-5 vs 1.5e-5).

**Step 12 (HORA reward_scale=0.01):** Applied HORA's core design element: multiply all rewards
by 0.01 before computing returns. Reduces surrogate gradient 100x, which should keep KL low
and LR high even with encoder.

| Step | Config | Roll/Pitch | noise_std | LR | z_range | KL |
|------|--------|-----------|-----------|-----|---------|-----|
| 8a (baseline) | Q1Q3 | 23/20 | 0.96 CEIL | 1.5e-5 | [-0.86,0.85] | 0.03 |
| 12a (enc norm) | reward_scale=0.01 | **17.9**/43.8 | **0.92 ↓** | **4.0e-5** | [-0.96,0.95] | 0.04 |
| 12b (no enc) | reward_scale=0.01 | 20.5/29.8 | **0.93 ↓** | **5.1e-5** | [-0.99,0.99] | 0.05 |

**First positive signals across all encoder experiments:**
- LR: 2.7-3.4x higher than baseline (4-5e-5 vs 1.5e-5)
- noise_std: declining for the first time (0.92-0.93, trend v\\\), all previous CEILING
- Roll improved in 12a (17.9 vs 23)

**Remaining issues:**
- Pitch degradation in 12a (43.8 deg, increasing trend) -- arm at extreme position (jnt_pos=5.38)
- z_range wider than baseline (0.95 vs 0.85)
- Encoder KL spike delayed but larger (~20 at iter 2-3 vs 0.88 at iter 1 in baseline)
- value_loss extremely low (7.2e-4) due to scaled rewards -- critic may not learn effectively

### Added
- `agents/rsl_rl_ppo_cfg.py`: `_StdUpdateAlgorithmCfg` (use_encoder_update=False),
  `ALBCDebugPPOStdUpdateRunnerCfg` (Step 11a), `ALBCDebugPPOStdUpdateNoEncNormRunnerCfg`.
- `agents/rsl_rl_ppo_cfg.py`: `_RewardScaleAlgorithmCfg` (reward_scale=0.01),
  `ALBCDebugPPORewardScaleRunnerCfg` (Step 12a), `ALBCDebugPPORewardScaleNoEncNormRunnerCfg`.
- `__init__.py`: Registered 4 new tasks: `Isaac-Constrained-ALBC-Debug-PPO-StdUpdate-v0`,
  `Isaac-Constrained-ALBC-Debug-PPO-StdUpdate-NoEncNorm-v0`,
  `Isaac-Constrained-ALBC-Debug-PPO-RewardScale-v0`,
  `Isaac-Constrained-ALBC-Debug-PPO-RewardScale-NoEncNorm-v0`.

### Changed
- `rsl_rl/algorithms/ppo.py` (external dep, not git-tracked): Added `use_encoder_update`
  (default True) and `reward_scale` (default 1.0) parameters. `use_encoder_update=False`
  bypasses `_update_encoder_ppo()`, routing encoder through standard `update()`.
  `reward_scale` applied in `process_env_step()` before storing rewards (HORA line 332).

### Notes
- Step 11 conclusively proved `_update_encoder_ppo()` is NOT the bug. Its mu/sigma refresh
  is a net positive. The custom path should be kept.
- reward_scale=0.01 is the most promising direction so far. Only 120 iters run.
- Pitch degradation in 12a may be transient (noise_std still declining, policy learning).
- All no-enc-norm variants continue to show z saturation -- encoder_obs_normalization
  is definitively required for 23D privileged input.
- ppo.py changes (use_encoder_update, reward_scale) need reapply on container rebuild.

---

## [2026-03-29] Q1+Q3 Encoder Fix: HORA-style Normalization + Strided Proprio History

### Context

Systematic analysis of why encoder destabilizes PPO training identified two structural
issues: (1) `actor_obs_normalizer` applies EmpiricalNorm to `cat([o_t, z])`, normalizing
the already-bounded softsign output z with non-stationary running stats, and (2) z/actor_input
ratio of 48.1% (13D z / 27D total) causes excessive mu shift per encoder update.

HORA reference comparison revealed: HORA normalizes only policy obs, passes z raw (no
double normalization), and has z/input ratio of 1.4-7.7%. HORA also does NOT normalize
privileged info before encoder (raw p_t to MLP).

Two experiments run (Steps 8a, 8b), both with 4096 envs:

| Step | Config | Roll/Pitch | noise_std | z_range | LR | KL iter-0 |
|------|--------|-----------|-----------|---------|-----|-----------|
| 8a | Q1+Q3 (enc norm kept) | 23/20 deg | 0.96 CEIL | [-0.86, 0.85] | 1.5e-5 | ~0.8 |
| 8b | Q1+Q3 (enc norm removed) | 26/32 deg | 0.98 CEIL | [-1.00, 1.00] SAT | 4.0e-5 | ~0.8 |

Both experiments show same failure pattern as Steps 4-7: iter-0 KL spike (~0.8) crashes
adaptive LR, noise_std stays at ceiling, policy effectively random. The Q1 normalize fix
did prevent z saturation in 8a (z_range [-0.86, 0.85] vs previous [-0.99, 0.99]), but
KL spike magnitude unchanged. Removing p_t normalization (8b) caused z saturation,
confirming encoder_obs_normalizer is necessary for 23D privileged input.

Key finding: Q1 and Q3 alone are insufficient. The iter-0 KL spike is caused by the
encoder's first gradient step magnitude, not by normalization or z/input ratio.
Next steps: Q2 (critic gradient to encoder) and Q4 (KL management).

### Added
- `config.py`: `proprio_history_stride` field on `ALBCEnvCfg` (default 1). Controls
  stride for proprioceptive history recording. stride=N records every N-th control step.
- `config.py`: `ALBCDebugEncoderHistStrideEnvCfg` -- Step 8a env config with
  `proprio_history_len=15`, `proprio_history_stride=5` (10Hz sampling, 1.5s window).
- `agents/rsl_rl_ppo_cfg.py`: `_Q1Q3EncoderPolicyCfg` (proprio_hist_dim=120,
  z_bounds_coef=0.0), `_Q1Q3AlgorithmCfg` (PPO, entropy_coef=0.0, desired_kl=0.02),
  `ALBCDebugPPOQ1Q3RunnerCfg` (Step 8a runner).
- `agents/rsl_rl_ppo_cfg.py`: Phase 1b configs: `_Q1Q3NoEncNormPolicyCfg`
  (encoder_obs_normalization=False), `ALBCDebugPPOQ1Q3NoEncNormRunnerCfg` (Step 8b).
- `__init__.py`: Registered `Isaac-Constrained-ALBC-Debug-PPO-Q1Q3-v0` (Step 8a)
  and `Isaac-Constrained-ALBC-Debug-PPO-Q1Q3-NoEncNorm-v0` (Step 8b).

### Changed
- `encoder/actor_critic_encoder.py`: HORA-style normalization -- `actor_obs_normalizer`
  now covers only `o_t + hist` dimensions (excludes z). Normalization moved inside
  `_get_actor_obs()`: normalizes obs part, then concatenates raw z. Previously normalized
  full `cat([o_t, z])` including the bounded softsign output.
  - `__init__`: `EmpiricalNormalization(num_actor_obs)` -> `EmpiricalNormalization(num_actor_obs_norm)`
    where `num_actor_obs_norm = policy_obs_dim + proprio_hist_dim` (z excluded).
  - `_get_actor_obs()`: builds `obs_part = cat([o_t, hist])`, normalizes it, then `cat([obs_normed, z])`.
  - `act()`, `act_inference()`, `evaluate()`: removed external `actor_obs_normalizer()` call.
  - `update_normalization()`: updates normalizer on `o_t + hist` only (not z).
  - `load_state_dict()`: added migration logic for normalizer dimension change (old->new reset).
- `albc_env.py`: Strided proprioceptive history recording. Added `_proprio_step_counter`
  (per-env torch.long). `_update_proprio_hist()` now increments counter and only records
  on stride boundary (`counter % stride == 0`). Counter reset on episode reset.

### Notes
- Step 8a confirmed: removing z from EmpiricalNorm prevents z saturation (z_range
  [-0.86, 0.85] vs previous saturated runs). This fix is sound and should be kept.
- Step 8b confirmed: encoder_obs_normalization is necessary for 23D p_t (removing it
  causes z saturation to [-1.00, 1.00]). Unlike HORA's 9D p_t, 23D benefits from normalization.
- Next experiments: Q2 (critic gradient to encoder via value loss path) and Q4
  (KL management: desired_kl, min_lr, init_lr adjustments).

---

## [2026-03-29] Q4 KL Management: HORA-style LR Range (FAILED)

### Context

Hypothesis: the iter-0 KL spike that crashes adaptive LR could be survived with HORA-style
LR range (init_lr=5e-3, min_lr=1e-6, giving 22 halvings vs ALBC's 9). Detailed iter-by-iter
analysis of Step 8a revealed the failure mechanism:

Step 8a (init_lr=3e-4) timeline:
- iter 0: encoder untrained -> KL artificially low (0.008) -> adaptive LR doubles to 1.5e-3
- iter 1: encoder first gradient -> z_std triples (0.15->0.48) -> KL=0.894 -> LR crashes

Q4 hypothesis: with init_lr=5e-3, post-spike LR should settle at ~3e-4 (viable), not 2.6e-5.

**Result: FAILED.** Higher init_lr made iter-0 KL WORSE (7.13 vs 0.008) because the first
gradient step itself was 17x larger. The encoder gets its gradient at iter 0 (not iter 1 as
previously thought). Both Q1Q3 and Q4 converge to the same equilibrium LR (~2.6e-5).

Two experiments run (Steps 9a, 9b), both with 4096 envs, 500 max iters:

| Step | Config | Roll/Pitch | noise_std | z_range | Eq. LR | KL iter-0 |
|------|--------|-----------|-----------|---------|--------|-----------|
| 8a | Q1Q3 (lr=3e-4) | 23/20 deg | 0.96 | [-0.86, 0.85] | 1.7e-5 | 0.008 |
| 9a | Q4 (lr=5e-3, enc norm) | 31/27 deg | 0.97 | [-0.91, 0.92] | 2.4e-5 | **7.13** |
| 9b | Q4 (lr=5e-3, no enc norm) | 26/26 deg | 0.96 | [-1.00, 1.00] SAT | 2.6e-5 | 5.77 |

Key finding: **equilibrium LR is determined by network dynamics, not init_lr or min_lr.**
The encoder gradient creates a fixed KL/LR relationship: at any LR where KL > desired_kl,
LR halves until reaching ~2.6e-5 where KL stabilizes near 0.03. This equilibrium is
independent of starting conditions.

Surrogate loss exploded to 11M-24M at iter 1-2 (policy ratios extreme due to massive
policy shift), normalizing only by iter 10+.

### Added
- `agents/rsl_rl_ppo_cfg.py`: `_Q4AlgorithmCfg` (learning_rate=5e-3, min_lr=1e-6, max_lr=1e-2),
  `ALBCDebugPPOQ4RunnerCfg` (Step 9a), `_Q4NoEncNormPolicyCfg` + `ALBCDebugPPOQ4NoEncNormRunnerCfg`
  (Step 9b).
- `__init__.py`: Registered `Isaac-Constrained-ALBC-Debug-PPO-Q4-v0` (Step 9a) and
  `Isaac-Constrained-ALBC-Debug-PPO-Q4-NoEncNorm-v0` (Step 9b).

### Changed
- `rsl_rl/algorithms/ppo.py` (external dep, not git-tracked): Added `min_lr` (default 1e-5) and
  `max_lr` (default 1e-2) constructor parameters. Replaced hardcoded `max(1e-5, ...)` and
  `min(1e-2, ...)` in both standard and encoder PPO update paths. Backwards compatible (defaults
  match original values). Needs reapply on container rebuild.

### Notes
- Q4 (LR range adjustment) alone cannot solve the encoder KL problem. The equilibrium
  LR is network-determined, not hyperparameter-determined.
- Step 9b confirms again: enc norm removal -> z saturation (consistent with 8b).
- ppo.py min_lr/max_lr change is still useful for future experiments but insufficient alone.
- Remaining approaches: encoder gradient scaling/clipping, separate encoder LR, delayed
  encoder start, or init_noise_std increase (KL proportional to 1/sigma^2).

---

## [2026-03-29] Encoder Gradient Scaling (FAILED - Root Cause Revision)

### Context

Hypothesis: encoder gradient dominates KL, causing low equilibrium LR (~2.6e-5).
Scaling encoder gradient by 0.1x should reduce encoder KL contribution by 100x (scale^2),
raising equilibrium LR toward the encoder-free level.

**Result: FAILED.** Results identical to Step 8a baseline to 5 significant figures.
encoder_grad_scale=0.1 verified: config saved correctly, code loaded correctly, gradient
scaling mechanics validated in isolation (encoder grads scaled exactly 0.1x, actor grads
untouched). Yet iter-by-iter KL, LR, z_std, and noise_std trajectories are byte-for-byte
identical between Step 10a (scaled) and Step 8a (unscaled).

| Step | Config | Roll/Pitch | noise_std | z_range | Eq. LR | KL iter-0/1 |
|------|--------|-----------|-----------|---------|--------|-------------|
| 8a | Q1Q3 baseline | 23/20 | 0.96 | [-0.86,0.85] | 1.7e-5 | 0.008/0.89 |
| 10a | EncScale 0.1x, enc norm | 22/19 | 0.97 | [-0.86,0.85] | 1.8e-5 | 0.008/0.88 |
| 10b | EncScale 0.1x, no enc norm | 23/32 | 0.97 | [-1.00,1.00] SAT | 2.6e-5 | - |

Checkpoint comparison at iter 50: encoder weights differ by only 1-2% between 8a and 10a,
confirming gradient scaling IS applied but encoder weight changes are negligible relative
to initial weights (20 gradient steps with effective LR ~1e-4 change weights by ~0.1%).

**Critical root cause revision:** The KL spike is NOT from encoder gradient magnitude.
Comparison with Step 4d (encoder-free PPO+History, SUCCEEDS at 3.3 deg) reveals the real
difference is the **PPO update path**, not the encoder:

| Factor | Step 4d (success) | Steps 8a/10a (fail) |
|--------|------------------|---------------------|
| Update path | `update()` standard | `_update_encoder_ppo()` |
| LR adaptation | per-minibatch (20x/iter) | per-epoch (5x/iter) |
| mu/sigma refresh | none (old from rollout) | per-minibatch (rolling) |
| iter-1 KL | 0.014 | 0.89 |
| iter-1 LR | 6.7e-3 (healthy) | 2.0e-4 (crashing) |

The `_update_encoder_ppo()` path uses per-EPOCH LR adaptation (5 adjustments per iteration),
which is 4x slower at reacting to KL spikes than standard per-MINIBATCH adaptation (20
adjustments). Within an epoch, 4 high-KL gradient steps proceed unchecked before LR halves.
The per-minibatch mu/sigma refresh compounds this: each minibatch measures fresh KL that
may be individually acceptable, but cumulative weight drift across the epoch is uncontrolled.

### Added
- `agents/rsl_rl_ppo_cfg.py`: `_EncScaleAlgorithmCfg` (encoder_grad_scale=0.1, lr=3e-4,
  min_lr=1e-6), `ALBCDebugPPOEncScaleRunnerCfg` (Step 10a),
  `_EncScaleNoEncNormPolicyCfg` + `ALBCDebugPPOEncScaleNoEncNormRunnerCfg` (Step 10b).
- `__init__.py`: Registered `Isaac-Constrained-ALBC-Debug-PPO-EncScale-v0` (Step 10a)
  and `Isaac-Constrained-ALBC-Debug-PPO-EncScale-NoEncNorm-v0` (Step 10b).

### Changed
- `rsl_rl/algorithms/ppo.py` (external dep, not git-tracked): Added `encoder_grad_scale`
  parameter (default 1.0) to `__init__`. In `_update_encoder_ppo()`, after `loss.backward()`,
  scales encoder-named parameter gradients by `encoder_grad_scale` before `clip_grad_norm_`
  and `optimizer.step()`. Needs reapply on container rebuild.

### Notes
- encoder_grad_scale is a valid mechanism (verified) but ineffective because encoder
  gradient is NOT the KL spike cause.
- The real bottleneck is the per-epoch LR adaptation in `_update_encoder_ppo()`.
- Step 10b confirms again: enc norm removal -> z saturation (consistent across 8b/9b/10b).
- Next approach: investigate per-minibatch vs per-epoch LR adaptation difference, or
  run encoder config through standard `update()` path instead of `_update_encoder_ppo()`.

---

## [2026-03-27] Encoder Ablation Study (Steps 0-7)

### Summary

Systematic ablation to isolate why full constrained ALBC (TRPO+IPO+Encoder+DR) stagnates at
17-27 deg attitude error. Components added incrementally: PPO (0.7 deg) -> +DR (3.7 deg)
-> +TRPO (5.1 deg) -> +Barrier (6.3 deg) -> **+Encoder (45 deg, DIVERGED)**. 14 encoder
experiments across TRPO, PPO, shared/separate backbone, large/small encoder, and with/without
history all failed with the same pattern: encoder update at iter 1 creates ~0.14 KL (7x
desired_kl), crashing adaptive LR. History-only PPO (no encoder, 254D input) converges to
3.3 deg, confirming encoder integration as the sole problem.

### Steps 0-3: Baseline Components

| Step | Config | Roll | Pitch | Iters | Verdict |
|------|--------|------|-------|-------|---------|
| 0 | Pure PPO (no DR/encoder/constraints) | 0.6 | 0.7 | 75 | PASS |
| 1 | PPO + DR | 3.9 | 3.7 | 66 | PASS |
| 2 | TRPO + DR | 5.4 | 5.1 | 83 | PASS (slower) |
| 3 | TRPO + DR + Barrier (4 constraints) | 8.4 | 6.3 | 162 | PASS (tighter=slower) |

Barrier works correctly: 0 spikes, all margins positive. Constraint budgets tightened from
ablation data (torque 0.20->0.08, velocity 0.10->0.02, yaw_vel 0.785->0.40).
Nominal position (0,pi)->(0,pi/2) tested: no difference (asymmetry from encoder, not kinematics).

### Step 4: TRPO+Encoder (FAIL -- Pitch Diverges)

Roll 16.4, pitch 45.2 deg (diverged in 54 iters). Encoder and actor share TRPO KL budget
(max_kl=0.005). Fisher info ~0 for encoder params + CG damping=0.1 amplifies encoder gradient
10x, consuming KL budget and leaving actor unable to improve.

### Step 4b: PPO+Encoder (FAIL -- Different Mechanism)

| Metric | Step 4 (TRPO+Enc) | Step 4b (PPO+Enc) |
|--------|-------------------|--------------------|
| Roll / Pitch | 14.7 / 46.2 deg | 32.5 / 26.3 deg |
| z_std | 0.265 | 0.975 (saturated) |
| LR | N/A (TRPO) | 1e-5 (crashed) |
| Failure mode | Fisher amplification | z saturation -> KL -> LR death |

PPO: 20 steps/iter (5 epochs x 4 minibatches) cause z_std 0.17->0.63 in 10 iters, KL to 0.04
(4x desired), LR crashes to 1e-5.

**z/actor_input ratio -- root cause of sensitivity:**

| | HORA | ALBC |
|--|------|------|
| Base obs / z / Actor input | 96D / 8D / 104D | 14D / 13D / 27D |
| z ratio | 7.7% | 48.1% |

Solution: add proprio history (30x8D=240D) -> z ratio 48.1% -> 4.9%.

### Step 4c: PPO+Encoder+History -- 6 Ablations (All Failed)

**HORA vs ALBC key differences:**

| Parameter | HORA | ALBC |
|-----------|------|------|
| entropy_coef | 0.0 | 0.01 |
| init_lr / min_lr | 5e-3 / 1e-6 | 3e-4 / 1e-5 |
| kl_threshold | 0.02 | 0.01 |
| horizon | 8 | 64 |
| normalize_value | yes | no |
| reward_scale | 0.01x | 1x |

HORA's init_lr=5e-3 allows 21 consecutive LR decreases before min_lr; ALBC's 3e-4 dies after 9.

**Single-variable ablation (all 267D actor input):**

| Exp | Changed | LR death | Roll | Pitch | Observation |
|-----|---------|:--------:|-----:|------:|-------------|
| baseline | (none) | YES | 41.5 | 32.5 | noise_std 0.97 ceiling |
| 4c-1 | entropy_coef=0.0 | YES | 29.8 | 37.2 | noise_std downtrend, LR=5.1e-5 |
| 4c-2 | ent=0+lr=5e-3 | YES | 16.2 | 47.1 | roll improved, pitch worsened |
| 4c-3 | desired_kl=0.02 | YES | 15.9 | 40.3 | Best reward, z SAT returned |
| 4c-4 | steps_per_env=8 | YES | 13.3 | 53.0 | Anti-phase oscillation, NaN |
| 4c-5 | normalize_value | YES | 24.3 | 22.0 | Best balanced (both improved) |
| 4c-6 | fixed schedule | N/A | NaN | NaN | Diverged -- adaptive LR was safety net |

All noise_std > 0.94 (policy effectively random).

### Step 4d: History-Only PPO -- No Encoder (SUCCESS)

Actor: policy(14D) + history(240D) = 254D. Standard ActorCritic, no encoder.

| Metric | ent=0.01 | ent=0.0 |
|--------|----------|---------|
| Roll / Pitch | 3.57 / 3.27 | 3.03 / 3.83 |
| reward | -6.71 | -5.57 |
| noise_std | 0.81 (rising) | 0.20 (falling) |

entropy_coef=0.0 (matching HORA) resolved sigma plateau. 254D input works fine.

### Steps 5-7: Architecture Experiments (All Failed)

**Step 5a-5b: Shared backbone (6 variants)**

Consistent pattern: iter 0 KL ~0.02 -> iter 1 KL 0.3-1.5 -> LR crashes -> pitch diverges.

| Variant | Key change | KL iter 1 | Result |
|---------|-----------|-----------|--------|
| 5a-v1 | 2-group opt, lr=1e-3 | 0.318 | LR death |
| 5a-v2 | single group | 0.517 | NaN (surr 5.9e22) |
| 5a-v3 | +log_ratio clamp | 0.835 | LR death |
| 5a-v4 | +per-minibatch refresh | 0.835 | LR death |
| 5b-v1 | +history(10) | 0.367 | LR death |
| 5b-v2 | +asymmetric LR | 0.367 | LR death |

Root cause: value loss shifts backbone features -> mu shifts -> unbounded KL not bounded by
surrogate advantage. At 2D actions, KL concentrates on 2 dims (HORA's 16D disperses it).

**Step 6: Separate network + per-minibatch refresh + combined hyperparams**

iter 1: KL=0.139 (7x desired), LR crashes to 5.9e-5. Pitch 19->48 deg.
Per-minibatch refresh reduced iter-1 KL from shared backbone's 0.835 to 0.139 (6x), still
insufficient.

**Step 7: Small encoder [256,128]->8D (15% of policy, matching HORA fraction)**

iter 1: KL=0.144, nearly identical to Step 6's 0.139. Encoder SIZE is not the differentiator.

### All 14 Experiments Summary

| Step | Architecture | Encoder | KL iter1 | Outcome |
|------|-------------|---------|----------|---------|
| 0 | PPO | none | - | 0.7 deg (PASS) |
| 1 | PPO+DR | none | - | 3.7 deg (PASS) |
| 2 | TRPO+DR | none | - | 5.1 deg (PASS) |
| 3 | TRPO+DR+Barrier | none | - | 6.3 deg (PASS) |
| 4 | TRPO+DR+Enc | [256,128,64]->13 | N/A | 45 deg (FAIL) |
| 4b | PPO+Enc | [256,128,64]->13 | high | z sat + LR death |
| 4c (x6) | PPO+Enc+Hist | [256,128,64]->13 | high | 6 ablations all failed |
| 4d | PPO+Hist (no enc) | none | - | 3.3 deg (PASS) |
| 5a (x6) | PPO+Enc shared BB | various | 0.3-1.5 | shared BB amplifies KL |
| 6 | PPO+Enc+Hist separate | [256,128,64]->13 | 0.139 | LR death |
| 7 | PPO+Enc+Hist separate | [256,128]->8 | 0.144 | LR death |

**Invariant finding**: Encoder update at iter 1 creates KL ~0.14 (7x desired_kl=0.02)
regardless of encoder size, architecture, or optimizer configuration.

### Unresolved Directions

- (a) Cosine-decaying encoder LR (starts high, decays to near-zero)
- (b) Freeze encoder for N iterations, let actor converge, then unfreeze
- (c) Encoder inside actor MLP as conditional input (not concatenated)
- (d) Abandon online encoder; use offline system identification

### Added

- `encoder/actor_critic_encoder.py`: `shared_backbone` mode (backbone MLP + linear heads),
  `z_bounds_loss()` method (soft quadratic penalty on |z| > 0.85)
- `encoder/actor_critic_constrained.py`: ActorCritic + cost critic wrapper (no encoder)
  for barrier-only ablation
- `algorithms/ppo.py`: `_update_encoder_ppo()` with per-minibatch mu/sigma refresh,
  per-epoch LR adaptation. Single optimizer group. Log-ratio clamp(-20, 20).
- `config.py`: `proprio_history_len` (default 0), `proprio_feature_dim` (8);
  debug env configs: `ALBCDebugEnvCfg`, `ALBCDebugDREnvCfg`, `ALBCDebugBarrierEnvCfg`,
  `ALBCDebugEncoderEnvCfg`, `ALBCDebugEncoderHistEnvCfg` (4c, history_len 30->10),
  `ALBCDebugHistOnlyEnvCfg` (4d, `state_space=0`)
- `albc_env.py`: `_get_proprio_features()` (8D per step), `_update_proprio_hist()` ring buffer,
  `_get_observations()` exposes `proprio_hist` as flat `(N, 240)`
- `encoder/actor_critic_encoder.py`: `proprio_hist_dim`, `_proprio_hist_key` parsing,
  `_get_actor_obs()` concatenates `cat([o_t, hist_flat, z])`.
  Added `nan_to_num` + `clamp(-10, 5)` on `log_std`.
- `agents/rsl_rl_ppo_cfg.py`: Runner/algorithm configs for Steps 4b/4c/4d/5a/5b/6/7.
  `_PPOHistOnlyAlgorithmCfg` (`entropy_coef=0.0`),
  `_PPOEncoderHistAlgorithmCfg` (4c ablation)
- `runners/constraint_encoder_runner.py`: `normalize_value` flag with Welford running mean/std
- `__init__.py`: Registered ablation tasks: `Isaac-Constrained-ALBC-Debug-v0` (0),
  `-DR-v0` (1), `-TRPO-v0` (2), `-Barrier-v0` (3), `-Encoder-v0` (4),
  `-PPO-Encoder-v0` (4b), `-PPO-Enc-Hist-v0` (4c), `-PPO-Hist-Only-v0` (4d),
  `-PPO-SB-v0` (5a), `-PPO-SB-Hist-v0` (5b), `-PPO-Sep-Enc-Hist-v0` (6)

### Changed

- `config.py`: `nominal_joint_pos` (0,pi)->(0,pi/2); constraint budgets tightened
  (torque 0.20->0.08, velocity 0.10->0.02, yaw_vel 0.785->0.40);
  reward weights (k_tau -0.01->-0.005, k_s -0.2->-0.1)
- `albc_env.py`: `_get_observations()` flattens `proprio_hist` to `(N,240)`.
  Guard `compute_all_costs()` with `num_constraints > 0`.
- `encoder/actor_critic_encoder.py`: `_get_actor_obs()` no longer flattens hist (already flat)
- `algorithms/constraint_trpo.py`: `num_constraints > 0` guards in `act()`,
  `process_env_step()`, `compute_returns()`, `_update_values()`

### Key Lessons

1. **RL fundamentally sound**: PPO solves 2-DOF in <75 iters (0.7 deg). All complexity from
   encoder integration.
2. **Encoder destabilizes any optimizer**: TRPO (Fisher amplification), PPO (z expansion ->
   KL -> LR death). Same iter-1 KL ~0.14 regardless of architecture.
3. **z/actor_input ratio**: HORA 7.7% vs ALBC 48.1%. History reduces to 4.9%, insufficient.
4. **Shared backbone incompatible with 2D actions**: value gradient -> unbounded KL via
   backbone feature shift. HORA's 16D disperses KL.
5. **HORA success non-transferable**: 16D actions, 16384 envs, reward_scale=0.01, horizon=8
   provide stability margins ALBC cannot match.
6. **Per-minibatch mu/sigma refresh**: reduces KL 6x, insufficient alone.
7. **entropy_coef=0.0 required**: positive entropy_coef pushes sigma up while LR death
   prevents pushing down. Resolved sigma plateau in Step 4d.
8. **Adaptive LR death = failure mode AND safety net**: prevents learning but also NaN.
9. **normalize_value**: only single variable improving both roll and pitch simultaneously.

---

## [2026-03-27] Action Parameterization & Reward Tuning

### Summary

Three sequential fixes addressing action jitter and constraint feasibility:
(1) Torque constraint measured PD controller's unbounded internal computation instead of actual
motor output, making it 100% violated and unsatisfiable. (2) Gaussian policy noise in absolute
joint targets created 115 deg/step jitter, causing 91% effort saturation. Switched to delta
action where noise is bounded per step. (3) Tuned delta_scale and reward weights from first
delta run analysis.

### Fix: Torque Constraint (computed_torque -> applied_torque)

`torque_limit_cost()` checked `computed_torque` (PD output, 326-554 Nm) against 9.5 Nm limit
-- 100% violated on every step, fundamentally unsatisfiable.

| Metric | computed_torque | applied_torque |
|--------|----------------|----------------|
| Range | 326-554 Nm | 12.0-12.5 Nm |
| Violation rate | ~100% | ~70-80% (improvable) |
| effort_saturation | 78-95% | - |

Impact: constant barrier gradient with no directional info, dominated reward signal (4:1),
collapsed exploration (noise_std 0.61->0.41), encoder grad_norm spikes to 19680.

#### Fixed
- `mdp/constraints.py`: `torque_limit_cost()` uses `applied_torque` instead of `computed_torque`

#### Notes
- Velocity constraint (limit=4.189 rad/s) is correct: checks actual joint_vel against motor max.
- Reward `joint_torque` already correctly used `applied_torque`.

### Switch: Absolute -> Delta Action Parameterization

With `action_scale=pi` and `noise_std=0.64`, per-step target jump = 0.64*pi = 2.0 rad = 115 deg.
PD (Kp=100) needs position error < 0.095 rad (5.4 deg) for torque < 9.5 Nm. Even at min_std=0.2,
noise = 0.2*pi = 36 deg -- 7x constraint-feasible range.

Reference: TDC achieves 0.2-6 deg using small incremental IK deltas. NORBC uses sigma_a=0.4
(8x smaller), but absolute scaling doesn't suit continuous-rotation arm.

Delta action: limits per-step change, allows any absolute position via accumulation. At 50Hz
with delta_scale=0.05, max velocity = 2.5 rad/s (within 4.189 constraint). With min_std=0.2,
noise = 0.65 deg/step (within PD tracking range).

#### Changed
- `config.py`: `action_scale: float = pi` -> `delta_scale: float = 0.05`
- `albc_env.py`: `_apply_joint_pd_action()` from absolute (`q_des = q_nominal + scale * a_t`)
  to delta accumulation (`q_des += delta_scale * a_t`, clamped to joint limits)

#### Notes
- Smoothness reward now penalizes acceleration (change in velocity command) rather than
  change in absolute position -- more physically meaningful with delta actions.
- delta_scale=0.10 rejected: PD torque = 10 Nm exceeds 9.5 limit.

### Tune: delta_scale and Reward Weights

First delta run (`2026-03-27_02-40-36`, 139 iters) -- dynamics success, attitude regression:

| Category | Metric | Absolute | Delta |
|----------|--------|----------|-------|
| Dynamics | effort_saturation | 91% | 2.2% |
| | applied_torque_max | 12.3 Nm | 6.5 Nm |
| | torque cost_return | 92 | 4.5 (within budget!) |
| | velocity cost_return | 91 | 0.02 |
| Attitude | Roll / Pitch | 17 / 13 deg | 21.6 / 18.8 deg |
| Reward | command:smoothness:torque | 97.3%:2.3%:0.5% | - |

Issues: delta_scale=0.05 too slow (0.62s to reach 90 deg offset), 160:1 reward imbalance.

#### Changed
- `config.py`: `delta_scale` 0.05 -> 0.08 (bandwidth +60%, 0.39s to 90 deg, PD torque 8.0 Nm
  within 9.5 limit)
- `config.py`: `k_tau` -0.001 -> -0.01 (10x), `k_s` -0.05 -> -0.2 (4x).
  Target ratio: command ~85%, smoothness ~10%, torque ~5%.

### Key Lessons

1. **Constraint must measure actual output**: computed_torque (PD internal) is unbounded;
   applied_torque (post-clamp) is the physical quantity.
2. **Gaussian noise in absolute action = structural jitter**: noise amplitude > 7x
   constraint-feasible range even at min_std. Delta action bounds per-step change.
3. **Reward weight balance matters**: 97%:2%:0.5% gives no incentive for smoothness/efficiency.

---

## [2026-03-27] TRPO+IPO Algorithm Fixes (NORBC Paper Alignment)

### Summary

Six structural fixes aligning ConstraintTRPO with the NORBC paper (Muller et al., ICML 2025).
Fixes applied in order: (1) logging artifact, (2) cost critic normalization + encoder
starvation, (3) encoder trust region integration, (4) missing 1/(1-gamma) factor,
(5) cost advantage standardization, (6) barrier_alpha tuning.

Combined effect: reward -78.80 -> -37.36 (2x), roll 29.2 -> 18.0 deg (38%), pitch 26.5 ->
11.9 deg (55%), z saturation eliminated ([-0.99,0.99] -> [-0.53,0.40]).

### Fix 1: Line Search Logging Artifact

`surrogate()` closure sets `_last_barrier_penalty` and `_last_mean_entropy` on every call.
During backtracking (up to 10 attempts), monitoring vars retain last rejected candidate's
values -- inflated barrier from near-constraint-boundary proposals.

#### Fixed
- `algorithms/constraint_trpo.py`: Recalculate `surrogate()` with reverted params after
  line search failure

### Fix 2: Cost Critic d_k^2 Normalization + Encoder LS Gating

**d_k^2 normalization**: Intended to prevent large-budget constraints from dominating.
Actually ineffective: yaw_vel (d_k=78.5, d_k^2=6162) contributed 98.6% of loss. Raw MSE
scales O(d_k^2), division merely cancels scaling. Non-standard -- OmniSafe, CPO, FOCOPS,
IPO all use plain MSE.

**Encoder LS gating**: Encoder received zero gradient on line search failure. No precedent
in HORA/RMA/Extreme Parkour/RSL-RL/PPG. Creates starvation loop: bad z -> constraint
violation -> LS fails -> encoder frozen -> worse z. Longest freeze: 8 iters, reward dropped
4.3x faster.

#### Changed
- `algorithms/constraint_trpo.py`: Cost value loss `(per_k_mse / d_k^2).mean()` ->
  `per_k_mse.mean()`
- `algorithms/constraint_trpo.py`: Removed `ls_success` gate on encoder update

### Fix 3: Encoder Integration into TRPO Trust Region

Separate Adam encoder update (5 epochs, lr=3e-4) was destroying trust region:
- Pre-encoder KL: 0.0035 avg (within budget)
- Post-encoder KL: 0.138 avg (**27.6x budget**, max 1153.4x)
- 11.4% of iterations: barrier_penalty = -inf

NORBC trains encoder jointly with actor (same optimizer, same KL constraint). Moved encoder
params into TRPO CG + line search.

#### Changed
- `algorithms/constraint_trpo.py`: Encoder params moved from separate Adam into
  `_policy_params`. Added `_encoder_param_offset`, `_encoder_param_count` for monitoring.
- `utils/logging.py`: `log_encoder_metrics()` reads `_last_encoder_grad_norm` from TRPO
- `runners/constraint_encoder_runner.py`: Removed encoder optimizer save/load, replaced
  `pre_encoder_kl` with `encoder_grad_norm`
- `agents/rsl_rl_ppo_cfg.py`: Removed `num_encoder_epochs`, `encoder_lr` config fields

#### Removed
- `algorithms/constraint_trpo.py`: `_update_encoder()` (22 lines), `encoder_optimizer`,
  `_encoder_params`, `_has_encoder_params`, `_last_pre_encoder_kl`

#### Notes
- CG Fisher matrix automatically captures encoder's KL contribution via natural gradient
  curvature. Encoder weight_decay (1e-5 in Adam) now omitted.

### Fix 4: Missing 1/(1-gamma) in IPO Barrier Cost Surrogate

Paper Eq. 10: `margin_k = d_k^i - J_Ck - [1/(1-gamma)] * E[ratio * A_Ck]`

With cost_gamma=0.99, factor = 100. Barrier estimated margin change 100x too small.
Example (attitude, d_k=1.0, barrier_base=0.5):
- Paper: 0.5 - 100*0.003 = 0.2 (detects shrinking)
- Code: 0.5 - 0.003 = 0.497 (sees no change)

Reward surrogate intentionally omits factor (constant scale, direction-only). Cost term INSIDE
log() changes barrier argument, not just scale.

#### Fixed
- `algorithms/constraint_trpo.py`: Added `inv_one_minus_gamma = 1/(1-cost_gamma)` to
  `cost_surrs` in barrier surrogate

#### Notes
- `margin.clamp(min=1e-8)` kills gradient at margin <= 0 (OK at ratio=1, may need smooth
  barrier if value function accuracy is poor)

### Fix 5: Per-Constraint Cost Advantage Standardization (NORBC Sec IV-B)

Removed during paper-aligned architecture overhaul (`8ba1827c`). Without it, constraints with
different scales (binary 0/1 vs continuous |omega_z|) have vastly different gradient magnitudes.
When deeply infeasible (96% violation), accurate cost value predictions make raw cost advantages
near-zero (A_Ck ~ 0.04) -- barrier gradient dominated by noise.

NORBC Sec IV-B: `A_hat_Ck = (A_Ck - mu) / sigma` per constraint k.

#### Fixed
- `algorithms/constraint_trpo.py`: Restored `(A_Ck - mean) / (std + 1e-8)` per constraint.
  Originally added in `332eff85`, removed in `8ba1827c`.

### Fix 6: Barrier Alpha Adjustment

With 1/(1-gamma)=100 and barrier_t=100, effective barrier weight = 1/margin_k. Four deeply
infeasible constraints at floor margins (0.20-1.57) gave total barrier weight = 9.2 vs
reward = 1.

Increased barrier_alpha 0.02 -> 0.05: enlarges floor margin (alpha*d_k), self-deactivating
when constraints become feasible.
- torque: 0.40->1.0, velocity: 0.20->0.50, yaw_vel: 1.57->3.93
- Total barrier weight: 9.19 -> 2.26 (ratio 2.3:1)

**3-run progression:**

| Run | Changes | reward | noise | entropy | enc_grad max |
|-----|---------|--------|-------|---------|-------------|
| 00-09-23 | baseline | -78.80 | 0.64 | 1.41 | 1.0 |
| 01-15-43 | +Fix 3,4 | -38.77 | 0.60 | 1.29 | 322 |
| 01-38-08 | +Fix 5 | -37.36 | 0.44 | 0.82 | 14097 |

#### Changed
- `agents/rsl_rl_ppo_cfg.py`: `barrier_alpha` 0.02 -> 0.05

### Key Lessons

1. **Separate encoder optimizer nullifies TRPO trust region**: encoder added 27.6x KL budget
   per iteration. Joint CG + line search is mandatory.
2. **1/(1-gamma) is critical in IPO barrier**: without it, barrier 100x too weak to detect
   constraint-violating steps.
3. **Cost advantage standardization required**: raw advantages near-zero when deeply infeasible.
   NORBC Sec IV-B standardization provides balanced gradient across constraints.
4. **d_k^2 normalization non-standard and ineffective**: raw MSE scales O(d_k^2), division
   merely cancels. Use plain MSE (OmniSafe/CPO convention).
5. **Encoder starvation from LS gating**: no precedent in literature, creates positive
   feedback loop.
6. **Monitoring must reflect accepted state**: in IPM, log(margin) diverges as margin->0.
   Rejected candidates' metrics are misleading.
7. **barrier_alpha controls deeply-infeasible behavior**: self-deactivating when feasible.
   Preferable to barrier_t for infeasibility management.
