# Encoder Integration Ablation Study

2026-03-27 ~ 2026-03-30. 20+ experiments systematically isolating why encoder
destabilizes PPO/TRPO training in 2-DOF constrained ALBC.

**Final root cause:** `sample().clamp(-1,1)` in `ActorCriticEncoder.act()` concentrates
actions at boundaries when noise_std ~1.0 (~32%), creating sharp log_prob gradients that
amplify KL 100x, crashing adaptive LR. Secondary issue: env-level clamp positive feedback
loop prevents noise_std recovery when encoder makes advantages noisy.

**Resolution:** Offline encoder pipeline (collect rollouts -> supervised training -> frozen
encoder fine-tuning). Three additional bugs found in frozen encoder path (see ../changelog_constrained_albc.md
2026-03-30 entry).

---

## Baseline Components (Steps 0-3)

Incremental complexity to establish where encoder breaks the system.

| Step | Config | Roll (deg) | Pitch (deg) | Iters | Notes |
|------|--------|-----------|-------------|-------|-------|
| 0 | PPO (no DR/encoder/constraints) | 0.6 | 0.7 | 75 | Baseline |
| 1 | PPO + DR | 3.9 | 3.7 | 66 | DR alone manageable |
| 2 | TRPO + DR | 5.4 | 5.1 | 83 | TRPO slower but works |
| 3 | TRPO + DR + Barrier (4 constraints) | 8.4 | 6.3 | 162 | Constraints slow convergence |

Barrier verified: 0 spikes, all margins positive. Constraint budgets tightened from
ablation data (torque 0.20->0.08, velocity 0.10->0.02, yaw_vel 0.785->0.40).

---

## Encoder Failure Pattern (Steps 4-7)

All encoder experiments share the same failure: encoder update creates KL ~0.14 (7x
desired_kl=0.02) at iter 1, crashing adaptive LR. noise_std stays at ceiling (~0.97),
policy effectively random.

### Step 4: TRPO + Encoder -> FAIL

Roll 16.4, pitch 45.2 deg. Fisher info ~0 for encoder params + CG damping=0.1 amplifies
encoder gradient 10x, consuming KL budget.

### Step 4b: PPO + Encoder -> FAIL (different mechanism)

| Metric | TRPO+Enc (Step 4) | PPO+Enc (Step 4b) |
|--------|-------------------|-------------------|
| Roll / Pitch | 14.7 / 46.2 | 32.5 / 26.3 |
| z_std | 0.265 | 0.975 (saturated) |
| Failure mode | Fisher amplification | z saturation -> KL -> LR death |

z/actor_input ratio identified as structural issue: HORA 7.7% vs ALBC 48.1%.

### Step 4c: PPO + Encoder + History -> 6 ablations, ALL FAILED

Single-variable ablation against HORA differences:

| Exp | Changed | LR death | Roll | Pitch |
|-----|---------|:--------:|-----:|------:|
| baseline | (none) | YES | 41.5 | 32.5 |
| 4c-1 | entropy_coef=0.0 | YES | 29.8 | 37.2 |
| 4c-2 | ent=0 + lr=5e-3 | YES | 16.2 | 47.1 |
| 4c-3 | desired_kl=0.02 | YES | 15.9 | 40.3 |
| 4c-4 | steps_per_env=8 | YES | 13.3 | 53.0 |
| 4c-5 | normalize_value | YES | 24.3 | 22.0 |
| 4c-6 | fixed schedule | N/A | NaN | NaN |

### Step 4d: History-Only PPO (no encoder) -> SUCCESS

Actor: policy(14D) + history(240D) = 254D. Standard ActorCritic, no encoder.

| Metric | ent=0.01 | ent=0.0 |
|--------|----------|---------|
| Roll / Pitch | 3.57 / 3.27 | 3.03 / 3.83 |
| noise_std | 0.81 (rising) | 0.20 (falling) |

**Conclusion:** 254D history-only works. Encoder integration is the sole problem.

### Steps 5-7: Architecture variations (all failed)

| Step | Architecture | KL iter-1 | Result |
|------|-------------|-----------|--------|
| 5a (x6) | Shared backbone (6 variants) | 0.3-1.5 | LR death |
| 5b (x2) | Shared backbone + history | 0.367 | LR death |
| 6 | Separate network + per-minibatch refresh | 0.139 | LR death |
| 7 | Small encoder [256,128]->8D | 0.144 | LR death |

**Invariant finding:** Encoder update at iter 1 creates KL ~0.14 regardless of encoder
size, architecture, or optimizer configuration.

---

## HORA Alignment (Steps 8-12)

Systematic replication of HORA design elements.

### Steps 8a-8b: Q1+Q3 fixes (HORA-style normalization)

Q1: actor_obs_normalizer excludes z (HORA passes z raw).
Q3: strided proprio history (15x5, 1.5s window).

| Step | Config | Roll/Pitch | noise_std | z_range | LR |
|------|--------|-----------|-----------|---------|-----|
| 8a | Q1+Q3 (enc norm kept) | 23/20 | 0.96 | [-0.86, 0.85] | 1.5e-5 |
| 8b | Q1+Q3 (enc norm removed) | 26/32 | 0.98 | [-1.00, 1.00] SAT | 4.0e-5 |

**Result:** Q1 prevents z saturation. encoder_obs_normalization confirmed required for 23D
input. KL spike unchanged.

### Steps 9a-9b: Q4 LR range (FAILED)

HORA-style init_lr=5e-3, min_lr=1e-6 (22 halvings vs ALBC's 9).

| Step | Config | Roll/Pitch | noise_std | Eq. LR | KL iter-0 |
|------|--------|-----------|-----------|--------|-----------|
| 9a | Q4 (enc norm) | 31/27 | 0.97 | 2.4e-5 | 7.13 |
| 9b | Q4 (no enc norm) | 26/26 | 0.96 | 2.6e-5 | 5.77 |

**Finding:** Equilibrium LR determined by network dynamics, not init_lr or min_lr.

### Step 10: Encoder gradient scaling 0.1x (FAILED)

Results byte-for-byte identical to Step 8a. Encoder weight changes are negligible
relative to initial weights (20 steps at effective LR ~1e-4 changes weights ~0.1%).

### Steps 11-12: Update path + HORA reward scaling

| Step | Config | Roll/Pitch | noise_std | LR | KL |
|------|--------|-----------|-----------|-----|-----|
| 11a | Standard update() | 23/18 | 0.98 | 1.0e-5 | 0.08 |
| 12a | reward_scale=0.01 | 17.9/43.8 | 0.92 | 4.0e-5 | 0.04 |

Step 11: **DISPROVED** -- `_update_encoder_ppo()` is actually BETTER (per-minibatch
mu/sigma refresh reduces KL). Step 12: **First positive signal** -- LR 2.7x higher,
noise_std declining for first time ever. But pitch degraded to 43.8.

---

## Root Cause Isolation (Steps 13-19)

### Steps 13-14: Static norm + Encoder freeze (both DISPROVED)

| Step | Hypothesis | Result |
|------|-----------|--------|
| 13 | EmpiricalNorm z drift causes KL | DISPROVED (static norm identical) |
| 14 | Encoder weight changes cause KL | DISPROVED (frozen encoder fails identically) |

Step 14 critical: even with encoder **completely frozen** (z_std=0.08, fixed random noise),
PPO fails identically (LR=1.8e-5, noise_std=0.97). Root cause is structural in the
ActorCriticEncoder class itself, not encoder learning.

### Steps 15-17: Structural isolation -> BREAKTHROUGH

| Step | Hypothesis | Roll/Pitch | noise_std | LR | KL iter-0 | Result |
|------|-----------|-----------|-----------|-----|-----------|--------|
| 15 | Asymmetric critic | 22.9/21.0 | 0.97 | 1.8e-5 | 0.65 | DISPROVED |
| 16 | log_std parameterization | 22.9/18.6 | 0.98 | 2.6e-5 | 0.88 | DISPROVED |
| **17** | **Action clamp** | **10.7/9.6** | **148** | **0.01** | **0.003** | **ROOT CAUSE** |

**Step 17: Action clamp confirmed as root cause.**
- First time encoder KL < 0.015 (100x reduction from 0.88)
- LR stable at max (0.01) throughout training
- But noise_std exploded to 148 (log_std exp(5.0) upper clamp)

**Mechanism:** `sample().clamp(-1,1)` piles actions at [-1,1] boundaries -> small mu shift
causes large log_prob change -> amplified surrogate gradient -> KL spike -> LR crash.

### Step 18: Scalar std + no clamp (combined fix)

noise_std=18.5 (exploded, same as Step 17 direction). **scalar_std hypothesis DISPROVED**
-- both parameterizations explode with encoder. Reverted to log_std.

Root cause of std explosion identified: **env-level clamp positive feedback loop.**
- `albc_env.py:320` clamps actions to [-1,1]
- `ppo.py` stores/evaluates UNCLAMPED actions
- When std > ~1.5: 62%+ actions clamped -> identical physical outcomes for different
  unclamped actions -> score function gradient loses corrective signal -> positive feedback

HORA has the same mismatch but avoids it via reward_scale=0.01 (reduced gradient ->
smaller encoder z changes -> std decreases before reaching ~1.5 threshold).

### Step 19: HORA-aligned (reward_scale=0.01)

| Metric | Value | vs Step 4d target |
|--------|-------|-------------------|
| noise_std | 7.66 (oscillating) | 0.20 |
| Roll/Pitch | 16.8/14.5 | 3.0/3.8 |

noise_std OSCILLATES (periodic drops to ~0.3, then spikes to 6-11). reward_scale provides
corrective signal but insufficient to prevent re-entry into positive feedback regime.

---

## Hypothesis Scorecard

| # | Hypothesis | Step | Result |
|---|-----------|------|--------|
| 1 | EmpiricalNorm z drift | 13 | DISPROVED |
| 2 | Encoder gradient magnitude | 10 | DISPROVED |
| 3 | Encoder weight changes (freeze) | 14 | DISPROVED |
| 4 | Init LR / min LR range | 9 | DISPROVED |
| 5 | Update path (_update_encoder_ppo) | 11 | DISPROVED (actually helps) |
| 6 | History size/stride | 8 | insufficient |
| 7 | Critic asymmetry | 15 | DISPROVED |
| 8 | Normalization method (static vs emp) | 13 | DISPROVED |
| 9 | noise_std type (scalar vs log) | 16,18 | DISPROVED |
| 10 | **Action clamp in act()** | **17** | **ROOT CAUSE (KL)** |
| 11 | **Env-level clamp + unclamped buffer** | **18,19** | **ROOT CAUSE (std explosion)** |
| 12 | reward_scale=0.01 | 12,19 | partial (oscillation) |

---

## Code Changes Summary

### Added to `encoder/actor_critic_encoder.py`
- `noise_std_type` param ("log" default, "scalar" option)
- `clamp_actions` param (default True, False removes act() clamp)
- `symmetric_critic` param (default False)
- Static min-max normalization (`encoder_obs_lower`/`encoder_obs_upper`)
- HORA-style normalization: `actor_obs_normalizer` excludes z dimensions

### Added to `agents/rsl_rl_ppo_cfg.py`
- 15+ runner configs for ablation steps (see `__init__.py` for task registration)
- 23D privileged obs bounds for static normalization
- `_FrozenEncoderAlgorithmCfg`, `_HoraAlignedAlgorithmCfg`

### Added to `rsl_rl/algorithms/ppo.py` (external, not git-tracked)
- `use_encoder_update`, `reward_scale`, `min_lr`, `max_lr`, `encoder_grad_scale` params
- **Needs reapply on container rebuild**

### Added env configs
- `ALBCDebugEnvCfg` through `ALBCDebugEncoderHistStrideEnvCfg` (Steps 0-8)
- `ALBCHardDR*` configs (offline pipeline)
- `ALBCHardDRFrozenEncoderEnvCfg` (frozen encoder fine-tuning)

### Added scripts
- `scripts/analysis/collect_rollouts.py` -- rollout data collection
- `scripts/analysis/train_offline_encoder.py` -- supervised encoder training

### Added classes
- `ActorCriticFrozenEncoder` -- frozen encoder with warm-start support

---

## Key Architectural Lessons

1. **2D action space breaks HORA assumptions**: KL concentrates on 2 dims (sigma consumes
   ~33% of budget vs ~4-8% in 16D). All HORA stability margins unavailable.

2. **Action clamp is poison for encoder co-training**: boundary concentration amplifies KL
   by 100x. ActorCritic (Step 4d) avoids this because it has no clamp in act().

3. **Env-level clamp creates positive feedback**: unclamped actions in buffer + clamped
   physics outcomes = credit assignment mismatch. Threshold ~1.5 std.

4. **Encoder destabilizes ANY optimizer**: TRPO (Fisher amplification), PPO (z expansion ->
   KL -> LR death). Same failure regardless of architecture, size, or optimizer config.

5. **Offline encoder is the viable path**: online co-training structurally unstable in 2D.
   Offline encoder quality verified: z explains 70.3% additional V_critic variance.
