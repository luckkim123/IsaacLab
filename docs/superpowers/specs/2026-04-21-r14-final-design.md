# r14 Final Training Run — Design

**Date**: 2026-04-21
**Status**: Approved by user (Option B) — proceeding to writing-plans
**Baseline**: r13_B (latent=16) — `logs/rsl_rl/fulldof_albc/2026-04-20_20-08-39_r13_B`

## Context

r13 A/B were an encoder latent dim ablation (the *only* config diff). After deep post-eval analysis (both `eval_dr` tracking and `eval_dr_switching` zero-cmd), r13_B shows the more production-ready profile:

| Axis | r13_A (latent=9) | r13_B (latent=16) | Winner |
|------|------------------|-------------------|--------|
| Roll SS (eval_dr/switching) | 0.44-1.08° / 0.22-0.67° | 0.25-1.14° / 0.19-0.52° | B (slight edge, large tail edge) |
| Roll heavy-tail %>10° (hard) | 28% | **19%** | B |
| Pitch SS | **0.14-0.28°** | 0.25-0.31° | A |
| Yaw SS | **0.0018°** | 0.0058° | A |
| Yaw overshoot %>20° (n_gt20) | 11.5% | **5.5%** | B |
| Y-bias (switching SS) | **-2 to -5 mm** | -6 to -11 mm | A |
| Pos p99 (hard, switching) | 0.57m | **0.42m** | B |

r13_B wins on *peak/heavy-tail/transient* dimensions; r13_A wins on *steady-state/Y-bias*. Since production-readiness requires controlled peaks more than the last mm of SS, r13_B is the chosen baseline.

User also observed current HardDR is too easy (survival 100% + clean SS on all levels) — policy capacity is under-utilized. Final run widens DR ranges aggressively (1.5-3x) to fully exercise the policy; evaluation will filter/rescale the extreme tail during analysis.

## Goals

1. **Finalize r13_B as r14**, applying the two minimum interventions identified by root-cause analysis.
2. **Push DR range 1.5-3x wider** on 17 physics parameters to test control at/beyond nominal robot limits.
3. **Commit compute** — single run, 8x more samples than r13 (4096 envs × 20000 iter vs 2048 × 5000).

## Root-Cause Findings (from r13_A/B analysis)

### Roll oscillation at DR=none (0.68-0.87 Hz limit cycle)

- **NOT min_std floor**: per-dim min_std is (arm 0.10, thruster 0.05). Final thruster log_std = 0.22-0.34 (4-6x above floor). Actor voluntarily maintains high noise.
- **Root cause**: `entropy_coef=0.003` keeps actor noise elevated. Noise × TAM(roll_arm=0.007m, 20x weaker than pitch=0.145m) × weak roll damping = visible roll limit cycle even at DR=none.
- **Fix**: `entropy_coef 0.003 → 0.001`. Lets actor shrink thruster std toward 0.10-0.15 region. Expected roll amplitude reduction ~50%. Longer training (20000 iter) absorbs aggressive coef without exploration collapse.

### Yaw overshoot at r13_A = 17.6% (vs r11_emabias 11.1%)

- Traced to cumulative r11→r13 diff: HardDR `ocean_current_strength_range=(0,1)` added between r12_base and r13. r11_emabias (same policy config, NO ocean HardDR) achieved yaw_os 11.1%.
- **r13_B with ocean HardDR still achieves yaw_os 12.5%** — latent=16 absorbs the ocean current transient better than latent=9.
- For r14 (latent=16), ocean HardDR is kept AND expanded to (0, 2.0) — the capacity is there.

### DR range too narrow (survival 100% on all levels)

- Current HardDR ranges are within policy's easy-solve zone.
- Aggressive expansion (below) pushes past the "always succeeds" regime so that remaining training (20k iter) meaningfully utilizes the 4x sample budget.

## Config Changes (from r13_B)

### Compute

| Param | r13_B | r14 | Rationale |
|-------|-------|-----|-----------|
| `num_envs` | 2048 | **4096** | 2x samples/iter |
| `max_iterations` | 5000 | **20000** | 4x iterations |
| `save_interval` | 50 | **100** | 20000/100 = 200 ckpts |

### Algorithm

| Param | r13_B | r14 | Rationale |
|-------|-------|-----|-----------|
| `entropy_coef` | 0.003 | **0.001** | Attack roll limit cycle |
| DORAEMON `step_interval` | 250 | **500** | 40 updates in 20k iter — stable cadence |
| `kl_ub` (TRPO) | 0.04 | 0.04 (keep) | Safety bound, not stat bound |
| `min_std` / `min_std_per_dim` | 0.05 / (0.10,0.10,0.05×6) | keep | Not binding |
| `max_std` | 2.0 | keep | |
| `learning_rate` | KL-adaptive | keep | Self-regulating |

### Policy

| Param | r13_B | r14 |
|-------|-------|-----|
| `encoder_latent_dim` | 16 | **16** (keep) |
| `encoder_hidden_dims` | [256, 128, 64] | keep |
| `actor_hidden_dims` | [256, 128, 64] | keep |
| `critic_hidden_dims` | [512, 256, 128] | keep |
| `activation` | elu | keep |
| Encoder output | softsign | keep |
| EMA bias `k_bias`, `bias_ema_alpha`, `bias_weights` | -2.0, 0.99, (1.5,1,1,1,1,1) | keep |

### HardDR expansion (17 params, 1.5-3x wider)

| Param | r13_B | r14 |
|-------|-------|-----|
| `thrust_coefficient_scale` | (0.7, 1.3) | **(0.3, 1.5)** |
| `time_constant_scale` | (0.7, 1.3) | **(0.3, 2.0)** |
| `yaw_damping_scale` | (0.5, 1.5) | **(0.2, 2.0)** |
| `body_mass_scale` | (0.75, 1.25) | **(0.5, 1.5)** |
| `added_mass_scale` | (0.5, 1.5) | **(0.3, 1.8)** |
| `linear_damping_scale` | (0.4, 1.7) | **(0.2, 2.2)** |
| `quadratic_damping_scale` | (0.4, 1.7) | **(0.2, 2.2)** |
| `volume_scale` | (0.75, 1.25) | **(0.6, 1.4)** |
| `inertia_scale` | (0.4, 2.0) | **(0.3, 3.0)** |
| `joint_effort_limit_range` | (0.7, 1.0) | **(0.3, 1.0)** |
| `ocean_current_strength_range` | (0.0, 1.0) | **(0.0, 2.0)** |
| `payload_mass_range` | (0.0, 3.0) | **(0.0, 5.0)** |
| `water_density_range` | (995, 1025) | **(970, 1050)** |
| `joint_stiffness_range` | (30, 150) | **(20, 200)** |
| `joint_damping_range` | (0.3, 7.0) | **(0.1, 10.0)** |
| `joint_static_friction_range` | (0.0, 0.03) | **(0.0, 0.1)** |
| `joint_viscous_friction_range` | (0.0, 0.2) | **(0.0, 0.5)** |

**Unchanged**: `payload_cog_offset_xy_radius=0.08` (per user — radius 0.05 deferred).

### Non-DORAEMON DR expansion

| Param | r13_B | r14 | Rationale |
|-------|-------|-----|-----------|
| `noise_scale` (observation) | (0.1, 0.1, 0.05, 0, 0, 0) | **(0.2, 0.2, 0.1, 0.05, 0.05, 0.05)** | Sensor noise widened across all channels incl. angular velocity |
| OU `delta_scale` (current time-variance) | 0.1 | **0.2** | 2x faster current drift |

### New DR: action latency (port from hero_agent)

hero_agent (deprecated) had a fully working `action_latency_range` DR via an `_action_history` buffer. Port to `constrained_full_albc`:

| Param | r14 |
|-------|-----|
| `action_latency_range` | **(0, 6)** physics steps = 0-30 ms delay |

**Implementation outline** (~40 lines in `albc_env.py`):
1. On `__init__`: allocate `_action_history = torch.zeros(num_envs, max_latency+1, action_dim)`, `_action_latency = torch.zeros(num_envs, dtype=long)`.
2. On `_reset_idx(env_ids)`: `_action_latency[env_ids] = randint(lo, hi+1)`.
3. In `_pre_physics_step(actions)`: roll history, insert new action at index 0, fetch delayed action `_action_history[env_idx, _action_latency]` — use this instead of raw action.
4. Register `action_latency_range` in `RandomizationCfg`.

Reference code: `source/isaaclab_tasks/isaaclab_tasks/direct/hero_agent/base_env.py:421-430, 561, 1144-1147`.

**Rationale**: real-hardware actuator/communication delay. Expected 5-15 ms in production. Training range (0, 30 ms) pushes beyond hardware limit to build robustness margin. Directly attacks yaw overshoot (delayed feedback = more underdamping tendency; policy must learn predictive control).

## Evaluation Strategy

Because the aggressive HardDR exposes the policy to scenarios beyond physical robot limits, standard eval metrics at the full hard level will look worse than r13 — this is expected and not a regression.

**Eval protocol for r14**:
1. Run `eval_dr_fulldof.py` with default 4 levels.
2. For summary metrics reporting, **use none/soft/medium only**. Hard is kept for survival/heavy-tail tail inspection, not SS/overshoot comparison.
3. Alternative: rescale `DR_SCALE` to `{none:0, soft:0.2, medium:0.5, hard:0.8}` so "hard" stays within previous r13 hard range — gives clean comparison to r13_B.
4. Run `eval_dr_switching.py` — zero-command DR-switching test. Filter analogously.

## Success Criteria

1. **Roll SS at none DR < 0.3°** (r13_B: 0.41° → target 25%+ reduction via entropy_coef drop).
2. **Yaw overshoot n_gt20% at r13_B-equivalent hard ≤ 6%** (r13_B: 5.5%, maintain or improve with longer training).
3. **Survival ≥ 95% at medium DR** (r14 medium ≈ r13 medium, expect near-100%).
4. **Survival ≥ 70% at r14 hard** (aggressive DR, 70% is acceptable limit-test target).
5. **Y-bias at soft DR < 10 mm** (r13_B: 7 mm baseline).
6. Thruster log_std final ≤ 0.18 mean (r13_B: 0.25 mean — lower = entropy reduction worked).

## Out of Scope (deferred)

- Payload `cog_offset_xy_radius` 0.08→0.05 (Y-bias fix) — defer to r15 if r14 still shows Y-bias issue.
- Replacement DR: per-axis yaw damping vs roll damping asymmetry.
- Action-rate penalty tuning — rely on entropy reduction first.

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Entropy_coef=0.001 too aggressive → exploration collapse early | 20k iter + DORAEMON curriculum provides recovery runway. If entropy drops below -5 before iter 5000, flag. |
| Expanded DR causes DORAEMON to stall at low difficulty | Curriculum stops scaling when reward < performance_lb=90. Worst case: policy stays at r13-like DR, no regression. |
| 4x iter + 2x envs = 8x wall clock (est 30-40h on one GPU) | Single run acceptable; user launches and polls. |
| Save_interval=100 × 20000 iter = 200 checkpoints × several MB | ~several GB storage. Monitor disk. |
| Ocean current 2x too strong for inner-loop vel tracking | vel_cmd saturation in cascade PID ensures policy still seeks zero; worst case mid-episode drift, no divergence. |

## Files to modify (for writing-plans)

- `source/isaaclab_tasks/isaaclab_tasks/direct/constrained_full_albc/config.py` (HardDR ranges, DORAEMON step_interval)
- `source/isaaclab_tasks/isaaclab_tasks/direct/constrained_full_albc/agents/rsl_rl_ppo_cfg.py` (entropy_coef, save_interval)
- Launch script args (num_envs, max_iterations)

No code changes to algorithm/encoder/runner required.
