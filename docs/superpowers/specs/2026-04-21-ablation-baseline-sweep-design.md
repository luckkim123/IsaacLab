# Ablation & Baseline Sweep — Design Spec

**Date**: 2026-04-21
**Status**: Draft (awaiting user review)
**Author**: brainstorming session output

## Purpose

Establish the comparison experiment set to support claims about the main method `encoder + constrained (IPO) + TRPO`. Priority of claims:

1. **Primary (A)**: Encoder contributes to DR adaptation.
2. **Secondary (B)**: Constrained formulation (IPO) outperforms reward-only shaping.

Secondary goals: bracket the main method against standard RL (PPO) and separate TRPO-vs-PPO effect from IPO-vs-reward effect.

## Scope

- 5 policy variants trained end-to-end, evaluated with `eval_dr_fulldof` and `eval_dr_switching`.
- Sequential training on GPU 0 only. GPU 1 is reserved for the user's parallel experiment. When GPU 1 becomes available, remaining phases may be parallelized (user will notify).
- Single seed per variant (`seed=30`). Multi-seed is out of scope for this experiment and deferred to a later statistical-rigor pass.

Out of scope:
- Hyper-parameter search per variant. All variants reuse the main method's hyper-parameters verbatim, minus algorithm-specific terms that do not apply.
- Adding constraint-equivalent reward penalties for the no-constraint variants. The fair comparison is "same reward across all variants", matching the convention in CPO/IPO/SafeExploration benchmarks.

## Variant Matrix

| # | Run name | Encoder | Constraint | Algorithm | Registered task | Implementation |
|---|---|---|---|---|---|---|
| 1 | `main` | Yes | Yes (IPO) | TRPO | `Isaac-FullDOF-TRPO-v0` | Existing, reuses best hist-series checkpoint |
| 2 | `noenc` | No | Yes (IPO) | TRPO | `Isaac-FullDOF-NoEncoder-v0` | Existing, re-train on current env |
| 3 | `nocstr` | Yes | No | TRPO | `Isaac-FullDOF-TRPO-NoIPO-v0` | NEW: requires `ALBCNoConstraintEnvCfg` + runner cfg |
| 4 | `ppoenc` | Yes | No | PPO | `Isaac-FullDOF-PPO-Enc-v0` | NEW: PPO runner + `ActorCriticEncoder`, compatibility smoke test required |
| 5 | `pureppo` | No | No | PPO | `Isaac-FullDOF-PPO-v0` | Existing, re-train on current env |

## Baseline (#1) Selection Procedure

**Initial selection (2026-04-21, completed)**: `r13_A` selected as baseline from `{r13_A, hist5, hist5_act3, hist10}` four-way eval_dr + eval_dr_switching comparison.

Rationale (from user's analysis):
- r13_A balanced across axes: pitch/vx/yaw/switching peak all 1st or tied-1st.
- hist_len expansion did not produce decisive wins — hist10 only improved vz (CV 231%→69%); other axes tied or slightly worse.
- hist5_act3 best on settling ss_roll (0.59°) and vy nominal but regressed on vx CV (292%).
- layernorm excluded (reward regression -33%, pos_drift +3278%).

Root-cause hypothesis (user's analysis): encoder bottleneck (`latent_dim=9`) + asymmetric critic's privileged-obs leakage together suppress marginal value of longer history. All 5 hist-series runs used `encoder_latent_dim=9`.

### Baseline Challenger: hist5_act3 + `encoder_latent_dim=16`

To verify the bottleneck hypothesis before committing to r13_A, train one challenger run with `hist_len=5, hist_action_len=3, observation_space=121` and `encoder_latent_dim=16`. This is the same hist5_act3 configuration but with 78% more encoder latent capacity.

**Outcome-driven baseline selection:**
- If challenger wins aggregate score vs r13_A → baseline = challenger, canonical env cfg = hist5_act3 + latent=16, variants #2–#5 inherit that config.
- If challenger loses → baseline = r13_A, canonical env cfg reverts to `hist_len=3, hist_action_len=2, observation_space=87, encoder_latent_dim=9`, variants #2–#5 inherit that config.

**Comparison rule**:
- Primary: sum of `ss_error` across all `{none, soft, medium, hard} × {roll, pitch, vx, vy, vz, yaw}`, lowest wins.
- Tie-break (within 5%): sum of `ss_error_std` (CV proxy).
- Tertiary: `n_gt20` sum + switching seg1-9 `peak_roll` mean.

Documented in `logs/rsl_rl/fulldof_albc/ablation_sweep/baseline_selection.md`.

### Variable-Control Implication

The selected final baseline's env cfg becomes canonical for variants #2–#5. Specifically:
- `hist_len`, `hist_stride`, `hist_action_len`, `observation_space` copied from final baseline's `params/env.yaml`.
- `policy_obs_dim` and `encoder_latent_dim` in `rsl_rl_ppo_cfg.py` updated to match.
- Reward weights, DR, DORAEMON, noise, ocean current, action latency held at their current (post-r14, k_bias=-2.0, OU-drift-on) values.

All edits are made **from the main isaaclab repo directly**. No worktree install swap. Before editing `config.py`/`rsl_rl_ppo_cfg.py`, confirm with the user that no other training (student policy, etc.) is constructing env instances, since mid-run config changes could corrupt imports.

## Variable Control (critical)

All variants share the current `ALBCEnvCfg` verbatim. This is the "single-variable" discipline that makes the ablation interpretable.

| Component | Setting | Rationale for sharing |
|---|---|---|
| Observation | 87D (26 current + 55 history + 6 integral) | `hist_len=3, hist_stride=3, hist_action_len=2` |
| Privileged | 24D | Encoder input (variants #1, #3, #4); critic input (all) |
| Action | 8D (2D arm + 6D thruster) | `delta_scale=0.10`, thruster wrench normalized |
| Reward | 7 terms (`lin_vel, att_rp, yaw_vel, torque, thruster, smoothness, bias_ema`) | `k_bias=-2.0`, same weights across all variants |
| DR | `HardDomainRandomizationCfg` (r14 widening) | hydro, COG/COB, inertia, mass, thruster, ocean current, actuator, payload, action latency |
| DORAEMON | Enabled (`kl_ub=0.06, performance_lb=90.0, step_interval=500`) | Adaptive DR curriculum |
| Noise | `_OBS_NOISE_STD`, `_OBS_BIAS_MAG` | Observation noise + additive bias |
| Ocean current | `max_velocity=(0.5, 0.5, 0.25, 0, 0, 0)`, `ou_enable=True` | Mid-episode drift active |

Only one deviation exists: variant #3 (`TRPO-NoIPO`) requires `constraints=ALBCConstraintCfg(terms=[])` to disable the `ConstraintEncoderRunner` auto-sync that forces `num_constraints=10` into the IPO algorithm. Without this, the IPO barrier cannot be cleanly disabled. The env still reports the same reward, same DR, same observation, same action, same DORAEMON; only the constraint bookkeeping is removed.

## Hyper-parameter Policy

All variants inherit the main method's hyper-parameters. Algorithm-specific parameters are kept only where they apply:

| Parameter | Value | Applies to |
|---|---|---|
| `seed` | 30 | All |
| `num_envs` | 2048 (CLI override) | All |
| `max_iterations` | 5000 (CLI override) | All |
| `num_steps_per_env` | 64 | All |
| `save_interval` | 100 | All |
| `actor_hidden_dims` | [256, 128, 64] | All |
| `critic_hidden_dims` | [512, 256, 128] | All |
| `activation` | elu | All |
| `encoder_hidden_dims` | [256, 128, 64] | #1, #3, #4 |
| `encoder_latent_dim` | 9 or 16 (dynamic) | #1, #3, #4 |
| `encoder_output_norm` | True (LayerNorm before softsign) | #1, #3, #4 |
| `critic_uses_z` | True | #1, #3, #4 |
| `init_noise_std` | 0.7 | All |
| `min_std` / `max_std` | 0.05 / 2.0 | TRPO variants (#1, #2, #3) |
| `min_std_per_dim` | (0.10, 0.10, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05) | TRPO variants |
| `entropy_coef_per_dim` | (0.01, 0.01, 0.0005, ..., 0.0005) | TRPO variants |
| `entropy_coef` (PPO) | 0.003 | PPO variants (#4, #5) |
| `max_kl` | 0.005 | TRPO variants |
| `cg_iters` | 10 | TRPO variants |
| `cg_damping` | 0.1 | TRPO variants |
| `line_search_max_backtracks` | 10 | TRPO variants |
| `line_search_kl_margin` | 1.5 | TRPO variants |
| `barrier_t` | 100.0 | IPO-only (#1, #2) |
| `barrier_alpha` | 0.05 | IPO-only (#1, #2) |
| `learning_rate` (PPO) | 3e-4 | PPO variants |
| `schedule` (PPO) | adaptive | PPO variants |
| `desired_kl` (PPO) | 0.01 | PPO variants |
| `clip_param` (PPO) | 0.2 | PPO variants |
| `value_lr` (TRPO) | 1e-3 | TRPO variants |
| `gamma` / `lam` | 0.99 / 0.95 | All |
| `cost_gamma` / `cost_lam` | 0.99 / 0.95 | IPO-only |
| `value_loss_coef` | 1.0 | All |
| `cost_value_loss_coef` | 1.0 | IPO-only |
| `max_grad_norm` | 1.0 | All |

## Implementation Required

### Variant #3 (`TRPO-NoIPO`, new)

1. New env config class `ALBCNoConstraintEnvCfg(ALBCEnvCfg)` with `constraints=ALBCConstraintCfg(terms=[])`.
2. New runner cfg `FullDOFTRPONoIPORunnerCfg` reusing `_FullDOFPolicyCfg` and `RslRlConstraintTRPOAlgorithmCfg` with `num_constraints=0`. `ConstraintEncoderRunner` auto-sync will confirm `num_constraints=0` because env has empty constraint list.
3. Register `Isaac-FullDOF-TRPO-NoIPO-v0` in `__init__.py`.

### Variant #4 (`PPO-Enc`, new)

1. New policy cfg `_FullDOFPPOEncPolicyCfg`: inherits `_FullDOFPolicyCfg`, overrides `class_name="FullDOFActorCriticEncoder"`, sets `num_constraints=0`.
2. New algorithm cfg reusing `_FullDOFPPOAlgorithmCfg`.
3. New runner cfg `FullDOFPPOEncRunnerCfg` that uses standard `OnPolicyRunner` (not `ConstraintEncoderRunner`; otherwise auto-sync forces `num_constraints=10`).
4. Smoke test: verify standard rsl-rl `OnPolicyRunner` + `ActorCriticEncoder` compatibility. `ActorCriticEncoder` inherits `PolicyBase` which exposes `act, act_inference, evaluate, get_actions_log_prob, update_normalization, load_state_dict, action_mean, action_std, entropy` — all methods `OnPolicyRunner` requires. Remaining risk: PPO kwargs dispatch to the custom policy class.
5. If smoke test fails, add a minimal `EncoderOnPolicyRunner` subclass that preserves PPO semantics while handling encoder metrics logging.
6. Register `Isaac-FullDOF-PPO-Enc-v0` in `__init__.py`.

### Variants #2 (`NoEncoder`) and #5 (`PurePPO`) — re-verification

Git log (2026-04-21 analysis):
- Both tasks introduced on **2026-04-08** in the `eafca264` refactor.
- Policy cfgs kept in sync during R7/R8 integral-obs addition (`policy_obs_dim=81→87`) and r14 tuning (`entropy_coef_per_dim` thruster 0.001→0.0005).
- `ActorCriticAsymConstrained` class body has not been edited since 2026-04-08.

Env-side changes since 2026-04-08 that affect both variants (inherited automatically via shared `ALBCEnvCfg`):
- Reward: `k_att_rp 9.0`, `bias_ema_alpha=0.99, k_bias=-2.0` (R11 EMA bias), `lin_vel` / `yaw_vel` saturating tanh term
- Constraint set: 10 terms (current), previously experimented with thruster_rate (removed), thruster_sat (reverted to thruster_util)
- Observation: 6D integral obs (R7/R8), action_hist layout refined
- DR: r14 Hard DR widening (inertia 0.75→0.3~3.0, payload_mass 1.0→5.0, thrust_coeff widened, action_latency 0→6 physics steps)
- Mid-episode: OU current drift enabled, angular ocean current channels activated
- DORAEMON: `kl_ub=0.06, performance_lb=90.0, step_interval=500`

Risk: even though `policy_obs_dim=87` matches current env, `ActorCriticAsymConstrained` and standard rsl-rl `ActorCritic` have **never been run for 5000 iter under the post-r14 env** end-to-end. Latent bugs may exist in interactions with action latency ring buffer, k_bias reward term, or mid-episode OU drift.

Mitigation: Phase 0.5 adds a 500-iter sanity run per variant before committing to full 5000-iter runs (see Training Schedule).

## Training Schedule (sequential, GPU 0 only)

| Phase | Variant | Task id | Est. time |
|---|---|---|---|
| 0 | Baseline selection (completed from user analysis) | — | r13_A selected as initial baseline |
| 0.6 | Baseline challenger: hist5_act3 + `encoder_latent_dim=16` | `Isaac-FullDOF-TRPO-v0` (env temporarily reconfigured) | ~5 hr train + 30 min eval |
| 0.7 | Final baseline decision + canonical env cfg lock | — | ~15 min analysis |
| 0.5 | Pre-flight 500-iter sanity (`#2`, `#5`, + smoke of `#3`, `#4`) under locked cfg | all 4 tasks | ~40 min per variant × 2 full + 2 smoke = ~1.5 hr |
| 1 | `#2 noenc` | `Isaac-FullDOF-NoEncoder-v0` | ~5 hr |
| 2 | `#3 nocstr` | `Isaac-FullDOF-TRPO-NoIPO-v0` | ~5 hr (+ 1 hr implementation) |
| 3 | `#4 ppoenc` | `Isaac-FullDOF-PPO-Enc-v0` | ~5 hr (+ 1 hr implementation) |
| 4 | `#5 pureppo` | `Isaac-FullDOF-PPO-v0` | ~5 hr |

Phase 0.6 rationale: user's 4-way analysis showed no hist-length variant decisively beat r13_A despite 2-4x more obs dim. Hypothesis: `encoder_latent_dim=9` bottleneck squeezes extra history through a fixed 9-dim manifold. Testing this requires one run with doubled latent capacity under the best hist variant's input (`hist5_act3`). Outcome decides canonical env cfg for the rest of the sweep.

Phase 0.5 rationale: `ActorCriticAsymConstrained` and standard rsl-rl `ActorCritic` have never been verified in 5000-iter runs under the current post-r14 env. A 500-iter run per variant catches interaction bugs with action latency, bias_ema reward, OU drift, etc., before committing 5 hr of compute. Acceptance: reward trend positive at iter 500, no crashes, constraint cost bookkeeping (where applicable) intact. Phase 0.5 runs under the canonical env cfg locked at Phase 0.7.

Order constraint: Phase 0.6 → 0.7 → 0.5 → 1–4. Phase 0.5 cannot start until Phase 0.7 locks final cfg.

Total: ~25 GPU-hr training + ~1.5 hr sanity + ~2 hr implementation. Each phase finishes fully before the next starts.

If the user frees GPU 1 during the schedule, remaining phases may be reassigned to GPU 1 in parallel. This is a runtime decision, not a spec requirement.

### Hard prohibition: no editable install swap

Per operational feedback 2026-04-21: no `./isaaclab.sh --install` calls to swap editable package paths during the sweep. All ablation training and eval must use the **main isaaclab repo's current install**. When baseline selection requires evaluating a checkpoint whose `policy_obs_dim` differs from the current `ALBCEnvCfg.observation_space`:

1. If the user indicates no other training is active, edit main's `ALBCEnvCfg` and `rsl_rl_ppo_cfg.py` to match the target checkpoint's env (temporary commit allowed), run eval, then revert / keep depending on Phase 0 Task 0.4 outcome.
2. If other training is active, defer that checkpoint from the candidate set and note it in `baseline_selection.md`.

Never run the editable install command while student-policy training (or any other training) is running.

## Evaluation Protocol (per phase)

After each training finishes (`model_4999.pt` written):

1. `eval_dr_fulldof.py --task <variant-task> --num_envs 64 --headless --checkpoint <ckpt>` → `<run>/eval_dr/`
2. `eval_dr_switching.py --task <variant-task> --num_envs 64 --headless --checkpoint <ckpt> --seed 42 --segment_duration 5.0 --num_segments 10` → `<run>/eval_dr_switching/`
3. Inspect `summary_att.png, summary_lin_vel.png, summary_yaw.png` + `enhanced_summary.json`.
4. Run `train-analyze` skill on the training run for phase-complete report.

After all 5 variants finished:
- `analyze_eval_dr.py --dirs <#1> <#2> <#3> <#4> <#5> --labels main noenc nocstr ppoenc pureppo --levels none soft medium hard` for heavy-tail and sample-mean divergence.
- 5-way `enhanced_summary.json` comparison table (mean ± across-env std per DR level × axis).
- Claim-specific reports:
  - A (encoder): `#1 vs #2` at each DR level; delta in roll/pitch/vx SS error; variance reduction under `hard`.
  - B (IPO): `#1 vs #3` at each DR level; constraint violation rates logged for both (env still computes constraints for #1; #3 reports from offline re-scoring).
  - Algorithm effect: `#3 vs #4` (TRPO vs PPO, both encoder+no-IPO), `#1 vs #5` (full vs naive).

## Acceptance Criteria

1. All 5 variants reach `model_4999.pt` without crashes.
2. `eval_dr/summary_*.png` and `enhanced_summary.json` exist for every variant.
3. Claim-specific reports produced and committed to `logs/rsl_rl/fulldof_albc/ablation_sweep/`.
4. For each newly-introduced env/runner config, a commit in the main branch documenting the variant.

## Risks and Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| #4 PPO+ActorCriticEncoder kwargs mismatch | Medium | Smoke test before full run; fallback to `EncoderOnPolicyRunner` subclass |
| `ConstraintEncoderRunner` auto-sync overrides `num_constraints=0` | High, known | Variant #3 uses empty `constraints.terms=[]` env config |
| #5 PurePPO fails to learn under hard DR + no-encoder | Medium | Accept as finding: reinforces need for encoder + IPO |
| Baseline (#1) chosen run has regressions in a DR level specific axis | Medium | Record selection rationale and per-axis breakdown; re-discuss if selected baseline has severe asymmetry |
| GPU 0 OOM at num_envs=2048 | Low | Fall back to num_envs=1024 for the affected variant; document |

## Lifecycle

- All new training logs written to `/workspace/isaaclab/logs/rsl_rl/fulldof_albc/` directly (main repo, source of truth).
- No worktree needed for these variants; env changes are narrow enough to keep in main.
- After the last phase, delete `/workspace/run_r13a_hist.sh` and any stale worktrees from the previous hist sweep (`/workspace/isaaclab-r13a_*/`).
- Commit the new env/runner cfgs plus the `__init__.py` task registrations in a single, well-scoped PR.

## Changelog Entry (to add on completion)

```
2026-04-21 Ablation & baseline sweep design
  - Baseline candidates: 4-way eval_dr comparison → r13_A initial pick (user analysis)
  - Baseline challenger: hist5_act3 + encoder_latent_dim=16 (bottleneck hypothesis test)
  - Baseline matrix: main(1), noenc(2), nocstr(3), ppoenc(4), pureppo(5)
  - New tasks: Isaac-FullDOF-TRPO-NoIPO-v0, Isaac-FullDOF-PPO-Enc-v0
  - ALBCNoConstraintEnvCfg for clean IPO ablation
  - Sequential GPU 0 schedule, single seed, 5000 iter each
```
