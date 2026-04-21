# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## Historical changelogs

- [Round 8: Error-Gated Integration](docs/hero/experiments/round8_gated_integral.md) (BEST POLICY)
- [Round 7: Integral Observation + EpsSmooth](docs/hero/experiments/round7_integral_obs.md)
- [Round 6: Axis-Specific Shape Calibration](docs/hero/experiments/round6_axis_calibration.md)
- [Round 5: Constraint Budget Tuning](docs/hero/experiments/round5_constraint_tuning.md)
- [Round 4: Saturating Penalty](docs/hero/experiments/round4_saturating_penalty.md)
- [Round 3: SS Error Structural Fixes](docs/hero/experiments/round3_ss_structural.md)
- [Round 2: PerDimEnt Validation](docs/hero/experiments/round2_perdiment_validation.md)
- [Round 1: Per-Dim Noise Comparison](docs/hero/experiments/round1_noise_comparison.md)
- [Pre-Round Infrastructure](docs/hero/experiments/pre_round_infrastructure.md) (2026-04-04 ~ 2026-04-13)
- [Full ALBC early development](docs/hero/changelog_full_albc_early.md) (2026-03-31 ~ 2026-04-02)
- [Constrained ALBC development](docs/hero/changelog_constrained_albc.md) (2026-03-27 ~ 2026-03-31)
- [Legacy development](docs/hero/changelog_legacy.md) (2026-03-05 ~ 2026-03-26)
- [Encoder ablation study](docs/hero/experiments/encoder_ablation.md) (Steps 0-19)

---

## [2026-04-21] Ablation & Baseline Sweep — Spec, Plan, Baseline Selection, Challenger Launch

### Context

Main method (`encoder + IPO + TRPO`) needs controlled comparisons to substantiate claims:
- **Claim A (primary)**: encoder contributes to DR adaptation
- **Claim B (secondary)**: IPO outperforms reward-only shaping

Five-variant matrix designed: `main(1) / noenc(2) / nocstr(3) / ppoenc(4) / pureppo(5)`. Reward / DR / DORAEMON held constant, single seed (`seed=30`), sequential on GPU 0. Student-policy training is live on GPU 1 throughout; any operation touching main repo must be safe against concurrent imports.

### Experiments

**4-way baseline candidate analysis (eval_dr + eval_dr_switching, hard-DR focus).** Candidates: r13_A, hist5, hist10, hist5_act3 (all `encoder_latent_dim=9`). layernorm excluded (reward -33%, pos_drift +3278%).

Per-axis ss_error at hard-DR:

| Run | roll | pitch | vx | vy | vz | yaw | Heavy-tail (CV>=150%) |
|---|---|---|---|---|---|---|---|
| r13_A | 1.08° | **0.28°** | **0.004** | 0.006 | 0.018 | **0.002** | vz 231%, roll 184% |
| hist5 | 1.25° | 0.31° | 0.008 | 0.008 | 0.015 | 0.003 | roll 220%, vx 241%, vz 230% |
| hist10 | **1.04°** | 0.35° | 0.008 | 0.009 | **0.007** | 0.004 | vx 193% (vz 69%!) |
| hist5_act3 | 1.09° | 0.38° | 0.007 | **0.006** | 0.015 | 0.003 | roll 244%, vx 292%, yaw 183% |

Switching seg1-9 peak_roll / ss_roll / pos_drift: r13_A wins peak (7.63°), hist5_act3 wins ss_roll (0.59°), hist10 wins pos_drift (0.070).

Finding: **no hist variant decisively beats r13_A**; hist10 only wins vz (Markovian-weakest axis), other axes tied or slightly worse.

**Three structural explanations considered for hist underperformance:**
1. Encoder bottleneck: obs 87→178 squeezed through fixed z∈R^9 latent.
2. Asymmetric critic leakage: critic reads privileged 23D directly, so hist's role (recovering hidden state) is marginal.
3. Markovian dynamics: quat/omega/linvel already sufficient; hist marginal value saturates early.

**Falsifiable predictions**: latent_dim=9→16 unmasks bottleneck → hist variants should win; no change → critic leakage / Markovian dominance.

**Runtime risk check for Phase 2 / Phase 3 (code-level verification).** Before committing to PPO+Encoder and TRPO-NoIPO variants:
- `ConstraintTRPO`: `num_constraints > 0` guards at lines 241, 262, 277, 593 — cleanly skips cost-critic paths when K=0.
- `ActorCriticEncoder.__init__`: `num_constraints > 0` guard at line 191 creates no cost_critic; `load_state_dict` guarded at line 287.
- `ConstraintEncoderRunner`: `getattr(self.alg, "num_constraints", 0) > 0` at line 121.
- `PolicyBase` / `ActorCriticEncoder` implement all PPO-required methods: `act, evaluate, act_inference, get_actions_log_prob, update_normalization, action_mean, action_std, entropy, is_recurrent=False`. `**_kwargs` swallows PPO's extra `masks`/`hidden_state` args.
- `OnPolicyRunner` uses `eval(class_name)` scope which finds `FullDOFActorCriticEncoder` via the module-level injection in `rsl_rl_ppo_cfg.py:22`.

Result: both variants should train without crashes; smoke test deferred until GPU 0 frees.

**Challenger smoke test (Phase 0.6 Task 0.6.2 Step 5)**: 5 iters with `Isaac-FullDOF-TRPO-ChallengerEnc16-v0` + num_envs=64 completed in 7.08s. Reward components, encoder metrics, obs shape=121 all correct.

**Challenger training launched (Phase 0.6 Task 0.6.3)**: `challenger_hist5_act3_enc16`, num_envs=2048, max_iterations=5000, GPU 0, seed=30. Background run; monitor set to fire on every 500-iter milestone + any crash signature. ETA ~5 hr.

**Challenger env drift discovered and corrected (hotfix, aborted first launch at ~iter 4)**. User challenged "is it actually identical to hist5_act3 except encoder_latent_dim?" — answer was No. Initial `ALBCChallengerEnc16EnvCfg` inherited from current main, which has drifted post-hist5_act3 via r14 across 20+ fields:

- `ou_enable` (False → True), `ou_sigma` (0.05 → 0.10) — r14 mid-episode OU drift doubled
- 18 HardDR ranges widened 1.5-3x: `added_mass_scale` (0.5,1.5)→(0.3,1.8), `inertia_scale` (0.4,2.0)→(0.3,3.0), `payload_mass_range` (0,3)→(0,5), `thrust_coefficient_scale` (0.7,1.3)→(0.3,1.5), `time_constant_scale` (0.7,1.3)→(0.3,2.0), `ocean_current_strength_range` (0,1)→(0,2), `joint_stiffness_range` (30,150)→(20,200), etc.
- `action_latency_range` added in r14 HardDR (0,6); field absent pre-r14 → effective (0,0)
- `doraemon.step_interval` (250 → 500)
- `save_interval` (50 → 100, runner-side)

Root cause: hist5_act3 was trained in a worktree that was deleted per lifecycle policy; only `params/env.yaml` snapshot survived. Inheriting from current main picked up ALL post-hist5_act3 changes. Under this drift, Phase 0.7 would have compared challenger-with-r14 against r13_A-no-r14 — not a latent_dim test but a 20-variable confound.

**Fix**: rewrote `config_challenger_enc16.py` to reconstruct hist5_act3's exact env — override `ou_enable/ou_sigma`, pair a new `ChallengerHardDomainRandomizationCfg` with all 18 r13-era HardDR ranges, override `doraemon.step_interval=250` and `save_interval=50`. Verified via yaml diff: smoke-run env.yaml has 0 behavioral diffs vs hist5_act3 (only action_latency_range field syntactic presence + CLI num_envs); agent.yaml diff = exactly `encoder_latent_dim: 9 → 16` (intended single variable) + CLI max_iterations. Relaunched.

### Decisions

- **Baseline = r13_A (initial pick)** from 4-way analysis. Rationale: most balanced (pitch/vx/yaw/switching peak_roll 1st or tied-1st); hist variants only win narrow axes while regressing CV. Alternatives (hist10 vz breakthrough, hist5_act3 switching ss_roll) rejected because gains are axis-specific and aggregate score does not favor them.

- **Add Phase 0.6 baseline challenger** before committing to variants. Challenger = `hist5_act3 + encoder_latent_dim=16`, doubling latent capacity (78% more). Rationale: the three competing explanations for hist underperformance are entangled; one challenger run falsifies the bottleneck hypothesis cheaply. If challenger wins aggregate score → baseline + canonical env cfg switch. Else r13_A stays.

- **Subclass approach over main cfg editing for challenger** (Phase 0.6.2 re-plan). Rationale: student policy training is live on GPU 1 and reads `ALBCEnvCfg` at import; in-place edits to main `config.py` carry non-zero risk if reset/eval reconstructs env and reads new config. New isolated task `Isaac-FullDOF-TRPO-ChallengerEnc16-v0` with `ALBCChallengerEnc16EnvCfg` + `FullDOFTRPOChallengerEnc16RunnerCfg` keeps main untouched. Same pattern applied to variants #3/#4 via `ALBCNoConstraintEnvCfg`.

- **No editable install swap during sweep** (carried forward from earlier session). Student-policy training must not be disrupted. All work from `/workspace/isaaclab` main repo only.

- **Single seed (`seed=30`) not multi-seed.** Rationale: user prioritizes faster iteration over statistical rigor for this claim set; published work will acknowledge this limitation. Multi-seed defensible as future statistical-rigor pass.

- **Reward kept constant across all variants** (no "replacement penalties" for no-constraint variants). Rationale: matches literature convention (CPO, IPO, SafeExploration benchmarks); adding penalties would conflate constraint effect with reward-shaping effect.

- **Phase execution order: 0 → 0.6 → 0.7 → 0.5 → 1–4 → 5.** Rationale: Phase 0.5 sanity runs (500-iter pre-flight for variants #2 and #5) must use canonical env cfg locked at Phase 0.7 — otherwise they test the wrong thing.

- **No new worktrees for experiments (policy going forward).** Rationale: worktree-based experiments (April 2026 hist/layernorm sweep) were deleted post-training per lifecycle policy, leaving only `params/env.yaml` snapshots. When later work attempted to "reproduce hist5_act3 + latent=16", main config had drifted via r14 and the subclass didn't override the drift — creating a 20-variable confound. Saved as `feedback_no_worktree_experiments` memory. Going forward: subclass-based env/runner cfg isolation in main repo only; subclass must override every field that differs from the intended reference run (verify via yaml diff before launch); worktree only acceptable for spec/plan writing, never for training.

- **Subclass must full-diff against reference, not just override intended axis.** Rationale: the challenger drift bug happened because only 3 fields (hist_len, hist_action_len, observation_space) were overridden; the remaining main drift (OU, HardDR widening, latency, step_interval) leaked through silently. Going forward, for any "reproduce historical run X with one change" experiment: parse the reference `params/env.yaml`, run a programmatic full-field diff against the subclass's effective cfg, and assert diff count equals the intended variable count before launching.

### Open Questions

- Does challenger win Phase 0.7 head-to-head vs r13_A? Answer determines canonical env cfg and whether latent=9 vs 16 is the correct baseline for variants.
- If challenger loses: does symmetric-critic follow-up ablation belong in this sweep or a later one? Per spec, deferred — but flag if Phase 0.7 result strongly implicates critic leakage.
- PPO + ActorCriticEncoder runtime compatibility verified by code inspection only; actual train loop interaction may still surface issues (e.g., `hidden_state=None` handling inside PolicyBase). Phase 0.5 smoke will confirm.
- Student-policy training ETA on GPU 1 is not tracked by Claude; if it finishes mid-sweep, later phases could parallelize on GPU 1 for throughput.

---

## [2026-04-21] r13_A Ablation Sweep — hist_len / hist_action_len / LayerNorm

### Context

Four single-variable branches forked from `r13_A` baseline (commit `bafe23f4`) to probe whether adding proprioceptive history or swapping the actor obs normalizer improves command tracking and DR-switching robustness. Each branch modifies exactly one config axis; everything else (reward, constraints, encoder latent=9, DORAEMON, HardDR) held fixed. All four trained 5000 iter, num_envs=4096, seed=42, in isolated worktrees (`/workspace/isaaclab-r13a_*`).

### Experiment-specific diffs vs `r13_A` baseline (documented before worktree removal)

- **r13a_hist5** (commit `a557d4e1`): `hist_len: 3 → 5`. `policy_obs_dim: 87 → 113`. Updated `_OBS_NOISE_STD` and `_OBS_BIAS_MAG` tuples (joint hist 12→20, body hist 27→45). `hist_stride=3`, `hist_action_len=2` unchanged.
- **r13a_hist10** (commit `5a26bf91`): `hist_len: 3 → 10`. `policy_obs_dim: 87 → 178`. Same obs-tuple expansion (joint hist 12→40, body hist 27→90).
- **r13a_hist5_act3** (commit `4063466b`): `hist_len: 3 → 5` + `hist_action_len: 2 → 3`. `policy_obs_dim: 87 → 121`. Joint hist 20, body hist 45, action hist 24.
- **r13a_layernorm** (commit `962cb9af`): in `encoder/actor_critic_encoder.py`, actor obs normalizer swapped `EmpiricalNormalization(num_actor_obs_norm) → nn.LayerNorm(num_actor_obs_norm)`. Also guarded `.update()` call with `hasattr` since LayerNorm has no `update`. `policy_obs_dim` unchanged at 87.

### Experiments (5 runs analyzed: r13_A baseline + 4 ablations)

eval_dr 4 DR levels (none/soft/medium/hard, num_envs=64, ckpt=model_4999) — per-axis `ss_error` at **hard DR**:

| Run | roll° | pitch° | vx m/s | vy m/s | vz m/s | yaw rad/s | CV heavy-tail |
|-----|-------|--------|--------|--------|--------|-----------|----------------|
| r13_A | 1.08 | **0.28** | **0.004** | 0.006 | 0.018 | **0.002** | vz 231%, roll 184% |
| hist5 | 1.25 | 0.31 | 0.008 | 0.008 | 0.015 | 0.003 | vx 241%, vz 230% |
| hist10 | **1.04** | 0.35 | 0.008 | 0.009 | **0.007** | 0.004 | vz 69% (!), vx 193% |
| layernorm | 1.53 | 1.04 | 0.040 | 0.035 | 0.060 | 0.037 | broad failure |
| hist5_act3 | 1.09 | 0.38 | 0.007 | **0.006** | 0.015 | 0.003 | vx 292%, yaw 183% |

- Survival=100% all runs / all DR — no run destabilizes.
- **hist10 vz breakthrough**: ss_error 0.018 → 0.007, CV 231% → 69%. Only hist10 wins an axis clearly. Matches the hypothesis that vz is the least Markovian axis (buoyancy + added-mass dynamics are slow), so history extension helps.
- **layernorm fails across the board**: pitch 4x worse (0.28 → 1.04), all linear velocities 5-10x worse, yaw 20x worse. Not marginal — structural.

eval_dr_switching (zero cmd, 5s x 10 segs with DR re-sample @ seg 1-9, same seed):

| Run | seg0 peak_roll (PID-neutral) | seg1-9 hard peak_roll | seg1-9 hard ss_roll | seg1-9 hard pos_drift |
|-----|-----|-----|-----|-----|
| r13_A | 14.8° | **7.63°** | 0.67° | 0.085 m |
| hist5 | **13.3°** | 8.70° | 1.28° | 0.121 m |
| hist10 | 15.4° | 8.60° | 0.68° | **0.070 m** |
| layernorm | 20.4° (worst) | 8.75° | 1.07° | 0.287 m (worst) |
| hist5_act3 | 14.8° | 7.72° | **0.59°** | 0.087 m |

- seg0 is PID-gain-neutral (common upright init): layernorm alone stands out as worst → policy failure, not controller artifact.
- seg1-9 hard vs none ratio varies 7-36x across runs; none-level values are too close to zero for ratios to be diagnostic.

### Decisions

- **r13_A remains the baseline**. None of the ablations unambiguously beats it. pitch/yaw/vx stay with r13_A; hist10 wins vz+pos_drift; hist5_act3 wins vy+settled roll. No single challenger dominates.
- **Reject LayerNorm actor obs normalizer**. Failure is structural across all axes, seg0 included (so not a PID-gain artifact). EmpiricalNormalization's running mean/var is load-bearing for this policy — LayerNorm's per-sample normalization breaks the conditioning the encoder + actor rely on.
- **Hist extension has marginal returns under the current architecture**. Explanation confirmed from config read + code audit:
  1. `encoder_latent_dim=9` is a hard bottleneck. Extended history (87→178D) still squeezes through z of 9-dim. Additional input ends up as either noise or ignored.
  2. Critic is **asymmetric** (receives privileged 23D directly), so actor does not have to reconstruct hidden state — history's DR-identification utility is already provided by privileged info at training time. This is the opposite of HORA's original setup (symmetric critic), which is why HORA-style history extension typically shows bigger gains in the literature.
  3. IMU-dominant axes (roll/pitch) are near-Markovian; only vz (slow hydrodynamic axis) shows meaningful gain from history.
- **Worktrees removed, branches kept**. Each ablation's single experiment commit is preserved on its branch (`r13a_hist5`, `r13a_hist10`, `r13a_hist5_act3`, `r13a_layernorm`) for future cherry-pick; only the filesystem worktrees under `/workspace/isaaclab-r13a_*` are deleted. Log directories already migrated into `/workspace/isaaclab/logs/rsl_rl/fulldof_albc/` (hist5_act3 moved in this session from `full_dof_trpo/` subproject; the rest were already there).
- **Pip editable install restored to main** `/workspace/isaaclab/source/*` after the hist5_act3 eval run temporarily pointed it at the worktree (`feedback_editable_install_namespace` rule triggered: a partial reinstall from the worktree collided with an in-flight student training on GPU1 which silently died during namespace swap).

### Open Questions

- **Would `encoder_latent_dim=16` unlock hist10's gains on axes other than vz?** The latent bottleneck hypothesis predicts yes. Cheap to test: clone r13a_hist10 branch, bump latent_dim, rerun.
- **Would a symmetric critic (privileged removed) flip the result?** Expensive but the critic-info-leakage hypothesis says history extension would start paying off in axes beyond vz.
- **DR-switching seg1-9 peak_roll depends on cascade PID `Kp_pos/Kp_yaw=0.5`**. Different policies have different closed-loop dynamics; the seg0 metric sidesteps this, but absolute seg1-9 comparison across runs is only ordinally trustworthy. A `Kp_*` sweep per policy would quantify the artifact, but cost is ~5× eval time — deferred.


---

## [2026-04-21] R13 A/B Deep Analysis + DR-Switching Eval Mode + R14 Final Design

### Context

R13 completed: pure encoder latent dim ablation (the *only* config diff). r13_A=latent 9, r13_B=latent 16. Both trained from r12_baseline equivalent + added HardDR `ocean_current_strength_range=(0,1)`. Goal: decide r14 baseline, diagnose remaining residual issues (roll oscillation, yaw overshoot).

### Experiments

- **Added eval_dr_switching.py**: new eval mode. Fixed zero command (xyz=0, rpy=0) with cascade PID outer loop, DR switches per 5s segment (10 segs total). Tests disturbance-rejection robustness under identical DR sequence seed. Extends env randomization mid-episode via `randomize_physics_mid_episode()` (new method in albc_env.py). Also added analyze_dr_switching.py for cross-run comparison.
- **Cascade PID ported to play.py**: outer loop position/yaw error → vel_cmd/yaw_rate_cmd, inner loop policy tracks. Kp_pos/kp_yaw tunable CLI args.
- **Kp_pos/kp_yaw doubled from 0.5→1.0**: Y-bias r13_A improved 2.3→2.3 mm (unchanged, already tiny), r13_B 6-11mm (still worst — inherent payload asymmetry on gripper side).
- **r13_A vs r13_B comparison** (both eval_dr and eval_dr_switching):
  - r13_A wins: pitch SS ~2x better, yaw SS ~2x better, Y-bias 2-5mm vs 6-11mm
  - r13_B wins: roll SS edge, roll heavy-tail %>10° 28%→19% at hard, yaw peak 1.61°→0.83°, pos p99 0.57→0.42m
  - Env-level agreement (hard): spearman ρ=+0.79, same worst env (env 56) — DR seed reproducible
- **Per-dim log_std measurement** (r13_A final ckpt): arm 0.15/0.22, thrusters 0.22-0.31. All **4-6x above min_std floor (0.05)**. Policy voluntarily maintains high noise — min_std is NOT binding.
- **Roll oscillation frequency** (FFT on eval_dr_switching DR=none): r13_A 0.68 Hz @ 5.4 magnitude, r13_B 0.87 Hz @ 112 magnitude. Same TAM-driven limit cycle mechanism but latent=16 has 20x stronger amplitude (wider noise channel).
- **Yaw overshoot history scan across 14 runs**:
  - r11_emabias (latent=9, NO ocean HardDR, EMA bias k=-2): yaw_os **11.1%** — best
  - r11_encdim16 (latent=16, NO ocean HardDR): 11.4%
  - r13_B (latent=16, +ocean HardDR): 12.5%
  - r13_A (latent=9, +ocean HardDR): **17.6%** — worst latent=9 with EMA bias
  - r12_latent12 (latent=12, NO EMA bias, NO ocean): 20.2% — **latent=12 is NOT a middle ground**, worse than latent=9

### Decisions

- **r14 = r13_B baseline (latent=16)**. Reason: production-readiness prioritizes controlled peaks/heavy-tail over last-mm SS. Latent=16 consistently absorbs transient disturbances better; r13_A's SS/Y-bias advantages are smaller gains than r13_B's peak/tail advantages. User-approved "Option B".
- **entropy_coef 0.003 → 0.001**. Reason: final thruster std 0.22-0.34 (4-6x above floor) drives roll limit cycle. min_std floor is NOT constraining. Lower entropy coefficient allows actor to shrink toward 0.10-0.15 std. 20k iteration horizon absorbs aggressive coef without exploration collapse.
- **DORAEMON step_interval 250 → 500**. Reason: 4x longer training means 80 DR updates at current cadence — too frequent. 500 gives 40 updates, stable curriculum progression.
- **HardDR aggressive widening (17 params, 1.5-3x wider)**. Reason: r13_B achieves survival 100% + clean SS on all DR levels — under-utilizing policy capacity. User: "DR 범위에서도 너무 잘되잖아, 최대한 많은 환경을 보여주는거지". Eval filters extreme tail (use none/soft/medium or rescale `DR_SCALE` to `{0, 0.2, 0.5, 0.8}`).
- **Non-DORAEMON DR expanded**: observation `noise_scale` widened across all channels incl. angular velocity; OU current `delta_scale` 0.1→0.2.
- **Action latency DR ported from hero_agent**. Range (0, 6) physics steps = 0-30ms delay. Rationale: real-hardware communication lag; directly attacks yaw overshoot (delayed feedback → policy must learn predictive control). Implementation: ~40 lines in albc_env.py via `_action_history` buffer per hero_agent pattern.
- **payload_cog_offset_xy_radius 0.08 retained** (user choice). Y-bias fix deferred to r15 if needed.
- **Compute scale: num_envs 2048→4096, max_iterations 5000→20000, save_interval 50→100**. Single run, ~8x sample budget, ~30-40h wall clock.

### Open Questions

- Will 20k iter + entropy_coef=0.001 actually reduce thruster std below 0.15? Monitor final log_std per-dim and roll limit cycle amplitude in eval_dr_switching.

---

## [2026-04-21] Student Policy (TCN/GRU BC from r13_A Teacher) — Architecture Ablation

### Context

Building asymmetric student encoder via behavior cloning from frozen r13_A teacher (latent_dim=9): teacher has privileged->z->actor pipeline, student learns proprio_history->z_hat via MSE on both z_hat vs teacher z and teacher_actor(o, z_hat) vs teacher a. Dropped DORAEMON, forced HardDR during BC rollouts (per RMA Phase 2 protocol). Two encoder variants compared: TCN (window-based conv) and GRU (recurrent).

### Experiments

- **TCN history mismatch (initial design flaw)**: first TCN runs used tcn_history=50 (1.0 s at 50 Hz), 50x longer than teacher's embedded 180 ms proprio history (stride=3 x 3 steps, built into o_t's 87D). Fixed to tcn_history=9 stride=1 with kernels (3,3,3) → receptive field matches teacher's embedding span. ~1h wasted before correction.
- **Killed GRU run `2026-04-21_06-38-03_student_gru`** (iter 599, terminated during panic restart): loss_total v0=0.295 → Q1=0.173, flat thereafter (Q4=0.170). loss_latent v0=0.221 → Q1=0.114 → Q4=0.113 (zero improvement after Q1). grad_norm collapsed 0.79 → 0.05 by Q4 (optimizer effectively dead).
- **Root-cause diagnosis of GRU plateau** (code audit, not speculation):
  1. `runner.py:_compute_loss_gru` called `self.student(batch.obs_seq, hidden=None)` — every minibatch re-initialized GRU hidden to zero, capping effective context at 24 steps (0.48 s). DR params are ~static per episode, so 0.48 s is insufficient identification window.
  2. Student encoder received raw 87D obs directly. Teacher's actor consumes `actor_obs_normalizer(o)` (EmpiricalNorm trained over 6.5e8 steps, mean in [-2.15, 3.32], std in [0.006, 1.59]). Raw scale mismatch ~10^2 between e.g. angular velocity and integral terms degrades initial gradient signal.
  3. Collection-time `self.gru_hidden` was zeroed on done but never forwarded through student — vestigial variable, not functional.
- **GRU v2 `student_gru_h9v2` with fixes applied**: (a) obs normalized via teacher's frozen `actor_obs_normalizer`, (b) `train_hidden` snapshotted at rollout start and threaded per-env through BPTT chunks with done-env reset, re-computed via no-grad forward at iter-end. Result: v0=0.199 → Q1=0.114 → **Q4=0.113 (identical asymptote to killed run)**. Only effect: 10% faster initial descent (v0 0.220→0.199).
- **TCN H=9 `student_tcn_h9`** (no fixes, baseline for comparison): v0=0.169 → Q1=0.059 → **Q4=0.034** (loss_latent Q4=0.026, ~4.3x lower than GRU). grad_norm Q4=0.178 (healthy).
- **DR-switching eval (64 envs, zero cmd, 5s/seg x 10 segs, hard DR re-sampled each seg)**:

| DR level | axis | TCN H=9 | GRU v2 | ratio GRU/TCN |
|---|---|---|---|---|
| none | roll/pitch | 0.797° | 0.750° | 0.94x |
| none | vz | 0.0101 | 0.0252 | **2.50x** |
| soft | vz | 0.0099 | 0.0531 | **5.38x** |
| medium | roll/pitch | 1.156° | 0.987° | 0.85x |
| medium | vz | 0.0463 | 0.1074 | 2.32x |
| hard | roll/pitch | 4.785° | 4.298° | **0.90x** |
| hard | vx | 0.0534 | 0.0450 | 0.84x |
| hard | vz | 0.240 | 0.300 | 1.25x |
| hard | yaw | 0.0308 | 0.0385 | 1.25x |

### Decisions

- **Training loss is not a reliable proxy for eval quality in this BC setup**. GRU v2 loss_latent=0.113 (4.3x worse than TCN's 0.026) produced eval metrics COMPARABLE to TCN, with GRU actually 5-18% better on roll/pitch at all DR levels. Teacher's actor downstream is robust to imperfect z_hat within a broad manifold — the latent->action mapping absorbs substantial z-noise without behavior degradation. Conclusion: don't chase training loss floor; evaluate final policy.
- **TCN H=9 is the preferred student for this teacher**. Reason: matches teacher's embedded 180 ms history exactly, and vz tracking is substantially better (GRU 1.25-5.4x worse at vz across all DR). TCN also reaches higher training stability (grad_norm 0.18 healthy vs GRU 0.05 collapsed).
- **GRU architectural ceiling recognized, not fixed**. obs_norm + hidden threading did nothing measurable on the asymptote. The plateau at loss_latent=0.113 is the current GRU arch's capacity limit under 24-step BPTT + single-layer 128-hidden. Deeper architecture exploration (bigger hidden, more layers, richer head) deferred.
- **Fixes retained in code despite zero asymptote impact**: obs normalization is still theoretically correct and improves initial descent (-10% v0). Hidden threading still correctly implements RMA-canonical truncated BPTT vs the "simplicity" zero-init anti-pattern. Both are semantically right even if GRU architecture limits prevent them from mattering here.

### Open Questions

- Does the 0.026 TCN loss_latent represent observability ceiling (privileged->proprio mutual information limit) or also TCN capacity ceiling? Would deeper TCN / bigger head reach 0.01?
- **GRU vz failure mode**: which DR parameters is GRU's z_hat missing that TCN recovers? Likely heave-related (added mass, buoyancy delta). Per-dim z sensitivity sweep (`encoder_z_sweep.py`) on both students would localize the gap.
- Teacher r13_A on eval_dr_switching for apples-to-apples baseline: neither student reaches 4.8° @ hard but teacher reference unmeasured. Without it, "how far from teacher" is unquantified.
- Will aggressive DR widening cause DORAEMON to stall at easy difficulty? Mitigation: performance_lb=90 automatic stop prevents divergence; worst case = same DR reach as r13.
- Should action_latency DR be added to DORAEMON curriculum (vs fixed-range)? Initially fixed; revisit if curriculum makes it too dominant.
- Is the mid-episode DR switch (new `randomize_physics_mid_episode` method) fully equivalent to episode-boundary DR for DORAEMON's statistics? Eval-only usage so OK for now, but if used in training, needs curriculum integration.

### Root-Cause Summary (documented for future reference)

- **Roll oscillation at DR=none**: NOT min_std floor. entropy_coef too high keeps thruster std at 0.25; via TAM roll arm (0.007m, 20x weaker than pitch) this noise manifests as 0.68-0.87 Hz limit cycle. Fix: lower entropy_coef.
- **Yaw overshoot r13_A 17.6% (vs r11_emabias 11.1%)**: caused by cumulative addition of `ocean_current_strength_range=(0,1)` HardDR between r12 and r13. Latent=9 cannot absorb this transient as well as latent=16. Choosing latent=16 (r14 = r13_B) directly addresses this.
- **latent=12 worse than latent=9 for yaw transient** (r12_latent12 data). Dimension effect is non-linear. Between latent=9 and 16 is NOT a safe interpolation for all axes.

### Evening addendum: GRU deep-head / capacity sweep and LN(9) diagnostic

Follow-up on Decision #3 (GRU architectural ceiling deferred). Hypothesis tested: the loss_latent=0.113 floor is head capacity or output-normalization structure, not information ceiling.

**Diagnostic infrastructure added.** `scripts/analysis/diagnose_student_latent.py` logs per-step (l_hat, l_true) tensors for each DR level. `analyze_student_latent.py` decomposes MSE into bias² + variance and per-dim stds. Applied to TCN H=9 and GRU v2 ckpts.

**Per-dim std verification (from latent_log_*.npz, hard DR):**
- teacher per-dim std: `[0.47, 0.43, 0.42, 0.47, 0.46, 0.35, 0.18, 0.17, 0.29]`
- TCN student per-dim std: `[0.46, 0.46, 0.37, 0.51, 0.40, 0.31, 0.23, 0.17, 0.17]` — matches teacher in range
- GRU v2 student per-dim std: `[0.032, 0.031, 0.021, 0.019, 0.007, 0.014, 0.002, 0.003, 0.007]` — **10-100x smaller, near-constant**

**Bias²/variance breakdown (GRU v2):** none DR bias²=96% (output essentially constant), soft=54%, medium=14%, hard=5%. Confirms GRU output variance collapse, while TCN reaches bias²=8-51% (variance-dominated at higher DR).

**Structural asymmetry identified:** TCN head ends at `Linear(flat,128)→ELU→LN(128)→Linear(128,9)→softsign` (no LN on 9D output). GRU head ends at `Linear(h,64)→ELU→LN(64)→Linear(64,9)→LN(9)→softsign`. Per-sample LN across 9 dims normalizes to unit std across dims; when the 9 predictions are near-constant, this amounts to dividing by ~0 and destroys whatever signal remained.

**Failed runs (all hit identical loss_latent=0.11 plateau by iter ~30 regardless of capacity):**
- **v3 `student_gru_h9v3_deephead`** (deep head 128→64→9, hidden 128, BPTT 24): Q1 loss_latent=0.114, final ~0.108 (5% improvement over v2, within noise).
- **v4 `student_gru_h9v4_h256_bptt48`** (hidden 256, deep head, BPTT 48, num_envs 2048): killed at iter 399. Slope over iter 200-399: -0.006/1000iter. loss_latent = 0.112 (identical to v2/v3), grad_norm already dropped 0.46→0.04 by iter 100. **Capacity doubling + BPTT doubling = zero asymptote change.**
- **v5 `student_gru_h9v5_noLN_h256_bptt48`** (v4 config minus final LN(9)): iter 220 loss_latent=0.113, grad_norm sustained at 0.21-0.27 (3x higher than v4's 0.08 at same iter). LN removal does free gradient flow, but loss floor unmoved. Training in-flight to iter 999.

**Information-theoretic floor argument.** At hard DR, teacher latent per-dim variance sum ≈ 1.28 → mean-predictor baseline MSE = 0.143. Observed loss_latent = 0.11 is only 20% improvement over predicting the mean. The 9-step proprioceptive window appears to bottleneck on mutual information with DR params; no architectural tweak changes this budget.

### Decisions (addendum)

- **Architectural tuning for GRU is a dead-end.** TCN (no output LN, matches teacher std) and GRU (output LN, near-constant output) form a clean natural experiment. Even after matching structure (v5) and capacity-doubling (v4), the loss_latent floor at 0.11-0.12 is unchanged. The floor is an observability/data limit, not a capacity limit.
- **Don't chase loss_latent below 0.10 via architecture.** v2 eval results already proved loss is a weak proxy for downstream action quality. The 20% gap to mean-predictor is likely all the signal 9-step proprio obs can carry.
- **Next intervention direction shifts to information/distribution, not architecture.** Candidates: DAGGER (student rollouts re-labeled by teacher, fixes covariate shift), extended obs window (proprio >9 steps), or explicit DR-parameter regression auxiliary. Architecture sweeps discontinued.

### Open Questions (addendum)

- Is v5 final eval meaningfully better than v2 despite identical training loss? If output std is restored (no LN collapse) even at the same MSE, downstream actor may decode better. Verification pending v5 eval_dr + diagnostic npz.
- Why does TCN reach loss_latent=0.026 (4x below GRU floor) with same 87D obs? Differential architecture effect vs observability: TCN sees explicit 9-step window; GRU accumulates via hidden state. If hidden state initialization or regularization is destroying info, that's a separate fix path worth one experiment.
- Can teacher action noise floor (loss_action=0.057 ≈ teacher σ²=0.0625) be driven lower by noise-aware BC loss (KL against stochastic teacher distribution instead of L2 to mean)?

### Late-evening resolution: r14 DR mismatch was the root cause

**v5 eval confirmed LN(9) removal inert.** eval_dr HARD ss_error v5/v2/TCN: roll 2.28°/2.12°/2.30°, vz 0.304/0.292/0.247. All three within 5% — LN hypothesis finally rejected. Heavy-tail (tail 30%, peak>20°/0.5) also indistinguishable between v5 and v2. Training grad_norm did sustain 4x higher in v5 (0.22 vs 0.05) — gradient flow was restored but produced no downstream quality change.

**Root cause identified via pipeline audit.** Student training used `HardDomainRandomizationCfg` (imported in `student/runner.py`) that had been **widened 1.5–3x by commit `1e2d5771` (r14 prep)** on 2026-04-21 — after r13_A was trained (commit `f05ca6f5`, 2026-04-20). Teacher r13_A had never seen the expanded DR ranges: added_mass (0.5,1.5)→(0.3,1.8), inertia (0.4,2.0)→(0.3,3.0), payload (0.0,3.0)→(0.0,5.0), thrust_coef (0.7,1.3)→(0.3,1.5), plus new `action_latency_range`, `ou_enable=True`, `ou_sigma` doubled. **Every BC label from the teacher in those widened regions was OOD noise.** DORAEMON final Beta(1.93, 1.93) shows teacher was competent near the midpoint of its own training range — querying it in the r14 tails produced incoherent z_true and action targets, explaining the universal `loss_latent=0.11` plateau (≈ mean-predictor baseline).

**r14 removed, r13_A state exactly restored.** Reverted `config.py`, `albc_env.py`, `agents/rsl_rl_ppo_cfg.py` to commit `f05ca6f5` (sha256 byte-identical, verified). Deleted `scripts/launch_r14.sh`, `docs/superpowers/{plans,specs}/2026-04-21-r14-final-*.md`. Comment in `student/runner.py:40` updated to reference r13_A-era HardDR rather than "r14 aggressive". No other r14 refs remain outside one unrelated ablation file (`config_challenger_enc16.py`, another session).

**v6 `student_gru_h9v6_r13aDR_h256_bptt48` (launched 2026-04-21 20:28, GPU 1)**: same architecture as v5 (hidden 256, deep head 64, BPTT 48, no output LN), only variable changed is DR distribution (now exact r13_A HardDR). **Iter 15 loss_latent=0.0848 — already 25% below the 0.11 plateau floor that all of v2/v3/v4/v5 were stuck at for 1000 iters each.** Confirms DR mismatch was the dominant bottleneck, not architecture. Full training + eval pending.

**Separate eval-time fix.** `student/eval.py` now auto-infers `gru_hidden` and `gru_head_hidden` from checkpoint tensor shapes so eval scripts no longer require explicit CLI args for non-default architectures. Prevents v4/v5 style checkpoint-mismatch crashes.

### Decisions (late-evening)

- **Match student DR distribution to teacher's training distribution.** RMA Phase-2 canonical protocol. Rollouts beyond teacher competence produce OOD labels that look like noise to the BC loss, collapsing it toward the mean-predictor baseline regardless of architecture. Alternatives rejected: keeping r14 DR with DAGGER (doesn't fix OOD teacher); keeping r14 DR and re-training teacher (out of scope; r14 teacher doesn't exist).
- **Discontinue all architecture-only experiments for GRU.** v2–v5 swept head depth, hidden size (128→256), BPTT window (24→48), and output LN. None moved the asymptote by more than the noise floor. Demonstrated the plateau was an external (DR) constraint, not an internal (capacity) one. Future GRU changes only on top of DR-matched baseline.
- **Reject "switching-first" framing.** User observation: switching-eval errors are largely PID-tunable at deployment while eval_dr steady-state errors are fundamental to policy competence. TCN's 33% vz-heavy-tail advantage at HARD (20% vs 30% envs peak>0.5 m/s) remains the cleanest eval_dr signal; switching differences get lower weight.

### Open Questions (late-evening)

- Does v6 reach loss_latent ≈ TCN's 0.026, or will GRU plateau at an intermediate floor even with matched DR? Answer defines whether GRU has a residual architectural disadvantage vs TCN beyond the DR issue.
- eval_dr_fulldof.py defines "hard" via the system HardDomainRandomizationCfg at import time. Post-revert, "hard" now matches teacher's training range — so v6 "hard" eval is apples-to-apples with r13_A, but earlier v2/v5 "hard" eval used the wider r14 range. Retro-comparison tables across v2…v5 vs v6 hard metrics are not strictly apples-to-apples; need to note this when citing previous numbers.
- Should DORAEMON eventually be re-enabled for student rollouts (curriculum from soft to r13_A-hard)? Currently disabled because the intended reason (match teacher's final operating point) is now satisfied by fixed r13_A-HardDR; DORAEMON adds non-stationarity with unclear benefit for pure BC.

### Context

All R11 experiments completed. Initial R12 design naively stacked "all R11 winners"
(encdim16 + emabias) on r11_baseline. Session goal: validate whether this is truly
optimal via deep variance + mechanism analysis of R9~R11, then revise R12 if needed.
User also flagged confusion about `summary_*.png` plots: baseline runs show
sample-env line overlapping the mean line, while other runs show large divergence
in attitude (hard DR) and lin_vel (soft/medium DR).

### Experiments

**Variance ranking across r9~r11 (r11_yawratedot excluded as catastrophic):**

Env-to-env `ss_error_std` at hard DR - per-axis extremes:
- roll: min **r11_encdim16 (0.55)**, max r9_baseline (2.10), 3.8x ratio
- pitch: min **r11_encdim16 (0.24)**, max r9_symatt (0.64), 2.6x
- vx: min **r11_emabias (0.014)**, max r9_tightrates (0.057), 4.1x
- vy: min **r11_emabias (0.018)**, max r9_normval (0.044), 2.4x
- vz: min r9_baseline (0.077), max **r11_encdim16 (0.133)** - REVERSAL, 1.7x
- yaw: min **r11_emabias (0.0015)**, max r10_perflb_high (0.025), **16.8x**

Within-env `ss_jitter` at hard DR - per-axis extremes:
- roll/pitch/vx/vy: r9_tightrates minimum (rate constraint suppresses oscillation)
- vz: r9_normval max (value norm distorts critic targets for lin_vel)
- pitch: **r11_emabias max (0.083)** - bias_weights roll emphasis destabilizes pitch trim
- yaw: **r11_emabias min (0.0008)** - bias EMA directly dampens yaw oscillation

**encdim16 mechanism (TB evidence from run `2026-04-19_11-26-59`):**
- Policy/entropy: baseline 0.145 -> 0.282 (+94%). Larger latent preserves exploration.
- Reward/lin_vel: baseline 1.79 -> 1.12 (-37%). Encoder capacity shifts to attitude at
  lin_vel's expense. Directly explains vz SS +87% and vx SS +60% regressions.
- Track/att/roll_err_deg (training): 2.79 -> 2.50 (-10%), Train/mean_reward: 231.6 -> 209.7.
- Conclusion: encdim16 trades lin_vel tracking for attitude robustness. Real trade-off,
  not training variance.

**emabias mechanism (TB evidence from run `2026-04-19_15-55-14`):**
- Track/att/pitch_err_deg (training): 1.65 -> 1.89 (+14%). Pitch regression visible
  in training, not just eval.
- Reward/att_rp: 5.13 -> 4.83 (-6%). Bias term settled at -0.10.
- Root cause traced to `bias_weights=(1.5, 1, 1, 1, 1, 1)` with squared EMA: roll bias^2
  weighted 1.5x dominates sum. Roll cannot structurally improve much (TAM arm 0.007m
  vs pitch 0.145m, 20x authority gap), so policy optimizer finds indirect reduction
  via thrust-trim shifts, which offset pitch. Classic "balloon squeeze": penalizing
  a structurally-limited axis diverts optimization to collateral axes.

**Sample env plot divergence mechanism (eval_dr_fulldof.py:570-582 `_pick_sample_env`):**
- Sample env = median-attitude-error env per DR level per run. NOT fixed index 0,
  NOT random. Same env reused for attitude / lin_vel / yaw plots.
- Baseline runs: axes inter-correlated, so median-att env is typical in lin_vel/yaw.
  Sample approx mean.
- Intervention runs: policy develops axis-specific strengths. Median-att env can be
  outlier in lin_vel (att good but lin_vel bad). Sample line diverges from mean in
  lin_vel plot even though DR seed is identical.
- This is a **diagnostic signal**, not a bug: policy axis-decorrelation reveals
  itself as sample-vs-mean divergence. Quantifiable via per-axis CV comparison.

### Decisions

- **R12 adopts `k_bias=-1.0` (halved from r11_emabias -2.0)**, single-variable change
  vs initial R12 plan. Rationale: encdim16 already solves roll structurally (roll
  SS 0.900 -> 0.480 + std 1.74 -> 0.55), so heavy bias pressure on roll is redundant;
  halving preserves yaw gain (expected ~-27% still better than baseline -54%) while
  halving pitch regression (from +26% eval / +14% train toward ~+7%). Bias-term
  reward contribution drops -0.10 -> -0.05, still larger than smoothness -0.11, so
  emabias mechanism remains effective.
- **Rejected `k_bias=-2.0` stacking (initial R12 plan)**: measured pitch regression
  +124% std and +26% mean at hard DR on r11_emabias alone; combining with encdim16
  risks compounding without mitigation.
- **Rejected `bias_weights=(1.5, 1.5, 1, 1, 1, 1)`**: pitch squared-bias magnitude is
  ~10x smaller than roll (0.005^2 vs 0.016^2), so 1.5x weight keeps pitch term ~13%
  of roll contribution. Mechanism is gradient-dominance, not weight-lack.
- **Rejected `encdim16 alone`**: equals `r11_encdim16`, already evaluated. No new
  information gained.
- **Methodology captured in rules/skill**: `/workspace/.claude/rules/03-analysis-quality.md`
  added "Env-to-Env Variance Analysis" and "Sample Env Plot Divergence Explained"
  sections. `/workspace/.claude/skills/train-analyze/SKILL.md` added "eval_dr Env
  Variance Analysis" workflow. Future analyses MUST report `mean +/- std` across all
  four DR levels and six axes, and interpret sample-vs-mean divergence as a
  decorrelation diagnostic.

### Cleanup

- Removed 9 experiment worktrees (r9_baseline, r9_symatt, r9_tightrates, r10_perflb,
  r10_minstd, r11_baseline, r11_encdim16, r11_yawratedot, r11_emabias) and their
  branches, all clean (logs already migrated; cherry-picks captured in r12_baseline).
- Removed 15 per-experiment run scripts from `/workspace/`. Kept only
  `run_r12_baseline.sh` (active training).
- Archived 32K stub log from r11_baseline worktree (aborted 14s after successful
  r11_baseline run) to `logs/archive/rsl_rl/fulldof_albc/`.

### Open Questions

- **vz regression from encdim16 is unsolved**. Hard DR vz SS 0.030 -> 0.056 (+87%)
  in r11_encdim16, and env-std flipped (baseline 0.077 is now the MIN, encdim16
  0.133 is the MAX - only such reversal across all axes). Hypothesis: encoder
  capacity shifts to attitude at heave's expense. Partially addressed in afternoon
  by launching r12_latent12 (see below).
- R12 prediction (to validate when training completes): roll ~0.48 (encdim16),
  pitch ~0.33-0.35 (partial regression recovered), vz ~0.05 (encdim16 cost persists),
  yaw ~0.003 (half of r11_emabias win), vx net small improvement.
- Run `2026-04-20_13-07-07_r12_baseline` was launched with k_bias=-2.0 then killed
  at ~22 min after variance analysis identified the risk. Relaunched as
  `2026-04-20_13-24-19_r12_baseline` with k_bias=-1.0.

### Afternoon addendum: r8_gated comparison and r12_latent12 launch

**r8_gated (archived best pre-R9) vs r9~r11 hard DR:** Per-axis winners
- roll: r11_encdim16 **0.48** (r8_gated 0.86, -44%) -- real improvement beyond seed variance
- pitch: r9_tightrates 0.28 (r8_gated 0.32, -12%)
- vx: r11_emabias 0.010 (r8_gated 0.018, -42%)
- vy: r9_tightrates 0.012 (r8_gated 0.013, tied)
- vz: r9_tightrates/baseline 0.026 (r8_gated 0.040, -35%); r11_encdim16 0.056 REGRESSES
- yaw: **r8_gated 0.0017 still best**; r11_emabias 0.0022 closest challenger
- r9_baseline (same config as r8_gated) gave roll 1.09 vs r8_gated 0.86 = **+27% seed variance**,
  setting ~20% floor on what counts as a real improvement
- No single run dominates r8_gated on all six axes -> trade-off structure is fundamental

**r12_latent12 launched on idle GPU1** (branch `r12_latent12` commit `176e2d3f`).
Single variable change: `encoder_latent_dim 9 -> 12` (midpoint between r11_baseline=9
and r11_encdim16=16). No emabias -- isolates pure encoder capacity effect.
Completes 3-point sweep to identify whether vz regression is monotonic in
latent_dim or has a sweet spot at 12.

Decision tree after both R12 runs finish:
- latent=12 roll ~0.6 + vz ~0.045: midpoint compromise; combine with emabias for R13
- latent=12 roll ~0.48 + vz ~0.040: 12 is sweet spot; switch base to 12
- latent=12 roll ~0.85 (no improvement): capacity effect non-linear; R13 must use
  vz-specific intervention (not latent tuning)

Rejected alternatives for parallel GPU1 experiment:
- r12_bias2 (k_bias=-2.0 full emabias + encdim16): tests bias intensity but leaves
  vz regression unexplored -- less information gain
- r12_seed2 (R12 replicate): controls seed variance but generates no mechanism info
- r12_encdim_only: duplicate of r11_encdim16, zero information gain

Infrastructure note: fresh worktrees need `_isaac_sim` symlink to `/isaac-sim` AND
the `meshes/` directory symlinked to main repo (Agent.usd + configuration/*.usd
are gitignored). Took three relaunches to diagnose.

### Evening addendum: R12 sweep completion + 24-run retrospective + R13 parallel launch

**r12_latent12 completed** (run `2026-04-20_13-48-43_r12_latent12`, migrated to
`logs/rsl_rl/fulldof_albc/`). Hard DR: roll SS 0.791, pitch 0.441, vz 0.040,
yaw_ovr 20.2%. No heavy-tail (0% peak>20°). 3-point latent sweep (9/12/16)
result: **inconclusive**. 12 wins vz at low DR (absolute magnitude at noise floor,
<0.01 m/s) and soft/medium roll; 16 wins yaw overshoot consistently across DR
(7-11% vs 20-29%, 2-3x margin) AND hard-DR attitude (roll 0.48 vs 0.79). latent=12
is NOT a sweet spot. Also, r12_latent12 was built on r12_baseline base (confounded)
so the single data point cannot be read as a clean latent_dim sweep.

**Comprehensive 24-run retrospective** via composite rank (4 DR levels x 10
metrics = 40 rankings averaged per run, lower = better):
- rank 1: `r11_emabias` (5.90) -- latent=9 + k_bias=-2.0
- rank 2: `r11_encdim16` (7.10) -- latent=16 + no emabias
- rank 3: `r12_latent12` (7.42) -- latent=12 + weak emabias (confounded)
- rank 4: `r9_tightrates` (7.55)
- rank 5: `r10_thr_minstd` (7.92)
- rank 6: `r11_baseline` (8.38)
- **rank 7: `r12_baseline` (8.65)** -- latent=16 + k_bias=-1.0 (half strength) REGRESSED

Hard DR same ranking order. **r11_emabias is the demonstrably best run across
all 24 runs, not just R11.** Previously under-sold as "small DIV 4/24 -> 2/24
improvement" -- actual wins span roll (all 4 DR levels, -11 to -31%), hard yaw
(-55% ss_error, -88% std, -61% n_gt20), and n_gt20 across most axes.

**"Intervention causes divergence" hypothesis REJECTED by data.** User observed
large sample-vs-mean divergence in PNG plots for intervention runs but not
baselines. Refined DIV rate (rank<5% or >95%, or gap > 2xnoise_floor AND
|ratio-1|>0.3) per run:
- BASELINE (r8_gated, r9_baseline, r11_baseline, r12_baseline): 19/96 cells = **19.8%**
- INTERVENTION (8 runs): 43/192 = 22.4%
- INTERVENTION excluding `r11_yawratedot` (training-failed, yaw 100% peak>th): 32/168 = **19.0%**

One failed run (r11_yawratedot) was skewing the entire intervention group to look
divergence-heavy. Without it, baseline and intervention are statistically
indistinguishable. The "baseline looks clean" perception was driven by smaller
absolute scale (baselines have lower means so the identical divergence rate is
less visually striking).

**Config lineage discovery -- critical for R13:** Direct params diff between
r11_baseline, r11_encdim16, r11_emabias, r12_baseline revealed r12_baseline =
r11_baseline + (latent=9->16) + (emabias k_bias=0 -> **-1.0**, half of r11_emabias
-2.0). **This is the ONLY test of latent=16 + emabias combination to date**, and
it REGRESSED (hard roll 1.26 vs r11_encdim16 0.48, rank #7 vs #2). Failure
cause ambiguous: weakened k_bias vs. inherent latent=16 + emabias interaction.

### Decisions (continued)

- **R13_A launched on GPU0**: r11_emabias config (latent=9, k_bias=-2.0, full
  strength) + ocean_current DR DORAEMON-managed (nominal=0, Beta skewed to zero
  current, expands as curriculum advances). **Minimum change from rank-#1 run**:
  only ocean DR added. Rationale: preserves proven best config; measures pure
  ocean DR effect; low risk of regression.
- **R13_B launched on GPU1**: latent=16 + k_bias=-2.0 (full strength) + ocean
  DR. Clean ablation that r12_baseline didn't do (it used k_bias=-1.0). Decides
  whether r12_baseline failure was due to weak bias (then B should succeed) or
  inherent combination problem (then B replicates r12 regression). Either
  outcome produces actionable information.
- **Rejected R13 alternatives**:
  - `r11_baseline + latent=16 + ocean DR` (no emabias): duplicates r11_encdim16
    with only ocean DR added. Testable by applying r11_encdim16 to ocean scenario
    post hoc rather than burning a full run.
  - `r11_baseline + emabias + tightrates + latent=16 + ocean DR` (all-in):
    5-variable change, violates minimum-change; r12_baseline already showed this
    direction regresses.
  - Adding ocean DR to existing config without DORAEMON management: ocean
    current was already present at static [0, 0.5, 0.5, 0.25] m/s; unmanaged
    range made it a fixed-perturbation, not a curriculum variable.

### Open Questions (continued)

- Does latent=16 fundamentally conflict with emabias, or was r12_baseline's
  regression purely from k_bias=-1.0 halving? R13_B answers this directly.
- Does ocean DR DORAEMON curriculum converge within 5000 iters, or does it stay
  at nominal=0 end of Beta? Monitor `DORAEMON/ocean_current_strength_mean` in
  TB; if stays <0.3 by iter 3000, policy cannot handle any current -- different
  failure mode than sim only.
- Does the sample-mean divergence pattern in PNG plots disappear with better
  policies? Divergence rate 19% is axis-decorrelation artifact of
  `_pick_sample_env` (median-att env). Unresolved whether rank-#1 training
  actually reduces decorrelation.

### Cleanup done this session

- Migrated `r12_latent12` logs (713MB, 101 checkpoints) from
  `/workspace/isaaclab-r12latent12/` to
  `/workspace/isaaclab/logs/rsl_rl/fulldof_albc/2026-04-20_13-48-43_r12_latent12/`.
- Created R13 worktrees `/workspace/isaaclab-r13_A` (branch `r13_A`) and
  `/workspace/isaaclab-r13_B` (branch `r13_B`) off `r12_baseline` branch.
- Cherry-picks from r12_baseline to main still pending (ocean DR integration,
  r11/r12 code lineage) -- deferred until R13 results land, per
  `rules/02-operations.md` "Experiment Worktree Lifecycle".

### Methodology refinements

- `scripts/analysis/analyze_eval_dr.py` (added prior session) exercised extensively
  to compute heavy-tail (peak_max, %env peak>threshold) and sample-mean
  divergence (sample_ss vs mean_ss, rank%). Refined DIV criterion combines
  extreme rank with absolute gap > 2xnoise_floor to suppress false positives
  from low-magnitude axes (yaw 0.003 rad/s mean_ss has spurious high ratios).
- Composite 24-run ranking methodology: per-DR-level per-metric rank sum, then
  average across 4 DR x 10 metrics. Resistant to single-axis optimization
  bias. Use for future cross-run comparisons.

---

## [2026-04-19] R9 Partial Results + R10 Queued + R11 Designed

### Context

First two R9 runs finished (baseline, symatt). Evaluated both with `eval_dr_fulldof.py`, migrated logs/wandb to `/workspace/isaaclab/logs/rsl_rl/fulldof_albc/`, and designed two R10 experiments that auto-chain after r9_tightrates (GPU0) and r9_normval (GPU1) complete. Later in the session r9_tightrates finished, was analyzed, and four R11 experiments (r11_baseline + three feature ablations) were queued on top of a new accumulated-best baseline.

### Experiments

**r9_baseline** (`2026-04-18_21-27-44_r9_baseline`, iter 5000) — control, same config as R8-Gated:
- hard DR: roll SS=1.090 (+28% vs r8_gated 0.855), pitch SS=0.320 (=), vz SS=0.026 (-35%), vy SS=0.015, yaw SS=0.004 (+135%), OS 14.1% / 21.2%.
- Survival 100% all DR levels. Reward decayed 267 (iter 1000 peak) -> 216 (iter 5000), -19%.
- Seed variance bound: roll SS 0.855 -> 1.090 on identical config = ~30% run-to-run spread. Any R10 delta must beat ~30% to be real signal.

**r9_symatt** (`2026-04-18_21-43-13_r9_symatt`, iter 5000) — att_roll_weight 1.5 -> 1.0:
- vs r9_baseline: roll **jitter -33%**, roll SS -3%, roll OS -21%, vy SS +44% (roll-sway coupling regression), vz SS slightly worse but vz Jit -56%, yaw OS 21.2 -> 18.9%.
- Barrier↔reward first-differenced correlation 0.72 (was 0.35 in baseline) — softer reward makes constraint the dominant gradient.
- **Finding: reward weight controls oscillation amplitude (jitter), not SS floor.** Symmetric weighting eliminates the 1.5x competing signal without resolving the 20x TAM authority gap.

**r9_tightrates** (`2026-04-19_00-56-32_r9_tightrates`, iter 5000) — rp_rate 1.0->0.5, yaw_rate 0.7->0.55:
- hard DR vs r9_baseline: **roll SS 1.090 -> 0.829 (-24%, beats r8_gated 0.855)**, roll Jit 0.264 -> 0.130 (-51%), pitch SS -12%, pitch Jit -35%, pitch OS -27%, **yaw OS 21.2 -> 15.3% (-28%)**, yaw n>20 halved (25.5 -> 15.0). Only regression: vx SS 0.021 -> 0.026 (+24%, still small absolute).
- Open Question from morning entry resolved: constraint margins `rp_rate=9.17`, `yaw_rate=~10` at converged policy suggested tightening would only touch transients. In fact it also reduced the SS floor — the 30% seed-variance bound was exceeded. Constraint slack WAS permitting SS oscillation, not just overshoot.

**Per-env outlier analysis of r9_tightrates hard DR (new this session):**
- Per-env SS CV values: roll 2.18, pitch 1.47, vx 2.19, vy 1.92, vz 3.12, yaw 1.50. vz CV>3 means a low mean (0.026 m/s) hides a heavy tail of catastrophic envs.
- Top-6-worst-env overlap matrix: **roll ∩ pitch = 0** (completely disjoint outliers). Hypothesis "extreme DR combo fails everywhere" falsified. Axis-specific DR combos drive distinct failure modes.
- 8 envs fail in ≥2 axes: env 14 (roll +5° systematic + vz -0.5 m/s), env 23 (pitch +10° saturated), env 43 (vx +0.17 m/s x-offset), env 16 (yaw oscillatory). All show **systematic bias**, not oscillation. Per-step reward cannot see offsets smaller than its gradient scale (σ=0.10 rad / m/s).
- Physical arithmetic: 3 kg payload × 0.15 m CoG-xy radius = 4.5 Nm gravity torque, exceeding roll TAM authority = 4 × 50 N × 0.007 m = 1.4 Nm. Some DR combos are physically uncontrollable for roll.

**DORAEMON scope verification (this session):**
- User asked whether ocean current is DORAEMON-managed. Checked `doraemon.py:69-85`: 15-param list covers payload_mass, added_mass, damping, water_density, COG/COB offsets, inertia, body_mass only. **Ocean current is NOT DORAEMON-managed**.
- Also verified `eval_dr_fulldof.py:315-355` build_dr_config does not scale ocean_current across DR levels. All 4 levels share `max_velocity=(0.5, 0.5, 0.25)`. Hard DR initial spike (30° roll at t=0) is driven by physics DR extremes, not current.

**Evidence gathered for R10 design (from TB metrics + plots):**
- `DORAEMON/success_rate` saturated at 0.98+ from iter 500 onward (`perf_lb=90` trivially met because mean return 200-280 >> 90). DR Beta advances at full speed throughout training. Late-training reward decline is DR difficulty outpacing policy adaptation.
- `Constraint/margin/rp_rate=9.17`, `rp_vel_settling=11.58` — both far from budget, **constraints not binding at current oscillation levels**. Implication: r9_tightrates effect may be limited to transients, not SS.
- Per-env CV(SS_error) at hard DR: roll 169%, pitch 196%, yaw 285% (vs 77-78% at no-DR). n>40% catastrophes appear **only** at hard DR (roll 2%, vy 2.8%). Points to DORAEMON-saturation driven tail under-coverage, not general brittleness.
- Reward gradient analysis at err=1°: with σ=0.10 rad (5.73°), a 1° roll error costs only 1.54% reward loss. Small-error region is effectively flat — explains why symatt moved jitter but not SS.
- Entropy collapsed to -0.87 by iter 5000, noise at min_std floor. Thruster min_std=0.05 = 2.5N per-thruster random thrust -> ~0.035 Nm RMS roll torque via 0.007m arm, non-trivial forcing for the weak roll axis.

### Decisions

- **r10_perflb_high** (`config.py:378`, `performance_lb 90 -> 180`) because DORAEMON success saturated at 0.98+ from iter 500 and late reward drops -19% -- current perf_lb gates nothing, policy pushed into hard DR before mid-DR mastery. Prediction: success falls to 0.6-0.8 mid-training, n>40% < 0.5%, reward plateau sustained.
- **r10_thr_minstd** (`rsl_rl_ppo_cfg.py:213`, thruster floor 0.05 -> 0.03, arm 0.10 kept) because entropy has collapsed by iter 5000 so the thruster floor operates during all SS behavior, injecting ~0.035 Nm random roll torque forcing. 40% forcing RMS reduction predicted -> roll jitter -30%, SS -15%.
- **Rejected Run B candidates**:
  - `integral_leak 0.99 -> 0.995`: minor change, weak evidence (no observation that integral signal is the bottleneck).
  - `obs noise halved`: user correctly flagged "trivially predictable" — lower noise obviously improves sim SS but would widen sim2real gap. Any diagnostic value is swamped by the obvious direction.
  - `kl_ub 0.06 -> 0.04`: user intentionally set `kl_ub=0.06` to accelerate DORAEMON advancement during short (5000 iter) runs; long (~20000 iter) runs will use lower kl_ub. Preserving that design choice.

**R11 Experiments (queued after R10, on new accumulated-best baseline):**

- **r11_baseline** (branch `r11_baseline`): fold in r9_tightrates thresholds (rp_rate 0.5, yaw_rate 0.55) AND shrink `HardDomainRandomizationCfg.payload_cog_offset_xy_radius` 0.15 -> 0.08. **P1 rationale**: outlier-env analysis showed the worst roll/vz/pitch envs have payload × CoG combinations exceeding roll TAM authority (1.4 Nm). 0.08 caps gravity torque at 2.4 Nm — still above the 1.4 Nm limit so roll must work for it, but eliminates the physically-impossible tail that dominates SS_std. From R11 onward this is the reference baseline; R11 features measure their effect against this, not against R9.
- **r11_yawratedot** (branch `r11_yawratedot`): new `yaw_rate_dot_cost` average constraint, threshold 0.8 rad/s², budget 0.10. **P2 rationale**: magnitude-only `yaw_rate_cost` fires only after |ω_z| crosses 0.55, by which time overshoot has already happened. Derivative bound targets the aggressive torque swings (observed 1-2 rad/s² at step changes) that cause the overshoot, while leaving normal tracking (~0.2 rad/s²) unaffected. Uses existing `env._prev_root_ang_vel_z`.
- **r11_encdim16** (branch `r11_encdim16`): encoder `latent_dim 9 -> 16`. **P3-a rationale**: 24D privileged info compressed into 9D latent. Multi-axis outlier envs show distinct failure patterns (not a single "extreme combo"), suggesting encoder needs to represent a richer DR-conditional behavior space. 16D roughly doubles capacity. If z_std on added dims stays near zero, we'll know capacity wasn't the bottleneck.
- **r11_emabias** (branch `r11_emabias`): add EMA bias penalty reward, `k_bias=-2.0`, `alpha=0.99` (100-step / 2 s effective window), per-axis weights (roll 1.5, others 1.0). **P3-b rationale**: outlier envs show systematic per-env bias (env 14 roll +5°, env 23 pitch +10°, env 43 vx +0.17 m/s), not oscillation. Per-step tracking reward with σ=0.10 has gradient ~1.5% at 1° roll error — too flat to correct sustained offsets. EMA-squared penalty gradient grows with persistence, directly targeting this failure mode. Matches user's long-standing SS-error priority.
- **Rejected R11 candidates**:
  - `linear_damping / quadratic_damping` added to privileged obs per-axis: damping is a global scalar scale in `hydrodynamics.py:106`; knowing roll damping value + the known base ratio recovers all other axes. Adding them gives no new info.
  - `roll reward σ 0.10 -> 0.17`: widens gradient for large errors but flattens it for small errors, which would hurt the SS regime where r9_tightrates is already doing well. EMA-bias is a cleaner alternative for the same motivation.
  - Running R11 experiments without r11_baseline: would confound P1 (xy_radius) with each feature. r11_baseline added to the queue despite user only asking for 3 features — required for clean variable control, matches user-flagged principle.

### Variable Control (R11 ablation structure)

Each R11 experiment differs from r11_baseline by **exactly one variable** (see commits `417810ce`, `4cc2eede`, `402cb5c7`, `a69723f3`). The P1 contribution is measured via r11_baseline vs r9_tightrates; each feature via r11_X vs r11_baseline.

### Open Questions

- r9_tightrates SS-vs-transient question **resolved**: constraint tightening reduced SS floor, not just transients (roll SS -24%). Outcome of this resolves whether threshold slack allows oscillation at all; it does.
- r9_normval after the cost-GAE fix from 2026-04-18: does HORA-style value normalization stabilize critic targets with constraint/reward advantage mixing? Still running on GPU1 at session end.
- r9_symatt's vy SS +44% regression is still unexplained. Hypothesis: roll-sway coupling via body-frame rotations -- reduced roll weight frees roll motion that couples into Fy.
- R11 predictions to validate:
  - r11_baseline vs r9_tightrates: does the outlier tail (per-env SS_std for roll/vz) actually collapse when physically-impossible payload combos are removed, or was something else in the tail?
  - r11_yawratedot: does yaw OS drop below 15.3% without damaging yaw rise time or tracking?
  - r11_encdim16: does z_std increase across new dims (encoder using the extra capacity), and does any outlier SS metric improve — or do the extra dims collapse (unused)?
  - r11_emabias: does EMA-bias penalty drop roll/vz SS in outlier envs specifically (per-env CV), without hurting per-step tracking?

---

## [2026-04-18] Starting Point: Code Cleanup + Current Baseline

Previous 8 rounds of experiments (R1-R8) completed. R8-Gated confirmed as best policy.
Codebase underwent major simplification (16 files, -2071 / +565 lines).
This entry documents the current code state as the baseline for all future work.

### Current Best: R8-Gated

Error-gated 6D integral integration이 SS error와 overshoot를 동시에 개선한 유일한 configuration.
Model checkpoint는 log cleanup 사고로 소실 -- 재학습 필요.

Key results (eval_dr_fulldof):
- Aggregate: SS=0.131 (best), OS=13.1% (best), n>20%=16.0% (best)
- Attitude: SS=0.370 (-15% vs R7I), OS=9.3% (-48% vs R7I)
- Velocity: SS=0.014 (-53% vs R7I), OS=10.4% (-52% vs R7I)
- Yaw SS=0.001 (6D integral), Yaw OS=34.4% (sole remaining weakness)

### Architecture

```
Task: Isaac-FullDOF-TRPO-v0 (single registered task)
Action: 8D (2D arm revolute + 6D thruster wrench)
Observation: 87D policy (26D proprio + 55D temporal history + 6D integral)
Privileged: 24D (DR parameters, static min-max normalized)

Encoder:  p_t(24D) -> static_minmax -> MLP[256,128,64] -> LayerNorm -> softsign -> z(9D)
Actor:    cat([o_t(87D), z(9D)]) = 96D -> MLP[256,128,64] -> 8D (Gaussian)
Critic:   cat([o_t(87D), z(9D), p_t(24D)]) = 120D -> MLP[512,256,128] -> 1D (asymmetric)
Cost:     same 120D -> MLP[512,256,128] -> K (multi-head, one per constraint)
```

### Algorithm

- **ConstraintTRPO + IPO** (Interior-Point Optimization)
- max_kl=0.005, cg_iters=10, cg_damping=0.1
- GAE: gamma=0.99, lam=0.95
- Value: Adam lr=1e-3, 5 epochs, 4 mini-batches
- Barrier: t=100.0, alpha=0.05

### Entropy Management

- entropy_coef_per_dim: arm=(0.01, 0.01), thr=(0.001 x6) -- PerDimEnt, validated R2
- min_std_per_dim: arm=(0.10, 0.10), thr=(0.05 x6)
- max_std=2.0, min_std=0.05 (scalar fallback)
- init_noise_std=0.7

### Reward

```
r = r_att + r_lin + r_yaw + r_tau + r_thr + r_s

Tracking: r = k * (exp(-e^2/2s^2) - q*e^2)
  att_rp:  k=9.0, sigma=0.10, quad=0.833, roll_weight=1.5
  lin_vel: k=4.0, sigma=0.10, quad=1.0
  yaw_vel: k=3.5, sigma=0.10, quad=1.0

Saturating penalty fields (tanh_coef, arctan_coef) exist but default to 0.0.
Penalty: k_tau=-0.01, k_thr=-0.35, k_s=-0.1
```

### Constraints (10 terms: 5 Prob + 5 Avg)

| Type | Name | Budget |
|------|------|--------|
| Prob | attitude_limit (80 deg) | 0.01 |
| Prob | arm_torque (9.5 Nm) | 0.08 |
| Prob | arm_joint_vel (4.189 rad/s) | 0.02 |
| Prob | joint1_pos (4*pi rad) | 0.01 |
| Prob | cumulative_yaw (8*pi rad) | 0.01 |
| Avg | thruster_util | 0.40 |
| Avg | rp_rate (1.0 rad/s) | 0.10 |
| Avg | yaw_rate (0.7 rad/s) | 0.10 |
| Avg | rp_vel_settling (0.087 rad) | 0.20 |
| Avg | manipulability (w=0.3) | 0.05 |

### DORAEMON DR Curriculum

- kl_ub=0.04, performance_lb=90.0, step_interval=250
- SLSQP optimizer, log-space Beta parameterization, 15D physics-only
- Binary success criterion (episode_return >= performance_lb)

### Observation Detail

Current proprioception (26D):
- Command (6D): vel_cmd_lin(3), ang_cmd(3) [att_rp(2) + yaw_rate(1)]
- Body State (9D): euler(3), ang_vel(3), lin_vel(3)
- Arm State (5D): joint_pos(2), joint_vel(2), manipulability(1)
- Thruster (6D): filtered output (T0-T5)

Temporal history (55D): ring buffer, stride=3
- Joint tracking (12D): (q_des-q_actual, joint_vel) x 3 steps
- Body tracking (27D): (lin_vel_err, ang_err, rpy) x 3 steps
- Action (16D): full_action(8D) x 2 steps

Integral error (6D): leaky integrator (leak=0.99, clamp=+-2.0)
- 6 channels: roll_err, pitch_err, vx_err, vy_err, vz_err, yaw_rate_err
- Error-gated: accumulate only when |err| < sigma (R8-Gated configuration)

### Registered Tasks

| Task | Algorithm | Encoder | Purpose |
|------|-----------|---------|---------|
| Isaac-FullDOF-TRPO-v0 | ConstraintTRPO + IPO | Yes (24D->9D) | Production |
| Isaac-FullDOF-NoEncoder-v0 | ConstraintTRPO + IPO | No | Ablation baseline 1 |
| Isaac-FullDOF-PPO-v0 | Standard PPO | No | Ablation baseline 2 |

### Code Simplification (this session)

16 files modified (-2071 / +565 lines). Key changes:
- Removed experiment-specific task registrations (R5/R6/R7/R8 tasks)
- Consolidated runner configs into production + 2 ablation baselines
- Extracted shared PolicyBase for ActorCriticEncoder and ActorCriticAsymConstrained
- Simplified config.py: removed unused experiment configs
- Cleaned up reward, constraint, and observation modules

### Open Questions

- R8-Gated model needs retraining (checkpoint lost in log cleanup)
- Yaw OS (34.4%): channel-specific gate configuration
- Roll SS high per-env variance (std > mean at all DR levels)
- Entropy collapse in all R8 runs (Gated: 0.03): PerDimEnt tuning needed?

---

## [2026-04-18] R9 Plan: Roll Oscillation / Yaw Overshoot + Refactor Bug Discovery

### Context

R8-Gated eval_dr_fulldof revealed three residual problems that block a production-ready policy: roll oscillation (high SS jitter), vz undershoot, and isolated yaw overshoot. Needed to launch new experiments addressing these while also retraining baseline (R8-Gated checkpoint lost) and exercising the disabled `normalize_value` feature for the first time.

### Experiments

Hard-DR evidence from R8-Gated archive (enhanced_summary.json):
- roll SS=0.855 / jitter=0.275 / OS=14.6% vs pitch SS=0.320 / jitter=0.082 / OS=5.6% -> roll is ~3x worse across every metric despite same reward structure.
- vz SS=0.040 (std=0.099) / undershoot=4.19% / OS=14.1% -> vz is the worst lin_vel axis by 2-3x on every metric, consistent with buoyancy-F_bu 26.24 N + heave-added-mass being 10x smaller than surge/sway.
- yaw SS=0.0017 rad/s and rise=0.014 s (essentially perfect) but OS=23.8% with n_gt20=33% -> control authority is not the issue; yaw_rate constraint threshold of 0.7 rad/s authorizes the overshoot given cmd range +-0.5 rad/s.

R9 queue launched on GPU0/1 (2048 envs, 5000 iter, WandB project `fulldof_albc`):
- **r9_baseline** (GPU0 first): control, zero code change on top of r8_gated config.
- **r9_normval** (GPU1 first): `normalize_value=True` (HORA-style running mean/std for critic targets, previously-disabled path).
- **r9_tightrates** (GPU0 second): rp_rate soft_threshold 1.0 -> 0.5, yaw_rate 0.7 -> 0.55. Hypothesis: both rate constraints had 3x / 1.4x margin over command, leaving rate damping inactive.
- **r9_symatt** (GPU1 second): `att_roll_weight` 1.5 -> 1.0. Hypothesis: the 1.5x multiplier is in a middle zone where it neither compensates the 20x TAM moment-arm gap (0.007 m vs 0.145 m) nor avoids being a competing sharp signal.

### Decisions

- **Minimum-change per run, orthogonal hypotheses** over a single "combined fix" run. Separate isolation lets the decision tree for R10 read off cleanly (e.g., if tightrates helps roll/yaw but symatt does not, rate threshold was the bottleneck, not reward asymmetry).
- **Root-cause fix for refactor bugs** chosen over a lazy null-check guard, on user pushback. The proper contract is "after `_reset_idx`, env is in a valid observation state", restored by populating `_euler_cache` at reset-time just like `_get_dones` does per step. An `if is None` check would have masked the ordering violation.
- **Git worktree + PYTHONPATH prepend** over argparse CLI overrides for run isolation. Argparse route would have required modifying config parse logic, violating minimum-change. PYTHONPATH takes precedence over pip editable-install .pth entries, so each worktree's source is found first without touching the main install.
- **Sequential queue over 4 concurrent**. RTX 4060 (8GB) cannot host two runs simultaneously at ~6-7 GB each. Halving num_envs to fit 2-concurrent would double iterations to reach equivalent samples, yielding zero net wall-time gain.

### Latent bugs surfaced by first fresh training since refactor eafca264

The -2071 / +565 line refactor removed lazy-init guards that previously hid three bugs. R8-Gated only evaluated an archived checkpoint, so none of these had been exercised until this session's fresh train:

1. **`_euler_cache` uninitialized at first `_get_observations`**. Root: init=None in `__init__`, population only in `_get_dones` which does not run during `env.reset()`. Symptom: `TypeError: cannot unpack non-iterable NoneType`. Fix: populate at tail of `_reset_idx` so post-reset observation contract holds.
2. **Encoder static-min-max constants were plain Python attributes**. `_enc_obs_range` and `_enc_obs_midpoint` were assigned with `self.x = ...`, not `register_buffer`, so `module.to(cuda)` left them on CPU while inputs were on cuda. Symptom: `RuntimeError: tensors on different devices`. Fix: wrap both in `register_buffer`.
3. **`normalize_value` pipeline referenced a PPO-only attribute**. `self.alg.normalize_advantage_per_mini_batch` does not exist on `ConstraintTRPO`. Symptom: `AttributeError`. Only triggered when the flag was flipped on for the first time in r9_normval. Fix: `getattr(..., False)` fallback in `_compute_returns_with_value_norm`.

Lesson: checkpoint-reload eval passes do not validate init paths. Before trusting a refactor, run fresh training at least once with each toggleable feature enabled.

### Open Questions

- vz structural undershoot (4.19% hard DR, highest of all axes): candidate fixes deferred to R10 (per-axis lin_vel sigma, vz-only undershoot penalty, or revisiting added-mass DR bounds for heave).
- Rise time improvement: user marked low priority this round; may revisit after R9 results if rate-tightening regresses it.

### Follow-up mid-session: normalize_value wrapper had a second, silent bug

`r9_normval` appeared healthy at iter 200 (reward rising, ls_success=1.0) but `cost_val=2.4e-10` revealed the cost critic was producing essentially zero-magnitude outputs. User flagged: "constraints not changing at all".

Root cause: the wrapper `_compute_returns_with_value_norm` replaces `ConstraintTRPO.compute_returns` wholesale. The original did TWO things (reward GAE + cost GAE for K constraints); the wrapper only did reward GAE, silently skipping cost GAE. Consequence: `storage.cost_returns=0`, `cost_advantages=0`, IPO barrier gradients=0 -> training degenerated into unconstrained PPO while still logging as ConstraintTRPO. Not a mathematical problem with normalize_value; purely a method-override scoping bug.

Decision: killed r9_normval at iter ~230 rather than waste further compute. Added the missing `self.alg._compute_cost_returns(last_cost_values)` inside the wrapper. Cherry-picked the fix (plus the earlier `normalize_advantage_per_mini_batch` fallback) onto `feat/encoder-tdc-integration` so main is no longer carrying the latent bug set. Chained a waiter script so `r9_normval` auto-launches on GPU1 once `r9_symatt` completes (~02:05 KST), rather than restarting the full queue.

Lesson: when overriding a method with multiple side effects, verify which effects the override preserves. `cost_val` near zero when constraints are configured is a clear "cost critic frozen" smell -- future runs should treat this as a hard pre-flight check.

### Open Questions (continued)

- R10 will need to decide whether vz gets a dedicated fix or whether tightrates + symatt results reshape the priorities.
- When R9 runs complete, apply the `.claude/rules/02-operations.md` "Experiment Worktree Lifecycle" cleanup: migrate logs/checkpoints to `/workspace/isaaclab/logs/`, cherry-pick remaining useful worktree commits to main, then remove the four `isaaclab-r9*/` worktrees and temporary `run_gpu*.sh` scripts.

---
