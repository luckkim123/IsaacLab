# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## Historical changelogs

- [R9 → Student v1](docs/hero/changelog_r9_to_student_v1.md) (2026-04-18 ~ 2026-04-22)
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

<!-- Active entries go below. Previous session log archived to
     docs/hero/changelog_r9_to_student_v1.md on 2026-04-22. -->

## [2026-04-22] Dead code purge after r13_A baseline lock

### Context
Baseline locked to r13_A on 2026-04-22 (Phase 0.7 decision, challenger Enc16
disqualified). The repo still carried three layers of experimental code that
were no longer reachable from any active config: (1) the disqualified
challenger task, (2) deprecated sibling task dirs `hero_agent/` and
`constrained_albc/` kept for historical imports, and (3) pre-Full-DOF analysis
scripts. Objective: shrink the reachable surface to just r13_A + the four
live ablation variants (v2 NoEncoder, v3 TRPO-NoIPO, v4 PPO-Enc, v5 PurePPO)
while Round 1 training was still running on both GPUs.

### Experiments
- **Import surface survey**: grep for live `from isaaclab_tasks.direct.hero_agent`
  and `from isaaclab_tasks.direct.constrained_albc` found 6 live importers beyond
  the registry: `scripts/analysis/common.py` (hero_agent DR + encoder cfg),
  `scripts/analysis/{eval_dr,collect_rollouts}.py` (constrained_albc runners/env),
  `scripts/demos/test_hero_thruster.py`, and the `constrained_full_albc_tdc`
  classical baseline (imported TDC + kinematics from `hero_agent/controllers/`).
  Additional hidden surface: dynamic runner dispatch maps in
  `scripts/reinforcement_learning/rsl_rl/{train,play}.py` referenced 5 dead
  runner classes (`BaseRunner`, `EncoderRunner`, `AdaptRunner`,
  `ConstraintEncoderRunner`, `SACMPCRunner`). These never fired on current
  configs (all use `FullDOFConstraintEncoderRunner` or `OnPolicyRunner`) so
  deletion was safe.
- **TDC baseline disambiguation**: user initially said "remove the TDC dir if
  it is RL-based TDC". Reading `constrained_full_albc_tdc/__init__.py:6` and
  `controllers/thruster_pd.py` showed it is TDC (arm) + stateless PD (thruster)
  classical control, not RL. User reclassified on inspection and asked to keep
  it. Evidence checked for TDC+PID alternative elsewhere in repo: none exists
  — only TDC+P(D) in this single dir. No run logs for `Isaac-FullDOF-TDC-v0`
  ever produced.
- **Challenger audit**: `FullDOFTRPOChallengerEnc16RunnerCfg` +
  `ALBCChallengerEnc16EnvCfg` + `Isaac-FullDOF-TRPO-ChallengerEnc16-v0` were
  still wired through `agents/__init__.py` and would have shown up as a
  gym-registered task for anyone pulling the repo. The 14M
  `challenger_hist5_act3_enc16.log` was still at repo-adjacent `/workspace/`
  after the run dir already preserved it.
- **Checkpoint-fallback path already existed**: `encoder_z_sweep.py` had both
  a hero_agent-backed `build_nominal_obs()`/`build_sweep_params()` path and a
  `build_sweep_params_from_checkpoint()` fallback guarded by a `try/except
  ImportError`. For any Full-DOF checkpoint (24D privileged) the hero_agent
  path returned a 19D array, triggered the dim-mismatch branch, and fell
  through to the checkpoint path anyway — i.e. the hero_agent branch was
  dead for every current use case. Collapsing to the checkpoint-only path
  removed the last reason to keep the hero_agent DR/encoder cfg imports in
  `common.py`.

### Decisions
- **Moved TDC controllers into `constrained_full_albc_tdc/controllers/`** (not
  left in a shrunken hero_agent dir) because the TDC baseline is the only
  consumer of `tdc.py` and `kinematics.py`. This makes the classical baseline
  self-contained and unblocks full deletion of `hero_agent/`. Rejected
  alternative: keep a pared-down `hero_agent/controllers/` package — would
  leave a cross-package import and the deprecated dir in the gym registry.
- **Deleted challenger task and code outright** rather than marking as
  deprecated. Rationale: the Phase 0.7 decision explicitly disqualified it
  (pitch regression +208%, yaw +125% at hard DR, 1-env catastrophic outlier
  with `att_lv=+0.976` coupling). Keeping the code invited re-running it.
  Still preserved: `logs/rsl_rl/.../challenger_hist5_act3_enc16/` run dir
  (git-ignored logs, contains the evaluated checkpoint).
- **Kept running ablation code untouched** (v2 NoEncoder, v3 TRPO-NoIPO, v4
  PPO-Enc, v5 PurePPO). Round 1 training was live at the time of the purge
  (v2 and v5 at iter 500/2500 at completion, ETA ~80 min). Editing runner
  configs mid-run risks Round 2 launch failure when the orchestrator
  re-imports the module. Deferred this tier to post-training.
- **Left dynamic runner maps with only `FullDOFConstraintEncoderRunner` plus
  the `OnPolicyRunner`/`DistillationRunner` elif branches**. No current cfg
  uses any other `class_name`.
- **Did not delete `hero_agent_hydro_demo.py`, `analyze_hero_mass.py`,
  `check_usd_*.py`**. They reference `HeroAgentBuoyHydrodynamicsCfg` / USD
  structure from `isaaclab_assets.robots.uuv`, not from the deprecated task
  dir. Asset package is still load-bearing for r13_A (robot cfg,
  hydrodynamics constants).

### Lessons
- **Deprecation without deletion grows tentacles.** "Deprecated" dirs
  accumulated live imports in three different systems (classical baseline,
  analysis scripts, dynamic runner dispatch). Grep-based audit caught them
  but the `try/except ImportError` in `common.py` hid one branch from
  grep-for-imports — only grep-for-function-names revealed it.
- **Dynamic runner dispatch maps are a blind spot.** A class_name -> module
  path dict does not trigger `ImportError` at import time; the missing
  module would only fail at runtime when a specific class_name is requested.
  These stale entries had been undiagnosed dead code since the pre-Full-DOF
  era.
- **Verify classification before acting on user directives.** User asked to
  "remove the TDC+PID baseline if it's RL-based". Reading the actual
  controller math (`thruster_pd.py` line 142: "P on attitude error, D on body
  angular rate") revealed it is P(D), not PID, not RL. Literal execution of
  the instruction would have deleted the wrong thing. Reading the code
  before executing the delete caught it.

### Open Questions
- `docs/hero/plans/2026-03-*.md` still reference now-deleted modules
  (`hero_agent.encoder.HistoryTCN`, `constrained_albc.algorithms.ConstraintTRPO`,
  etc.). Not actionable: these are historical specs superseded by live
  implementations in `constrained_full_albc/`. Defer to legacy-changelog
  archival when Tier D (ablation code) is purged.
- Ablation code (v2/v3/v4/v5) must be removed after Round 2 + cross-variant
  analysis completes. Budget: ~8 hours from training start, bounded by
  2500-iter runs at ~2.0-2.6s/iter on RTX 4070/4060.

