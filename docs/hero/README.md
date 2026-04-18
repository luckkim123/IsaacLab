# Hero Agent Project Documentation

Hero Agent UUV (Underwater Vehicle) 프로젝트의 통합 문서 디렉토리.
hero_agent, constrained_albc, constrained_full_albc 관련 문서를 포함한다.

Active changelog: [/workspace/isaaclab/changelog.md](/workspace/isaaclab/changelog.md)

---

## Changelogs

| File | Period | Description |
|------|--------|-------------|
| [changelog_full_albc_early.md](changelog_full_albc_early.md) | 2026-03-31 ~ 2026-04-02 | Full ALBC initial development (DORAEMON, wrench-space, logging) |
| [changelog_constrained_albc.md](changelog_constrained_albc.md) | 2026-03-27 ~ 2026-03-31 | Constrained ALBC redesign Steps 1-8 |
| [changelog_legacy.md](changelog_legacy.md) | 2026-03-05 ~ 2026-03-26 | Initial development Phase 1-8 (85+ commits) |

---

## experiments/

Experiment records, ablation studies, root cause analyses.

### Full ALBC Experiment Rounds (2026-04-04 ~ 2026-04-18)

8 rounds of systematic experiments, from entropy management to error-gated integration.
각 문서는 hypothesis, setup, results, analysis, conclusions 구조로 정리.

| File | Period | Summary |
|------|--------|---------|
| [pre_round_infrastructure.md](experiments/pre_round_infrastructure.md) | 04-04 ~ 04-13 | DORAEMON 안정화, eval 도구, entropy 조사, ablation baseline, reward/constraint 확정 |
| [round1_noise_comparison.md](experiments/round1_noise_comparison.md) | 04-14 | Per-dim noise 3-run 비교. PerDimEnt (arm=0.01, thr=0.001) 최초 검증 |
| [round2_perdiment_validation.md](experiments/round2_perdiment_validation.md) | 04-14 ~ 04-15 | PerDimEnt harder DR 검증. Thr entropy reduction이 핵심 확인 |
| [round3_ss_structural.md](experiments/round3_ss_structural.md) | 04-16 | L1 + Settling. L1 SS/OS tradeoff, Settling catastrophic failure |
| [round4_saturating_penalty.md](experiments/round4_saturating_penalty.md) | 04-16 | Tanh/Arctan. Per-env OS metric 도입, TAM coupling 한계 확인 |
| [round5_constraint_tuning.md](experiments/round5_constraint_tuning.md) | 04-17 | Constraint budget + settling. Settling dead end 최종 선언 |
| [round6_axis_calibration.md](experiments/round6_axis_calibration.md) | 04-17 | Axis-specific shape. VelTanh c=0.3이 4/4 none-DR target 달성 |
| [round7_integral_obs.md](experiments/round7_integral_obs.md) | 04-17 ~ 04-18 | Integral obs (Hwangbo 2017). 50-67% SS 감소, reward shape 초월 |
| [round8_gated_integral.md](experiments/round8_gated_integral.md) | 04-18 | **Error-gated integration. SS+OS 동시 개선. BEST POLICY** |

### Other Experiments

| File | Description |
|------|-------------|
| [encoder_ablation.md](experiments/encoder_ablation.md) | Encoder integration ablation study (Steps 0-19, 20+ experiments) |
| [arm_freeze_analysis.md](experiments/arm_freeze_analysis.md) | Arm freeze root cause analysis (tanh saturation, H1-H6 hypotheses) |
| [dr_training_survey.md](experiments/dr_training_survey.md) | DR training strategies 문헌 조사 (ADR, curriculum, contrastive) |

---

## architecture/

System design, theory, algorithm 문서.

| File | Description |
|------|-------------|
| [system_overview.md](architecture/system_overview.md) | Hero Agent ALBC 전체 아키텍처 (UUV 물리, 디렉토리 구조, 시뮬레이션 루프) |
| [tdc_control_law.md](architecture/tdc_control_law.md) | TDC controller 수학적 유도 (roll/pitch body dynamics) |
| [tdc_literature_survey.md](architecture/tdc_literature_survey.md) | Time-Delay Control 이론 서베이 (Youcef-Toumi, Hsia & Gao) |
| [dynamics_analysis.md](architecture/dynamics_analysis.md) | ALBC dynamics model 분석 (adaptive M, added mass coupling) |
| [training_pipeline.md](architecture/training_pipeline.md) | HORA/RMA 2-phase training pipeline (Encoder-Base + Adapt-Base) |
| [reward_functions.md](architecture/reward_functions.md) | Reward function 설계 (Gaussian tracking + PBRS + penalties) |
| [theoretical_analysis.md](architecture/theoretical_analysis.md) | NORBC 논문 기준 theory-code alignment 검증 |

---

## environment/

Simulation environment, physics, domain randomization 문서.

| File | Description |
|------|-------------|
| [physics_environment.md](environment/physics_environment.md) | PhysX 안정성 (effort_limit, max_velocity, damping, added mass) |
| [domain_randomization.md](environment/domain_randomization.md) | DR 구현 (12 categories, 35+ parameters, curriculum) |
| [sim_to_real.md](environment/sim_to_real.md) | Sim-to-real gap 분석 (actuator, sensor, hydrodynamics) |

---

## plans/

Implementation plan 문서 (date-prefixed).

| File | Description |
|------|-------------|
| [2026-02-04-albc-task-integration-design.md](plans/2026-02-04-albc-task-integration-design.md) | ALBC task integration 초기 설계 |
| [2026-03-17-history-encoder-architecture.md](plans/2026-03-17-history-encoder-architecture.md) | History encoder architecture 설계 |
| [2026-03-17-lagrangian-baseline-3constraint.md](plans/2026-03-17-lagrangian-baseline-3constraint.md) | Lagrangian baseline + 3 constraints |
| [2026-03-17-analysis-toolkit-restructure.md](plans/2026-03-17-analysis-toolkit-restructure.md) | Analysis toolkit restructure |
| [2026-03-24-sigma-decoupling-yaw-removal.md](plans/2026-03-24-sigma-decoupling-yaw-removal.md) | Sigma decoupling + yaw removal |
| [2026-03-27-encoder-trpo-integration.md](plans/2026-03-27-encoder-trpo-integration.md) | Encoder-TRPO integration |
| [2026-03-31-constraint-redesign.md](plans/2026-03-31-constraint-redesign.md) | Constraint system redesign |
| [2026-03-31-full-dof-tracking-design.md](plans/2026-03-31-full-dof-tracking-design.md) | Full 6-DOF velocity tracking design (constrained_full_albc) |

---

## history/

Debugging log, tuning history, code cleanup 기록.

| File | Description |
|------|-------------|
| [tdc_tuning_history.md](history/tdc_tuning_history.md) | TDC gain tuning 36-combo sweep + root cause analysis |
| [tdc_debug_history.md](history/tdc_debug_history.md) | TDC controller debugging history |
| [code_review.md](history/code_review.md) | Hero Agent code review |
| [code_simplification_log.md](history/code_simplification_log.md) | Code simplification 진행 기록 (Steps 1-7) |

---

## archive/

Deprecated/abandoned approach 문서. 참고용으로 보존.

| File | Description |
|------|-------------|
| [rl_tdc_comparison.md](archive/rl_tdc_comparison.md) | **DEPRECATED** RL-TDC vs Encoder-TDC 비교 (2026-02-16) |
| [sac_mpc_monitoring.md](archive/sac_mpc_monitoring.md) | **ABANDONED** SAC-MPC training monitoring guide |
