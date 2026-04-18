# Pre-Round Infrastructure and Investigation

Period: 2026-04-04 ~ 2026-04-13.
Full ALBC 시스템 안정화 및 실험 인프라 구축 기간. DORAEMON DR curriculum 안정화,
eval_dr_fulldof 평가 도구 개발, entropy collapse 조사, reward/constraint 체계 확정.
Round 1~8 실험의 기반이 되는 모든 인프라 작업을 포함.

Active changelog `changelog.md`의 2026-04-04 ~ 2026-04-13 항목에서 정리.

---

## 1. DORAEMON DR Curriculum Stabilization

### Problem

DORAEMON (Tiboni et al., ICLR 2024) optimizer가 정상 작동하지 않음.
scipy `trust-constr`이 KL의 zero gradient (identity distribution)에서 stuck.

### Key Fixes (chronological)

| Date | Fix | Detail |
|------|-----|--------|
| 04-04 | trust-constr -> SLSQP | SQP linearization이 zero-grad 처리. Maxiter 50->200 |
| 04-04 | Log-space parameterization | Beta distribution의 72 box constraints 제거 |
| 04-04 | IS clamp 20->5, ESS min 0.05->0.01 | Importance sampling 안정화 |
| 04-04 | `build_param_specs(dr_cfg)` | Hardcoded bounds -> DR config에서 자동 추출 |
| 04-06 | kl_ub 1.5 -> 0.3 | Reference 대비 6x 공격적이었음 (step_interval=250 vs ref ~100k) |
| 04-06 | performance_lb 200 -> 110 | Command curriculum 없이 iter 0부터 학습 불가능 |
| 04-06 | Command scale 제거 (18D->15D) | DORAEMON이 command를 축소하는 degenerate solution 방지 |
| 04-07 | performance_lb 100 -> 80 | DORAEMON mode=-2 stuck 해소 |
| 04-09 | kl_ub 0.15 -> 0.08 | DR expansion 속도 감속 |
| 04-09 | performance_lb 80 -> 90 | 최종 안정화 값 |

### DORAEMON Trajectory Analysis (04-07)

7-phase trajectory 확인: mode -3 -> -2 -> 0 (expansion) -> 0 (catch-up) -> 0
(frozen, success binding) -> +1 (inverted) -> -2 (retreating). Phase 7에서 entropy
DECREASE (-18.35 -> -19.69) 확인 -- DORAEMON이 policy 열화 시 자동 retreat하지만,
retreat 속도가 열화 속도보다 느림.

### Final DORAEMON Configuration

```
kl_ub=0.04, performance_lb=90.0, step_interval=250
SLSQP optimizer, log-space Beta parameterization, 15D (physics only)
```

---

## 2. Evaluation Tooling (eval_dr_fulldof.py)

### Development Timeline

| Date | Change | Impact |
|------|--------|--------|
| 04-04 | Initial creation | 6-DOF step trajectory (14 segments), `--doraemon-dr` flag |
| 04-06 | Major overhaul | Warmup exclusion, block-aware crop, DR-separated layout, per-channel plots |
| 04-07 | **DR anchor bug fix** | `DomainRandomizationCfg` -> `HardDomainRandomizationCfg` as 100%-DR. 이전 eval은 ~40% 범위만 평가 |
| 04-07 | **DORAEMON clamp bug fix** | Learned distribution이 base-DR bounds로 truncate되던 문제 수정 |
| 04-07 | DR distributions visualization | `dr_distributions.png` 추가 |
| 04-07 | Trajectory update | 27->31 segments (zero-command + doubled att return) |
| 04-10 | SS metrics 확장 | SS jitter, zero-crossing count, trajectory overlay |
| 04-16 | Per-env OS metric | `recompute_eval_summary.py`: per-env OS/US distribution (기존 ensemble-mean 보완) |

### Critical Bug Impact

Bug fix 전 DR anchor 문제로 모든 이전 eval은 ~40% DR 범위에서 수행.
Fix 후 re-eval (run `2026-04-06_21-24-43`): att SS 1.9-2.3 deg, 100% survival 확인.
Hard DR 범위가 1.94-3.13x 확장됨.

### Per-Env OS Metric (04-16)

Stock `OS%`는 ensemble-mean trajectory의 peak 기반 -- 64 env의 peak timing 차이로
envelope smoothing 발생, undershoot는 0%로 처리됨.
Per-env OS: 각 env별 peak 기반 계산, OS/US 분포 + std + median + q90 제공.
Round 4 결론 재평가에 결정적 역할 (Arctan "승리" 판정 뒤집힘).

---

## 3. Reward and Constraint System

### Reward Structure Evolution

**Final form (04-09~):**
```
r = r_att + r_lin + r_yaw + r_tau + r_thr + r_s

Tracking: r = k * (exp(-e^2/2s^2) - q_quad*e^2 - q_lin*|e|)
  att_rp:  k=9.0, sigma=0.10, quad=0.833, roll_weight=1.5
  lin_vel: k=4.0, sigma=0.10, quad=1.0
  yaw_vel: k=3.5, sigma=0.10, quad=1.0

Penalty:
  k_tau=-0.01, k_thr=-0.35, k_s=-0.1
```

**Key changes:**
- 04-04: 3개 tracking reward 통합 (exp+quadratic). Sigma 0.15->0.10 (04-06)
- 04-06: att_roll_weight=1.5 추가 (TAM roll actuation weakness: 0.007m vs pitch 0.145m)
- 04-07: Linear penalty (lin_ratio=0.5) 시도 -> 10 deg에서 reward 음수, attitude tracking 포기. **Reverted**.
- 04-09: k_att_rp 6.0->9.0 (attitude gradient equilibrium shift)

### Constraint System

**Final: 10 terms (5 Probabilistic + 5 Average)**

| Type | Constraint | Budget | Note |
|------|-----------|--------|------|
| Prob | attitude_limit (80 deg) | 0.01 | |
| Prob | arm_torque (9.5 Nm) | 0.08 | |
| Prob | arm_joint_vel (4.189 rad/s) | 0.02 | |
| Prob | joint1_pos (4*pi rad) | 0.01 | |
| Prob | cumulative_yaw (8*pi rad) | 0.01 | |
| Avg | thruster_util | 0.40 | 유일하게 saturation 근접 (94%) |
| Avg | rp_rate (1.0 rad/s) | 0.10 | |
| Avg | yaw_rate (0.7 rad/s) | 0.10 | |
| Avg | rp_vel_settling (0.087 rad) | 0.20 | 04-09: settling-aware gating 추가 |
| Avg | manipulability (w=0.3) | 0.05 | |

**Removed/reverted:**
- `thruster_rate`: entropy_coef > 0에서 noise만으로 5x 위반. 구조적 비호환.
- `thruster_sat` -> `thruster_util`로 revert (Average, budget=0.40)
- `body_linear_velocity`: 항상 inactive (cr=0.00)

### rp_vel_settling Redesign (04-09)

**Before:** 매 step `|p|+|q|` penalize (transit 중에도 attitude command에 반대 작용).
**After:** `|att_err| <= settling_threshold (0.087 rad = 5 deg)` 일 때만 활성화.
Transit phase에서 zero cost -> settling phase에서만 angular rate 억제.

### Tested and Reverted

| Date | Change | Failure Mode |
|------|--------|-------------|
| 04-07 | Linear penalty lin_ratio=0.5 | err > 5.7 deg에서 reward 음수, attitude 포기 |
| 04-07 | rp_vel_settling budget 0.20->0.12 | 60-deg traverse가 8.7s IPO binding zone, att_rp reward sign flip |

---

## 4. Ablation Baselines (04-08)

세 가지 component ablation 실험. ALBCEnv (DR, reward, action space, DORAEMON) 동일.

| Phase | Task | Removes | Result |
|-------|------|---------|--------|
| 1 | `Isaac-FullDOF-NoEncoder-v0` | Encoder only | TRPO+IPO 유지, encoder 기여도 분리 |
| 2 | `Isaac-FullDOF-PPO-v0` | Encoder + IPO | Standard PPO, constraint 없음 |
| 3 | `Isaac-FullDOF-TDC-v0` | All RL | Classical TDC + 6-DOF PD (P-only) |

**Phase 3 (TDC baseline) eval:**
- 100% survival 전 DR level
- att SS: 2.8-7.1 deg (P-only floor)
- lin_vel: 0.11-0.40 m/s
- yaw: 0.013 (none) -> 0.13 (hard DR)

### Code Added
- `encoder/actor_critic_asym_constrained.py`: NoEncoder policy
- `constrained_full_albc_tdc/`: Phase 3 module (TDC env, thruster PD, single-step DLS IK)
- `__init__.py`: Phase 1+2 task registration

---

## 5. Entropy Collapse Investigation (04-10 ~ 04-13)

### Timeline

TRPO natural gradient가 구조적으로 noise를 감소시키는 문제. 8D 혼합 action space
(arm 2D + thruster 6D)에서 arm dim이 min_std floor에 도달하고, thruster는 발산.

| Date | Approach | Outcome |
|------|----------|---------|
| 04-10 | Adaptive entropy (SAC-style) | **FAILED.** Alpha가 0.003->0.0014로 decay (entropy가 target 위에서 시작). 구조적 불일치 |
| 04-10 | HardDR expansion (+30-50%) | **REVERTED.** Roll 4.59 vs 2.80 deg 열화 |
| 04-10 | **Log_std TRPO reintegration** | 핵심 수정. 모든 reference TRPO가 log_std를 natural gradient에 포함. 우리는 separate Adam. 그러나 이것만으로 불충분 (entropy -6.28까지 collapse) |
| 04-13 | ERC-TRPO absolute H (beta=0.01) | **FAILED.** 8D Gaussian dimension constant 11.35 -> noise max_std=2.0 폭발 |
| 04-13 | ERC-TRPO H-H_ref (beta=0.01) | **FAILED.** Hard entropy floor at H_ref - 0.75. Line search 0% at iter 53. 이 task는 H 8.5->3.1 감소 필요, ERC-TRPO는 0.75 nats 이상 감소 방지 |
| 04-13 | **Per-dim min_std** | arm(0,1)=0.10, thr(2-7)=0.05. Floor crash 방지하지만 상향 압력 없음 |
| 04-13 | **entropy_coef=0.003 복원** | **ROOT CAUSE 확정.** 04-09 run (coef=0.003): noise 0.36->0.55 회복. 04-10 run (coef=0): 0.12까지 collapse. Entropy bonus가 유일한 상향 메커니즘 |

### Root Cause Analysis

`sigma_step_mean` 분석: 전체 10k iter 중 85.6% negative step (100%가 아닌 것은 확인).
Arm dims (0-1)이 collapse 주도: 0-1% positive steps, 4-5x larger magnitude.
Thruster dims (2-7) oscillate (17-36% positive) but arm dominates aggregate.

`TRPO Fisher(log_sigma) = 2I` (constant) -> natural gradient = vanilla gradient / 2.
KL limit이 per-step 변화를 ~2.5%로 제한하지만 cumulative decline은 방지 불가.

### Definitive Fix

`entropy_coef` + `per-dim min_std` + `TRPO-integrated sigma` 조합:
- entropy_coef: 유일한 상향 압력 (surrogate loss에 +coef*H)
- per-dim min_std: arm floor 비대칭 보호 (arm sensitivity > thruster)
- TRPO-integrated: KL constraint가 sigma 변화도 포함

### Literature Context

- **EnTRPO** (Xu et al., 2021), **ERC-TRPO** (Guo et al., 2024): TRPO entropy 관리
- TRPO entropy bonus: PPO에서는 standard이지만 TRPO에서는 non-standard
- Constrained TRPO + entropy: 선행 연구 없음 (NORBC, CPO, IPO, FOCOPS 모두 entropy 무시)
- 이 프로젝트의 low-dim mixed action space (8D)가 entropy collapse를 유발하는 고유 조건일 수 있음

---

## Summary: Infrastructure State Before Round 1

Round 1 시작 시점(04-14)의 시스템 상태:

| Component | State |
|-----------|-------|
| DORAEMON | kl_ub=0.04, perf_lb=90, SLSQP, log-space, 15D physics-only |
| Eval | eval_dr_fulldof with correct DR anchor, 31 segments, per-channel plots |
| Reward | 3-term exp+quad (att k=9.0, lin k=4.0, yaw k=3.5), sigma=0.10 |
| Constraints | 10 terms (5 prob + 5 avg), settling-aware rp_vel |
| Entropy | entropy_coef=0.003, per-dim min_std (arm=0.10, thr=0.05), log_std in TRPO |
| Noise | init_noise_std=0.7, min_std=0.05, max_std=2.0 |
| Observation | 81D policy (26D proprio + 55D temporal history), 24D privileged |
| Encoder | 24D->9D, elu + LayerNorm + softsign, static min-max normalization |
| Network | Actor [256,128,64], Critic [512,256,128] (asymmetric + z), Cost Critic same |
