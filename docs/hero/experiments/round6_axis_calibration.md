# Round 6: Axis-Specific Shape Calibration

Period: 2026-04-17.
Round 4의 5-way 비교 데이터에 기반하여 axis별 최적 penalty shape를 적용.
coef=0.3으로 recalibrate (Round 4 coef=1.0의 magnitude 문제 수정).

---

## Hypothesis

Round 4 5-way 데이터에서 각 axis의 winning shape가 다름:
- **Attitude (roll)**: Arctan이 유일한 roll SS winner (1.42 vs Control 1.68)
- **Velocity (vy, yaw)**: Tanh/L1이 winner (vy 0.043-0.045, yaw 0.019-0.021)

coef=1.0은 e=0 gradient가 L1의 6.7x (Tanh) / 4.2x (Arctan) -> reward 40% 잠식.
coef=0.3으로 recalibrate: Arctan e=0 gradient 0.191, Tanh 0.3 (L1의 0.15 근방).

## Baseline

Round 2 PerDimEnt control (`perdiment_kl06`).

## Experimental Setup

| Run | Task | Config | Target |
|-----|------|--------|--------|
| R6-AttArctan (`r6_attarctan_c03`, wandb xtjmnwbk) | Isaac-FullDOF-R6-AttArctan-v0 | att_rp_arctan_coef=0.3, eps=0.10 | roll SS < 1.40 |
| R6-VelTanh (`r6_veltanh_c03`, wandb txroyh8u) | Isaac-FullDOF-R6-VelTanh-v0 | lin_vel_tanh_coef=0.3, yaw_vel_tanh_coef=0.3, eps=0.10 | vy SS < 0.045, yaw SS < 0.022 |

Both: 10 constraints (rp_vel_settling 유지), per-dim entropy, kl_ub=0.06, num_envs=2048, 5000 iters, seed=30.

## Results

### R6-AttArctan

| Metric | Control | R6-AttArctan | Change |
|--------|---------|-------------|--------|
| **Pitch SS** | 0.82 | **1.33** | **+62%** (Cohen's d=+1.05, large) |
| Pitch OS | 9.37 deg | **18.82 deg** | 2x (std=4.67, systematic) |
| vy SS | - | -31% | Unexpected improvement |
| yaw SS | - | -36% | Unexpected improvement |

**Attitude penalty가 attitude를 악화시킴.** Overcorrection -> velocity oscillation -> pitch settling
degradation. Attitude에 직접 penalty를 적용하는 것이 구조적으로 역효과.

### R6-VelTanh

| Metric | Control | R6-VelTanh | Change | Target | Pass? |
|--------|---------|-----------|--------|--------|-------|
| Roll SS (none) | - | 1.05 | - | < 1.25 | Yes |
| Pitch SS (none) | 0.82 | 0.69 | -16% | < 1.30 | Yes |
| vy SS (none) | - | 0.037 | - | < 0.045 | Yes |
| yaw SS (none) | - | 0.011 | - | < 0.022 | Yes |

**None-DR에서 4/4 target 달성한 최초의 run.** L1만이 유일한 다른 4/4 run.
Cohen's d: yaw improvement large (d=-0.98), 나머지 small (d~0.3).
Roll marginal (1.05 vs target 1.25, medium DR에서 1.29로 fail).

### Noise Collapse Analysis

Per-dim NoiseStd 추출 (6개 completed runs). Control (PerDimEnt)이 arm noise를
0.179-0.191 (well above 0.10 floor)에서 4000+ iter 유지, 그러나 roll은 여전히 악화
(10.42->14.12 deg) + reward decline (201->165).
**Noise preservation은 necessary but NOT sufficient.** Reward shape과 observation structure가
binding constraints.

### Per-Segment Trajectory Analysis (R6-VelTanh)

Roll SS의 critical pattern:
- **target=0**: SS 개선 (dSS -0.20 to -0.61)
- **target=+-15 deg**: SS 악화 (dSS +0.09 to +0.71)

Physical mechanism: off-equilibrium attitude에서 buoyancy torque가 velocity coupling 발생.
Tanh velocity penalty가 coupled velocity response를 억제 -> non-zero roll 유지를 주저.
Pitch는 면역 (pitch actuation 20x 강함: TAM 0.145m vs roll 0.007m).

## Conclusions

- **R6-VelTanh를 새 baseline으로 채택.** None-DR 4/4 target 달성, yaw SS large improvement
- **R6-AttArctan 폐기.** Attitude에 direct penalty가 overcorrection-oscillation 유발
- **coef=0.3 calibration 유효**: Round 4 coef=1.0의 reward 잠식 문제 해소
- Roll은 medium DR에서 target 초과 (1.29 vs 1.25) -- off-equilibrium buoyancy coupling이 원인
- Axis-specific shape (velocity에만 tanh) > uniform shape의 효과 확인

## Impact on Next Round

Round 7: VelTanh baseline에서 두 가지 방향:
1. Parameter tuning: tanh_eps 확대 (0.10->0.20)로 off-equilibrium roll penalty 완화
2. Observation structure: Integral error obs (Hwangbo 2017) -- reward shape 튜닝을 초월하는 구조적 해결
