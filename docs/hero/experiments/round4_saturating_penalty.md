# Round 4: Saturating Penalty (Tanh / Arctan)

Period: 2026-04-16.
L1의 zero-gradient-at-zero fix를 유지하면서 far-field saturation으로 OS를 제어.
Per-env OS metric 도입으로 기존 평가 방법론의 한계를 발견.

---

## Hypothesis

`rho(e) = coef*eps*tanh(|e|/eps)` 또는 `coef*eps*(2/pi)*arctan(|e|/eps)`는:
- e=0 근처: non-zero gradient (L1과 동일한 SS 개선 효과)
- far-field: saturation (L1과 달리 OS 유발하지 않음)

Gradient 비교 (at e=0, sigma=0.10): L1 0.150, Tanh 1.000 (coef=1.0), Arctan 0.637.
At e=5*sigma: L1 0.150 (persistent), Tanh 0.018, Arctan 0.024 (vanish).

### Literature

- Slotine & Li 1991 (Applied Nonlinear Control): tanh/arctan = standard SMC chatter reduction
- Hwangbo et al. 2019 (ANYmal, Science Robotics): logistic kernel with near-linear + saturation
- CAPS (Mysore et al. 2021, ICRA): action-rate penalties for anti-overshoot

## Baseline

Round 2 PerDimEnt control (`perdiment_kl06`).

## Experimental Setup

| Run | Config | Gradient at e=0 |
|-----|--------|----------------|
| Tanh (`2026-04-16_16-32-12_exp_tanh_ss`) | coef=1.0, eps=0.10 on lin_vel+yaw | 1.000 |
| Arctan (`2026-04-16_16-32-44_exp_arctan_ss`) | coef=1.0, eps=0.10 on lin_vel+yaw | 0.637 |

Both: kl_ub=0.06, num_envs=2048, 5000 iters, seed=30.
**Note:** Attitude tracking은 변경하지 않음 (lin_vel + yaw만 적용).

## Results

### Training

| Metric | Control | Tanh | Arctan |
|--------|---------|------|--------|
| Reward | 151 (4.854/step) | 134 (4.256, **-12%**) | 145 (4.690, -3%) |
| lin_vel reward | - | **-41%** | **-40%** |
| yaw_vel reward | - | -27% | neutral |

### Per-Env Overshoot (hard DR) -- Enhanced Metric

기존 stock OS%는 ensemble-mean trajectory peak 기반.
Per-env OS는 각 env별 peak -> mean 계산 (envelope smoothing 제거, undershoot 감지).

| Axis | Control | L1 | Tanh | Arctan |
|------|---------|------|------|--------|
| roll | 13.7 | 13.9 | 13.6 | **17.1** |
| pitch | 9.7 | 12.8 | 9.9 | 9.3 |
| vx | 20.0 | 22.5 | 19.3 | 19.9 |
| vy | 16.2 | **22.8** | **22.7** | 16.7 |
| vz | 20.0 | 18.6 | **16.6** | **17.3** |
| yaw | 33.4 | 41.5 | 37.8 | 42.0 |

### Deep Analysis Findings

1. **Tanh vy trajectory drift**: vy +0.25 step에서 persistent SS drift (median 0.280 vs target 0.250, +12%).
   vy reversal (-0.25)에서 undershoot (peak -0.218, -12.9% miss)
2. **DORAEMON 동일**: 4개 run 모두 동일 DR entropy trajectory 수렴. 성능 차이는 DR difficulty가 아닌
   penalty shape에서 기인
3. **Thruster usage**: Tanh +2.7% norm, +4.0% rate (aggressive). Arctan -6.2% norm, -7.4% rate (smooth)

## Analysis

1. **"Tanh failed / Arctan succeeded" 프레이밍 오류 수정**: Per-env metric 도입 후 재평가.
   - Tanh: vy OS +40% (real degradation), 그러나 vx/vz neutral-or-better
   - Arctan: roll OS 17.1% (**전 run 중 최악**), yaw OS +26%. "Smooth winner" 판정 뒤집힘
   - L1: 전 axis degradation 확인 (vy +41%, yaw +24%, pitch +32%)
2. **Saturating penalty가 lin_vel reward를 40% 잠식** (Tanh/Arctan 모두).
   penalty가 exp kernel을 cut into하지만 SS error는 변화 없음
3. **TAM coupling이 penalty shape보다 dominant**: vz (독립 vertical thruster)는 Tanh/Arctan
   모두 개선 (-17%, -13%). vx/vy/yaw (shared horizontal thrusters)는 mixed/degraded.
   Reward-penalty tuning은 TAM-coupled axes에서 비효과적
4. **coef=1.0이 과다**: e=0 gradient 1.000 (L1의 6.7x). 이 크기가 reward 잠식의 원인

## Conclusions

- **어떤 단일 run도 전 axis에서 개선을 달성하지 못함** -- TAM coupling의 구조적 한계
- Per-env OS metric이 이전 stock metric의 결론을 뒤집음 -- methodology 중요성 확인
- coef=1.0은 과도. 0.3 수준으로 recalibrate 필요 (L1 gradient 0.15 근방)
- **SS error의 근본 원인은 observation structure** (integral error obs 없음), reward shape이 아님

## Impact on Next Round

Round 5: Constraint budget tuning (rp_vel tightening, lin/yaw settling reactivation).
Round 6: coef=0.3으로 recalibrate + axis-specific shape (5-way 데이터 기반).
