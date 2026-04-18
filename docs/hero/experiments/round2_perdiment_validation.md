# Round 2: PerDimEnt Validation Under Harder DR

Period: 2026-04-14 ~ 2026-04-15.
PerDimEnt의 thr entropy 감소 효과를 harder DR에서 검증하고,
arm entropy boost vs thr entropy reduction의 기여도를 분리.

---

## Hypothesis

PerDimEnt의 이점이 arm entropy boost (arm=0.01)에서 오는가, thr entropy reduction
(thr=0.001)에서 오는가? 낮은 entropy (0.51)가 harder DR에서 문제가 되는가?

## Baseline

Round 1 PerDimEnt 결과. DORAEMON kl_ub 0.04->0.06으로 상향하여 DR stress 증가.

## Experimental Setup

- 3-run, num_envs=2048, 5000 iters, kl_ub=0.06, seed=30
- Round 1 대비: env 수 2x, iteration 2x, DR 더 공격적

| Run | Config | Purpose |
|-----|--------|---------|
| PerDimEnt (`2026-04-14_18-55-20_perdiment_kl06`) | arm=0.01, thr=0.001 | 전체 per-dim |
| ArmOnly (`2026-04-14_18-55-29_armonly_kl06`) | arm=0.01, thr=0.003 | Arm boost만 |
| Baseline (`2026-04-14_22-33-43_baseline_kl06`) | uniform 0.003 | Control |

## Results

### Training Metrics (5000 iters)

| Metric | PerDimEnt | ArmOnly | Baseline |
|--------|-----------|---------|----------|
| Reward | **151.3** | 130.6 | 137.9 |
| Smoothness | **-0.090** | -0.326 | -0.192 |
| Thruster | **-0.074** | -0.222 | -0.142 |
| Entropy | -0.26 | 7.63 | 3.12 |
| DORAEMON success | **0.811** | 0.787 | 0.775 |

### Noise Dynamics (5000 iters)

| Dim | PerDimEnt | ArmOnly | Baseline |
|-----|-----------|---------|----------|
| arm0 | 0.157 | 0.158 | 0.100 (floor) |
| arm1 | 0.244 | 0.255 | 0.114 |
| thr7 | 0.332 | **1.360** | 0.893 |

### eval_dr_fulldof (PerDimEnt vs ArmOnly)

| Metric | PerDimEnt | ArmOnly |
|--------|-----------|---------|
| Attitude SS (none) | ~1.5 deg | ~1.5 deg |
| Attitude SS (hard) | ~2.4 deg | ~2.4 deg |
| **Yaw SS (hard)** | **0.019 rad/s** | **0.070 rad/s** (3.7x worse) |
| ArmOnly zero-crossings | - | 4.8 (medium DR, yaw oscillation) |
| Survival | 100% | 100% |

## Analysis

1. **ArmOnly는 역효과**: Baseline보다 나쁨 (reward 130.6 vs 138.0). Arm entropy boost가
   TRPO update를 통해 thruster dim으로 전파, thr=0.003이 divergence를 억제하지 못함.
   thr7: 0.621->1.360 (2.19x divergence) vs Baseline 0.606->0.893
2. **Thruster entropy reduction이 핵심**: PerDimEnt와 ArmOnly의 유일한 차이는 thr coef
   (0.001 vs 0.003). thr=0.001이 reward +15.9%, smoothness 3.6x 차이의 원인
3. **PerDimEnt entropy collapse (-0.26)는 문제 아님**: 가장 높은 DORAEMON success (0.811),
   가장 높은 reward (151.3). Low entropy = precise policy = robust to DR
4. **Yaw SS 차이가 극명**: PerDimEnt 0.019 vs ArmOnly 0.070 (3.7x). Thruster noise
   divergence가 yaw control을 직접 훼손 (horizontal thrusters 공유)

## Conclusions

- **PerDimEnt를 default로 채택** (arm=0.01, thr=0.001)
- Arm entropy boost는 단독으로는 유해. Thr entropy reduction과 반드시 함께 적용
- DORAEMON kl_ub=0.06에서도 PerDimEnt 안정적 — harder DR이 entropy collapse를 악화시키지 않음
- Baseline eval_dr: roll_deg 10.9 (training metric 최고) vs PerDimEnt 14.6 — eval_dr SS는 유사 (~1.5 deg)

## Impact on Next Round

PerDimEnt가 default entropy config로 확정. 이후 모든 round에서 arm=0.01, thr=0.001 사용.
SS error 감소가 다음 과제 -> Round 3 시작.
