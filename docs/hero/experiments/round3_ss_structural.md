# Round 3: Structural Fixes for SS Error

Period: 2026-04-16.
SS error의 reward gradient dead zone 문제에 대한 구조적 해결 시도:
L1 linear penalty와 settling constraint.

---

## Hypothesis

`r(e) = exp(-e^2/2s^2) - q*e^2` 형태에서 `dr/de = 0 at e=0` (수학적 증명).
Weight tuning은 모든 gradient를 동일 비율로 scaling할 뿐 zero-at-zero 구조를 변경 불가.
구조적 변경 필요: (a) L1 term으로 constant gradient 추가, (b) settling constraint로 반가 압력.

### Reward Dead Zone (mathematical proof)

```
dr/de = -e × (1/σ² × exp(-e²/2σ²) + 2q)
At e=0: gradient = 0 regardless of weights
At e=0.01: gradient = 16% of peak (at e=σ=0.10)
```

### Constraint Asymmetry Discovery

`rp_vel_settling_cost`가 `|att_err| < 5°` 일 때 `|p|+|q|` penalize -> attitude anti-overshoot.
lin_vel과 yaw에는 동등한 mechanism 없음 -> attitude는 양호, lin/yaw는 overshoot 발생.

## Baseline

Round 2 PerDimEnt control (`2026-04-14_18-55-20_perdiment_kl06`).

## Experimental Setup

| Run | Task | Config | Target |
|-----|------|--------|--------|
| Exp-L1 (`exp_l1_ss`) | Isaac-FullDOF-Exp-L1-v0 | lin_vel_lin_ratio=0.15, yaw_vel_lin_ratio=0.15 | SS error 감소 |
| Exp-Settling (`exp_settling_overshoot`) | Isaac-FullDOF-Exp-Settling-v0 | lin_vel_settling + yaw_settling constraints | OS 감소 |

Both: kl_ub=0.06, num_envs=2048, 5000 iters.

## Results

### eval_dr_fulldof (64 envs x 4 DR levels)

**Exp-L1:**

| Metric | Control | Exp-L1 | Change |
|--------|---------|--------|--------|
| vx SS | - | -15 to -21% | Improved |
| vy SS | - | -18 to -24% | Improved |
| yaw SS | - | -10 to -21% | Improved |
| att OS | - | **+25-60%** | Degraded |
| vx OS | - | **+49-86%** | Degraded |
| vy OS | - | **+51-70%** | Degraded |
| Rise time | - | 20-38% faster | Aggressive |
| Reward | 148 | 154 | Stable |

Mechanism 확인: L1의 constant far-field gradient가 aggressive controller 학습.
SS 개선과 OS 악화의 classic L1 tradeoff.

**Exp-Settling:**

| Metric | Control | Settling | Change |
|--------|---------|----------|--------|
| yaw SS | 0.012-0.019 | **0.272-0.337** | **+20x 악화** |
| Reward | 148 | 103 | -31% |
| Entropy | -0.30 | -3.08 | 10x deeper collapse |

**Catastrophic failure.** Policy가 yaw tracking을 완전히 포기.
yaw_rate = -0.2 rad/s (target = +0.3 rad/s, 반대 방향).

### Root Cause: Settling Constraint 3-fold Failure

1. `yaw_settling` cost가 iter 50부터 budget 초과 (1.077 vs d_k=0.5).
   Barrier gradient가 policy 학습 전에 활성화
2. threshold=0.04 rad/s가 typical yaw error range (0.1-0.3)보다 tight.
   near_target gate가 정상 tracking 중 거의 활성화되지 않음
3. **Perverse incentive**: yaw_rate_err ≈ 0.32에서 local optimum 발견.
   exp kernel reward ~0.006 (포기) but yaw_settling gate = 0 (cost 없음).
   Target 회피가 target 도달보다 용이

`lin_vel_settling`은 cost=0.010 vs budget=0.50 (2% utilization) -- threshold 0.04 m/s가
너무 tight하여 아무 정보도 제공하지 못함.

## Conclusions

- **L1**: SS 개선 mechanism은 확인됐으나, far-field constant gradient로 인한 OS 악화가 불가피.
  Pure L1은 SS/OS tradeoff에서 양쪽을 동시에 개선 불가
- **Settling constraint: 구조적 dead end 선언.** Binary gate `(err < thr)*|dv|`의 perverse
  incentive는 parameter tuning으로 수정 불가. rp_vel_settling (attitude)이 작동하는 이유는
  attitude exp kernel이 충분히 강해서 dominate하기 때문
- Smoothness penalty는 구조적으로 약함: -0.090/episode (reward의 0.06%). 10x 증가해도 0.6%

## Impact on Next Round

Round 4: L1의 zero-at-zero gradient fix는 유효하나 far-field force가 문제.
-> Saturating penalty (Tanh/Arctan): near-zero에서 gradient + far-field에서 saturation.
