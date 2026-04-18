# Round 1: Per-Dim Noise Comparison

Period: 2026-04-14.
Per-dim entropy coefficient vs max_std cap의 noise 관리 효과 비교.

---

## Hypothesis

Arm dim(0-1)과 thruster dim(2-7)의 noise dynamics가 근본적으로 다르므로
(arm: collapse, thruster: divergence), uniform entropy_coef 대신 per-dim coefficient를
사용하면 양쪽 문제를 동시에 해결할 수 있다.

## Baseline

기존 uniform entropy_coef=0.003 (entropy collapse 조사 후 복원된 값).

## Experimental Setup

- 3-run comparison, num_envs=1024, 2500 iters, kl_ub=0.04, seed=30
- 짧은 run으로 noise dynamics의 초기 분화를 관찰

| Run | Config | Key Change |
|-----|--------|-----------|
| Baseline (`2026-04-14_15-23-49_baseline`) | entropy_coef=0.003 uniform | Control |
| PerDimEnt (`2026-04-14_15-22-17_perdiment`) | arm=0.01, thr=0.001 | Per-dim entropy |
| MaxStd1 (`2026-04-14_15-22-40_maxstd1`) | max_std=1.0 (was 2.0) | Upper cap |

## Results

### Training Metrics (2500 iters)

| Metric | Baseline | PerDimEnt | MaxStd1 |
|--------|----------|-----------|---------|
| Reward | 193.8 | **204.7** | 187.4 |
| att_rp (deg) | 5.53 | **5.03** | 5.72 |
| Smoothness | -0.24 | **-0.11** | -0.23 |
| Thruster penalty | -0.17 | **-0.09** | -0.14 |
| Entropy | 2.81 | 0.51 | 2.39 |
| DORAEMON success | >0.95 | >0.95 | >0.95 |

### Noise Dynamics

| Dim | Baseline | PerDimEnt | MaxStd1 |
|-----|----------|-----------|---------|
| arm0 | 0.100 (floor) | **0.144** (above floor) | 0.100 (floor) |
| arm1 | 0.114 | **0.225** | 0.118 |
| thr6 | 0.731 | **0.370** | 0.635 |
| thr7 | 0.893 | **0.349** | 0.856 |

PerDimEnt: arm noise가 floor 위에서 equilibrium 형성 (arm0: slight recovery 0.142->0.147
after iter 750). Thruster noise 공격적으로 감소 (0.25-0.37 range).

## Analysis

1. **PerDimEnt가 명확히 우수**: reward +5.6%, attitude +9.1%, smoothness 2.2x 개선
2. **Arm noise equilibrium 확인**: arm=0.01이 arm net gradient를 +0.003으로 역전,
   collapse 방지. arm0=0.144는 floor(0.10) 위에서 자연 균형점 형성
3. **Thruster noise 급감**: thr=0.001이 divergence 억제. 전 dim 0.25-0.37 (vs baseline 0.54-0.82)
4. **MaxStd1 효과 미미**: dim7=0.856 (cap 1.0 미도달). Reward 최하위. 상한 cap은 divergence 억제 불충분
5. **Smoothness reward가 가장 명확한 차별 지표**: PerDimEnt -0.25->-0.11 (지속 개선),
   Baseline/MaxStd1 flat (-0.25->-0.24/-0.23)

## Conclusions

- **PerDimEnt 채택 결정** (arm=0.01, thr=0.001). Max_std cap은 보조 수단으로만 유지.
- PerDimEnt의 낮은 entropy (0.51)가 DR 적응에 지장을 주는지는 Round 2에서 검증 필요.
- 짧은 2500 iter에서 DORAEMON kl_ub=0.04는 DR stress 부족 (전 run >0.95 success).

## Impact on Next Round

Round 2: PerDimEnt를 harder DR (kl_ub=0.06)에서 검증 + arm vs thr contribution 분리.
