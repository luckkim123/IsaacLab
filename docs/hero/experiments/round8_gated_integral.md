# Round 8: Error-Gated Integration (BEST POLICY)

Period: 2026-04-18.
R7-Integral의 overshoot 증가를 error-gated conditional integration으로 해결.
**R8-Gated가 SS error와 overshoot를 동시에 개선한 최초의 run.**

---

## Hypothesis

R7-Integral의 OS 증가 원인: policy가 "slow start + integral-driven push" 패턴을 학습.
Integral이 target 근처에서도 계속 누적되어 overshoot을 유발.

1. **Error-gated**: `|err| < reward_sigma` 일 때만 integration. Target crossing 시
   integral이 ~9x 감소 (simulation). SS correction은 유지 (SS zone에서만 적분)
2. **Faster leak**: leak 0.99->0.95 (tau 2.0s->0.39s). Integral decay를 빠르게 하여
   windup 억제. 간단하지만 SS correction도 약화될 risk
3. **6D baseline**: R7의 3D -> 6D 확장 (roll, pitch, vx, vy, vz, yaw_rate).
   yaw SS 악화 해결용 ungated reference

## Baseline

R7-Integral (`r7_integral`, 3D integral, 84D obs).

## Experimental Setup

All: 5000 iters, 2048 envs, 6D integral obs (87D total), tanh velocity penalty, PerDimEnt.

| Run | Config | Key Difference |
|-----|--------|---------------|
| R8-Gated (`2026-04-18_03-48-21_r8_gated`) | Error-gated 6D integral | Gate: accumulate only when \|err\| < sigma |
| R8-Baseline (`2026-04-18_08-17-35_r8_baseline`) | Ungated 6D integral | Always accumulate |
| R8-FastLeak (`2026-04-18_03-48-21_r8_fastleak`) | Ungated 6D, leak=0.95 | Faster decay (tau=0.39s vs 2.0s) |

## Results

### Aggregate Performance

| Metric | R7-Integral | R8-Gated | R8-Baseline | R8-FastLeak |
|--------|------------|----------|-------------|-------------|
| **SS error** | - | **0.131** | 0.206 | 0.275 |
| **OS %** | - | **13.1%** | 16.2% | **26.0%** |
| **n > 20%** | - | **16.0%** | - | 37.7% |
| Rise time | - | 0.318s | - | - |
| Entropy | - | 0.03 | -0.32 | -0.98 |

### Per-Axis Results (R8-Gated vs R7-Integral)

| Axis | Metric | R7-Integral | R8-Gated | Change |
|------|--------|-------------|----------|--------|
| Attitude | SS | 0.433 | **0.370** | -15% |
| Attitude | OS | 18.0% | **9.3%** | **-48%** |
| Velocity | SS | 0.030 | **0.014** | **-53%** |
| Velocity | OS | 21.7% | **10.4%** | **-52%** |
| Pitch | OS (none) | - | **6.2%** | Near-eliminated |
| Yaw | SS | 0.021 | **0.001** | **-95%** (6D integral fixed) |
| Yaw | OS | 32.5% | 34.4% | +6% (sole weakness) |

### R8-Baseline (6D ungated)

Middling results. SS=0.206, OS=16.2%.
- yaw OS=0.0% (integral의 over-damping으로 OS 완전 제거, but undershoot=5.0%)
- yaw SS=0.017 (R8-Gated 0.001의 17x)
- thruster_util=40.06 > budget 40.0 (constraint violation)

### R8-FastLeak: WORST

| Metric | R8-FastLeak | Note |
|--------|-------------|------|
| SS | 0.275 | Worst |
| OS | 26.0% | Worst |
| n > 20% | 37.7% | Worst |
| Pitch SS | 0.852 | 5x worse than R8-Gated (0.167) |
| Entropy | -0.98 | Deep collapse |

**Root cause**: Faster leak (tau=0.39s)이 SS correction을 파괴 (integral이 error offset 전에 decay)
+ transient에서 여전히 accumulate (no gating) -> **worst of both worlds**.

### Variance Analysis (R8-Gated vs R7-Integral)

48 comparisons (4 DR x 6 axis x {SS, OS}):
- **36/48 (75%) statistically significant** in favor of R8-Gated (t>2.0, p<0.05)
- **1/48 significantly worse**: none/yaw/OS (R8G 34.4% vs R7I 32.5%, d=+0.62)
- **11/48 not significant** (roll SS, vy SS, hard DR roll OS)

**Strongly significant results:**
- Yaw SS: d=-11.75, t=66.5 (massive improvement)
- Pitch OS: d=-5.10, t=28.9
- vx OS: d=-1.65, t=9.3

**Caveats:**
- Roll SS: 어떤 DR level에서도 significant하지 않음. std=0.405 > mean=0.341 (CV=1.19).
  특정 DR parameter combination에서 integral이 roll tracking을 destabilize할 수 있음
- Hard DR roll OS: std=22.5%, t=1.22. DR이 per-env variance를 amplify하여 signal mask

## Conclusions

- **R8-Gated = CURRENT BEST POLICY.** SS error와 OS를 동시에 개선한 최초이자 유일한 run
- Error-gated integration이 핵심 mechanism: SS zone에서만 적분하여 correction 유지 +
  transient에서 적분 차단하여 windup 방지
- **FastLeak approach 폐기.** Faster leak은 gating보다 strictly worse.
  SS correction과 transient damping을 모두 파괴
- **6D integral이 yaw SS를 해결**: R7 yaw SS 0.021 -> R8-Gated 0.001 (-95%)
- **Yaw OS (34.4%)가 유일한 남은 약점.** R8-Baseline의 yaw OS=0.0%는 over-damping으로 달성.
  Channel-specific gate configuration (ungated yaw / wider gate threshold)이 후속 과제

## Open Questions (R8 종료 시점)

1. Yaw OS 34.4%: channel-specific gate (ungated yaw, or 2*sigma threshold)로 개선 가능?
2. 전 R8 run에서 entropy collapse (Gated 0.03, FastLeak -0.98). PerDimEnt tuning 필요?
3. Roll SS의 high per-env variance (std > mean): 특정 DR parameter와의 correlation 분석 필요
4. Roll OS 14.5% (target 10% 이상): 별도 조사 필요

## Cross-Round Summary Table

전 라운드 hard-DR SS 비교 (핵심 metric):

| Round | Run | Roll SS | Pitch SS | vy SS | Yaw SS | Note |
|-------|-----|---------|----------|-------|--------|------|
| R2 | Control (PerDimEnt) | 1.68 | 1.38 | 0.059 | 0.025 | Baseline |
| R3 | Exp-L1 | 1.91 | 1.47 | 0.044 | 0.019 | SS down, OS up |
| R4 | Tanh | 1.53 | 1.46 | 0.045 | 0.021 | Reward erosion |
| R4 | Arctan | 1.42 | 1.44 | 0.051 | 0.028 | Roll-only win |
| R6 | VelTanh c=0.3 | 1.05 | 0.69 | 0.037 | 0.011 | 4/4 none-DR pass |
| R7 | Integral | ~0.98 | ~0.49 | 0.016 | 0.021 | SS -50-67%, OS up |
| **R8** | **Gated** | - | **0.370** | **0.014** | **0.001** | **SS+OS simultaneous** |

*Note: R8 SS 단위가 이전과 다를 수 있음 (aggregate vs per-axis). att_norm SS로 비교.*
