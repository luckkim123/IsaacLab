# Domain Randomization

> **Status**: 2026-02-11 | **Source**: `config.py`, `mdp/events.py`, `base_env.py`
>
> Hero Agent ALBC 환경의 Domain Randomization(DR) 구현 전체 검토.
> 10개 카테고리, 30+ 파라미터, Fossen 모델 기반 물리적 랜덤화.

---

## Overview

DR은 reset-time에만 적용된다 (에피소드 중 변경 없음). 이는 DR이 "다른 로봇 인스턴스"를 나타내는 것이지, 시변 동역학이 아니기 때문이다.

---

## DR Items

### A. Initial Pose (6 parameters)

| Item | Range | Distribution | Physical Meaning |
|:---|:---|:---|:---|
| position_x | [-0.5, 0.5] m | Uniform | 수평 오프셋 |
| position_y | [-0.5, 0.5] m | Uniform | 수평 오프셋 |
| position_z | [4.0, 5.0] m | Uniform | 초기 깊이 |
| roll | [-0.785, 0.785] rad (+-45deg) | Uniform | 초기 roll 기울기 |
| pitch | [-0.785, 0.785] rad (+-45deg) | Uniform | 초기 pitch 기울기 |
| yaw | [-pi, pi] rad | Uniform | 초기 방향 |

Quaternion 기반 회전 (gimbal lock 방지). Position은 기본값에 대한 additive offset.

+-45deg 범위는 의도적으로 공격적이다. DLS IK가 singularity 근처를 자연스럽게 처리하므로, 넓은 초기 자세에서의 robust policy 학습이 가능하다.

### B. Hydrodynamic Parameters (7 categories, main body + buoy 각각 적용)

| Item | Range | Method | DOF | Physical Meaning |
|:---|:---|:---|:---|:---|
| added_mass_scale | [0.7, 1.3] | Multiplicative | 6 (independent) | Added mass 불확실성 (+-30%) |
| linear_damping_scale | [0.7, 1.3] | Multiplicative | 6 (independent) | 마찰 감쇠 불확실성 (+-30%) |
| quadratic_damping_scale | [0.6, 1.4] | Multiplicative | 6 (independent) | 형상 항력 불확실성 (+-40%) |
| volume_scale | [0.9, 1.1] | Multiplicative | scalar | 부력 불확실성 (+-10%) |
| cob_offset | +-1cm (xy), +-4cm (z) | Additive | 3 | 부력 중심 오차 |
| cog_offset | +-1cm (xy), +-4cm (z) | Additive | 3 | 질량 중심 오차 |
| inertia_scale | [0.8, 1.2] | Multiplicative | 3 (independent) | 관성 모멘트 불확실성 (+-20%) |

Implementation: `_randomize_hydro_model()` in `mdp/events.py`. Base tensor는 `_HydroBaseCache`로 캐싱하여 4096 병렬 환경에서의 성능 보장.

### C. Ocean Current (3 active + 3 disabled)

| Item | Range | Distribution |
|:---|:---|:---|
| linear_x/y | [-0.2, 0.2] m/s + N(0, 0.05) | Uniform + Gaussian |
| linear_z | [-0.1, 0.1] m/s + N(0, 0.02) | Uniform + Gaussian |
| angular_x/y/z | 0 (disabled) | - |

Main body와 buoy에 동일한 해류 적용 (동일 수역). 에피소드 중 일정 (reset-time only). 에피소드 길이 (~15s)가 해류 변동 시간 스케일보다 짧으므로 시변 모델링은 불필요.

### D. Joint Initial State (2 parameters)

| Item | Range | Note |
|:---|:---|:---|
| joint1_pos | [-pi, pi] rad | 관절 한계 내 클램핑, 전 범위 |
| joint2_pos | [-pi, pi] rad | Target buffer도 동기화 |

### E. Payload (4 parameters, `enable_payload=True`일 때만)

Payload는 **gripper body**에 적용된다 (base에 고정 조인트로 연결, 오프셋 (0, 0.0881, -0.185)). PhysX가 고정 조인트를 통해 힘을 자동 전파한다.

| Item | Range | Note |
|:---|:---|:---|
| mass | [0.0, 1.0] kg | Weight 모델만 (drag 없음), 0=페이로드 없음 |
| cog_offset_x | [-0.50, 0.50] m | 부착점 기준 CoG 오프셋 |
| cog_offset_y | [-0.50, 0.50] m | 부착점 기준 CoG 오프셋 |
| cog_offset_z | [-0.20, 0.0] m | 부착점 아래 방향 오프셋 |

Implementation: `randomize_payload()` in `mdp/events.py`.
- Payload force: $F = mg$, gripper body frame으로 변환
- Payload torque: $\tau = (\mathbf{r}_{attach} + \mathbf{r}_{cog}) \times F$
- CoG offset은 페이로드의 질량 분포 불확실성 모델링 (비대칭 도구, 긴 막대 등)

### F. Joint Actuator Gains (2 parameters)

| Item | Range (Base RL) | Range (TDC) | Note |
|:---|:---|:---|:---|
| stiffness (Kp) | [80.0, 120.0] | [160.0, 240.0] | Asset default: 100.0 / TDC optimal: 200.0 |
| damping (Kd) | [2.4, 3.6] | [8.0, 12.0] | Asset default: 3.0 / TDC optimal: 10.0 |

동일 환경 내 두 ALBC 관절에 같은 값 적용. TDC 환경은 별도 게인 범위 사용.

### G. Body Mass (1 parameter, multiplicative scale)

| Item | Range | Note |
|:---|:---|:---|
| body_mass_scale | [0.9, 1.1] | 모든 rigid body에 동일 스케일 (+-10%) |

PhysX `set_masses()` API 사용. 제조 공차 모델링. 관성은 hydro DR의 `inertia_scale`로 별도 랜덤화.

### H. Water Density (1 parameter)

| Item | Range | Note |
|:---|:---|:---|
| water_density | [995.0, 1025.0] kg/m^3 | 담수~해수 전 범위 |

Per-env tensor. 부력 ($F_b = \rho V g$)과 항력 ($F_d = 0.5 \rho C_d A v^2$) 모두에 영향.

### I. Sensor Noise (IMU bias + white noise)

| Item | Range | Note |
|:---|:---|:---|
| euler noise (3D) | N(0, 0.01 rad) | White noise per step |
| euler bias (3D) | U(-0.005, 0.005 rad) | Per-episode 샘플링 |
| ang_vel noise (3D) | N(0, 0.02 rad/s) | White noise per step |
| ang_vel bias (3D) | U(-0.01, 0.01 rad/s) | Per-episode 샘플링 |
| other dims (7D) | 0 | att_error, joint_pos, prev_actions에는 노이즈 없음 |

`NoiseModelWithAdditiveBiasCfg` 사용. Bias는 리셋 시 샘플링 (per-episode gyro drift 모델), white noise는 매 스텝 추가. Obs dims 0-5 (IMU)에만 적용, dims 6-12는 정확.

### J. Joint Friction (2 parameters)

| Item | Range | Note |
|:---|:---|:---|
| static_friction | [0.0, 0.05] | Coulomb 마찰 계수 |
| viscous_friction | [0.0, 0.3] | 속도 비례 저항 |

두 ALBC 관절에 동일 값 적용.

---

## Per-Environment DR Activation

| Environment | Task ID | DR | Current | Payload | Noise | Target DR |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| HeroAgentEnvCfg (debug) | Isaac-HeroAgent-v0 | OFF | OFF | OFF | OFF | OFF |
| HeroAgentTrainEnvCfg | Isaac-HeroAgent-Base-v0 | ON | ON | ON | ON | ON |
| HeroAgentEncoderTrainEnvCfg | Isaac-HeroAgent-Encoder-Base-v0 | ON | ON | ON | ON | ON |
| HeroAgentTDCEnvCfg | Isaac-HeroAgent-TDC-v0 | Pose(15deg) | OFF | OFF | OFF | OFF |
| HeroAgentEncoderTDCEnvCfg | Isaac-HeroAgent-Encoder-TDC-v0 | ON | ON | ON | ON | ON |
| HeroAgentAdaptTDCEnvCfg | Isaac-HeroAgent-Adapt-TDC-v0 | ON | ON | ON | ON | ON |

---

## DR Application Sequence

All DR은 reset-time에만 적용된다 (runtime randomization 없음).

```
_reset_idx() execution order:
  1. Logging (episode metrics)
  2. Component reset (robot, action buffers)
  3. Episode length decorrelation (full batch: full range, individual: 10% jitter)
  4. Hydrodynamics reset + DR
     - hydro.reset() + buoy_hydro.reset() (density also reset)
     - payload reset to defaults
     - randomize_hydrodynamics() [if enabled] (includes water density)
     - randomize_body_mass() [if enabled]
     - randomize_payload() [if enabled]
     - randomize_ocean_current() [if has current]
  5. Attitude task reset
  6. Robot state reset
     - randomize_joint_positions() [if DR]
     - randomize_robot_pose() [if DR]
  7. Joint actuator DR (always applied -- resets to defaults when DR disabled)
     - randomize_joint_gains()
     - randomize_joint_friction()
  8. Potential initialization
```

---

## Privileged Observations (Encoder)

HORA encoder 훈련을 위한 24D privileged information:

```
Main body (10D): [volume(1), CoG(3), CoB(3), inertia(3)]
Buoy body (10D): [volume(1), CoG(3), CoB(3), inertia(3)]
Payload    (4D): [mass(1), cog_offset(3)]
```

| Category | Included | Excluded | Rationale |
|:---|:---|:---|:---|
| Hydrostatic | Volume, CoB, CoG, inertia | - | 자세 제어의 핵심 파라미터 |
| Hydrodynamic | - | Added mass, damping | 동적 응답 속도에 영향, 정상상태 자세에는 미미 |
| External | Payload (mass + CoG) | Ocean current | 페이로드는 복원 토크 직접 변경 |
| Sensor | - | Noise/bias | 관측 노이즈는 policy robustness로 처리 |

---

## Implementation Quality

### Strengths

1. **물리적 원칙 기반**: Fossen 모델의 질량, 감쇠, Coriolis, 부력을 정확히 분리
2. **이중 수력학 DR**: Main body와 buoy를 독립적으로 랜덤화 (buoy 부력 = 제어 권한)
3. **CoG 보정 토크**: PhysX nominal CoG와 DR CoG 차이를 정확히 보상
4. **Caching**: `_HydroBaseCache`로 텐서 재생성 방지 (4096 병렬 환경 성능)
5. **Reset-time only**: 물리적으로 정확 (다른 로봇 인스턴스 모델)
6. **Episode decorrelation**: 초기 분산 + jitter로 환경 동기화 방지

### Resolved Issues

| Issue | Description | Resolution |
|:---|:---|:---|
| Body mass 미랜덤화 | Net buoyancy 불확실성 비대칭 | Section G: PhysX `set_masses()` (+-10%) |
| Water density 고정 | 담수/해수 간 전이 불가 | Section H: Per-env tensor (995-1025) |
| Sensor noise 없음 | Sim-to-real gap 주요 원인 | Section I: IMU bias + white noise |
| Joint friction 없음 | 관절 저항 미모델링 | Section J: Static + viscous friction |

### Remaining Minor Issues

| Issue | Description | Verdict |
|:---|:---|:---|
| Main/buoy 동일 DR 범위 | Scale factor가 multiplicative이므로 base 값 차이가 자동 반영 | Minor, 필요시 `BuoyDRCfg` 분리 |
| Ocean current 시불변 | 에피소드 ~15s < 해류 변동 시간 | Acceptable |
| Damping 미포함 (privileged obs) | 자세 제어에는 hydrostatic이 지배적 | Design choice |

---

## Base Parameter Reference

### Main Body (HeroAgentHydrodynamicsCfg)

| Parameter | Value |
|:---|:---|
| Geometry | Cylinder R=0.09m, L=0.325m, m=9.18kg |
| Water density | 998 kg/m^3 (default) |
| Volume | 0.00827 m^3 |
| Buoyancy / Weight | 80.9N / 90.1N (net: -9.2N, negatively buoyant) |
| Added mass | (0.6, 5.76, 5.76, 0.04, 0.05, 0.05) |
| Linear damping | (2.0, 4.0, 4.0, 0.1, 0.1, 0.1) |
| Quadratic damping | (26.0, 26.0, 10.7, 1.5, 1.5, 0.01) |
| CoB | (0.0, 0.0, 0.0) |
| CoG | (0.0, 0.0, -0.10) |
| Inertia | (0.0994, 0.0994, 0.0372) |

### Buoy Body (HeroAgentBuoyHydrodynamicsCfg)

| Parameter | Value |
|:---|:---|
| Geometry | Cylinder R=0.085m, H=0.118m, m=0.93kg |
| Volume | 0.00268 m^3 |
| Buoyancy / Weight | 26.2N / 9.1N (net: +17.1N, positively buoyant) |
| Added mass | (0.15, 1.5, 1.5, 0.01, 0.01, 0.01) |
| Linear damping | (0.5, 0.5, 0.5, 0.01, 0.01, 0.01) |
| Quadratic damping | (4.6, 4.6, 4.6, 0.1, 0.1, 0.1) |
| CoB / CoG | (0.0, 0.0, 0.0) / (0.0, 0.0, 0.0) |
| Inertia | (0.00278, 0.00278, 0.00336) |

### System Total

| Parameter | Value |
|:---|:---|
| Total buoyancy | 80.9 + 26.2 = 107.1 N |
| Total weight | ~104.1 N (approximate) |
| Net | ~+3.0 N (slightly positively buoyant) |

---

## Related Documents

- [ARCHITECTURE.md](./ARCHITECTURE.md): 환경 구조 및 시뮬레이션 설정
- [SIM_TO_REAL.md](./SIM_TO_REAL.md): Sim-to-real gap 분석 및 배포
- [TRAINING_PIPELINE.md](./TRAINING_PIPELINE.md): HORA encoder의 privileged obs 사용

---

**Created**: 2026-02-11
**Updated**: 2026-02-11 (Consolidated from research note 09. Issues B/C resolved, privileged obs 22D->24D.)
