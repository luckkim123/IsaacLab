# RL-TDC vs Hero Agent Encoder-TDC - Comparative Analysis

> **Status**: 2026-02-16 | Comparative study + stability reward feasibility analysis
>
> RL-based Adaptive TDC (RL-TDC) 논문과 Hero Agent Encoder-TDC 아키텍처의 비교 분석.
> Section 7: M_bb(gamma) 시뮬레이터 계산 및 stability-aware reward 구현 검토 (2026-02-16 추가).
> 기반 논문: Baek et al. "A Reinforcement Learning-based Adaptive Time-Delay Control
> and Its Application to Robot Manipulators" (ACC 2022).
> 참조 디렉토리: `/workspace/references/RL-TDC - RL-based Adaptive Time-Delay Control/`

---

## 1. Overview

두 방법 모두 **end-to-end RL이 아닌, TDC 제어기의 파라미터를 RL로 적응시키는** 전략을 취한다.
핵심 차이는 **"RL이 무엇을 출력하는가"**에 있다.

| 항목 | RL-TDC (Taefi et al.) | Hero Agent Encoder-TDC |
|:-----|:----------------------|:-----------------------|
| RL 출력 | 관성 게인 M_hat(t) | PD 게인 [Kp, Kd] |
| M_hat 결정 | RL 정책이 직접 출력 | Encoder가 privileged info로 추정 |
| RL 알고리즘 | SAC (off-policy) | PPO (on-policy) |
| 적용 시스템 | 2-DOF manipulator (육상) | 6-DOF UUV with ALBC (수중) |
| 학습 단계 | 1단계 (SAC end-to-end) | 2단계 (Phase1 Encoder+Actor, Phase2 Adaptation) |
| Domain Randomization | 없음 | 12종류, 35+ 파라미터 |
| 배포 전략 | 학습된 정책 그대로 | Phase2 AdaptTConv가 privileged info 대체 |

---

## 2. RL-TDC: Core Approach

### 2.1 Problem Statement

기존 TDC는 고정 관성 게인 M_bar를 사용하며, 이는 근본적 trade-off를 만든다:

- **큰 M_bar**: 추적 성능 향상, 노이즈 증폭
- **작은 M_bar**: 노이즈 감소, 추적 성능 저하

기존 Adaptive TDC (ATDC)는 수학적으로 고정된 적응 법칙 (gradient descent 등)을 사용하여,
추적 오차만 고려하는 보수적인 적응만 가능했다.

### 2.2 Solution: Structure-Free Adaptive Law

고정된 적응 법칙을 **SAC로 학습된 신경망 정책**으로 대체한다.
정책은 시스템 상태에 기반하여 시간 변동 관성 게인 M_hat(t)를 출력한다.

### 2.3 Control Law

$$\tau_t = \hat{M}_t [\ddot{q}_{d,t} + K_d \dot{e}_t + K_p e_t] + \underbrace{\tau_{t-L} - \hat{M}_{t-L} \ddot{q}_{t-L}}_{\hat{N}_t \text{ (TDE)}}$$

여기서:
- $\hat{M}_t$: RL 정책이 출력하는 **시간 변동 관성 게인** (핵심)
- $K_p, K_d$: 고정 PD 게인 (모든 baseline 동일)
- $\hat{N}_t$: Time Delay Estimation (미모델링 동역학 추정)
- $L$: 샘플링 주기 (1 ms)

기존 TDC와의 차이: M_bar (상수) 대신 M_hat(t) (시간 변동, 상태 의존적).

### 2.4 MDP Formulation

**State (5n-D, n=DOF)**:

$$s_t = [\dot{e}_t, e_t, \tau_{t-L}, \hat{M}_{t-L}, \ddot{q}_{t-L}]$$

| 요소 | 차원 | 기존 ATDC 사용 여부 |
|:-----|:-----|:-------------------|
| e_dot (오차 미분) | n | O |
| e (추적 오차) | n | O |
| tau_prev (이전 토크) | n | X -- 신규 |
| M_hat_prev (이전 관성) | n | X -- 신규 |
| q_ddot_prev (이전 가속도) | n | X -- 신규 |

**확장 상태의 핵심 의의**: 제어 법칙에 등장하는 모든 변수를 포함하여,
정책이 TDE 오차의 크기를 간접적으로 파악할 수 있게 한다.

**Action**: M_hat(t), squashing으로 양수 보장

$$\hat{M}_t = \frac{a_{\max}}{2} (\tanh(a_t) + 1)$$

**Reward**: 안정성 인식 보상

$$r_t = \begin{cases} C \cdot \exp(-K \|e_t\|) & \text{if stability condition met} \\ 0 & \text{otherwise} \end{cases}$$

**Stability Condition** (시간 변동 관성에 대한 확장):

$$\left\| \left[ I - M_t^{-1} \bar{M} \right] \frac{\hat{M}_t}{\hat{M}_{t-L}} \right\| < 1$$

---

## 3. Hero Agent Encoder-TDC: Core Approach

### 3.1 Architecture

HORA (History-based Online Robust Adaptation) 프레임워크에 TDC를 통합.
RL은 PD 게인을 출력하고, encoder는 privileged info로부터 M_hat을 추정한다.

```
Privileged Info (24D) --> Encoder --> z (6D latent)
    |-- z[3:5] --> M_hat (roll, pitch inertia) --> TDC Controller
    |-- z (full 6D) + policy_obs (13D) --> Actor (19D) --> 4D gains
                                                            |
                                           [Kp_roll, Kp_pitch, Kd_roll, Kd_pitch]
                                                            |
                                                     TDC Controller
                                                            |
                                                   p_EE --> DLS IK --> Joint Targets
```

### 3.2 Control Law

$$\tau = \hat{M} \cdot u_{pd} + \hat{U} + \Delta T_b$$

여기서:
- $\hat{M}$: Encoder가 추정한 관성 행렬 (z[3:5], 물리적 의미 보존)
- $u_{pd} = K_p e + K_d \dot{e}$: RL이 출력한 게인으로 계산
- $\hat{U} = \Lambda_{\text{prev}} p_{EE,\text{prev}} - \hat{M} \dot{\nu}_{\text{prev}}$: TDE
- $\Delta T_b = T_{b,\text{prev}} - T_b$: 복원 토크 변화량 보상

### 3.3 State/Action Design

**Policy Observations (13D)**:

| 인덱스 | 내용 | 차원 |
|:-------|:-----|:-----|
| 0:3 | euler angles (roll, pitch, yaw) | 3 |
| 3:6 | angular velocity (body frame) | 3 |
| 6:9 | attitude error (target - current) | 3 |
| 9:11 | joint positions (normalized) | 2 |
| 11:13 | previous actions (Kp only) | 2 |

+ Encoder z (6D) = 총 19D 입력

**Privileged Observations (24D)**:
- Main body (10D): volume, CoG(3), CoB(3), inertia(3)
- Buoy body (10D): volume, CoG(3), CoB(3), inertia(3)
- Payload (4D): mass, cog_offset_xyz(3)

**Action (4D)**: sigmoid scaling으로 범위 보장

$$K_p = K_{p,\min} + \sigma(\text{raw}) \cdot (K_{p,\max} - K_{p,\min}), \quad K_p \in [10, 100]$$
$$K_d = K_{d,\min} + \sigma(\text{raw}) \cdot (K_{d,\max} - K_{d,\min}), \quad K_d \in [2, 30]$$

### 3.4 Two-Phase Training (HORA)

**Phase 1**: Encoder + Actor + Critic (PPO)
- Encoder가 privileged info를 z로 압축
- Actor가 z + policy_obs로 TDC 게인 출력
- Critic은 actor와 동일한 19D 입력 (symmetric) -- encoder 학습 강제

**Phase 2**: Adaptation Module (Supervised L2)
- ProprioAdaptTConv: proprioception history (30 step x 12D) --> z_hat (6D)
- Phase 1 encoder/actor를 freeze하고, adapt module만 학습
- On-policy data collection (adapt z_hat이 actor를 구동)

---

## 4. Detailed Comparison

### 4.1 RL 출력의 역할

| 관점 | RL-TDC | Hero Agent |
|:-----|:-------|:-----------|
| **무엇을 적응** | M_hat (관성 게인 = TDE의 핵심 파라미터) | Kp, Kd (PD 피드백 게인) |
| **M_hat 역할** | 제어 변수 (물리적 의미 불필요) | 물리 파라미터 추정값 (encoder가 학습) |
| **적응 속도** | 매 step 변경 (펄스형 가능) | 매 step 변경 (sigmoid로 연속적) |
| **정보원** | 확장 상태 벡터 (과거 토크/가속도) | Encoder latent (privileged info 압축) |

RL-TDC에서 M_hat은 "제어를 위한 변수"이지 물리적 관성이 아니다.
Hero Agent에서 M_hat은 encoder가 실제 물리 파라미터를 추정한 값이며,
TDE의 정확도와 직접 연결된다.

### 4.2 상태 공간 설계

RL-TDC는 **제어 법칙에 나타나는 모든 변수**를 상태에 포함시킨다:
- tau_prev: 이전 제어 토크 (TDE의 입력)
- M_hat_prev: 이전 관성 게인 (TDE의 입력)
- q_ddot_prev: 이전 가속도 (TDE의 입력)

이를 통해 정책이 현재 TDE 오차의 크기를 간접적으로 파악 가능.

Hero Agent는 이 대신 **encoder z를 통해 동역학 정보를 전달**한다.
encoder가 privileged info를 6D latent로 압축하므로,
정책은 시스템의 현재 동역학 상태를 파악할 수 있다.
단, TDE 오차 자체에 대한 직접적 관측은 없다.

### 4.3 안정성 보장

| 전략 | RL-TDC | Hero Agent |
|:-----|:-------|:-----------|
| 방식 | Soft constraint (reward=0) | Hard constraint (sigmoid bounds) |
| 이론적 근거 | 확장 안정성 조건 (Eq.16) | 게인 범위 경험적 설정 |
| 실제 행동 | 안정성 경계 근처에서 공격적 제어 | 항상 안전 범위 내 |
| M_hat 변동 | 급격한 변동 허용 (펄스형) | softplus(z) >= 0.1, 연속적 |

RL-TDC의 안정성 조건은 M_hat(t)/M_hat(t-L) 비율을 포함하여,
급격한 파라미터 변화를 자연스럽게 억제한다.

### 4.4 Domain Randomization

| 항목 | RL-TDC | Hero Agent |
|:-----|:-------|:-----------|
| DR 전략 | 없음 | Reset-time + Per-step DR |
| 강인성 검증 | 학습 외 궤적 + 페이로드 제거 | DR 범위 내 일반화 |
| 범위 | N/A | added_mass +/-50%, body_mass +/-15%, etc. |
| 외란 | 없음 | 해류, 페이로드, 센서 노이즈, 액션 지연 |

Hero Agent의 DR은 이론적 TDE 안정성 경계를 **의도적으로 초과**하여
(관성비 > 2) 강인한 적응을 강제한다.

### 4.5 Sim-to-Real 전략

| 항목 | RL-TDC | Hero Agent |
|:-----|:-------|:-----------|
| 배포 | 학습된 정책 그대로 | Phase2 AdaptTConv (proprio only) |
| Privileged info | 불필요 (모든 상태 관측 가능) | Phase2가 대체 |
| 실제 검증 | 시뮬레이션만 | 시뮬레이션 (실물 실험 계획 중) |
| 센서 요구사항 | joint encoder + force sensor | IMU + joint encoder |

---

## 5. Performance Comparison Context

### RL-TDC Results (2-DOF Manipulator)

| 방법 | NRT (nominal) | HFRT (high-freq) | DRT (discontinuous) | Payload Removal |
|:-----|:-------------|:-----------------|:-------------------|:----------------|
| TDC | 0.889 deg | 1.163 deg | 2.421 deg | Unstable |
| ATDC | 0.856 deg | 1.132 deg | 2.385 deg | Unstable |
| ATDC-SMC | 0.827 deg | 1.057 deg | 2.317 deg | Unstable |
| **RL-TDC** | **0.815 deg** | **1.054 deg** | **1.873 deg** | **0.779 deg** |

핵심 관찰: DRT에서 ~20% 개선, payload removal에서 유일하게 안정.
M_hat(t)가 충격 시 펄스형으로 증가하여 강인한 대응.

### 비교 제약

직접적인 수치 비교는 불가능하다:
- 시스템이 근본적으로 다름 (2-DOF 매니퓰레이터 vs 6-DOF UUV)
- 제어 대상이 다름 (joint-space tracking vs end-effector attitude stabilization)
- 외란 특성이 다름 (마찰/중력 vs 부력/해류/수동역학)

---

## 6. Lessons and Potential Improvements

### 6.1 RL-TDC에서 배울 수 있는 점

1. **확장 상태 벡터**: Hero Agent policy_obs에 TDE 관련 변수 추가 고려
   - tau_prev (이전 제어 토크)
   - TDE 잔차 크기 (U_hat의 norm)
   - M_hat(t) / M_hat(t-1) 비율
   - 이를 통해 정책이 TDE 오차를 인지하고 게인을 보상적으로 조정 가능

2. **안정성 인식 보상**: 현재 Hero Agent에는 TDC 안정성 기반 보상이 없음
   - TDE 잔차가 특정 임계를 초과하면 보상 감소
   - 게인 변화율 (Kp(t) - Kp(t-1)) 페널티 (이미 action_rate로 일부 구현)

3. **M_hat의 동적 조정**: 현재 encoder M_hat은 episode 내에서 상대적으로 안정적
   - RL-TDC처럼 충격 시 일시적 M_hat 보정을 RL이 추가로 학습하는 하이브리드 가능

### 6.2 Hero Agent이 이미 우월한 점

1. **Sim-to-Real 파이프라인**: HORA 2-phase가 실제 배포를 가능하게 함
2. **물리적 M_hat**: Encoder가 물리 파라미터에서 추정 -> TDE 정확도 향상
3. **공격적 DR**: RL-TDC의 단일 nominal 학습 대비 훨씬 강인
4. **복잡한 시스템**: 수중 환경의 비선형 동역학 (부력, 해류, ALBC)
5. **이중 적응**: M_hat (encoder) + Kp/Kd (RL) 동시 적응

### 6.3 Potential Hybrid Architecture

```
Encoder (privileged -> z -> M_hat)  [물리 기반 추정]
    |
    v
RL Policy (z + policy_obs + TDE_info -> [Kp, Kd, delta_M])  [적응적 보정]
    |
    v
TDC Controller (M_hat + delta_M, Kp, Kd)
    |
    v
DLS IK -> Joint Targets
```

여기서 delta_M은 RL-TDC의 아이디어를 차용하여,
encoder M_hat에 대한 **실시간 보정값**을 RL이 추가로 학습하는 구조.
이를 통해:
- Encoder의 물리적 추정이 부정확할 때 RL이 보상
- 급격한 외란 시 펄스형 M_hat 조정 가능
- 안정 상태에서는 delta_M -> 0 (encoder 추정만 사용)

---

## 7. Stability-Aware Reward: Feasibility Analysis

> 2026-02-16 추가. RL-TDC의 stability-aware reward를 Hero Agent에 적용할 수 있는지 검토.

### 7.1 M_bb(gamma) 시뮬레이터 계산

시뮬레이터에서 M_bb(gamma)를 **정확히 재구성하는 것은 가능**하다.
단, PhysX의 mass matrix만으로는 불완전하고, added mass를 별도로 결합해야 한다.

**PhysX가 제공하는 것**:
- `root_physx_view.get_generalized_mass_matrices()`: rigid body generalized mass matrix M(q)
- URDF 기반 각 body의 질량, 관성 텐서
- Configuration에 따른 커플링 자동 계산

**PhysX가 모르는 것**:
- Added mass (수력학적 효과) -- `HydrodynamicsModel`이 외력으로 별도 적용
- 부력제의 added mass m_A ~ 1.83 kg이 M_bb 변화의 **지배적 요인**

**M_bb(gamma) 계산식** (Parallel axis theorem, [DYNAMICS_ANALYSIS.md](./DYNAMICS_ANALYSIS.md) Section 3.7):

$$M_{bb}(\Gamma) = M_{ROV} + m_A \cdot H_{bu}^T \cdot H_{bu}$$

Roll/pitch 대각 성분만 추출하면:

$$I_p(\Gamma) = I_{p,ROV} + m_A(y_{bu}^2 + h^2), \quad I_q(\Gamma) = I_{q,ROV} + m_A(x_{bu}^2 + h^2)$$

여기서 $(x_{bu}, y_{bu})$는 FK로 계산:

$$x_{bu} = l_1\cos\gamma_1 + l_2\cos(\gamma_1 + \gamma_2), \quad y_{bu} = l_1\sin\gamma_1 + l_2\sin(\gamma_1 + \gamma_2)$$

### 7.2 Reward 시점에 필요한 텐서 접근 경로

| 필요 데이터 | 접근 경로 | 비고 |
|:---|:---|:---|
| $\gamma_1, \gamma_2$ | `env._robot.data.joint_pos[:, joint_ids]` | 매 step 가용 |
| $(x_{bu}, y_{bu})$ | `env._kin.forward(joint_pos)` | FK 1회 호출 |
| $h$ | `env._tdc_cfg.h` (= 0.180 m) | 상수 |
| $m_A$ | `env._buoy_hydro._added_mass_matrix[:, 0, 0]` | DR 후 per-env 값 |
| $I_{ROV}$ | `env._hydro._rigid_body_inertia[:, :2]` | DR 후 per-env 값 |
| $\hat{M}$ (m_hat) | `env._tdc._m_hat` | per-env (num_envs, 2) |
| $\hat{M}_{t-L}$ | 별도 히스토리 버퍼 필요 | TDC에 미구현 |

### 7.3 RL-TDC Stability Condition 적용 검토

**RL-TDC 확장 안정성 조건** (Eq. 16):

$$\left\| \left[ I - M_t^{-1} \bar{M} \right] \cdot \hat{M}_t \cdot \hat{M}_{t-L}^{-1} \right\| < 1$$

우리 시스템에 적용하면:

| 기호 | RL-TDC | Hero Agent 대응 |
|:---|:---|:---|
| $M_t$ | 실제 관성 (시뮬레이터 내부) | $M_{bb}(\Gamma)$ (위 수식으로 계산) |
| $\bar{M}$ | 튜닝 상수 | 해당 없음 (M_hat이 직접 사용됨) |
| $\hat{M}_t$ | RL action 출력 | Encoder z[3:5] (softplus >= 0.1) |
| $\hat{M}_{t-L}$ | 이전 step의 RL action | 이전 step의 z[3:5] |

**구조적 차이**: RL-TDC에서 M_bar는 "튜닝 상수" (안정성 조건만 관여)이고,
M_hat이 실제 제어 게인으로 사용된다. 반면 우리 시스템에서 M_hat은
TDE 추정에 직접 사용되는 물리 파라미터 추정값이다.

따라서 우리 시스템에 적합한 **기본 안정성 조건**은 확장 조건이 아닌,
원래의 TDC 안정성 조건이다:

$$\| I - \hat{M}^{-1} M_{bb}(\Gamma) \| < 1$$

대각 행렬이므로 element-wise로:

$$\left| 1 - \frac{I_p(\Gamma)}{\hat{M}_p} \right| < 1 \quad \Leftrightarrow \quad \frac{1}{2} I_p(\Gamma) < \hat{M}_p < 2 I_p(\Gamma)$$

이 조건이 만족되면 M_hat이 true inertia의 50-200% 범위 내에 있다는 뜻이다.

### 7.4 구현 방안

#### Option A: Soft Penalty (PPO 친화적, 권장)

```python
def stability_violation_penalty(env, **_kwargs):
    """Soft penalty for TDC stability condition violation.

    Computes true M_bb(gamma) from FK + added mass, then measures
    how far M_hat deviates from the stable region [0.5*M_true, 2*M_true].
    Zero inside stable region, quadratic growth outside.
    """
    joint_pos = env._robot.data.joint_pos[:, env._albc_joint_ids]
    p_EE = env._kin.forward(joint_pos)
    x_bu, y_bu = p_EE[:, 0], p_EE[:, 1]
    h = env._tdc_cfg.h
    m_A = env._buoy_hydro._added_mass_matrix[:, 0, 0]
    I_ROV = env._hydro._rigid_body_inertia[:, :2]

    M_true = torch.stack([
        I_ROV[:, 0] + m_A * (y_bu**2 + h**2),
        I_ROV[:, 1] + m_A * (x_bu**2 + h**2),
    ], dim=-1)

    ratio = M_true / env._tdc._m_hat.clamp(min=1e-4)
    violation = (torch.abs(1.0 - ratio) - 1.0).clamp(min=0.0)
    return violation.sum(dim=-1)  # (num_envs,)
```

특성:
- 안정 영역 내에서 penalty = 0 (gradient 소실 없음)
- 위반량에 비례하여 증가 (smooth, PPO-safe)
- dt-scaled, negative weight 사용

#### Option B: Accuracy Bonus (Gaussian kernel)

```python
def mhat_accuracy_reward(env, sigma=0.5, **_kwargs):
    """Bonus for M_hat close to true M_bb(gamma).

    Gaussian kernel: exp(-||M_hat - M_true||^2 / (sigma^2 * M_true^2))
    """
    # ... M_true 계산 동일 ...
    rel_error_sq = ((env._tdc._m_hat - M_true) / M_true.clamp(min=1e-4))**2
    return torch.exp(-rel_error_sq.sum(dim=-1) / sigma**2)
```

특성:
- [0, 1] 범위, positive weight 사용
- Encoder가 M_hat을 정확히 추정하도록 직접 유도
- Tracking reward와 동일한 Gaussian 구조

#### Option C: Hard Gate (RL-TDC 원본)

```python
def stability_gated_reward(tracking_reward, stability_met):
    """Zero reward when stability condition violated."""
    return torch.where(stability_met, tracking_reward, torch.zeros_like(tracking_reward))
```

특성:
- RL-TDC 논문 그대로
- PPO에서 위험: 학습 초기 대부분 reward=0 -> gradient 소실
- SAC (replay buffer + off-policy)에 적합한 설계

### 7.5 RL-TDC의 M_hat vs 우리 시스템의 M_hat/Kp,Kd 분리

RL-TDC 제어 법칙을 전개하면:

$$\tau = \hat{M}_t \cdot (K_p e + K_d \dot{e}) + \text{TDE}$$

여기서 Kp, Kd는 **고정 상수**이므로, $\hat{M}_t$가 사실상 **adaptive gain multiplier**이다:

$$\tau_{\text{feedback}} = \hat{M}_t K_p \cdot e + \hat{M}_t K_d \cdot \dot{e} = K_p^{\text{eff}}(t) \cdot e + K_d^{\text{eff}}(t) \cdot \dot{e}$$

즉 RL-TDC에서 M_hat은 물리적 관성 추정이 아니라, **Kp와 Kd를 동시에 스케일링하는
단일 변수**이다. 이는 설계를 단순화하지만, Kp/Kd 비율(= damping ratio)을 변경할 수 없다.

| 자유도 | RL-TDC | Hero Agent Encoder-TDC |
|:---|:---|:---|
| M_hat | 1개 스칼라 (per DOF) | Encoder z[3:5] (물리 기반) |
| Kp | 고정 | RL 출력 [10, 100] |
| Kd | 고정 | RL 출력 [2, 30] |
| 실효 자유도 | 1 (스케일만) | 3 (M, Kp, Kd 독립) |
| Damping ratio | 고정 (Kd / sqrt(Kp)) | **가변** (RL이 조절) |

우리 시스템이 자유도 면에서 더 유연하지만, 그만큼 학습 난이도가 높다.
RL-TDC는 단일 변수로 제어하므로 SAC가 빠르게 수렴하는 반면,
우리 시스템은 4D action + 6D encoder latent 조합을 PPO로 학습해야 한다.

### 7.6 권장 접근

**단기 (검증)**: Stability metric을 **diagnostic으로만 로깅**

```python
# TDC compute() 이후, logging에서 매 step 기록
stability_ratio = M_true / M_hat  # (num_envs, 2)
max_violation = |1 - stability_ratio|.max(dim=-1)
log("stability/max_violation", max_violation.mean())
log("stability/m_true_roll", M_true[:, 0].mean())
log("stability/m_true_pitch", M_true[:, 1].mean())
```

이를 통해 학습 중 안정성 조건이 얼마나 위반되는지 확인한 후,
위반이 유의미하면 **Option A (Soft Penalty)**를 reward에 추가.

**중기 (구현)**: Soft Penalty를 `EncoderTDCRewardCfg`에 추가

```python
stability_violation_weight: float = -0.5  # curriculum 적용 가능
```

HORA Phase 1의 reconstruction loss와 보완적으로 작동:
- Reconstruction loss: "z를 정확히 복사하라" (latent space 거리)
- Stability penalty: "M_hat이 물리적으로 유효한 범위에 있으라" (task space 제약)

---

## 8. Summary

RL-TDC는 **"M_hat 자체를 RL로 학습"**하는 단일 모듈 접근이고,
Hero Agent는 **"M_hat은 encoder로 추정 + 게인은 RL로 적응"**하는 이중 모듈 접근이다.

RL-TDC의 이론적 기여 (structure-free adaptive law, 확장 안정성 조건)는 가치가 있으나,
실제 배포를 고려할 때 Hero Agent의 HORA 파이프라인 + 공격적 DR이 더 실용적이다.

두 접근의 장점을 결합한 하이브리드 (encoder M_hat + RL delta_M 보정)는
향후 연구 방향으로 고려할 만하다.

**M_bb(gamma) 기반 stability reward** (Section 7)는 시뮬레이터에서 구현 가능하며,
HORA reconstruction loss와 보완적으로 작동할 수 있다.
단, PPO 학습 안정성을 위해 RL-TDC의 hard gate가 아닌 soft penalty를 권장한다.
먼저 diagnostic 로깅으로 실제 위반 빈도를 확인한 후 reward 추가를 결정하는 것이 안전하다.

---

## References

- Baek et al. (2022). "A Reinforcement Learning-based Adaptive Time-Delay Control
  and Its Application to Robot Manipulators." American Control Conference (ACC).
- Kumar et al. (2021). "RMA: Rapid Motor Adaptation for Legged Robots." (HORA basis)
- Qi et al. (2023). "HORA: Hand-Object Interaction with Online Robust Adaptation."
- See also: [TDC_CONTROL_LAW.md](./TDC_CONTROL_LAW.md), [TRAINING_PIPELINE.md](./TRAINING_PIPELINE.md),
  [TDC_LITERATURE_SURVEY.md](./TDC_LITERATURE_SURVEY.md), [DYNAMICS_ANALYSIS.md](./DYNAMICS_ANALYSIS.md)

---

**Created**: 2026-02-16
**Updated**: 2026-02-16
**Status**: Active reference -- Sections 1-6 comparative analysis, Section 7 stability reward feasibility
