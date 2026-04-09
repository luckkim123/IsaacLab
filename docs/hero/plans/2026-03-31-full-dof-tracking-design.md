# Full 6-DOF Velocity Tracking with Thruster + ALBC: Design Document

> Task: `Isaac-FullDOF-TRPO-v0`
>
> Package: `constrained_full_albc` (forked from `constrained_albc`)
>
> Date: 2026-03-31
>
> Status: Step 2 Implemented (velocity tracking, verified 14/14)

---

## 1. Motivation

현재 constrained ALBC는 2D action space (arm delta joint)로 roll/pitch 자세만 제어한다.
이 시스템은 **underdetermined** -- 2개의 actuator로 2개의 DOF만 제어하므로 나머지 4 DOF
(x, y, z, yaw)는 통제되지 않는다. 이로 인해:

- Yaw drift가 발생하여 yaw_vel constraint (budget=0.40)가 필요
- 위치가 고정되지 않아 ocean current에 의한 drift 발생
- Encoder가 학습하기 어려움 -- DR parameter 변화에 따른 동역학 차이가 uncontrolled DOF로
  dissipate되면 policy가 일관된 응답을 보이기 어려움

**가설**: 6개 thruster를 추가하여 8D action space (2D arm + 6D thruster)로 확장하고,
velocity command tracking을 수행하면:
1. System이 fully determined되어 더 안정적인 제어 가능
2. Encoder에게 더 풍부한 gradient signal 제공 (더 많은 actuator = 더 다양한 response)
3. TRPO KL budget에서 sigma 점유율이 ~33% (2D) -> ~8% (8D)로 감소

---

## 2. Architecture: Cascaded Control

Step 1에서 position+attitude tracking을 구현했으나, reward 충돌 발견:
- Position tracking reward는 target으로 이동 요구 (velocity 필요)
- Velocity regulation reward는 속도 억제 요구
- Transit 구간에서 두 reward가 상반됨

**해결**: Velocity command tracking으로 전환 (cascaded control).

```
[Training]
Random vel_cmd ────────────────> RL Policy ──> 8D action
                                     ^
[Deployment]                         |
Position/Attitude error ──> Outer PID ──> vel_cmd ──> RL Policy ──> 8D action
```

- **학습**: Policy가 velocity command를 tracking (reward 충돌 없음)
- **배포**: Outer PID가 position/attitude error -> velocity command -> Policy
- PID 튜닝이 학습과 완전 분리 (학습은 random command, PID는 별도)

---

## 3. Implementation Approach

`constrained_albc` 폴더 전체를 `constrained_full_albc`로 복사하고, TRPO+IPO+Encoder
pipeline 하나만 남긴 뒤 코드를 직접 수정. 기존 코드에 영향 없이 독립적으로 디버깅 가능.

---

## 4. Action Space (8D)

| Index | Content | Range | Mechanism |
|-------|---------|-------|-----------|
| [0:2] | Arm delta joint targets | [-1, 1] | `q_des += 0.08 * a_t`, PhysX PD |
| [2:8] | Thruster commands (T0-T5) | [-1, 1] | 1st-order lag -> TAM -> body wrench |

- T0-T3: 수평 45도 vectored (surge, sway, yaw)
- T4-T5: 수직 (heave, pitch)
- Thrust coefficient: 40 N/unit, max: 50N, time_constant: 0.1s up / 0.05s down

---

## 5. Velocity Command Structure

학습 시 velocity command를 주기적으로 resampling:
```
lin_vel_cmd ~ Uniform(-0.5, 0.5) m/s per axis (body frame)
ang_vel_cmd ~ Uniform(-0.5, 0.5) rad/s per axis (body frame)
```

- **Resampling**: 5초(250 steps)마다 새 command sampling
- Legged locomotion 논문과 동일 패턴 (3-5초 유지 후 재샘플링)
- 속도 전환(transition) 학습 -> 배포 시 PID 출력 변화에 적응
- Station-keeping(cmd=0)도 sampling 범위에 자연 포함

---

## 6. Policy Observation o_t (28D)

기능별 그룹핑. 실제 로봇에서 측정 가능 (IMU, DVL, motor encoder, ESC feedback).

### Command (6D)

| Index | Dim | Content | Source | Noise std |
|-------|-----|---------|--------|-----------|
| [0:3] | 3 | linear velocity error (body) | vel_cmd - root_lin_vel_b | 0.04 |
| [3:6] | 3 | angular velocity error (body) | vel_cmd - root_ang_vel_b | 0.04 |

### Body State (9D)

| Index | Dim | Content | Source | Noise std |
|-------|-----|---------|--------|-----------|
| [6:9] | 3 | euler angles (roll, pitch, yaw) | IMU quaternion -> euler | 0.02 |
| [9:12] | 3 | angular velocity body (p, q, r) | IMU gyro | 0.04 |
| [12:15] | 3 | linear velocity body (u, v, w) | DVL / root_lin_vel_b | 0.04 |

### Arm State (7D)

| Index | Dim | Content | Source | Noise std |
|-------|-----|---------|--------|-----------|
| [15:17] | 2 | joint positions (normalized [-1,1]) | motor encoder | 0.02 |
| [17:19] | 2 | joint velocities | motor encoder | 0.04 |
| [19:21] | 2 | previous arm actions | internal buffer | 0.0 |
| [21] | 1 | manipulability index w (normalized [0,1]) | computed from theta2 | 0.0 |

### Thruster State (6D)

| Index | Dim | Content | Source | Noise std |
|-------|-----|---------|--------|-----------|
| [22:28] | 6 | thruster filtered output (T0-T5) | ESC RPM feedback | 0.0 |

### Design Decisions

- **Velocity error as command**: vel_err = cmd - actual. Reward 충돌 없음 (target이 velocity)
- **lin_vel in body state**: vel_err만으로는 absolute velocity 복원 불가. Drag/current 보상에 필요
- **Manipulability 추가 (1D)**: `w = sqrt(|l1*l2*sin(theta2)|) / sqrt(l1*l2)`, [0,1] 정규화.
  Arm singularity 인식 제공. Reward/constraint가 아닌 정보 제공으로 policy 판단에 위임
- **Thruster state 유지**: ESC RPM feedback으로 실제 로봇에서 측정 가능.
  1st-order lag dynamics (tau_up=0.1s, 5 steps)가 무시할 수 없음
- **Euler angles 유지**: 물리적 상태 인코딩 (buoyancy 방향, 감쇠 특성)
- **prev_arm_actions 유지**: stride=5 history에서 최근 5 step은 미포함될 수 있음

---

## 7. Privileged Observation p_t (23D)

기존 constrained_albc와 동일. Phase 2에서 thruster DR params, ocean current 추가 예정.

| Index | Dim | Content |
|-------|-----|---------|
| [0:3] | 3 | Main body: volume, CoG_z, CoB_z |
| [3:6] | 3 | Buoy: volume, CoG_z, CoB_z |
| [6:8] | 2 | Main inertia: Ixx, Iyy |
| [8:10] | 2 | Buoy inertia: Ixx, Iyy |
| [10:12] | 2 | Linear damping: roll, pitch |
| [12:14] | 2 | Quadratic damping: roll, pitch |
| [14] | 1 | Body mass |
| [15] | 1 | Added mass surge |
| [16] | 1 | Payload mass |
| [17:20] | 3 | Payload CoG offset (x, y, z) |
| [20] | 1 | Joint stiffness Kp |
| [21] | 1 | Joint damping Kd |
| [22] | 1 | Water density |

---

## 8. Proprioceptive History (210D)

14D features x 15 steps, stride=5 (50Hz -> 10Hz sampling, 1.5s window).

| Dim | Content | Purpose |
|-----|---------|---------|
| [0:3] | roll, pitch, yaw | Attitude trajectory for dynamics inference |
| [3:6] | p, q, r | Angular velocity (yaw rate included for thruster yaw control) |
| [6:9] | u, v, w (body frame) | Linear velocity for translation dynamics (ocean current, added mass) |
| [9:11] | joint_pos_norm (2D) | Arm state |
| [11:13] | prev_arm_actions (2D) | Arm command history |
| [13] | manipulability w (1D) | Arm configuration quality over time |

### Design Decisions

- **16D -> 14D 변경**: pos_err_body(3D) 제거 (velocity tracking에서 불필요), manipulability(1D) 추가
- **Thruster state 미포함**: lin_vel 시계열에서 간접 추론 가능, 차원 절약
- **1.5s window 유지**: dynamics 추론에 충분 (ocean current 변화 주기 ~2s)

---

## 9. Encoder Architecture

```
Encoder: p_t(23D) -> select(15D) -> static_minmax_norm -> MLP[256,128,64] -> LayerNorm -> softsign -> z(9D)
Actor:   cat([o_t(28D), hist(210D), z(9D)]) = 247D -> MLP[256,128,64] -> 8D mean
Critic:  cat([o_t(28D), hist(210D), z(9D), p_t(23D)]) = 270D -> MLP[512,256,128] -> 1D value
Cost:    same 270D -> MLP[512,256,128] -> 6D (multi-head, one per constraint)
```

Algorithm: TRPO (actor+encoder) + decoupled Adam (sigma, std_lr=1e-3) + Adam (critic+cost_critic)

---

## 10. Reward (5 terms, all dt-scaled)

| Term | Weight | Formula | Purpose |
|------|--------|---------|---------|
| lin_vel | k_lin = -4.0 | `\|\|lin_vel_err\|\|^2` | Linear velocity command tracking |
| ang_vel | k_ang = -8.0 | `\|\|ang_vel_err\|\|^2` | Angular velocity command tracking |
| torque | k_tau = -0.005 | `mean(applied_torque^2)` | Arm energy |
| thruster | k_thr = -0.01 | `mean(thruster_cmd^2)` | Thruster energy |
| smoothness | k_s = -0.1 | `mean(da^2) + mean(d2a^2)` | Action smoothness (8D) |
| termination | -10.0 | one-shot | |

### Design Decisions

- **ang_vel weight > lin_vel weight**: Attitude 제어가 더 중요 (buoyancy stability)
- **lin/ang 분리**: 제어 난이도가 다름 (linear: drag+current vs angular: inertia+arm coupling)
- **Position/attitude tracking reward 제거**: Velocity tracking으로 대체, reward 충돌 해소

---

## 11. Constraints (6 terms, IPO log-barrier)

| # | Name | Type | Formula | Budget D_k |
|---|------|------|---------|------------|
| 0 | attitude | Probabilistic | `I(max(\|roll\|,\|pitch\|) > 80 deg)` | 0.01 |
| 1 | torque | Probabilistic | `I(any \|tau_j\| > 9.5 Nm)` | 0.08 |
| 2 | velocity | Probabilistic | `I(any \|q_dot_j\| > 4.189 rad/s)` | 0.02 |
| 3 | yaw_vel | Average | `\|w_z\|` | 0.10 |
| 4 | position | Probabilistic | `I(\|\|pos_err\|\| > 3.0 m)` | 0.01 |
| 5 | depth | Probabilistic | `I(z_local < 1m or z_local > 8m)` | 0.01 |

**Note**: Velocity tracking 전환 후 constraint 재검토 필요 (yaw_vel, position_bound).
Position_bound는 현재 dummy buffer(항상 0)로 비활성 상태.

---

## 12. Domain Randomization

Standard DR (현재 사용). HardDR도 정의되어 있으나 미활성.

| Category | Parameter | Range |
|----------|-----------|-------|
| Added mass | scale | 0.85-1.15 |
| Damping | linear/quad scale | 0.5-1.5 |
| Volume | scale | 0.9-1.1 |
| CoG/CoB | offset xyz | +-1-2 cm |
| Inertia | scale | 0.75-1.3 |
| Body mass | scale | 0.9-1.1 |
| Water density | absolute | 995-1025 kg/m3 |
| Joint Kp/Kd | absolute | 40-120 / 0.5-5.0 |
| Payload | mass | 0-1 kg |
| Thruster coeff | scale | 0.8-1.2 |
| Thruster tau | scale | 0.8-1.2 |
| Ocean current | velocity | xy: 0.5 m/s, z: 0.25 m/s |

---

## 13. Termination Conditions

| Condition | Threshold |
|-----------|-----------|
| Excessive angular velocity (rp) | > 180 deg/s |
| Excessive tilt (rp) | > 90 deg |
| NaN/Inf in state | any |
| Depth out of bounds | < 0.5m or > 10.0m |
| Excessive linear velocity | > 2.0 m/s |
| Timeout | 30s (1500 steps) |

**Note**: Position drift termination (> 5m) 제거됨 -- velocity tracking에서 position target 없음.

---

## 14. Simulation Parameters

| Parameter | Value |
|-----------|-------|
| Physics dt | 0.005s (200Hz) |
| Decimation | 4 (50Hz env step) |
| Control decimation | 1 (50Hz policy) |
| Episode length | 30s |
| Num envs (default) | 4096 |
| Env spacing | 4.0m |
| Vel cmd resample | 250 steps (5s) |

---

## 15. Verification Results (2026-03-31, Step 2)

14/14 tests passed:
- Smoke test: 200 random 8D steps without NaN
- Obs dims: policy=(N,28), privileged=(N,23), proprio_hist=(N,210)
- Thruster motion: forward 0.65m, vertical 0.83m (50 steps)
- Reward: zero_cmd=0.0000, large_cmd=-0.0200 (vel_err increases -> reward decreases)
- Manipulability: nominal(pi/2)=1.0, singularity(0)=0.1
- Command resampling: unchanged before N steps, changed after
- Constraints: depth_bound fires at z=0.5m

---

## 16. Phase 2+ (Not Yet Implemented)

- Constraint 상세 재검토 (yaw_vel, position_bound 재설계)
- Privileged obs 확장 (23D -> ~43D: thruster DR, ocean current, translation damping)
- Encoder input selection 재설계 (z-sweep 후)
- DORAEMON adaptive DR with velocity tracking success criterion
- Mid-episode DR 변경 (payload, ocean current 동적 변화)
  - Encoder 역할 전환: "정적 DR 추정" -> "동적 파라미터 추정"
- Outer PID loop 구현 (배포용)
- HardDR config variant
- Network architecture tuning
- Training and evaluation
