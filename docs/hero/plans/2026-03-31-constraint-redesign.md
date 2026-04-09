# Constraint & Termination Redesign for Velocity Tracking (Step 3)

> Package: `constrained_full_albc`
>
> Date: 2026-03-31
>
> Status: Implemented

---

## 1. Motivation

Step 2에서 task를 velocity command tracking으로 전환한 후, 기존 6개 constraint 중 2개가
velocity tracking과 호환되지 않음:

- `yaw_velocity_cost`: raw `|w_z|`를 penalize → yaw tracking command와 직접 충돌
- `position_bound_cost`: dummy zero buffer 참조 → 항상 0 (비활성)

참고 논문(NORBC, ICML 2025)의 constraint 설계 철학을 적용하여 재설계:
- **Probabilistic** (5개): hard physical limits (binary indicator)
- **Average** (4개, ReLU-style): continuous operational limits with threshold
- Ratio 5:4는 논문의 6:5와 유사
- **Adaptive Constraint Thresholding** (수식 11) 이미 구현 확인

Termination, 초기화, position tracking 잔재도 함께 정리.

---

## 2. Joint Continuous Rotation

ALBC arm의 joint motor는 실제로 continuous rotation motor (물리적 position limit 없음).
기존 URDF의 `revolute` +-2pi limit은 인위적 제약이었음.

### Changes
- URDF: joint1, joint2 모두 `revolute` -> `continuous`
- `_apply_joint_pd_action()`: position clamp 제거. Delta action (`q_des += 0.08 * a_t`)이
  유일한 rate limiter
- Joint1 cable wrapping 보호: `joint1_position_cost` constraint (soft boundary)

### Joint Position Observation

Continuous joint은 cumulative angle을 보고 (PhysX). 예: 3회전 = 6*pi.
Observation에는 **rotation cycle 내 위치**만 필요 (dynamics는 angular position에 의존):

```python
# cumulative angle -> rotation cycle position -> [-1, 1]
joint_pos_norm = atan2(sin(theta), cos(theta)) / pi
```

| 용도 | 데이터 | 변환 |
|------|--------|------|
| Observation | cumulative angle | `atan2(sin,cos)/pi` -> [-1,1] |
| Constraint (joint1) | cumulative angle | `I(\|theta1\| > 4*pi)` |
| Action | delta accumulation | `q_des += 0.08 * a_t` (clamp 없음) |

---

## 3. Constraint Configuration (9 total)

### Probabilistic (5) -- Hard Physical Boundaries

| # | Name | Formula | Budget D_k | Purpose |
|---|------|---------|------------|---------|
| 0 | attitude | `I(max(\|r\|,\|p\|) > 80deg)` | 0.01 | Tilt safety |
| 1 | arm_torque | `I(any \|tau_j\| > 9.5 Nm)` | 0.08 | Motor stall limit |
| 2 | arm_joint_vel | `I(any \|q_dot_j\| > 4.189)` | 0.02 | Motor speed limit |
| 3 | joint1_pos | `I(\|theta1\| > 4*pi)` | 0.01 | Cable wrapping (joint1 only) |
| 4 | cumul_yaw | `I(\|yaw_accum\| > 8*pi)` | 0.01 | Tether wrapping (body yaw) |

### Average (4) -- Continuous Operational Limits (ReLU-style)

| # | Name | Formula | Budget D_k | Purpose |
|---|------|---------|------------|---------|
| 5 | yaw_rate | `max(0, \|w_z\| - 1.0)` | 0.10 | Excessive yaw rate |
| 6 | body_lin_vel | `max(0, \|\|v\|\| - 1.0)` | 0.10 | Tether tension (rapid movement) |
| 7 | thruster_util | `max(\|T_i\|)` | 0.40 | Thruster saturation / battery |
| 8 | manipulability | `max(0, 0.3 - w)` | 0.05 | Arm singularity proximity |

### Why ReLU-style for Average Constraints

논문의 average constraint는 "잘 걷으면 자연적으로 ~0"인 물리량 (접촉 미끄러짐, 직교 속도)에
raw value를 사용. 우리 시스템에서 `|w_z|`, `||v||`는 velocity tracking 중 **자연적으로 0이
아님** (command 추종 중). Raw value를 쓰면 원하는 행동 자체가 penalty를 받음.

ReLU threshold는 "정상 운용 범위"와 "과도한 범위"를 분리:
- Threshold 이하: cost=0, gradient=0 (tracking 방해 없음)
- Threshold 이상: cost=excess, gradient 존재 ("더 줄여라")
- Threshold = 1.0 > vel_cmd_range(0.5)이므로 정상 tracking에 영향 없음

### Infeasible Target Handling

Target이 물리적으로 달성 불가능할 때 (heavy payload + large velocity command):
- **Adaptive Constraint Thresholding** (이미 구현): `d_k^i = max(d_k, J_Ck + alpha*d_k)`
- Constraint가 물리적 한계를 정의 → reward는 그 안에서 최적화
- Policy가 "할 수 있는 범위 내에서" 점진적으로 개선 (log-barrier가 터지지 않음)

---

## 4. Cumulative Yaw Tracking

Body yaw의 누적 회전량을 추적하여 tether wrapping 방지.

```python
# euler_xyz_from_quat returns yaw in [-pi, pi] (wrapping)
# delta 계산 시 wrapping boundary(+-pi) 보정 필요:
delta_yaw = current_yaw - prev_yaw
delta_yaw = where(delta_yaw > pi, delta_yaw - 2*pi, delta_yaw)
delta_yaw = where(delta_yaw < -pi, delta_yaw + 2*pi, delta_yaw)
cumulative_yaw += delta_yaw  # grows unbounded
```

Episode reset 시 `cumulative_yaw = 0`, `prev_yaw = current_yaw`.

---

## 5. Termination Conditions (4 total)

Step 2까지의 termination을 6DOF에 맞게 정리. Constraint와 역할 분리:
- **Termination**: 시뮬레이션이 유효하지 않은 상태 (PhysX 불안정, 극단적 속도) -> 즉시 리셋
- **Constraint**: 정상 운용 범위 이탈 -> IPO log-barrier로 소프트 제동

| # | Name | Formula | Threshold | Purpose |
|---|------|---------|-----------|---------|
| 0 | too_fast_ang | `max(\|p\|, \|q\|, \|r\|) > limit` | pi rad/s | Angular velocity 과다 (3축) |
| 1 | too_fast_lin | `\|\|v_world\|\| > limit` | 2.0 m/s | Linear velocity 과다 |
| 2 | bad_state | NaN/Inf on pos, quat, lin_vel, ang_vel | - | PhysX 실패 감지 |
| 3 | excessive_tilt | `\|roll\| > limit \| \|pitch\| > limit` | pi/2 (90 deg) | Lambda 부호 반전 방지 |

이전 대비 변경:
- `too_fast`: roll/pitch 2축 -> 전체 3축 (yaw 포함), `too_fast_ang`으로 이름 변경
- `too_fast_lin`: `too_fast_linear`에서 이름 변경, 로깅 추가
- `bad_state`: `root_ang_vel_b` NaN/Inf 체크 추가
- `out_of_depth`: 제거 (수중 로봇에 depth termination 불필요)
- `depth` constraint: 제거 (유저 요청하지 않은 항목)

---

## 6. Environment Initialization

### Position Tracking 잔재 제거

Velocity tracking task에 불필요한 position tracking 시절 코드 제거:
- `target_attitude`, `randomize_target_attitude`, `target_attitude_range` config 필드
- `_target_euler`, `_base_attitude`, `_target_range`, `_randomize_targets`, `_attitude_error` 버퍼
- `compute_attitude_error()`, `_get_attitude_error()`, `_update_attitude_error()` 메서드
- Attitude target 랜덤화 블록 in `_reset_task_and_state()`
- `Attitude_Error/` 로깅 -> `Attitude/roll_deg`, `pitch_deg` (절대값 모니터링)으로 교체

### Robot Pose Reset

매 에피소드 로봇은 env_origin에서 정립/정지 상태로 시작 (pose DR 제거):
- Position: env_origin + (0, 0, 0)
- Orientation: identity quaternion (정립)
- Velocity: 0 (정지)
- `initial_height` config 필드 제거
- `randomize_robot_pose` 호출 제거 -> 항상 `reset_robot_pose_default`

Joint만 DR 유지: (-pi, pi) 랜덤 초기화 + actuator gain/effort/friction DR.

---

## 7. Files Modified

| File | Action | Key Changes |
|------|--------|-------------|
| `meshes/agent.urdf` | EDIT | joint1, joint2: revolute -> continuous |
| `mdp/constraints.py` | REWRITE | 3 removed, 6 added = 9 constraint functions |
| `mdp/__init__.py` | EDIT | Export updates |
| `mdp/observations.py` | EDIT | joint_pos_norm: linear -> atan2 wrapping |
| `mdp/events.py` | EDIT | joint clamp removed, `reset_robot_pose_default` simplified |
| `config.py` | EDIT | 9-term constraints, attitude target removed, initial_height removed |
| `albc_env.py` | EDIT | termination 4개, attitude 잔재 제거, pose DR 제거, cumul_yaw 추가 |
| `test_full_dof_env.py` | EDIT | shape=9, updated constraint tests |

**No changes needed**: `agents/rsl_rl_ppo_cfg.py` (auto-synced by runner),
`algorithms/constraint_trpo.py` (generic K-constraint), `encoder/` (generic cost_critic heads).

---

## 8. Command, Action, Reward Tuning (Step 4)

### Velocity Command
- `vel_cmd_zero_prob: float = 0.1`: 리샘플 시 10% env에 zero command 부여 (hovering 학습)
- Anymal의 `rel_standing_envs`와 동일 원리, hovering이 더 중요한 UUV 특성에 맞게 비율 상향

### Action Space
- `delta_scale`: 0.08 -> **0.10** rad/step
- 이전: max velocity 4.0 rad/s < `arm_joint_vel` constraint (4.189 rad/s) -> constraint 발동 불가
- 이후: max velocity 5.0 rad/s > constraint -> constraint가 실효성을 가짐

### Reward
- `ang_vel_axis_weights: tuple = (2.0, 2.0, 1.0)`: roll/pitch rate error에 yaw 대비 2x 가중치
- ALBC arm이 roll/pitch 안정화를 담당하므로 더 큰 penalty 합리적

### Files Modified
| File | Changes |
|------|---------|
| `config.py` | `vel_cmd_zero_prob=0.1`, `delta_scale=0.10` |
| `albc_env.py` | `_sample_velocity_command()` zero-command 로직 |
| `mdp/rewards.py` | `ang_vel_axis_weights`, `ang_vel_tracking()` weighted sum |

---

## 9. Observation Redesign (Step 5)

### Motivation

기존 o_t (28D) + proprio_hist (210D) = 238D 구조의 문제점:
1. **Velocity error-as-observation**: `cmd - vel`에 독립 noise 추가 → 물리적으로 부정확 (실제는 vel sensor noise만 존재)
2. **atan2 wrapping**: cumulative joint angle 소멸 → cable wrapping constraint 사전 회피 불가
3. **Previous action in obs**: temporal history에 action이 있으면 중복
4. **Command/history 분리**: 개념적으로 통합이 자연스러움
5. **History 과대**: 14D×15steps×stride5 = 210D, 1.5초 → Phase 2 adaptation 미계획이므로 불필요

### New o_t Structure (81D = 26D current + 55D history)

#### Current Proprioception (26D)

| Slice | Element | Dim | Noise |
|-------|---------|-----|-------|
| [0:3] | vel_cmd_lin | 3D | 없음 (우리 명령) |
| [3:6] | vel_cmd_ang | 3D | 없음 |
| [6:9] | roll, pitch, yaw | 3D | std=0.02 |
| [9:12] | ang_vel (p, q, r) | 3D | std=0.04 |
| [12:15] | lin_vel (u, v, w) | 3D | std=0.04 |
| [15:17] | joint_pos (raw cumulative) | 2D | std=0.02 |
| [17:19] | joint_vel | 2D | std=0.04 |
| [19] | manipulability | 1D | 없음 |
| [20:26] | thruster_state | 6D | std=0.02 |

#### Temporal History (55D, ring buffer, stride=3)

| Category | Per step | Steps | Total | Span |
|----------|----------|-------|-------|------|
| Joint tracking: `q_des_{t-1} - q_actual`, `q_dot` | 4D | 3 | 12D | 0.18s |
| Body tracking: `cmd_lin - vel`, `cmd_ang - vel`, rpy | 9D | 3 | 27D | 0.18s |
| Action: full 8D (`prev_actions`, causal) | 8D | 2 | 16D | 0.12s |

History feature 21D per step: [joint_track(4) | body_track(9) | action(8)]
Buffer shape: `(N, 3, 21)`, slice `[:,:,:13]` for joint+body, `[:,-2:,13:]` for action.

### Design Decisions

1. **Command vs error**: Command(no noise) + measured velocity(noise) 분리. 네트워크가 내부적으로 error 계산.
   물리적으로 정확 (DVL noise만 반영). Anymal도 raw command 사용.

2. **Raw cumulative joint_pos**: atan2 wrapping 제거. Cable wrapping constraint (4*pi)
   접근을 policy가 인지 가능. Manipulability가 cycle position geometric info 보완.

3. **History error terms**: Joint tracking error (`q_des_{t-1} - q_actual`) + body tracking error
   (`vel_cmd - vel`) = "target - actual" 패러다임 통일. 응답 특성 (시간 상수, 감쇠비) 추론 가능.

4. **Stride=3**: 수중 dynamics가 느려서 (added mass + damping, τ~0.5-1.0s)
   연속 0.02s 샘플은 높은 상관성. 0.06s 간격이 더 유의미한 변화 캡처. Ablation으로 최적화 예정.

5. **History action = `prev_actions`**: `q_des_{t-1} - q_actual`과 인과적으로 일치하는 action
   (이전 action이 현재 state를 만듦).

6. **Thruster noise 추가**: ESC feedback 측정 noise 모사 (std=0.02, bias=+-0.01).

### Network Dimensions

```
Actor:  EmpNorm(o_t(81D)) + z(9D) = 90D → MLP[256,128,64] → 8D
Critic: cat[o_t(81D), z(9D), p_t(23D)] = 113D → MLP[512,256,128] → 1D
Cost:   same 113D → MLP[512,256,128] → K heads
```

### Files Modified

| File | Changes |
|------|---------|
| `mdp/observations.py` | `compute_policy_obs` 28D→26D: cmd(not error), raw jpos, no prev_actions |
| `config.py` | `observation_space=81`, 81D noise model, `hist_len/stride/feature_dim` |
| `albc_env.py` | `_hist_buf(3×21D)`, `_get_hist_features()`, `_update_hist()`, unified 81D obs |
| `agents/rsl_rl_ppo_cfg.py` | `policy_obs_dim=81`, obs_groups 2-key, `proprio_hist_dim` 제거 |
| `encoder/actor_critic_encoder.py` | proprio_hist 분리 로직 제거, obs 처리 단순화 |
| `test_full_dof_env.py` | obs shape 81D 검증 |

---

## 10. Domain Randomization Redesign (Step 6)

### Changes

1. **Position/orientation DR removed**: config에 정의만 있고 실제 미호출 (dead code). 6개 필드 삭제, `randomize_robot_pose()` 삭제.

2. **DORAEMON 확장 (7 -> 15 params)**: hydrodynamics, CoB/CoG, inertia/mass, payload 전체 DORAEMON 관리.

   | # | Parameter | Bounds | Nominal | Category |
   |---|-----------|--------|---------|----------|
   | 0 | payload_mass | [0, 1.0] kg | 0.5 | payload |
   | 1 | added_mass_scale | [0.85, 1.15] | 1.0 | hydrodynamics |
   | 2 | linear_damping_scale | [0.5, 1.5] | 1.0 | hydrodynamics |
   | 3 | quadratic_damping_scale | [0.5, 1.5] | 1.0 | hydrodynamics |
   | 4 | water_density | [995, 1025] kg/m3 | 1010 | hydrodynamics |
   | 5 | cog_offset_z | [-0.02, 0.02] m | 0.0 | geometry |
   | 6 | cob_offset_z | [-0.02, 0.02] m | 0.0 | geometry |
   | 7 | volume_scale | [0.9, 1.1] | 1.0 | hydrodynamics |
   | 8 | cob_offset_x | [-0.01, 0.01] m | 0.0 | geometry |
   | 9 | cob_offset_y | [-0.01, 0.01] m | 0.0 | geometry |
   | 10 | cog_offset_x | [-0.01, 0.01] m | 0.0 | geometry |
   | 11 | cog_offset_y | [-0.01, 0.01] m | 0.0 | geometry |
   | 12 | inertia_scale | [0.75, 1.3] | 1.0 | mass |
   | 13 | body_mass_scale | [0.9, 1.1] | 1.0 | mass |
   | 14 | payload_cog_offset_z | [-0.03, 0.0] m | -0.015 | payload |

3. **Hard DR 확장**: actuator/thruster override 추가.
   - joint_stiffness: (40,120) -> (30,150)
   - joint_damping: (0.5,5.0) -> (0.3,7.0)
   - thrust_coefficient_scale: (0.8,1.2) -> (0.7,1.3)
   - time_constant_scale: (0.8,1.2) -> (0.7,1.3)

4. **Success metric 수정**: `success_threshold_deg` (deg, deg2rad 변환) -> `success_threshold` (m/s, 직접 비교). 실제 settling error가 velocity error (m/s)이므로 단위 일치.

5. **Encoder bounds 업데이트**: Hard DR 범위에 ~10% margin 적용.

### Design Rationale

- Terrain curriculum 분석: DORAEMON의 entropy maximization + trust region이 이미 terrain curriculum의 핵심 아이디어 (adaptive difficulty, forgetting prevention)를 포함. Per-group 독립 커리큘럼은 보류.
- DORAEMON PARAM_SPECS bounds = base DomainRandomizationCfg 범위. Hard DR 사용 시 `DoraemonCfg.param_overrides`로 확대.
- Checkpoint backward compat: old 7D checkpoint 로드 시 first 7 dims 복원, new dims는 init_concentration.

### Files Modified

| File | Changes |
|------|---------|
| `doraemon.py` | PARAM_SPECS 7->15, success metric rename, checkpoint compat |
| `config.py` | Remove pos/orient, add Hard DR actuator/thruster |
| `mdp/events.py` | Generalize xyz offset for DORAEMON, add sampled to body_mass/payload |
| `mdp/__init__.py` | Remove randomize_robot_pose export |
| `albc_env.py` | Pass sampled to body_mass, fix success metric |
| `agents/rsl_rl_ppo_cfg.py` | Update encoder bounds for Hard DR |

---

## 11. Privileged Obs Redesign (Step 7)

### Motivation

기존 23D privileged obs의 문제:
1. **중복**: main/buoy body에 같은 DR scale 적용 → 13개 dim이 실질 7개 독립변수
2. **누락**: CoG/CoB x/y, thruster params, ocean current 미반영
3. **불균형**: DORAEMON 15 params 중 4개(CoG/CoB x/y)가 priv obs에 없음

### Design Principles

Legged locomotion Teacher-Student 논문의 privileged info 설계 철학 적용:
- sim에서만 측정 가능하고 proprioception으로는 직접 얻을 수 없는 정보
- 제어에 유용한 물리 파라미터 (dynamics equation terms)
- Raw DR parameter 스타일: 각 dim = 하나의 독립 random variable

### New 24D Layout

| Index | Category | Parameter | Source |
|-------|----------|-----------|--------|
| 0 | Hydro | main volume | `_hydro.volume` |
| 1-3 | Hydro | main CoG (x,y,z) | `_hydro.center_of_gravity` |
| 4-6 | Hydro | main CoB (x,y,z) | `_hydro.center_of_buoyancy` |
| 7 | Dynamics | Ixx | `_hydro.rigid_body_inertia[:, 0]` |
| 8 | Dynamics | linear damping roll | `_hydro.linear_damping[:, 3]` |
| 9 | Dynamics | quadratic damping roll | `_hydro.quadratic_damping[:, 3]` |
| 10 | Dynamics | body mass | `_hydro.body_mass` |
| 11 | Dynamics | added mass surge | `_hydro.added_mass_matrix[:, 0, 0]` |
| 12 | Payload | payload mass | `_payload_mass` |
| 13-15 | Payload | CoG offset (x,y,z) | `_payload_cog_offset` |
| 16 | Actuator | joint stiffness Kp | `robot.data.joint_stiffness` |
| 17 | Actuator | joint damping Kd | `robot.data.joint_damping` |
| 18 | Actuator | thrust coefficient | `_thruster._thrust_coeff` |
| 19 | Actuator | time constant up | `_thruster._time_constant_up` |
| 20 | Env | water density | `_hydro.water_density` |
| 21-23 | Env | ocean current (x,y,z) | `_hydro._current_velocity[:, :3]` |

### Changes from 23D

- **Removed (9D redundancy)**: buoy volume/CoG_z/CoB_z (same scale as main), buoy Ixx/Iyy,
  main Iyy, linear/quadratic damping pitch (same scale as roll)
- **Added (10D)**: CoG x/y (2), CoB x/y (2), thrust_coeff (1), time_const_up (1),
  ocean_current xyz (3), payload_cog_z was already present
- **Encoder**: all 24D direct input (no index selection), static min-max norm

### Encoder Architecture

```
p_t(24D) -> static_minmax[-1,1] -> MLP[256,128,64] -> LN -> softsign -> z(9D)
Actor: cat[EmpNorm(o_t), z] = 90D -> MLP[256,128,64] -> 8D
Critic: cat[o_t, z, p_t] = 114D -> MLP[512,256,128] -> 1D
Cost: same 114D -> MLP[512,256,128] -> 9D
```

### Files Modified

| File | Changes |
|------|---------|
| `mdp/observations.py` | 24D `compute_privileged_obs()`, removed `_hydro_privileged()` |
| `agents/rsl_rl_ppo_cfg.py` | 24D bounds, removed 15D indices, `privileged_dim=24` |
| `config.py` | `state_space=24` |

---

## 12. Mid-Episode Dynamics (Step 8)

### Motivation

Legged locomotion Teacher-Student 논문에서 terrain curriculum (episode-level)과 per-step
perturbation (friction, push)을 분리하는 패턴을 UUV에 적용:
- DORAEMON = episode-level difficulty (terrain curriculum analog)
- Mid-episode dynamics = per-step external disturbances

### Payload Toggle (Binary Pick/Place)

Episode 시작 시 payload 유무를 확률적으로 결정하고, episode 중간에서 1회 toggle:

| Scenario | Start | Mid-Episode | End | Fraction |
|----------|-------|-------------|-----|----------|
| A | mass=0 | PICK | mass=m | (1-p_start)(1-p_notoggle) |
| B | mass=0 | no toggle | mass=0 | (1-p_start)(p_notoggle) |
| C | mass=m | DROP | mass=0 | (p_start)(1-p_notoggle) |
| D | mass=m | no toggle | mass=m | (p_start)(p_notoggle) |

Config: `payload_toggle_steps=-1` (midpoint), `payload_start_with_prob=0.5`,
`payload_no_toggle_prob=0.2`.

DORAEMON 호환: DORAEMON이 episode-level xi[0] (payload_mass)를 관리. Toggle은
환경 난이도의 일부로 xi->success 매핑에 흡수.

### Ocean Current OU Drift

Ornstein-Uhlenbeck process로 매 step ocean current를 연속 변동:

```
dx = -theta * (x - mu) * dt + sigma * sqrt(dt) * N(0,1)
```

- mu = reset 시 샘플된 base current (episode 내 고정)
- theta = 0.15 (1/s), sigma = 0.05 (m/s per sqrt(s))
- Linear components (xyz)만 적용, angular 유지 0
- Main/buoy hydro 동일 current 공유
- Clamp: max_velocity * 1.05 (encoder bounds 내)

### Files Modified

| File | Changes |
|------|---------|
| `config.py` | 6 config fields (3 payload toggle + 3 OU) |
| `albc_env.py` | 7 buffers, 4 new methods, _pre_physics_step logic, logging |

### Impact

Observations, rewards, constraints, encoder 모두 변경 없음.
Privileged obs [12:16] (payload), [21:24] (current)이 버퍼를 직접 참조하므로 자동 반영.

---

## 13. Phase 2+ (Not Yet Implemented)

- Encoder z-sweep analysis on 24D composition
- Outer PID loop (deployment)
- Training and evaluation
