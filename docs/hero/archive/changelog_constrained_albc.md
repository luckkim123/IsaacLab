# Constrained ALBC Changelog (2026-03-27 ~ 2026-03-31)

All notable changes to the constrained_albc project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

For the active changelog (constrained_full_albc), see [/workspace/isaaclab/changelog.md](/workspace/isaaclab/changelog.md).
For entries before 2026-03-27, see [changelog_legacy.md](changelog_legacy.md).
For the encoder ablation study (Steps 0-19), see
[encoder_ablation.md](experiments/encoder_ablation.md).

---

## [2026-03-31] Mid-Episode Dynamics (Step 8)

### Context

Episode 내 payload와 ocean current가 고정되어 있던 문제 해결. Legged locomotion 논문의
per-step perturbation 패턴을 UUV에 적용: payload toggle (pick/place event) +
ocean current OU drift.

### Added
- Payload toggle: episode 중간 (midpoint)에서 binary pick/place event 발생.
  4가지 시나리오 (start with/without x toggle/no-toggle). Config:
  `payload_toggle_steps`, `payload_start_with_prob`, `payload_no_toggle_prob`.
- Ocean current OU drift: Ornstein-Uhlenbeck process로 매 step current 연속 변동.
  Config: `ou_theta`, `ou_sigma`, `ou_enable`.
- 4개 logging 메트릭: `MidEp/payload_toggled_frac`, `payload_mass_final`,
  `current_drift_norm`, `current_mag_final`.

### Changed
- `config.py`: ALBCEnvCfg에 Mid-Episode Dynamics 섹션 추가 (6 fields).
- `albc_env.py`: 7 buffers, 4 new methods (`_setup_payload_toggle`,
  `_apply_payload_toggle`, `_clamp_payload_cog`, `_step_ocean_current_ou`,
  `_sample_stashed_cog_offset`), `_pre_physics_step` 확장.

### Design Decisions
- Payload toggle 1회/episode: 30초 내 다회 pick/place 비현실적.
- DORAEMON 호환: stash 방식으로 xi->success attribution 보존.
- OU clamp: max_velocity * 1.05 (encoder bounds 내).
- 기본 비활성: `payload_toggle_steps=0`, `ou_enable=False` -> 기존 동작 무변경.

---

## [2026-03-31] Privileged Obs Redesign (Step 7)

### Context

23D privileged obs의 구조적 문제 해결: 중복 제거 (buoy 쌍, pitch/roll 쌍),
누락 파라미터 추가 (CoG/CoB x/y, thruster, ocean current).
Legged locomotion Teacher-Student 설계 철학 적용: raw DR parameter, non-redundant.

### Changed
- `mdp/observations.py`: `compute_privileged_obs()` 23D -> 24D non-redundant layout 재작성.
  Removed `_hydro_privileged()` helper (no longer needed).
- `agents/rsl_rl_ppo_cfg.py`: `privileged_dim=24`, 24D encoder bounds (Hard DR + 10% margin),
  15D index selection 제거 (encoder gets all 24D directly).
- `config.py`: `state_space=24`.

### Added (new priv obs dimensions)
- CoG x/y, CoB x/y (4D): 횡방향 복원력 비대칭, DORAEMON 관리 파라미터 관측 가능
- thrust_coefficient, time_constant_up (2D): 추진기 효율/응답속도 불확실성
- ocean_current xyz (3D): 외란 (world frame)

### Removed (redundant dimensions)
- Buoy volume, CoG_z, CoB_z (3D): main body와 같은 DR scale
- Buoy Ixx, Iyy (2D): main body와 같은 inertia_scale
- Main Iyy (1D): Ixx와 같은 inertia_scale
- Linear/quadratic damping pitch (2D): roll과 같은 damping_scale

---

## [2026-03-31] Domain Randomization Redesign (Step 6)

### Context

DR 파라미터 관리 체계 개선: DORAEMON 커버리지 확장, dead code 제거, success metric 수정.
Terrain curriculum 분석 결과 DORAEMON이 이미 핵심 기능 (adaptive difficulty, forgetting prevention)을
갖추고 있어 파라미터 확장에 집중.

### Added
- `doraemon.py`: 8개 신규 PARAM_SPECS (volume_scale, cob/cog_offset_x/y, inertia_scale,
  body_mass_scale, payload_cog_offset_z). 총 7 -> 15 params.
- `doraemon.py`: Checkpoint backward compat (차원 불일치 시 partial restore + warning)
- `config.py` (HardDomainRandomizationCfg): joint_stiffness (30,150), joint_damping (0.3,7.0),
  thrust_coefficient_scale (0.7,1.3), time_constant_scale (0.7,1.3) Hard DR overrides

### Changed
- `mdp/events.py`: `_apply_xyz_offset_with_doraemon()` 일반화 (x_key, y_key, z_key 지원)
- `mdp/events.py`: volume_scale, inertia_scale -> `_sample_or_uniform()` (DORAEMON 지원)
- `mdp/events.py`: `randomize_body_mass()` sampled parameter 추가
- `mdp/events.py`: `randomize_payload()` payload_cog_offset_z DORAEMON 지원
- `albc_env.py`: `randomize_body_mass()` 호출에 `sampled` 전달
- `doraemon.py`: success metric rename (`success_threshold_deg` -> `success_threshold`, m/s 단위)
- `agents/rsl_rl_ppo_cfg.py`: encoder bounds Hard DR 범위 + 10% margin 적용

### Removed
- `config.py` (DomainRandomizationCfg): position_x/y/z_range, roll/pitch/yaw_range 6개 필드
- `mdp/events.py`: `randomize_robot_pose()` 함수 (dead code, 미호출)
- `mdp/__init__.py`: `randomize_robot_pose` export

---

## [2026-03-31] Observation Redesign (Step 5)

### Context

o_t 28D + proprio_hist 210D (238D total) 구조를 통합된 81D (26D current + 55D history)로 재설계.
센서 노이즈 물리 정합성, cable wrapping observability, history 효율성 개선.

### Added
- `config.py`: `hist_len`, `hist_stride`, `hist_feature_dim`, `hist_action_len` config fields
- `albc_env.py`: `_hist_buf(N,3,21)` ring buffer, `_get_hist_features()` (21D: joint tracking
  + body tracking + action), `_update_hist()` (stride-based recording)
- `config.py`: thruster state noise (std=0.02, bias=+-0.01), 81D noise model with per-component
  noise for current proprio + temporal history

### Changed
- `mdp/observations.py`: `compute_policy_obs` 28D -> **26D**
  - velocity error -> velocity command (no noise, 물리적 정확성)
  - atan2 joint wrapping -> raw cumulative angle (cable wrapping observability)
  - previous arm actions 제거 (history에 포함)
- `config.py`: `observation_space` 28 -> **81** (26D current + 55D history)
- `albc_env.py`: `_get_observations()` unified 81D policy obs (separate `proprio_hist` key 제거)
- `encoder/actor_critic_encoder.py`: `proprio_hist_dim` 파라미터 및 분리 로직 제거,
  actor/critic obs 빌드 단순화 (Actor 247D->90D, Critic 270D->113D)
- `agents/rsl_rl_ppo_cfg.py`: `policy_obs_dim=81`, obs_groups `["policy","privileged"]` (2-key)
- `test_full_dof_env.py`: obs shape 81D 검증, `proprio_hist` key 없음 확인

### Removed
- `_prev_actions_obs` buffer (current proprioception에서 prev action 제거)
- `_proprio_hist`, `_proprio_step_counter` buffers
- `proprio_history_len`, `proprio_feature_dim`, `proprio_history_stride` config fields
- `proprio_hist_dim` in encoder/agent config
- obs_groups의 `"proprio_hist"` key

### Observation Layout (81D)
| Slice | Element | Dim |
|-------|---------|-----|
| [0:6] | vel command (lin+ang) | 6D |
| [6:15] | body state (rpy+pqr+uvw) | 9D |
| [15:19] | joint state (pos+vel) | 4D |
| [19] | manipulability | 1D |
| [20:26] | thruster state | 6D |
| [26:38] | joint tracking history (3 steps) | 12D |
| [38:65] | body tracking history (3 steps) | 27D |
| [65:81] | action history (2 steps) | 16D |

---

## [2026-03-31] Command, Action, Reward Tuning (Step 4)

### Context

Step 3에서 constraint/termination 정리 후, command/action/reward 검토 및 튜닝.

### Added
- `config.py`: `vel_cmd_zero_prob: float = 0.1` -- 리샘플 시 10% env에 zero command
  (hovering/station-keeping 학습, Anymal `rel_standing_envs` 원리)
- `mdp/rewards.py`: `ang_vel_axis_weights: tuple = (2.0, 2.0, 1.0)` -- roll/pitch rate
  error에 yaw 대비 2x 가중치 (ALBC arm의 roll/pitch 안정화 중요도 반영)

### Changed
- `config.py`: `delta_scale` 0.08 -> **0.10** rad/step.
  이전: max vel 4.0 rad/s < `arm_joint_vel` constraint 4.189 -> constraint 발동 불가.
  이후: max vel 5.0 rad/s -> constraint 실효성 확보
- `albc_env.py`: `_sample_velocity_command()` zero-command mask 로직 추가
- `mdp/rewards.py`: `ang_vel_tracking()` uniform sum -> per-axis weighted sum

---

## [2026-03-31] Constraint, Termination & Env Redesign (Step 3)

### Context

Step 2의 velocity tracking 전환 후 constraint/termination/환경 초기화 전면 재설계.
참고 논문(NORBC)의 constraint 설계 철학 적용, 6DOF에 맞게 termination 정리,
position tracking 잔재 제거.

### Added
- `mdp/constraints.py`: 6개 새 constraint 함수
  - Probabilistic: `joint1_position_cost` (cable wrapping, limit=4*pi),
    `cumulative_yaw_cost` (tether wrapping, limit=8*pi)
  - Average ReLU-style: `yaw_rate_cost` (threshold=1.0 rad/s),
    `body_linear_velocity_cost` (threshold=1.0 m/s),
    `thruster_utilization_cost` (peak |T_i|),
    `manipulability_cost` (w_threshold=0.3)
- `albc_env.py`: cumulative yaw tracking buffers + `_update_cumulative_yaw()`
- `albc_env.py`: `_term_too_fast_lin` termination flag + 로깅

### Changed
- `agent.urdf`: joint1, joint2 모두 `revolute` -> `continuous`
- `mdp/observations.py`, `albc_env.py`: joint position normalization
  linear -> **angular wrapping** (`atan2(sin, cos) / pi`)
- `albc_env.py`: `_apply_joint_pd_action()` position clamp 제거
- `mdp/events.py`: joint clamp 제거, `reset_robot_pose_default` 단순화
  (env_origin 기준 (0,0,0), identity orientation, zero velocity)
- `config.py`: `_FULL_DOF_CONSTRAINT_TERMS` 6 -> 9 terms (5 prob + 4 avg)
- Termination 6DOF 정리:
  - `too_fast` -> `too_fast_ang` (roll/pitch 2축 -> 전체 3축)
  - `bad_state`: `root_ang_vel_b` NaN/Inf 체크 추가
  - 모든 termination 조건 로깅 (`too_fast_ang`, `too_fast_lin`, `bad_state`,
    `excessive_tilt`)
- `_collect_episode_metrics()`: `Attitude_Error/` -> `Attitude/roll_deg`,
  `pitch_deg` (절대값 모니터링)
- DORAEMON settling error: `_attitude_error` -> `_lin_vel_err` 기반
- Pose DR 제거: `_reset_task_and_state()`에서 항상 `reset_robot_pose_default` 호출.
  Joint DR만 유지
- `test_full_dof_env.py`: constraint tests (shape=9, 인덱스 조정)

### Removed
- `yaw_velocity_cost`, `position_bound_cost`, `depth_bound_cost` (3개 constraint 함수)
- `_position_error_body`, `_joint_limits_lower/upper/range` buffers
- `out_of_depth` termination condition
- Position tracking 잔재: `target_attitude`, `randomize_target_attitude`,
  `target_attitude_range` config; `_target_euler`, `_base_attitude`, `_target_range`,
  `_randomize_targets`, `_attitude_error` buffers; `compute_attitude_error()`,
  `_get_attitude_error()`, `_update_attitude_error()` methods
- `initial_height` config (pose가 항상 env_origin)
- `randomize_robot_pose` import in `albc_env.py`
- `min_depth`, `max_depth` config fields

### Constraint Layout (9 total)
| # | Name | Type | Budget |
|---|------|------|--------|
| 0 | attitude | Prob | 0.01 |
| 1 | arm_torque | Prob | 0.08 |
| 2 | arm_joint_vel | Prob | 0.02 |
| 3 | joint1_pos | Prob | 0.01 |
| 4 | cumul_yaw | Prob | 0.01 |
| 5 | yaw_rate | Avg | 0.10 |
| 6 | body_lin_vel | Avg | 0.10 |
| 7 | thruster_util | Avg | 0.40 |
| 8 | manipulability | Avg | 0.05 |

### Termination Layout (4 total)
| # | Name | Formula | Threshold |
|---|------|---------|-----------|
| 0 | too_fast_ang | `max(\|p\|,\|q\|,\|r\|) > limit` | pi rad/s |
| 1 | too_fast_lin | `\|\|v_world\|\| > limit` | 2.0 m/s |
| 2 | bad_state | NaN/Inf on pos, quat, lin_vel, ang_vel | - |
| 3 | excessive_tilt | `\|roll\| > limit \| \|pitch\| > limit` | pi/2 |

---

## [2026-03-31] Velocity Tracking Task Redesign (Step 2)

### Context

Step 1의 position+attitude tracking 환경에서 **reward 충돌** 발견:
position tracking reward는 target으로 이동 요구(velocity 필요), velocity regulation
reward는 속도 억제 요구. Transit 구간에서 상반됨.

해결: **cascaded velocity tracking** 아키텍처로 전환.
- 학습: Policy가 random velocity command를 tracking
- 배포: Outer PID(pos/att error -> vel_cmd) -> Policy(vel_cmd -> action)
- PID 튜닝과 학습이 완전 분리

추가로 **Yoshikawa manipulability index**를 o_t에 1D 추가하여 arm singularity 인식.

### Added
- `albc_env.py`: velocity command buffers (`_vel_cmd_lin`, `_vel_cmd_ang`), velocity error
  buffers (`_lin_vel_err`, `_ang_vel_err`), manipulability buffer, command resampling logic
  (250 steps = 5s 주기)
- `albc_env.py`: `_update_manipulability()` -- Yoshikawa index `w = sqrt(|l1*l2*sin(theta2)|)`,
  정규화 [0,1]. `_sample_velocity_command()` with resampling counter
- Logging: `Vel_Tracking/lin_vel_err_norm`, `ang_vel_err_norm`, `cmd_norm`,
  `Arm/manipulability_mean`, `manipulability_min`

### Changed
- `mdp/rewards.py`: 6-term -> **5-term** velocity tracking reward
  - 제거: `attitude_tracking`, `position_tracking`, `velocity_regulation`
  - 추가: `lin_vel_tracking` (k=-4.0), `ang_vel_tracking` (k=-8.0)
  - 유지: `joint_torque`, `thruster_energy`, `action_smoothness`
- `mdp/observations.py`: 27D -> **28D** velocity tracking obs
  - command: att_err(3)+pos_err(3) -> lin_vel_err(3)+ang_vel_err(3)
  - body: euler(3)+ang_vel(3)+lin_vel(3) (lin_vel 추가, drag/current 보상에 필요)
  - arm: jpos(2)+jvel(2)+prev_arm(2)+**manipulability(1D)** (신규)
  - thruster: 6D 유지 (ESC RPM feedback으로 실측 가능)
- `config.py`: `observation_space=28`, `proprio_feature_dim=14`,
  vel_cmd ranges/resample config 추가, position target configs 제거
- `albc_env.py`: `_get_proprio_features()` 16D -> **14D**
  - 제거: `pos_err_body(3D)` (velocity tracking에서 불필요)
  - 추가: `manipulability(1D)` (singularity 추이 관찰)
- `agents/rsl_rl_ppo_cfg.py`: `policy_obs_dim=28`, `proprio_hist_dim=210`
  - Actor: 247D input, Critic: 270D input
- `albc_env.py`: `_get_dones()` -- position drift termination 제거 (position target 없음)
- `albc_env.py`: `_reset_task_and_state()` -- `_sample_target_position()` -> `_sample_velocity_command()`

### Removed
- Position tracking: `_target_pos_w`, `_position_error_body` 능동 사용 제거
  (constraint 호환을 위해 dummy zero buffer 유지)
- `max_position_error` config 제거
- `target_pos_offset_xy`, `target_pos_offset_z` config 제거

### Verification
- 14/14 tests passed: smoke(2), obs_dims(3), thruster_motion(2), reward(1),
  manipulability(2), resampling(2), constraints(2)
- obs: (N,28), privileged: (N,23), proprio_hist: (N,210)
- Manipulability: nominal(pi/2)=1.0, singularity(0)=0.1

### Design Document
- [full_dof_tracking_design.md](plans/2026-03-31-full-dof-tracking-design.md)

---

## [2026-03-31] Full 6-DOF Environment: constrained_full_albc (Step 1)

### Context

`constrained_albc`를 `constrained_full_albc`로 fork하여 full 6-DOF position+attitude
tracking 환경을 구축. 기존 2D arm-only attitude 제어가 underdetermined system이라는
가설에서 출발: 8D action (2D arm + 6D thruster)으로 확장하면 fully determined system이
되어 더 나은 수렴과 encoder 학습이 가능할 것으로 기대.

TRPO+IPO+Encoder pipeline만 남기고 나머지 5개 task (HistOnly, FrozenEncoder,
SharedBackbone, AsymmetricEncoder, Production Encoder)를 제거. 단일 task
`Isaac-FullDOF-TRPO-v0`으로 정리.

### Added
- `constrained_full_albc/` 전체 패키지 (fork from `constrained_albc`)
- `mdp/rewards.py`: `position_tracking()`, `velocity_regulation()`, `thruster_energy()` 3개 reward 함수. `FullDOFRewardManager` 6-term reward (attitude, position, velocity, torque, thruster, smoothness)
- `mdp/constraints.py`: `position_bound_cost()` (I(||pos_err|| > 3m)), `depth_bound_cost()` (I(z < 1m or z > 8m)). Total 6 constraints (기존 4 + 신규 2)
- `albc_env.py`: `_target_pos_w`, `_position_error_body` buffers. `_sample_target_position()` (reset 시 current_pos + random offset). Position/depth/velocity termination conditions
- `config.py`: `FullDOFEnvCfg` -- 8D action, 27D obs, 30s episode, 6 constraints, ocean current enabled, thruster enabled
- `scripts/demos/test_full_dof_env.py`: 5-category verification script (10 tests)

### Changed
- `mdp/observations.py`: `compute_policy_obs()` 재구성 33D -> 27D
  - 기능별 그룹핑: command(6D), body_state(9D), arm(6D), thruster(6D)
  - Attitude error 3D 통합 (기존: rp 2D + yaw 1D 분리)
  - `prev_thruster_actions` (6D) 제거 (thruster state와 중복)
- `albc_env.py`: `_get_proprio_features()` 8D -> 16D
  - 추가: yaw, r (ang_vel_yaw), lin_vel_body(3D), pos_err_body(3D)
  - Proprio history: 120D -> 240D (16D x 15 steps, stride=5, 1.5s window)
- `mdp/rewards.py`: `command_tracking()` -> `attitude_tracking()` (yaw 포함 3D)
- `agents/rsl_rl_ppo_cfg.py`: 단일 `FullDOFTRPORunnerCfg`만 유지
  - policy_obs_dim=27, proprio_hist_dim=240
  - Actor input: 276D, Critic input: 299D

### Removed
- 5개 unused task registrations, 관련 config/runner classes
- `encoder/actor_critic_encoder_constrained.py`, `encoder/actor_critic_frozen_encoder.py`
- `docs/` (원본 constrained_albc에 유지)

### Verification
- 10/10 tests passed: smoke(2), obs_dims(3), thruster_motion(2), reward(1), constraints(2)
- obs: (N,27), privileged: (N,23), proprio_hist: (N,240)
- Thruster forward: 0.65m/50steps, vertical: 0.83m/50steps

### Design Document
- [full_dof_tracking_design.md](plans/2026-03-31-full-dof-tracking-design.md)

---

## [2026-03-30] Thruster Integration (Stage 1) + TRPO Gradient Decomposition

### Context

Added 6 thrusters to the Hero Agent UUV simulator, matching the real robot's actuator
layout (4 horizontal vectored + 2 vertical). Motivation: the 2D ALBC-only action space
is too simple -- adding thrusters enables position/velocity/heading control and expands
action space to 8D, which also improves KL budget distribution for RL (sigma occupancy
drops from ~33% to ~8%).

Thruster positions and TAM (Thrust Allocation Matrix) derived from the real robot's
`hero_agent_control/config/TAM.yaml` and `actuators.xacro`, verified against the URDF
geometry. Thruster parameters use BlueROV T200 as baseline (max_thrust=50N, coeff=40,
time_constant_up=0.1s) with DR covering real-robot differences.

Also added TRPO gradient decomposition logging to diagnose why encoder sensitivity
decreases over training (iter 550->750: most DR parameters lost 10-40% sensitivity).
Hypothesis: TRPO natural gradient actively reduces encoder sensitivity because the
policy finds a "robust average" strategy. The FIM may rotate the encoder gradient
direction. New `GradDecomp/` metrics will test this in the next training run.

Thruster visualization test confirmed correct force directions for all 6 DOF
(surge, sway, heave, roll, pitch, yaw) with minor position offsets acceptable
for DR coverage.

### Added
- `config.py`: `HeroAgentThrusterCfg` with 6-thruster TAM from real robot, BlueROV T200 parameters
- `config.py`: `ALBCEnvCfg.thrusters` field (default None = backward compatible, no thrusters)
- `config.py`: `DomainRandomizationCfg.thrust_coefficient_scale` and `time_constant_scale` for thruster DR
- `albc_env.py`: `_init_thrusters()` method using existing `ThrusterModel` from `isaaclab_tasks.models`
- `constraint_trpo.py`: 8 new `GradDecomp/` monitoring metrics -- vanilla/natural gradient norms for encoder vs actor, cosine similarity between vanilla and natural gradient (encoder), cosine similarity between vanilla gradient and step direction (encoder)
- `constraint_encoder_runner.py`: TB/WandB logging for `GradDecomp/` metrics
- `scripts/demos/test_hero_thruster.py`: Standalone thruster verification script with two modes: `viz` (static arrow visualization of thruster layout) and `test` (directional force test, 5s per direction)

### Changed
- `albc_env.py`: `_pre_physics_step()` now splits actions into arm (first 2D) and thruster (remaining 6D), applies thruster dynamics via `ThrusterModel.apply_dynamics()`
- `albc_env.py`: `_apply_action()` pre-combines hydro + thruster forces before single `set_forces_and_torques()` call (critical: `set` overwrites, cannot call twice on same body)
- `albc_env.py`: `_reset_physics()` resets thruster state and randomizes thruster parameters when DR enabled

### Notes
- Stage 1 (this session): thruster forces working, visualization confirmed, backward compatible
- Stage 2 (next): expand observation space (position, velocity, heading target), reward structure, constraints, encoder privileged obs for thruster DR parameters
- Target task: track xyzrpy + vxvyvz in body frame, angular velocity targets always 0
- URDF dimensions differ slightly between Isaac Lab (9.18kg, r=0.09m) and reference (8.6kg, r=0.0825m) -- TAM is still valid since it encodes thruster geometry, and DR covers parameter differences
- `GradDecomp/enc_cos_vanilla_natgrad < 0` would confirm FIM rotates encoder gradient (TRPO actively harms encoding)
- Encoder z sweep @750: sensitivity declined from @550 peak (most params -10 to -40%), suggesting the current TRPO co-training degrades encoding after noise_std drops sufficiently

---

## [2026-03-30] TRPO + IPO + Asymmetric Encoder: Integration and Sigma Decoupling

### Context

Combined the asymmetric encoder architecture (15D->9D, LayerNorm+softsign, critic_uses_z)
with TRPO + IPO constrained RL. Previously these existed separately: asymmetric encoder
used PPO, while TRPO+IPO used shared backbone multi-head critic. The new task
`Isaac-Constrained-ALBC-HardDR-AsymmetricEncoder-TRPO-v0` combines both.

**First run (no sigma decoupling, 334 iters):** TRPO consumed KL budget reducing sigma
instead of improving action mean. noise_std dropped 0.99->0.28 while attitude error
stagnated at ~20 deg. Encoder z_std remained constant at 0.4498 (no learning). Reward
worsened from -8 to -47. Root cause: 2D action space makes sigma changes ~33% of KL
budget, so TRPO preferentially reduces sigma over improving mean.

**Second run (sigma decoupling, 248 iters):** noise_std decays slower (0.0022/iter vs
0.0036/iter), reward slightly better (-44.9 vs -48.2 at iter 200). Sigma decoupling
partially effective. However encoder z_std still constant at 0.456 -- encoder receives
insufficient gradient through TRPO natural gradient. Attitude error ~20 deg, improving
at ~0.03 deg/iter.

**Remaining issue:** Encoder gradient (0.005-0.04) is too small for meaningful updates
within TRPO trust region. The encoder is essentially producing random z from initialization
weights. TRPO+IPO is operating as actor-only, without encoder benefit.

### Added
- `encoder/actor_critic_encoder.py`: `num_constraints` and `cost_critic_hidden_dims`
  params. Separate cost critic MLP in asymmetric mode: same input as reward critic
  `cat([o_t, hist, z, p_t])=166D -> MLP[512,256,128] -> K`. `evaluate_costs()` method.
  `load_state_dict()` backward compat for missing cost_critic. `num_constraints=0` and
  `cost_critic=None` in shared_backbone mode.
- `agents/rsl_rl_ppo_cfg.py`: `_AsymmetricEncoderConstrainedPolicyCfg` (adds
  num_constraints, cost_critic_hidden_dims to asymmetric policy).
  `ALBCHardDRAsymmetricEncoderTRPORunnerCfg` (TRPO+IPO algorithm + asymmetric encoder).
- `config.py`: `_STANDARD_CONSTRAINT_TERMS` module-level constant (DRY: shared between
  ALBCEnvCfg and constrained variants). `ALBCHardDRAsymmetricEncoderConstrainedEnvCfg`
  (Hard DR + constraints enabled).
- `__init__.py`: Registered `Isaac-Constrained-ALBC-HardDR-AsymmetricEncoder-TRPO-v0`.

### Changed
- `algorithms/constraint_trpo.py`: **Sigma decoupling** -- `log_std` removed from
  `_policy_params`, separate Adam optimizer (`std_lr=3e-3`). After TRPO step, re-snapshot
  baseline (IS ratio=1.0), compute reward-only score-function gradient, Adam step on
  log_std, then min_std clamp. TRPO KL budget now fully used for mean improvement.
- `algorithms/constraint_trpo.py`: **Binary constraint std clamp** -- `ca_std + 1e-8`
  changed to `ca_std.clamp(min=1.0)`. Prevents gradient explosion when binary constraints
  (attitude/torque/velocity) have near-zero std (all envs agree on 0 or 1).
- `algorithms/constraint_trpo.py`: **value_prefixes trailing dot** -- `"critic"` changed
  to `"critic."` to prevent `critic_obs_normalizer` from being misclassified as value
  param (pre-existing bug, not triggered when `critic_obs_normalization=False`).
- `agents/rsl_rl_ppo_cfg.py`: Added `std_lr: float = 3e-3` to
  `RslRlConstraintTRPOAlgorithmCfg`.
- `config.py`: `ALBCEnvCfg.constraints` now uses `_STANDARD_CONSTRAINT_TERMS` constant.

### Experimental Results

**Run 1: No sigma decoupling (334 iters, killed):**

| Metric | Iter 0 | Iter 150 | Iter 334 |
|--------|--------|----------|----------|
| Roll | 28.7 deg | 21.7 deg | 19.6 deg |
| Pitch | 22.5 deg | 22.4 deg | 22.3 deg |
| noise_std | 0.99 | 0.42 | 0.28 |
| Encoder z_std | 0.4498 | 0.4479 | 0.4493 |
| Reward | -8.0 | -52.6 | -41.3 |

**Run 2: Sigma decoupling applied (248 iters, ongoing):**

| Metric | Iter 0 | Iter 150 | Iter 248 |
|--------|--------|----------|----------|
| Roll | 28.7 deg | 22.7 deg | 21.3 deg |
| Pitch | 22.5 deg | 22.4 deg | 19.7 deg |
| noise_std | 1.00 | 0.65 | 0.46 |
| Encoder z_std | 0.450 | 0.449 | 0.456 |
| Reward | -8.0 | -49.3 | -54.8 |
| cost_value loss | 13.7 | 5.7 | 8.9 (increasing) |
| KL | 0.003 | 0.004 | 0.005 (at max_kl) |

### Notes
- Sigma decoupling slowed noise_std decay (0.0036 -> 0.0022/iter), confirming the
  KL budget was being consumed by sigma changes.
- Encoder is not learning in either run: z_std constant at ~0.45 = softsign(N(0,1))
  natural spread. Encoder gradient (0.005-0.04) is too small relative to actor gradient
  within the TRPO natural gradient framework.
- cost_value loss increasing after iter 100: cost return distribution is non-stationary
  (yaw_vel oscillates 7.7->47->27->43), critic can't track.
- At current improvement rate (~0.03 deg/iter), 2500 iters would reach ~10-11 deg.
  PPO hist-only baseline achieves 8.7 deg. Encoder not contributing.

## [2026-03-30] Encoder Decoupling Experiment + std_lr Tuning

### Context

Continued investigation into encoder not learning within TRPO. Two hypotheses tested:
1. TRPO natural gradient distorts encoder step direction (Fisher matrix is over action
   distribution, not feature space). Separating encoder to Adam should preserve encoding.
2. noise_std reaching min_std floor (0.2) too early (~500 iter) starves exploration.

**Run 3: Encoder decoupled (11-51-27 continued to 637 iter):**
Run 11-51-27 ran to completion with original settings (std_lr=3e-3, encoder in TRPO).
Z sweep at iter 450 vs iter 0 showed encoding COLLAPSE: most DR parameters lost
60-80% sensitivity. Main Volume max z range: 0.88->0.34, Body Mass: 0.70->0.29,
Added Mass: 0.66->0.20, Joint Damping: 0.66->0.18. Only Payload CoG X/Y survived.
noise_std hit floor (0.2) at ~500 iter, after which roll improvement slowed 5x
(0.032->0.006 deg/iter). Pitch continued improving (0.026 deg/iter at floor).
Final: roll 14.2, pitch 13.0 at iter 637.

**Run 4: Encoder decoupled from TRPO (12-27-19, 270 iter, killed):**
Separated encoder params from `_policy_params`, gave encoder its own Adam optimizer
(lr=3e-3, wd=1e-5) with post-TRPO re-snapshot and full surrogate (reward+barrier).
Result: FAILED. enc_grad dropped 85% (0.040->0.006) because actor->z gradient path
is inherently weak when decoupled. Z sweep @250 showed worse encoding than OLD @250:
Lin Damp Pitch 0/9 active (was 6/9), Payload CoG X 1/9 (was 5/9). Performance also
worse: reward -46.1 (vs -40.9 OLD), pitch 21.6 (vs 20.0 OLD).
Root cause: with re-snapshot, IS ratio starts at 1.0, and the gradient from surrogate
through actor->z to encoder is too attenuated. TRPO's CG solver at least gives
encoder an indirect signal through the shared Fisher Hessian.

**Run 5: std_lr reduced, encoder back in TRPO (12-47-41, ongoing):**
Reverted encoder decoupling. Changed std_lr from 3e-3 to 1e-3. At 171 iter:
noise_std=0.84 (vs OLD 0.65 at same iter). Floor estimated at ~824 iter (vs ~500).
Exploration preserved longer. Z sweep @150 showed sensitivity maintained (Joint
Damping 0.75, Joint Stiffness 0.69, Payload CoG Y 0.56). Z sweep @300 showed
sensitivity INCREASING: Body Mass 0.51->1.01, Joint Stiffness 0.69->1.03, Payload
CoG Y 0.56->0.99, Main Volume 0.39->0.87. This contrasts sharply with run 3 where
sensitivity collapsed. The key difference: slower noise_std decay means more diverse
actions, providing encoder with richer gradient signal.

Further change: min_std reduced from 0.2 to 0.01 (safety net only). With std_lr=1e-3,
score-function gradient naturally finds equilibrium without needing artificial floor.

### Changed
- `algorithms/constraint_trpo.py`: Encoder decoupling implemented then reverted.
  Final state: encoder in TRPO `_policy_params` (original design). `std_lr` default
  changed from 3e-3 to 1e-3. `min_std` default changed from 0.2 to 0.01.
- `agents/rsl_rl_ppo_cfg.py`: `std_lr` changed from 3e-3 to 1e-3. `min_std` changed
  from 0.2 to 0.01. Runner docstring updated with experiment results.

### Experimental Results

**Run 3: Original settings continued (11-51-27, 637 iters):**

| Metric | Iter 100 | Iter 300 | Iter 500 | Iter 637 |
|--------|----------|----------|----------|----------|
| Roll | 25.3 deg | 19.5 deg | 15.0 deg | 14.2 deg |
| Pitch | 23.6 deg | 19.0 deg | 16.5 deg | 13.0 deg |
| noise_std | 0.776 | 0.390 | 0.214 | 0.200 (floor) |
| Encoder z_std | 0.452 | 0.457 | 0.450 | 0.460 |
| Z sweep max | 0.91 (init) | -- | -- | 0.34 (collapsed) |

**Run 4: Encoder decoupled (12-27-19, 270 iters, killed):**

| Metric | @150 (OLD) | @150 (NEW) | Diff |
|--------|-----------|-----------|------|
| Roll | 22.7 deg | 18.2 deg | -4.5 (better) |
| Reward | -49.3 | -46.1 | worse |
| enc_grad | 0.040 | 0.006 | -85% |

**Run 5: std_lr=1e-3 (12-47-41, 300+ iters, ongoing):**

| Metric | Iter 100 | Iter 150 | Iter 300 (est) |
|--------|----------|----------|----------|
| Roll | 20.9 deg | 20.4 deg | ~19.5 deg |
| Pitch | 24.1 deg | 24.3 deg | ~23.8 deg |
| noise_std | 0.909 | 0.856 | ~0.75 |
| Z sweep max | -- | 0.75 | 1.03 (improving!) |

### Notes
- z_std=0.45 constant does NOT mean "encoder not learning". z_std is the statistical
  spread of softsign output, not encoding quality. Z sweep heatmaps are the only
  reliable measure of encoder learning.
- Encoder decoupling from TRPO failed because actor->z gradient path is too weak when
  using re-snapshot IS ratio. The CG solver in TRPO provides indirect but sufficient
  encoder signal through shared Hessian-vector products.
- Slower noise_std decay (std_lr 3e-3->1e-3) is the key enabler for encoder learning:
  more exploration = more diverse actions = richer gradient signal through actor->z->encoder.
- min_std floor (0.2->0.01): with slow std_lr, score-function gradient equilibrium
  naturally determines optimal sigma. Floor was only needed with fast std_lr=3e-3.
- PPO and TRPO use identical normalization: EmpiricalNorm on o_t(14D)+hist(120D),
  z(9D) raw (softsign bounded). critic_obs_normalization=False.

---

## [2026-03-30] Asymmetric Critic Test + Pre-Softsign LayerNorm

### Context

Two experiments to improve online encoder training:

**Experiment 1: Asymmetric critic (critic sees z + p_t).**
Tested whether critic receiving both z and raw privileged obs p_t would provide
encoder gradient from value loss while using separate actor/critic MLPs.
Result: **shortcut problem confirmed.** Critic immediately ignores z in favor of
p_t (easier path to value prediction). z_std goes to ~1 instantly and stays
constant -- encoder receives no meaningful gradient from either path. Actor also
can't leverage z (chicken-and-egg: z is noise -> actor ignores z -> no gradient
to shape z).

**Experiment 2: Pre-softsign LayerNorm (shared backbone).**
Root cause analysis of z saturation in original shared backbone run: encoder
weight growth causes pre-softsign MLP output to explode (|x| mean: 0.44 at init
-> 8.50 at iter 499). Softsign gradient = 1/(1+|x|)^2 vanishes (75% of outputs
have gradient < 0.05 by iter 350), trapping z near boundaries.

Added LayerNorm between encoder MLP output and softsign activation. LayerNorm
normalizes output to ~N(0,1), keeping softsign in its responsive range regardless
of weight magnitude.

Result: z saturation eliminated. Encoder learns meaningful representations --
Body Mass 12/13 active dims, Main Volume 10/13, CoG/CoB 10/13. However, noise_std
drops too fast (entropy_coef=0.0, no min_std floor), causing premature exploration
collapse. Final performance still worse than hist-only baseline.

### Added
- `config.py`: `ALBCHardDRAsymmetricEncoderEnvCfg` -- inherits SharedBackbone env
- `agents/rsl_rl_ppo_cfg.py`: `_AsymmetricEncoderPolicyCfg` (critic_uses_z=True,
  shared_backbone=False, critic_obs_normalization=False),
  `ALBCHardDRAsymmetricEncoderRunnerCfg`
- `__init__.py`: Registered `Isaac-Constrained-ALBC-HardDR-AsymmetricEncoder-v0`

### Changed
- `encoder/actor_critic_encoder.py`: Added `critic_uses_z` param -- when True,
  `_get_critic_obs()` includes z via `_encode(obs)`, making critic input
  cat([o_t, hist, z, p_t]). Added `encoder_output_norm` param -- when True,
  inserts `nn.LayerNorm(latent_dim)` between encoder MLP and softsign activation.
  Updated `_encode()` flow: MLP -> LayerNorm -> softsign.
- `agents/rsl_rl_ppo_cfg.py`: `_SharedBackbonePolicyCfg` now sets
  `encoder_output_norm: bool = True`. Both shared backbone and asymmetric configs
  use LayerNorm.
- `scripts/analysis/encoder_z_sweep.py`: Detects `_encoder_output_norm` in
  checkpoint and includes LayerNorm in reconstructed encoder Sequential.
  `build_encoder_mlp()` gains `output_norm` parameter.

### Experimental Results

**Shared Backbone + LayerNorm (500 iters) vs Hist-Only (500 iters):**

| Metric | Shared BB + LN | Hist-Only | Delta |
|--------|:--------------:|:---------:|:-----:|
| Roll | 9.96 deg | 8.74 deg | +1.22 |
| Pitch | 9.74 deg | 6.54 deg | +3.20 |
| Reward | -18.22 | -10.75 | -7.47 |
| noise_std | 0.15 | 0.15 | 0 |
| z_std | 0.56 | -- | -- |

**Encoder z sweep comparison (shared backbone, iter 499):**

| Condition | |z|>0.9 | Softsign grad mean | Body Mass active | Main Vol active |
|-----------|:------:|:-----------------:|:----------------:|:--------------:|
| No LayerNorm | 44% | 0.063 | 12/13 (saturated) | 10/13 (saturated) |
| With LayerNorm | ~0% | ~0.25 (healthy) | 12/13 (real variation) | 10/13 (real variation) |

**Root cause data (no LayerNorm, model_0 vs model_499):**

| Metric | Init (iter 0) | Trained (iter 499) |
|--------|:------------:|:-----------------:|
| Pre-softsign |x| mean | 0.44 | 8.50 |
| |x| > 3 fraction | 0% | 80% |
| Softsign gradient mean | 0.548 | 0.063 |
| Encoder weight std (hidden) | 0.04-0.07 | 0.12-0.13 |

**Asymmetric Critic + LayerNorm (500 iters):**

Previous asymmetric run (no LayerNorm) showed z_std -> 1 immediately (shortcut +
saturation confounded). Re-ran with LayerNorm to isolate the shortcut effect.
Result: encoder DOES learn -- z_std=0.70 (stable), z responds to DR parameters.
Shortcut is not total: actor gradient alone (with LayerNorm) is sufficient to
train the encoder. Encoder even shows broader DR sensitivity than shared backbone
(quad_damp_roll 0.06->0.47, water_density 0.04->0.26, buoy_cog_z 0.02->0.14).

| Metric | Hist-Only | Shared BB + LN | Asymmetric + LN |
|--------|:---------:|:--------------:|:---------------:|
| Roll | **8.74** | 9.96 | 9.77 |
| Pitch | **6.54** | 9.74 | 10.16 |
| Reward | **-10.75** | -18.22 | -19.74 |
| noise_std | 0.15 | 0.15 | 0.14 |
| z_std | -- | 0.56 | 0.70 |

Encoder z sweep (asymmetric + LN, iter 499): Body Mass 10/13 active (max 1.06),
Main CoG Z 8/13 (max 0.90), Quad Damp Roll 3/13 (max 0.47), Water Density 2/13
(max 0.26). More diverse than shared backbone but still no performance gain.

### Notes
- Asymmetric critic WITHOUT LayerNorm: shortcut + saturation -> encoder fails.
  WITH LayerNorm: encoder learns via actor gradient alone. The earlier "shortcut
  conclusively disproved" conclusion was wrong -- the issue was saturation, not
  shortcut exclusively.
- Both encoder architectures (shared BB, asymmetric) learn encoder representations
  but neither beats hist-only. Common bottleneck: noise_std collapse (entropy_coef=0,
  no min_std floor).
- encoder_z_sweep.py verified to produce identical output as training forward pass
  (max diff = 0.00e+00).

---

## [2026-03-30] Encoder Input Reduction + Hyperparameter Ablations

### Context

Continued encoder experiments from previous session. Three ablations tested on
the asymmetric critic + LayerNorm architecture to improve encoder-based policy:

**Experiment 3: entropy_coef=0.001 (asymmetric + LN, 23D->13D, 500 iters).**
Hypothesis: noise_std collapse (0.14, LOW) is the bottleneck. Small entropy bonus
should maintain exploration. Result: noise_std improved marginally (0.14->0.17) but
still LOW. Roll WORSENED (9.77->13.59 deg), pitch slightly better (10.16->9.69 deg).
Entropy bonus interfered with exploitation without sufficiently maintaining exploration.

**Experiment 4: encoder [128, 64] 2-hidden layer (asymmetric + LN, 23D->13D, 500 iters).**
Hypothesis: 3-layer [256,128,64] encoder (~49K params) is over-parameterized for
23D->13D compression. Smaller encoder should learn faster. Result: performance degraded
(not fully analyzed, user observed instability and moved on).

**Experiment 5: Reduced encoder input 15D->6D (asymmetric + LN, no ocean current).**
Based on z-sweep sensitivity analysis, dropped 8 input dims with near-zero encoder
response (buoy CoG/CoB Z, main/buoy Ixx/Iyy, payload CoG Z, water density). Kept
10 clearly important + 3 borderline/suspicious + 2 physically important (payload CoG XY).
Also removed ocean current from DR. Result: severe instability at 145 iters (roll 23 deg,
pitch 39 deg), compression ratio 2.5:1 likely too aggressive.

**Experiment 6: Increased output to 9D (15D->9D, asymmetric + LN, no ocean current, 500 iters).**
Raised latent dim from 6 to 9 (compression ratio 1.67:1, close to original 1.77:1).
Result: much better than 6D -- roll 12.96 deg, pitch 11.06 deg. Encoder z sweep shows
improved sensitivity to CoG Z (3.5x), CoB Z (2.9x), and Lin Damp Roll (0->0.46) vs
23D->13D. But performance still worse than hist-only and 23D->13D asymmetric.
noise_std=0.13 (LOW) remains the common bottleneck across ALL encoder experiments.

### Added
- `agents/rsl_rl_ppo_cfg.py`: 15D encoder bounds (`_ENC_OBS_INDICES_15D`,
  `_ENC_OBS_15D_LOWER`, `_ENC_OBS_15D_UPPER`) selected by z-sweep sensitivity analysis
- `scripts/analysis/common.py`: `_build_reduced_encoder_sweep()` for reduced encoder
  z sweep parameter mapping

### Changed
- `encoder/actor_critic_encoder.py`: Added `encoder_obs_indices` parameter.
  When provided, `_encode()` selects subset of privileged dims before normalization.
  Encoder input_dim matches len(indices), bounds validated against selected dims.
- `agents/rsl_rl_ppo_cfg.py`: `_AsymmetricEncoderPolicyCfg` updated to use 15D input
  (encoder_obs_indices), 9D output (encoder_latent_dim=9), entropy_coef=0.0 (restored).
  Encoder hidden dims restored to [256,128,64].
- `config.py`: `ALBCHardDRAsymmetricEncoderEnvCfg` now disables ocean current
  (max_velocity=0, noise_scale=0)
- `scripts/analysis/common.py`: `get_encoder_architecture_from_checkpoint()` detects
  softsign for 15D+ encoders with static bounds. `build_sweep_params_from_checkpoint()`
  routes non-23D static-bound encoders to reduced sweep builder.

### Experimental Results

**All experiments: asymmetric critic + LayerNorm, 500 iters unless noted:**

| Metric | Hist-Only | 23D->13D (ent=0) | 23D->13D (ent=0.001) | **15D->9D (ent=0)** |
|--------|:---------:|:----------------:|:--------------------:|:-------------------:|
| Roll | **8.74** | **9.77** | 13.59 | 12.96 |
| Pitch | **6.54** | **10.16** | 9.69 | 11.06 |
| Reward | **-10.75** | **-19.74** | -17.49 | -19.75 |
| noise_std | 0.15 | 0.14 | 0.17 | 0.13 |
| z_std | -- | 0.70 | 0.75 | 0.70 |

**15D->9D encoder z sweep improvements vs 23D->13D:**

| Parameter | 23D->13D max range | 15D->9D max range | Change |
|-----------|:------------------:|:-----------------:|:------:|
| Main CoG Z | 0.32 | **1.13** | 3.5x |
| Main CoB Z | 0.35 | **1.00** | 2.9x |
| Lin Damp Roll | 0.04 | **0.46** | 11x |
| Body Mass | 1.75 | 1.59 | -9% |
| Main Volume | 1.70 | 1.40 | -18% |

### Notes
- Input reduction improved encoder's sensitivity to secondary parameters (CoG, CoB,
  damping) by removing noise from uninformative dims. But performance did not improve.
- All encoder experiments share noise_std collapse (entropy_coef=0, no min_std floor).
  This is likely the fundamental bottleneck -- encoder produces good z but policy
  can't explore to exploit it.
- 15D->6D was too aggressive (pitch 39 deg at 145 iters). 15D->9D is viable.
- Ocean current removed from asymmetric env to focus on pure DR adaptation.

---

## [2026-03-30] Shared Backbone Encoder: Online End-to-End PPO

### Context

Offline encoder experiments showed that value prediction provides a strong learning
signal for the encoder (R^2 0.088 -> 0.791). However, the existing online encoder
architecture (separate mode) only gave the encoder gradient from the actor/policy
loss -- the critic used privileged obs directly, bypassing the encoder entirely.

Designed a shared backbone architecture where the encoder receives gradient from
BOTH actor and critic losses. The critic uses z (not raw privileged obs) as its
only path to privileged information, replicating the offline encoder's success
condition (value-prediction trains encoder) in online end-to-end training.

Previous online encoder experiments (Steps 5a-5b) failed with shared backbone due
to `sample().clamp(-1,1)` causing KL death (root cause identified in Step 17, clamp
since removed). With clamp removed + PPO single optimizer, shared backbone is stable.

### Added
- `config.py`: `ALBCHardDRSharedBackboneEnvCfg` -- Hard DR + 15-step strided
  history (stride=5, 120D) for shared backbone experiments
- `agents/rsl_rl_ppo_cfg.py`: `_SharedBackboneAlgorithmCfg` (PPO, use_encoder_update=False),
  `_SharedBackbonePolicyCfg` (shared_backbone=True, static min-max norm, proprio_hist_dim=120),
  `ALBCHardDRSharedBackboneRunnerCfg`
- `__init__.py`: Registered `Isaac-Constrained-ALBC-HardDR-SharedBackbone-v0`

### Changed
- `scripts/analysis/common.py`: Added `_build_constrained_albc_23d_sweep()` for 23D
  privileged obs z sweep. Fixed activation detection: `input_dim >= 23` -> softsign
  (was `>= 28`, missed 23D/27D). Added `enc_obs_lower`/`enc_obs_upper` params to
  `build_sweep_params_from_checkpoint()`.
- `scripts/analysis/encoder_z_sweep.py`: Rewritten to support static min-max
  normalization. Added `NormMode` dataclass, `load_encoder()` now detects and uses
  static bounds from checkpoint (`_enc_obs_lower`/`_enc_obs_upper` or top-level keys).

### Experimental Results

Training (shared backbone, 342 iters, PPO, HardDR):

| Metric | Shared Backbone | Hist-Only (500 iters) | Delta |
|--------|----------------|----------------------|-------|
| Roll | 8.56 deg | 8.74 deg | -0.18 |
| Pitch | 9.39 deg | 6.54 deg | +2.85 |
| Reward | -13.60 | -10.75 | -2.85 |
| noise_std | 0.21 | 0.15 | +0.06 |

Encoder z sweep (model_350):
- 75/299 active param-z pairs (range > 0.05)
- 0/13 saturated dims
- Top responsive: body_mass (10/13 active), main_vol (8/13), main_CoG_z (8/13)
- z at nominal: 12/13 dims near |z| > 0.7 (boundary bias), only z_11 (-0.17)
  has full dynamic range. Effective encoder capacity ~2-3 dimensions out of 13.

### Notes
- Training is stable (no KL death, no noise_std explosion) -- first successful
  online encoder training since ablation Steps 5a-5b
- Encoder IS learning domain info (z responds to DR parameters), but most z
  dimensions are near softsign boundary, reducing effective capacity
- Pitch 2.85 deg worse than hist-only, possibly due to history dimension gap
  (120D vs 240D) and/or symmetric critic limitation
- Offline encoder z sweep comparison was not properly validated -- inline analysis
  had softsign not applied, producing incorrect z ranges. Needs re-run with
  corrected `encoder_z_sweep.py` for fair comparison.

---

## [2026-03-30] Frozen Encoder: Normalization Mismatch Fix + z_init_scale Experiment

### Context

Encoder z sweep analysis revealed the frozen encoder was producing constant (saturated)
z output -- 13/13 dimensions pinned at |z| > 0.999 regardless of DR parameter variation.
Root cause: the offline encoder was trained WITH static min-max normalization
(`(2x - upper - lower) / (upper - lower)` -> [-1, 1]), but the frozen encoder deployment
did not load these bounds from the checkpoint. Raw privileged obs (body_mass~9,
stiffness~80, water_density~1010) caused extreme pre-activation values, saturating softsign.

The fix auto-loads normalization bounds from the offline encoder checkpoint via
`register_buffer()` in `_load_pretrained_encoder()`. After fix: 0/13 saturated dims,
116/299 active param-z pairs (vs 0/299 before).

Additionally, `z_init_scale` was changed from 0.01 to 1.0. The 0.01 scaling was designed
for hist-only warm-start (prevent z from disrupting pre-trained actor) but is
counterproductive when training from scratch -- it forces the actor to spend iterations
re-learning to upweight z.

### Fixed
- `encoder/actor_critic_frozen_encoder.py`: `_load_pretrained_encoder()` now auto-loads
  `enc_obs_lower`/`enc_obs_upper` from checkpoint and registers buffers + sets
  `_has_static_enc_norm=True`, even when base class was initialized without bounds.

### Changed
- `agents/rsl_rl_ppo_cfg.py`: `_FrozenEncoderPolicyCfg.z_init_scale` 0.01 -> 1.0

### Experimental Results

Encoder z sweep (offline encoder checkpoint):

| Condition | Active z-param pairs | Saturated dims |
|-----------|---------------------|----------------|
| With static normalization | 116/299 | 0/13 |
| Without normalization (bug) | 0/299 | 13/13 |

Training comparison (500 iterations, last 50 avg):

| Run | Roll (deg) | Pitch (deg) | Reward | Terminations |
|-----|-----------|------------|--------|-------------|
| Hist-Only baseline | **9.78** | **7.41** | **-12.18** | 4.0% |
| Frozen(norm bug) | 9.77 | 7.51 | -13.29 | 2.9% |
| Frozen(z=0.01, norm fix) | 10.27 | 7.97 | -13.76 | 3.3% |
| Frozen(z=1.0, norm fix) | 9.85 | 7.89 | -14.64 | 3.3% |

z_init_scale=1.0 improved roll by 0.41 deg over z=0.01 and stabilized convergence slope
(roll: +0.003/iter vs +0.007/iter). However, frozen encoder still does not beat hist-only
on attitude accuracy. Encoder z_std=0.74 (healthy), z_mean=-0.16 (centered) -- encoder
itself is functioning correctly.

### Notes
- The normalization bug affected ALL previous frozen encoder experiments (2026-03-29 ~
  2026-03-30). The encoder was always producing constant output.
- Despite the fix, frozen encoder does not yet outperform hist-only. Possible causes:
  (1) offline encoder trained to predict V_critic, not attitude error directly;
  (2) 240D history already encodes sufficient dynamics info, making z redundant;
  (3) actor needs warm-start from hist-only to leverage z effectively.
- Analysis plots saved to `logs/offline_encoder/encoder_analysis/`.

---

## [2026-03-30] Constrained ALBC Codebase Refactoring

### Context

Ablation study (Steps 0-19, 20+ experiments) accumulated 30 debug tasks, 33 RunnerCfgs,
12 EnvCfgs, and 4 ablation-only parameters. With ablation conclusions preserved in
`encoder_ablation.md`, removed all debug/ablation code, keeping only 3 production tasks.

### Changed
- `agents/rsl_rl_ppo_cfg.py`: 1625 -> 336 lines. Removed 30+ ablation RunnerCfgs.
  Renamed `_DebugPolicyCfg` -> `_HistOnlyPolicyCfg`. Removed `noise_std_type` from
  `_EncoderPolicyCfg`. Fixed pre-existing double `@configclass` on
  `_FrozenEncoderAlgorithmCfg`.
- `config.py`: 529 -> 410 lines. Removed 8 debug/ablation EnvCfg classes.
- `encoder/actor_critic_encoder.py`: 399 -> 357 lines. Removed ablation parameters
  (`noise_std_type`, `clamp_actions`, `symmetric_critic`, `z_bounds_coef`,
  `z_bounds_soft_bound`). Deleted `z_bounds_loss()` method. Hardcoded log_std,
  no-clamp, asymmetric critic.
- `__init__.py`: 379 -> 60 lines. Removed 30 `gym.register()` blocks.
- `encoder/__init__.py`: Removed `ActorCriticConstrained` export.

### Removed
- `encoder/actor_critic_constrained.py`: 43-line Step 3 ablation-only class deleted.
- 30 debug task registrations (e.g., `Isaac-Constrained-ALBC-Debug-*`)

### Notes
- Total ~1812 lines removed across 5 files (1 deleted, 4 rewritten).
- 3 production tasks retained: Encoder-v0, HardDR-HistOnly-v0, HardDR-FrozenEncoder-v0.
- Checkpoint backward compatibility maintained via `load_state_dict(..., strict=False)`
  and `**kwargs` catch in `ActorCriticEncoder.__init__`.

---

## [2026-03-30] Frozen Encoder: Three Critical Fixes

### Context

Frozen encoder fine-tuning (offline pipeline Step 3) had noise_std explosion preventing
any learning. Systematic investigation found three independent bugs.

**Bug 1: `_normalize_storage_values()` overwrote normalized advantages.**
`storage.compute_returns()` normalizes advantages, then `_normalize_storage_values()`
recomputed advantages as `returns_norm - values_norm`, introducing bias (mean ~-0.66).
Surrogate loss ~1.0 at iter 0 (normal: ~0.002) drove immediate noise_std explosion.

**Bug 2: Critic received less information than actor.**
`_get_critic_obs()` returned `cat([o_t, p_t])` = 37D while actor received
`cat([o_t, hist, z])` = 267D. Critic was blind to 240D of proprioceptive history.
All previous encoder experiments (Steps 4-19) were affected.

**Bug 3: Missing denormalization during rollout (HORA mismatch).**
HORA denormalizes critic output during rollout so GAE operates on raw-scale values.
Our implementation stored normalized values directly.

### Fixed
- `runners/constraint_encoder_runner.py`: Removed advantages recomputation in
  `_normalize_storage_values()`
- `encoder/actor_critic_encoder.py`: `_get_critic_obs()` now returns
  `cat([o_t, hist, p_t])` = 277D. `num_critic_obs` includes `proprio_hist_dim`.
- `runners/constraint_encoder_runner.py`: HORA-style value normalization --
  denormalize stored values before GAE, denormalize last_values for bootstrap,
  then normalize values/returns after GAE for critic targets.

### Changed
- `agents/rsl_rl_ppo_cfg.py`: `ALBCHardDRFrozenEncoderRunnerCfg` obs_groups
  critic now includes `proprio_hist`. Added `hist_only_checkpoint` field.
- `encoder/actor_critic_frozen_encoder.py`: `load_history_only_weights()` now
  copies `log_std`/`std` parameter from hist_only checkpoint.

### Experimental Results

| Metric | Frozen Encoder (499 iters) | Hist Only | Delta |
|--------|---------------------------|-----------|-------|
| Best roll | 6.9 deg | 7.0 deg | -0.1 |
| Best pitch | 5.7 deg | 5.6 deg | +0.1 |
| Final roll | 11.8 deg | 8.7 deg | +3.1 |
| Final pitch | 6.9 deg | 6.5 deg | +0.4 |
| noise_std | 0.065 | 0.153 | -0.088 |

Training stable. Best performance comparable but encoder z not yet providing measurable
advantage over history-only baseline. Final roll has more variance (7-12 deg oscillation).

### Notes
- All previous encoder experiments (Steps 4-19) had the critic bug (37D instead of 277D).
  The "encoder destabilizes training" conclusion may need revision.
- Offline encoder quality verified: z explains 70.3% additional V_critic variance
  (R^2: 0.088 -> 0.791).
- Untested: actor warm-start, encoder unfreezing after convergence, online encoder
  with fixed critic.

---

## [2026-03-29] Offline Encoder Pipeline

### Context

After 15+ online encoder experiments failed (see
[encoder_ablation.md](experiments/encoder_ablation.md)),
pivoted to offline training: (1) collect rollouts from trained history-only policy,
(2) train encoder supervised with value prediction bottleneck, (3) fine-tune actor
with frozen encoder.

Root cause of online failure: `sample().clamp(-1,1)` in `ActorCriticEncoder.act()`
concentrates actions at boundaries, amplifying KL 100x. Secondary: env-level clamp
positive feedback on noise_std when encoder makes advantages noisy.

### Added
- `scripts/analysis/collect_rollouts.py`: Rollout data collection from trained policy.
  Collects (o_t, privileged, V_critic) per step.
- `scripts/analysis/train_offline_encoder.py`: Supervised encoder training.
  Architecture: p_t(23D)->MLP[256,128,64]->softsign->z(13D),
  value head: cat([o_t, z])->Linear->V_hat, loss=MSE(V_hat, V_critic).
- `encoder/actor_critic_frozen_encoder.py`: `ActorCriticFrozenEncoder` -- encoder
  frozen (requires_grad=False), pre-trained weights loaded, z-related actor weights
  init to near-zero (scale=0.01). `load_history_only_weights()` for warm-start.
- `config.py`: `ALBCHardDRFrozenEncoderEnvCfg` -- Hard DR + state_space=23 +
  history(30, stride=1).
- `agents/rsl_rl_ppo_cfg.py`: `_FrozenEncoderAlgorithmCfg` (use_encoder_update=False),
  `_FrozenEncoderPolicyCfg`, `ALBCHardDRFrozenEncoderRunnerCfg`.
- `__init__.py`: Registered `Isaac-Constrained-ALBC-HardDR-FrozenEncoder-v0`.

### Experimental Results
- Rollout collection: 207,360 transitions (50 episodes, 512 envs).
- Offline encoder: val_loss 13.67->2.43, z_std=0.315 (non-trivial output).

---

## [2026-03-29] Hard DR Environment

### Context

History-only baseline achieves 3.0 deg with standard DR -- encoder has no gap to close.
Created "hard DR" environment where history-only degrades to ~10 deg, providing headroom
for encoder benefit.

### Added
- `config.py`: `HardDomainRandomizationCfg` -- aggressive DR:
  added_mass +-40% (was +-15%), body_mass +-25% (was +-10%), volume +-25% (was +-10%),
  CoG/CoB offsets doubled, inertia (0.5, 1.8), payload 0-2kg
- `config.py`: `ALBCHardDRHistOnlyEnvCfg` -- hard DR + history(30) + ocean current
- `agents/rsl_rl_ppo_cfg.py`: `ALBCHardDRHistOnlyRunnerCfg`
- `__init__.py`: Registered `Isaac-Constrained-ALBC-HardDR-HistOnly-v0`

---

## [2026-03-29] Encoder Ablation: Root Cause Found

### Summary

20+ experiments (Steps 0-19) systematically isolated why encoder destabilizes training.
Full details: [encoder_ablation.md](experiments/encoder_ablation.md).

**Root cause:** `sample().clamp(-1,1)` in `ActorCriticEncoder.act()`.
Actions pile at boundaries -> sharp log_prob gradients -> KL 100x amplification ->
adaptive LR death. Removing clamp (Step 17) reduced encoder KL from 0.88 to 0.003 --
but noise_std exploded due to env-level clamp positive feedback loop.

10 hypotheses tested and disproved (EmpiricalNorm, encoder gradient, encoder freeze,
init LR, update path, history, critic asymmetry, normalization, std type, action clamp
alone). Online co-training structurally unstable in 2D action space.

### Added
- `encoder/actor_critic_encoder.py`: `noise_std_type`, `clamp_actions`,
  `symmetric_critic` params. Static min-max normalization support.
  HORA-style `actor_obs_normalizer` (excludes z).
- `config.py`: `proprio_history_stride` field. Debug/ablation env configs (Steps 0-19).
- `agents/rsl_rl_ppo_cfg.py`: 15+ runner configs for ablation steps.
- `albc_env.py`: Strided proprioceptive history recording.
- `runners/constraint_encoder_runner.py`: `normalize_value` flag.
- `__init__.py`: 15+ debug task registrations.

### Changed (rsl_rl/algorithms/ppo.py -- external, not git-tracked)
- Added `use_encoder_update`, `reward_scale`, `min_lr`, `max_lr`, `encoder_grad_scale`.
- **Needs reapply on container rebuild.**

---

## [2026-03-27] Action Parameterization and Reward Tuning

### Summary

Three fixes: (1) torque constraint measured unbounded PD internal computation instead of
actual motor output (100% violated, unsatisfiable), (2) Gaussian noise in absolute joint
targets created 115 deg/step jitter (91% effort saturation), switched to delta action,
(3) tuned delta_scale and reward weights.

### Fixed
- `mdp/constraints.py`: `torque_limit_cost()` uses `applied_torque` (post-clamp, max
  9.5 Nm) instead of `computed_torque` (PD internal, 326-554 Nm)

### Changed
- `config.py`: `action_scale: float = pi` -> `delta_scale: float = 0.08`
- `albc_env.py`: `_apply_joint_pd_action()` from absolute to delta accumulation
  (`q_des += delta_scale * a_t`, clamped to joint limits)
- `config.py`: `k_tau` -0.001 -> -0.01, `k_s` -0.05 -> -0.2

### Experimental Results (delta action first run, 139 iters)

| Category | Metric | Absolute | Delta |
|----------|--------|----------|-------|
| Dynamics | effort_saturation | 91% | 2.2% |
| | torque cost_return | 92 | 4.5 (within budget) |
| Attitude | Roll / Pitch | 17/13 deg | 21.6/18.8 deg |

Delta action solved dynamics (effort/torque within limits) at the cost of slower attitude
convergence (delta_scale bandwidth). Tuned from 0.05 to 0.08 (0.39s to 90 deg).

---

## [2026-03-27] TRPO+IPO Algorithm Fixes (NORBC Paper Alignment)

### Summary

Six structural fixes aligning ConstraintTRPO with NORBC paper (Muller et al., ICML 2025).
Combined effect: reward -78.80 -> -37.36 (2x), roll 29.2 -> 18.0 deg, pitch 26.5 ->
11.9 deg, z saturation eliminated ([-0.99,0.99] -> [-0.53,0.40]).

### Fixes

1. **Line search logging artifact**: `surrogate()` closure overwrites monitoring vars on
   each backtracking attempt. Fixed: recalculate with reverted params after failure.

2. **Cost critic d_k^2 normalization**: Non-standard, ineffective (yaw_vel contributed
   98.6% of loss). Changed to plain MSE (OmniSafe/CPO convention).

3. **Encoder LS gating removed**: Encoder received zero gradient on line search failure,
   creating starvation loop. No precedent in HORA/RMA/RSL-RL.

4. **Encoder integrated into TRPO trust region**: Separate Adam encoder update destroyed
   trust region (post-encoder KL: 27.6x budget avg, max 1153.4x). Moved encoder params
   into TRPO CG + line search (matching NORBC joint training).

5. **Missing 1/(1-gamma) in IPO barrier**: With cost_gamma=0.99, factor=100. Barrier
   estimated margin change 100x too small without this factor.

6. **Per-constraint cost advantage standardization**: Restored NORBC Sec IV-B
   `(A_Ck - mean) / (std + 1e-8)`. Raw advantages near-zero when deeply infeasible.

### Changed
- `algorithms/constraint_trpo.py`: Cost value loss plain MSE, LS gating removed,
  encoder params in `_policy_params`, 1/(1-gamma) factor added, per-constraint
  standardization restored
- `agents/rsl_rl_ppo_cfg.py`: `barrier_alpha` 0.02 -> 0.05, removed
  `num_encoder_epochs`/`encoder_lr`

### Removed
- `algorithms/constraint_trpo.py`: `_update_encoder()`, `encoder_optimizer`,
  `_encoder_params`, `_has_encoder_params`, `_last_pre_encoder_kl`
